"""This script trains the DCT Pose Transformer model for human motion prediction.

Model and loss functions are located at src/models/dct_pose_transformer.py.

The training has multiple stages:
  1. Pose only: train the DCT pose transformer to only perform the human motion prediction,
      without any uncertainty estimation. Uses ADAM with initial learning rate 1e-4 and weight decay of 1e-6.
      N_epochs = 50. Clip max grad norm = 0.01.
      Uses pose_prediction_loss as loss function.
  2. Uncertainty head only: train the uncertainty head while keeping the transformer weights fixed.
      Detach the transformer output features from the computational graph before passing them
      to the uncertainty head, effectively treating these features as constant inputs.
      Loss = gaussian_nll_from_cholesky + lambda * pose_prediction_loss, where
      lambda starts at 1 and is gradually decreased to 0 within M=5 epochs.
  3. End-to-end finetuning: train the whole model end-to-end, with both the pose prediction loss and
      the uncertainty loss.
      Loss = gaussian_nll_from_cholesky + pose_prediction_loss.
"""

import os
import json
import argparse
from typing import Any, Dict, Optional, Tuple
from functools import partial
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import orbax.checkpoint
import numpy as np
import wandb

from src.models.dct_pose_transformer import (
    DCTPoseTransformer,
    pose_prediction_loss,
    gaussian_nll_from_cholesky,
)
from src.datasets.wrapper import dataloader_from_string
from human_pose_pipeline.motion_prediction.h36m_settings import (
    N_JOINTS,
    INPUT_HORIZON_LENGTH,
    PREDICTION_HORIZON_LENGTH
)
from human_pose_pipeline.utils.eval_utils import evaluate_pose_prediction_scores_jax, evaluate_uncertainty_coverage_jax

# Much slower and does not make a difference (at least for stage 1)
# jax.config.update("jax_enable_x64", True)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class TrainingConfig:
    """Configuration for training the DCT Pose Transformer."""

    def __init__(
        self,
        # Model hyperparameters
        input_dim: int = N_JOINTS * 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int = INPUT_HORIZON_LENGTH,
        seq_len_output: int = PREDICTION_HORIZON_LENGTH,
        unit_conversion: float = 1000.0,
        reduced_size: bool = False,  # Should always be False.

        # Training hyperparameters
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        max_grad_norm: float = 0.01,

        # Learning rate scheduling
        use_lr_schedule: bool = True,
        lr_schedule_type: str = "cosine",  # "cosine", "exponential", "constant"
        lr_warmup_epochs: int = 5,
        lr_min_factor: float = 0.01,  # Minimum LR as fraction of initial LR

        # Stage 1: Pose only training
        stage1_epochs: int = 50,

        # Stage 2: Uncertainty head training
        stage2_epochs: int = 20,

        # Stage 3: End-to-end training
        stage3_epochs: int = 30,

        # Data settings
        data_path: str = "../datasets",
        seed: int = 420,

        # Experiment tracking
        run_id: Optional[str] = None,
        wandb_project: str = "motion-prediction",
        wandb_entity: Optional[str] = None,
        use_wandb: bool = True,
    ):
        # Model hyperparameters
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.seq_len_output = seq_len_output
        self.unit_conversion = unit_conversion
        self.reduced_size = reduced_size

        # Training hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        # Learning rate scheduling
        self.use_lr_schedule = use_lr_schedule
        self.lr_schedule_type = lr_schedule_type
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_factor = lr_min_factor

        # Stage epochs
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_epochs = stage3_epochs

        # Data settings
        self.data_path = data_path
        self.seed = seed

        # Experiment tracking
        self.run_id = run_id
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.use_wandb = use_wandb

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return vars(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Load configuration from dictionary."""
        return cls(**config_dict)

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_lr_schedule_for_stage(
    config: TrainingConfig,
    steps_per_epoch: int,
    stage_epochs: int,
    learning_rate: float
):
    """Create learning rate schedule for a specific stage.

    Args:
        config: Training configuration
        steps_per_epoch: Number of training steps per epoch
        stage_epochs: Number of epochs for this stage
        learning_rate: Initial learning rate for this stage

    Returns:
        tuple: (lr_schedule, lr_fn) where lr_schedule is for optax and lr_fn computes current LR
    """
    if not config.use_lr_schedule:
        return learning_rate, lambda step: learning_rate

    assert config.lr_schedule_type == "cosine" or config.lr_schedule_type == "exponential"

    # Total steps for this stage
    total_steps = stage_epochs * steps_per_epoch
    warmup_steps = min(config.lr_warmup_epochs * steps_per_epoch, total_steps // 2)  # Cap warmup at half the stage

    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps
    )
    if config.lr_schedule_type == "cosine":
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=total_steps - warmup_steps,
            alpha=config.lr_min_factor  # Minimum learning rate as fraction of initial
        )
    elif config.lr_schedule_type == "exponential":
        # Calculate decay rate to reach lr_min_factor at the end
        decay_rate = (config.lr_min_factor) ** (1.0 / (total_steps - warmup_steps))
        lr_schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=1,
            decay_rate=decay_rate
        )
    full_schedule = optax.join_schedules(
        schedules=[warmup_schedule, lr_schedule],
        boundaries=[warmup_steps]
    )
    # Return both the schedule and a function to get the current LR
    return full_schedule, full_schedule


def create_train_state(
    rng: jax.Array,
    config: TrainingConfig,
    steps_per_epoch: int,
    stage_epochs: int,
    learning_rate: Optional[float] = None,
):
    """Create initial training state for a stage.

    Args:
        rng: Random key for initialization
        config: Training configuration
        steps_per_epoch: Number of steps per epoch
        stage_epochs: Number of epochs for the current stage
        learning_rate: Learning rate (defaults to config.learning_rate)

    Returns:
        tuple: (state, lr_fn) where state is TrainState and lr_fn computes current LR
    """
    if learning_rate is None:
        learning_rate = config.learning_rate

    # Initialize model
    model = DCTPoseTransformer(
        input_dim=config.input_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        seq_len=config.seq_len,
        seq_len_output=config.seq_len_output,
        unit_conversion=config.unit_conversion,
        reduced_size=config.reduced_size,
    )

    # Initialize parameters
    dummy_input = jnp.ones((1, config.seq_len, config.input_dim))
    variables = model.init(rng, dummy_input, train=True)
    params = variables['params']

    # Create learning rate schedule for this stage
    lr_schedule, lr_fn = get_lr_schedule_for_stage(config, steps_per_epoch, stage_epochs, learning_rate)

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(lr_schedule, weight_decay=config.weight_decay),
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # Store the LR function in the state for logging purposes
    return state, lr_fn


def update_optimizer_for_stage(
    state: TrainState,
    config: TrainingConfig,
    steps_per_epoch: int,
    stage_epochs: int,
    learning_rate: Optional[float] = None,
):
    """Update optimizer with a new learning rate schedule for a new stage.

    Args:
        state: Current training state
        config: Training configuration
        steps_per_epoch: Number of steps per epoch
        stage_epochs: Number of epochs for the new stage
        learning_rate: Learning rate (defaults to config.learning_rate)

    Returns:
        tuple: (new_state, lr_fn) with updated optimizer
    """
    if learning_rate is None:
        learning_rate = config.learning_rate

    # Create new learning rate schedule for this stage
    lr_schedule, lr_fn = get_lr_schedule_for_stage(config, steps_per_epoch, stage_epochs, learning_rate)

    # Create new optimizer with the new schedule
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(lr_schedule, weight_decay=config.weight_decay),
    )

    # Create new state with the same parameters but new optimizer
    new_state = TrainState.create(
        apply_fn=state.apply_fn,
        params=state.params,
        tx=optimizer,
    )

    return new_state, lr_fn


@partial(jax.jit, static_argnames=['use_uncertainty_head', 'lambda_weight', 'freeze_backbone'])
def train_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    use_uncertainty_head: bool,
    lambda_weight: float,
    freeze_backbone: bool = False,
) -> Tuple[TrainState, Dict[str, float]]:
    """Training step with optional backbone freezing.

    Args:
        state: Training state
        batch: Input and target batch
        use_uncertainty_head: Whether to use uncertainty head in loss
        lambda_weight: Weight for pose loss when using uncertainty head
        freeze_backbone: If True, only train uncertainty_head parameters (Stage 2)

    Returns:
        Updated state and metrics
    """
    input_pose, target_pose = batch

    def loss_fn(params):
        pred_poses, (cov, L) = state.apply_fn({'params': params}, input_pose, train=True)

        pose_loss = pose_prediction_loss(pred_poses, target_pose)

        if use_uncertainty_head:
            # Reshape target to match covariance shape [B, T, J, 3]
            batch_size = target_pose.shape[0]
            target_reshaped = target_pose.reshape(batch_size, -1, N_JOINTS, 3)
            pred_reshaped = pred_poses.reshape(batch_size, -1, N_JOINTS, 3)
            nll_loss = gaussian_nll_from_cholesky(pred_reshaped, target_reshaped, L)
            total_loss = nll_loss + lambda_weight * pose_loss
        else:
            nll_loss = 0.0
            total_loss = pose_loss

        return total_loss, {
            'loss': total_loss,
            'nll_loss': nll_loss,
            'pose_loss': pose_loss,
            'lambda': lambda_weight,
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)

    # If freeze_backbone is True, zero out gradients for all parameters except uncertainty_head
    if freeze_backbone:
        def freeze_grads(path, grad):
            # Only allow gradients for uncertainty_head parameters
            if 'uncertainty_head' in path:
                return grad
            else:
                # Zero out gradients for frozen parameters
                return jax.tree.map(jnp.zeros_like, grad)

        # Apply the freezing mask to gradients
        grads = jax.tree_util.tree_map_with_path(
            lambda path, grad: freeze_grads('/'.join(str(k.key) for k in path), grad),
            grads
        )

    state = state.apply_gradients(grads=grads)

    return state, metrics


@jax.jit
def eval_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Evaluation step."""
    input_pose, target_pose = batch

    pred_poses, (cov, L) = state.apply_fn({'params': state.params}, input_pose, train=False)

    # Reshape target to match covariance shape [B, T, J, 3]
    batch_size = target_pose.shape[0]
    target_reshaped = target_pose.reshape(batch_size, -1, N_JOINTS, 3)
    pred_reshaped = pred_poses.reshape(batch_size, -1, N_JOINTS, 3)

    nll_loss = gaussian_nll_from_cholesky(pred_reshaped, target_reshaped, L)
    pose_loss = pose_prediction_loss(pred_poses, target_pose)

    # Compute MPJPE
    mpjpe, std, per_time_errors, _, _, _ = \
        evaluate_pose_prediction_scores_jax(pred_reshaped, target_reshaped)

    uncertainty_coverage = evaluate_uncertainty_coverage_jax(
        pred_poses=pred_reshaped,
        true_poses=target_reshaped,
        L=L,
        std_multipliers=[1, 2, 3, 4]
    )

    return {
        'nll_loss': nll_loss,
        'pose_loss': pose_loss,
        'mpjpe': mpjpe,
        'mpjpe_std': std,
        'mpjpe_time_80ms': per_time_errors[1],
        'mpjpe_time_160ms': per_time_errors[3],
        'mpjpe_time_240ms': per_time_errors[5],
        'mpjpe_time_320ms': per_time_errors[7],
        'mpjpe_time_400ms': per_time_errors[9],
        'uncertainty_coverage std=1': uncertainty_coverage[0],
        'uncertainty_coverage std=2': uncertainty_coverage[1],
        'uncertainty_coverage std=3': uncertainty_coverage[2],
        'uncertainty_coverage std=4': uncertainty_coverage[3]
    }


def train_epoch(
    state: TrainState,
    train_loader,
    stage: int,
    epoch: int,
    config: TrainingConfig,
    lr_fn,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Train for one epoch."""
    epoch_metrics = []

    for batch in tqdm(train_loader, "Training Epoch {}".format(epoch + 1)):
        # Convert to JAX arrays
        input_pose = jnp.array(batch[0], dtype=jnp.float32)
        target_pose = jnp.array(batch[1], dtype=jnp.float32)

        # Select training step based on stage
        if stage == 1:
            # Stage 1: Train only pose prediction (no uncertainty head)
            state, metrics = train_step(
                state=state,
                batch=(input_pose, target_pose),
                use_uncertainty_head=False,
                lambda_weight=1.0,  # Not used here
                freeze_backbone=False
            )
        elif stage == 2:
            # Stage 2: Train ONLY uncertainty head (freeze backbone)
            # Calculate lambda decay
            lambda_weight = 0.0
            state, metrics = train_step(
                state=state,
                batch=(input_pose, target_pose),
                use_uncertainty_head=True,
                lambda_weight=lambda_weight,
                freeze_backbone=True  # FREEZE all params except uncertainty_head
            )
        elif stage == 3:
            # Stage 3: Train entire model end-to-end
            state, metrics = train_step(
                state=state,
                batch=(input_pose, target_pose),
                use_uncertainty_head=True,
                lambda_weight=1.0,
                freeze_backbone=False
            )

        epoch_metrics.append(metrics)

    # Average metrics over the epoch
    avg_metrics = {}
    for key in epoch_metrics[0].keys():
        avg_metrics["train/" + key] = float(jnp.mean(jnp.array([m[key] for m in epoch_metrics])))

    # Get current learning rate from the schedule function
    current_step = int(state.step)
    current_lr = float(lr_fn(current_step))
    avg_metrics["learning_rate"] = current_lr

    return state, avg_metrics


def evaluate(
    state: TrainState,
    eval_loader,
    epoch: int
) -> Dict[str, float]:
    """Evaluate the model."""
    eval_metrics = []

    for batch in tqdm(eval_loader, "Eval Epoch {}".format(epoch + 1)):
        # Convert to JAX arrays
        input_pose = jnp.array(batch[0], dtype=jnp.float32)
        target_pose = jnp.array(batch[1], dtype=jnp.float32)

        metrics = eval_step(state, (input_pose, target_pose))
        eval_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in eval_metrics[0].keys():
        avg_metrics["eval/" + key] = float(jnp.mean(jnp.array([m[key] for m in eval_metrics])))

    return avg_metrics


def save_checkpoint(
    state: TrainState,
    checkpoint_dir: str,
    step: int,
    stage: int,
    keep: int = 3,
):
    """Save checkpoint using Orbax.

    Args:
        state: Training state to save
        checkpoint_dir: Base checkpoint directory
        step: Step number for checkpoint name
        stage: Training stage (1, 2, or 3)
        keep: Number of checkpoints to keep (not currently used)
    """
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(state)

    # Create stage-specific subdirectory
    stage_dir = os.path.join(checkpoint_dir, f"stage_{stage}")
    os.makedirs(stage_dir, exist_ok=True)

    checkpoint_path = os.path.join(stage_dir, f"checkpoint_{step}")
    checkpointer.save(checkpoint_path, state, save_args=save_args, force=True)

    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    state: TrainState,
    checkpoint_dir: str,
    step: Optional[int] = None,
    stage: Optional[int] = None,
) -> TrainState:
    """Load checkpoint using Orbax.

    Args:
        state: Training state template for restoration
        checkpoint_dir: Base checkpoint directory
        step: Specific step to load (if None, finds latest)
        stage: Specific stage to load from (if None, searches all stages for latest)

    Returns:
        Restored training state
    """
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    if stage is None:
        # Search all stage directories for the latest checkpoint
        latest_step = -1
        latest_stage = None

        for s in [1, 2, 3]:
            stage_dir = os.path.join(checkpoint_dir, f"stage_{s}")
            if not os.path.exists(stage_dir):
                continue

            checkpoints = [d for d in os.listdir(stage_dir) if d.startswith("checkpoint_")]
            if checkpoints:
                steps = [int(c.split("_")[1]) for c in checkpoints]
                max_step = max(steps)
                if max_step > latest_step:
                    latest_step = max_step
                    latest_stage = s

        if latest_stage is None:
            raise ValueError(f"No checkpoints found in any stage subdirectory of {checkpoint_dir}")

        stage = latest_stage
        step = latest_step
        print(f"Found latest checkpoint in stage {stage}, step {step}")
    else:
        # Load from specific stage
        stage_dir = os.path.join(checkpoint_dir, f"stage_{stage}")
        if not os.path.exists(stage_dir):
            raise ValueError(f"Stage directory {stage_dir} does not exist")

        if step is None:
            # Find latest checkpoint in this stage
            checkpoints = [d for d in os.listdir(stage_dir) if d.startswith("checkpoint_")]
            if not checkpoints:
                raise ValueError(f"No checkpoints found in {stage_dir}")
            steps = [int(c.split("_")[1]) for c in checkpoints]
            step = max(steps)

    checkpoint_path = os.path.join(checkpoint_dir, f"stage_{stage}", f"checkpoint_{step}")
    restored_state = checkpointer.restore(checkpoint_path, item=state)

    print(f"Loaded checkpoint from {checkpoint_path}")
    return restored_state


def verify_frozen_params(state_before: TrainState, state_after: TrainState, stage: int):
    """Verify that frozen parameters didn't change during training.

    Args:
        state_before: Training state before update
        state_after: Training state after update
        stage: Current training stage
    """
    def check_params(path, before_val, after_val):
        path_str = '/'.join(str(k.key) for k in path)
        changed = not jnp.allclose(before_val, after_val, rtol=1e-6)

        if stage == 2:  # Stage 2: only uncertainty_head should change
            if 'uncertainty_head' in path_str:
                if not changed:
                    print(f"  ⚠️  WARNING: {path_str} didn't change (should be trainable)")
            else:
                if changed:
                    print(f"  ❌ ERROR: {path_str} changed (should be frozen!)")

        return changed

    # Compare parameters
    jax.tree_util.tree_map_with_path(
        lambda path, before, after: check_params(path, before, after),
        state_before.params,
        state_after.params
    )


def train_stage(
    state: TrainState,
    train_loader,
    valid_loader,
    stage: int,
    n_epochs: int,
    config: TrainingConfig,
    checkpoint_dir: str,
    lr_fn,
    start_epoch: int = 0,
    verify_freezing: bool = False,
) -> TrainState:
    """Train a single stage.

    Args:
        verify_freezing: If True, verify parameter freezing after first batch (for debugging)
    """
    stage_names = {1: "Pose Only", 2: "Uncertainty Head (Frozen Backbone)", 3: "End-to-End"}
    freeze_info = {
        1: "Training: All parameters",
        2: "Training: ONLY uncertainty_head | Frozen: transformer, decoders, embeddings",
        3: "Training: All parameters"
    }

    print(f"\n{'=' * 60}")
    print(f"Stage {stage}: {stage_names[stage]}")
    print(f"{freeze_info[stage]}")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, n_epochs):
        # Train
        state, train_metrics = train_epoch(
            state, train_loader, stage, epoch, config, lr_fn
        )

        # Evaluate
        eval_metrics = evaluate(state, valid_loader, epoch)

        # Combine metrics
        all_metrics = {**train_metrics, **eval_metrics, 'epoch': epoch, 'stage': stage}

        # Log to wandb
        if config.use_wandb:
            wandb.log(all_metrics)

        # Print progress
        print(f"Epoch {epoch + 1}/{n_epochs} - " + " - ".join(
            [f"{k}: {v:.6f}" for k, v in all_metrics.items() if k not in ['epoch', 'stage']])
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(state, checkpoint_dir, epoch + 1, stage)

    # Save final checkpoint for this stage
    save_checkpoint(state, checkpoint_dir, n_epochs, stage)

    return state


def main(args):
    """Main training function."""
    # Initialize wandb first to get run_id if not provided
    if args.use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.run_id,  # Will be None if not provided, wandb will generate one
            config=vars(args),
            resume="allow" if args.resume else False,
        )
        # Use wandb run id as the run_id
        run_id = wandb_run.id
        # Update the run name to match the id if it wasn't provided
        # if args.run_id is None:
        #     wandb_run.name = run_id
    else:
        # Generate a run_id if not provided and wandb is disabled
        if args.run_id is None:
            import datetime
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            run_id = args.run_id

    # Setup paths
    model_dir = os.path.join(root_dir, "human_pose_pipeline", "models", "motion_prediction", run_id)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    config_path = os.path.join(model_dir, "dct_pose_transformer_args.json")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load or create configuration
    if os.path.exists(config_path) and not args.new_config:
        print(f"Loading configuration from {config_path}")
        config = TrainingConfig.load(config_path)
    else:
        print("Creating new configuration")
        config = TrainingConfig(
            input_dim=N_JOINTS * 3,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            seq_len=args.seq_len,
            seq_len_output=args.seq_len_output,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            use_lr_schedule=args.use_lr_schedule,
            lr_schedule_type=args.lr_schedule_type,
            lr_warmup_epochs=args.lr_warmup_epochs,
            lr_min_factor=args.lr_min_factor,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            stage3_epochs=args.stage3_epochs,
            data_path=args.data_path,
            seed=args.seed,
            run_id=run_id,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            use_wandb=args.use_wandb,
        )
        config.save(config_path)
        print(f"Saved configuration to {config_path}")

    # Update wandb config with full configuration
    if config.use_wandb:
        wandb.config.update(config.to_dict(), allow_val_change=True)

    # Set random seeds
    np.random.seed(config.seed)
    rng = jax.random.PRNGKey(config.seed)

    # Load data
    print("Loading data...")
    dataset_name = "Human36mMotionDataset3D"
    train_loader, valid_loader, test_loader = dataloader_from_string(
        dataset_name,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        download=False,
        data_path=config.data_path,
    )

    # Calculate steps per epoch for learning rate scheduling
    steps_per_epoch = len(train_loader)
    print(f"Steps per epoch: {steps_per_epoch}")

    # Create or load training state
    print("Initializing model...")
    # Initialize with Stage 1 parameters
    state, lr_fn = create_train_state(rng, config, steps_per_epoch, config.stage1_epochs)

    if args.resume:
        try:
            load_stage = args.stage - 1 if args.stage > 1 else None
            state = load_checkpoint(state, checkpoint_dir, stage=load_stage)
            print("Resumed from checkpoint")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Starting from scratch")

    # Train stages: if stage 1 is done, proceed to stage 2, etc.
    if args.stage <= 1:
        print("\nStarting Stage 1: Pose Only Training")
        # Stage 1 already has the correct LR schedule from initialization
        state = train_stage(
            state, train_loader, valid_loader,
            stage=1,
            n_epochs=config.stage1_epochs,
            config=config,
            checkpoint_dir=checkpoint_dir,
            lr_fn=lr_fn,
        )

    if args.stage <= 2:
        print("\nStarting Stage 2: Uncertainty Head Training")
        # Create new optimizer with Stage 2 LR schedule
        print("Creating new LR schedule for Stage 2...")
        state, lr_fn = update_optimizer_for_stage(
            state, config, steps_per_epoch, config.stage2_epochs
        )
        state = train_stage(
            state, train_loader, valid_loader,
            stage=2,
            n_epochs=config.stage2_epochs,
            config=config,
            checkpoint_dir=checkpoint_dir,
            lr_fn=lr_fn,
        )

    if args.stage <= 3:
        print("\nStarting Stage 3: End-to-End Finetuning")
        # Create new optimizer with Stage 3 LR schedule
        print("Creating new LR schedule for Stage 3...")
        state, lr_fn = update_optimizer_for_stage(
            state, config, steps_per_epoch, config.stage3_epochs
        )
        state = train_stage(
            state, train_loader, valid_loader,
            stage=3,
            n_epochs=config.stage3_epochs,
            config=config,
            checkpoint_dir=checkpoint_dir,
            lr_fn=lr_fn,
        )

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate(state, test_loader, epoch=0)
    print("Test metrics:", test_metrics)

    if config.use_wandb:
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb.finish()

    print(f"\nTraining complete! Model saved to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCT Pose Transformer")

    # Run configuration
    parser.add_argument("--run_id", type=str, default=None,
                        help="Unique run identifier (defaults to wandb run id or timestamp)")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="Training stage to start from (1, 2, or 3)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--new_config", action="store_true",
                        help="Create new config even if one exists")

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--seq_len_output", type=int, default=10)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=0.01)

    # Learning rate scheduling
    parser.add_argument("--use_lr_schedule", action="store_true", default=True,
                        help="Use learning rate scheduling")
    parser.add_argument("--no_lr_schedule", dest="use_lr_schedule", action="store_false",
                        help="Disable learning rate scheduling")
    parser.add_argument("--lr_schedule_type", type=str, default="cosine",
                        choices=["cosine", "exponential", "constant"],
                        help="Type of LR schedule: cosine, exponential, or constant")
    parser.add_argument("--lr_warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs for learning rate")
    parser.add_argument("--lr_min_factor", type=float, default=0.01,
                        help="Minimum LR as fraction of initial LR (e.g., 0.01 = 1%)")

    # Stage epochs
    parser.add_argument("--stage1_epochs", type=int, default=50)
    parser.add_argument("--stage2_epochs", type=int, default=20)
    parser.add_argument("--stage3_epochs", type=int, default=30)

    # Data
    parser.add_argument("--data_path", type=str, default="../datasets")
    parser.add_argument("--seed", type=int, default=420)

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="motion-prediction")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")

    args = parser.parse_args()
    main(args)
