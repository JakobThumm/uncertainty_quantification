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
from typing import Any, Dict, Tuple
from functools import partial

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
from human_pose_pipeline.motion_prediction.h36m_settings import N_JOINTS


class TrainingConfig:
    """Configuration for training the DCT Pose Transformer."""

    def __init__(
        self,
        # Model hyperparameters
        input_dim: int = N_JOINTS * 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int = 50,
        seq_len_output: int = 10,
        unit_conversion: float = 1000.0,
        reduced_size: bool = False,

        # Training hyperparameters
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        max_grad_norm: float = 0.01,

        # Stage 1: Pose only training
        stage1_epochs: int = 50,

        # Stage 2: Uncertainty head training
        stage2_epochs: int = 20,
        stage2_lambda_decay_epochs: int = 5,

        # Stage 3: End-to-end training
        stage3_epochs: int = 30,

        # Data settings
        data_path: str = "../datasets",
        seed: int = 420,

        # Experiment tracking
        run_id: str = None,
        wandb_project: str = "motion-prediction",
        wandb_entity: str = None,
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

        # Stage epochs
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage2_lambda_decay_epochs = stage2_lambda_decay_epochs
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


def create_train_state(
    rng: jax.random.PRNGKey,
    config: TrainingConfig,
    learning_rate: float = None,
) -> TrainState:
    """Create initial training state."""
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

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(learning_rate, weight_decay=config.weight_decay),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@partial(jax.jit, static_argnames=['stage'])
def train_step_stage1(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[TrainState, Dict[str, float]]:
    """Training step for Stage 1: Pose only."""
    input_pose, target_pose = batch

    def loss_fn(params):
        pred_poses, _ = state.apply_fn({'params': params}, input_pose, train=True)
        loss = pose_prediction_loss(pred_poses, target_pose)
        return loss, {'loss': loss, 'pose_loss': loss}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, metrics


@partial(jax.jit, static_argnames=['lambda_weight'])
def train_step_stage2(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    lambda_weight: float,
) -> Tuple[TrainState, Dict[str, float]]:
    """Training step for Stage 2: Uncertainty head only.

    The transformer features are detached (stopped gradient) so only
    the uncertainty head parameters are updated.
    """
    input_pose, target_pose = batch

    def loss_fn(params):
        pred_poses, (cov, L) = state.apply_fn({'params': params}, input_pose, train=True)

        # Reshape target to match covariance shape [B, T, J, 3]
        batch_size = target_pose.shape[0]
        target_reshaped = target_pose.reshape(batch_size, -1, N_JOINTS, 3)
        pred_reshaped = pred_poses.reshape(batch_size, -1, N_JOINTS, 3)

        nll_loss = gaussian_nll_from_cholesky(target_reshaped, pred_reshaped, L)
        pose_loss = pose_prediction_loss(pred_poses, target_pose)

        total_loss = nll_loss + lambda_weight * pose_loss

        return total_loss, {
            'loss': total_loss,
            'nll_loss': nll_loss,
            'pose_loss': pose_loss,
            'lambda': lambda_weight,
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, metrics


@jax.jit
def train_step_stage3(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[TrainState, Dict[str, float]]:
    """Training step for Stage 3: End-to-end finetuning."""
    input_pose, target_pose = batch

    def loss_fn(params):
        pred_poses, (cov, L) = state.apply_fn({'params': params}, input_pose, train=True)

        # Reshape target to match covariance shape [B, T, J, 3]
        batch_size = target_pose.shape[0]
        target_reshaped = target_pose.reshape(batch_size, -1, N_JOINTS, 3)
        pred_reshaped = pred_poses.reshape(batch_size, -1, N_JOINTS, 3)

        nll_loss = gaussian_nll_from_cholesky(target_reshaped, pred_reshaped, L)
        pose_loss = pose_prediction_loss(pred_poses, target_pose)

        total_loss = nll_loss + pose_loss

        return total_loss, {
            'loss': total_loss,
            'nll_loss': nll_loss,
            'pose_loss': pose_loss,
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, metrics


@jax.jit
def eval_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
) -> Dict[str, float]:
    """Evaluation step."""
    input_pose, target_pose = batch

    pred_poses, (cov, L) = state.apply_fn({'params': state.params}, input_pose, train=False)

    # Reshape target to match covariance shape [B, T, J, 3]
    batch_size = target_pose.shape[0]
    target_reshaped = target_pose.reshape(batch_size, -1, N_JOINTS, 3)
    pred_reshaped = pred_poses.reshape(batch_size, -1, N_JOINTS, 3)

    nll_loss = gaussian_nll_from_cholesky(target_reshaped, pred_reshaped, L)
    pose_loss = pose_prediction_loss(pred_poses, target_pose)

    return {
        'val_nll_loss': nll_loss,
        'val_pose_loss': pose_loss,
    }


def train_epoch(
    state: TrainState,
    train_loader,
    stage: int,
    epoch: int,
    config: TrainingConfig,
) -> Tuple[TrainState, Dict[str, float]]:
    """Train for one epoch."""
    epoch_metrics = []

    for batch_idx, batch in enumerate(train_loader):
        # Convert to JAX arrays
        input_pose = jnp.array(batch[0].numpy(), dtype=jnp.float32)
        target_pose = jnp.array(batch[1].numpy(), dtype=jnp.float32)

        # Select training step based on stage
        if stage == 1:
            state, metrics = train_step_stage1(state, (input_pose, target_pose))
        elif stage == 2:
            # Calculate lambda decay
            lambda_weight = max(0.0, 1.0 - epoch / config.stage2_lambda_decay_epochs)
            state, metrics = train_step_stage2(state, (input_pose, target_pose), lambda_weight)
        elif stage == 3:
            state, metrics = train_step_stage3(state, (input_pose, target_pose))

        epoch_metrics.append(metrics)

    # Average metrics over the epoch
    avg_metrics = {}
    for key in epoch_metrics[0].keys():
        avg_metrics[key] = float(jnp.mean(jnp.array([m[key] for m in epoch_metrics])))

    return state, avg_metrics


def evaluate(state: TrainState, eval_loader) -> Dict[str, float]:
    """Evaluate the model."""
    eval_metrics = []

    for batch in eval_loader:
        # Convert to JAX arrays
        input_pose = jnp.array(batch[0].numpy(), dtype=jnp.float32)
        target_pose = jnp.array(batch[1].numpy(), dtype=jnp.float32)

        metrics = eval_step(state, (input_pose, target_pose))
        eval_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in eval_metrics[0].keys():
        avg_metrics[key] = float(jnp.mean(jnp.array([m[key] for m in eval_metrics])))

    return avg_metrics


def save_checkpoint(
    state: TrainState,
    checkpoint_dir: str,
    step: int,
    keep: int = 3,
):
    """Save checkpoint using Orbax."""
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(state)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    checkpointer.save(checkpoint_path, state, save_args=save_args, force=True)

    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    state: TrainState,
    checkpoint_dir: str,
    step: int = None,
) -> TrainState:
    """Load checkpoint using Orbax."""
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    if step is None:
        # Find latest checkpoint
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint_")]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        steps = [int(c.split("_")[1]) for c in checkpoints]
        step = max(steps)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    restored_state = checkpointer.restore(checkpoint_path, item=state)

    print(f"Loaded checkpoint from {checkpoint_path}")
    return restored_state


def train_stage(
    state: TrainState,
    train_loader,
    valid_loader,
    stage: int,
    n_epochs: int,
    config: TrainingConfig,
    checkpoint_dir: str,
    start_epoch: int = 0,
) -> TrainState:
    """Train a single stage."""
    stage_names = {1: "Pose Only", 2: "Uncertainty Head", 3: "End-to-End"}
    print(f"\n{'='*60}")
    print(f"Stage {stage}: {stage_names[stage]}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, n_epochs):
        # Train
        state, train_metrics = train_epoch(
            state, train_loader, stage, epoch, config
        )

        # Evaluate
        val_metrics = evaluate(state, valid_loader)

        # Combine metrics
        all_metrics = {**train_metrics, **val_metrics, 'epoch': epoch, 'stage': stage}

        # Log to wandb
        if config.use_wandb:
            wandb.log(all_metrics)

        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs} - " +
              " - ".join([f"{k}: {v:.6f}" for k, v in all_metrics.items() if k not in ['epoch', 'stage']]))

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(state, checkpoint_dir, epoch + 1)

    # Save final checkpoint for this stage
    save_checkpoint(state, checkpoint_dir, n_epochs)

    return state


def main(args):
    """Main training function."""
    # Setup paths
    model_dir = os.path.join("human_pose_pipeline", "models", "motion_prediction", args.run_id)
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
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            stage2_lambda_decay_epochs=args.stage2_lambda_decay_epochs,
            stage3_epochs=args.stage3_epochs,
            data_path=args.data_path,
            seed=args.seed,
            run_id=args.run_id,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            use_wandb=args.use_wandb,
        )
        config.save(config_path)
        print(f"Saved configuration to {config_path}")

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.run_id,
            config=config.to_dict(),
            resume="allow" if args.resume else False,
        )

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
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # Create or load training state
    print("Initializing model...")
    state = create_train_state(rng, config)

    if args.resume:
        try:
            state = load_checkpoint(state, checkpoint_dir)
            print("Resumed from checkpoint")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Starting from scratch")

    # Train stages
    if args.stage <= 1:
        print("\nStarting Stage 1: Pose Only Training")
        state = train_stage(
            state, train_loader, valid_loader,
            stage=1,
            n_epochs=config.stage1_epochs,
            config=config,
            checkpoint_dir=checkpoint_dir,
        )

    if args.stage <= 2:
        print("\nStarting Stage 2: Uncertainty Head Training")
        state = train_stage(
            state, train_loader, valid_loader,
            stage=2,
            n_epochs=config.stage2_epochs,
            config=config,
            checkpoint_dir=checkpoint_dir,
        )

    if args.stage <= 3:
        print("\nStarting Stage 3: End-to-End Finetuning")
        state = train_stage(
            state, train_loader, valid_loader,
            stage=3,
            n_epochs=config.stage3_epochs,
            config=config,
            checkpoint_dir=checkpoint_dir,
        )

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate(state, test_loader)
    print("Test metrics:", test_metrics)

    if config.use_wandb:
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb.finish()

    print(f"\nTraining complete! Model saved to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCT Pose Transformer")

    # Run configuration
    parser.add_argument("--run_id", type=str, required=True, help="Unique run identifier")
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
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=0.01)

    # Stage epochs
    parser.add_argument("--stage1_epochs", type=int, default=50)
    parser.add_argument("--stage2_epochs", type=int, default=20)
    parser.add_argument("--stage2_lambda_decay_epochs", type=int, default=5)
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
