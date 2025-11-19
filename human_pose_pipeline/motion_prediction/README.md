# Motion Prediction Model Training

This directory contains the training script for the DCT Pose Transformer model for human motion prediction with uncertainty quantification.

## Overview

The training script (`train_motion_prediction_model.py`) implements a three-stage training process:

1. **Stage 1 (Pose Only)**: Train the full pose prediction transformer without uncertainty estimation
   - **Trains**: All parameters (transformer, decoders, embeddings)
   - **Loss**: `MAE(predicted_poses, target_poses)`

2. **Stage 2 (Uncertainty Head Only)**: Train ONLY the uncertainty head with frozen backbone
   - **Trains**: `uncertainty_head` parameters ONLY
   - **Frozen**: Transformer blocks, pose decoders, frequency embeddings
   - **Loss**: `GaussianNLL + λ * MAE` (λ decays 1→0 over first 5 epochs)

3. **Stage 3 (End-to-End)**: Fine-tune the complete model end-to-end
   - **Trains**: All parameters
   - **Loss**: `GaussianNLL + MAE`

## Features

- **JAX/Flax Implementation**: Fully optimized with `jax.jit` and `jax.grad`
- **Checkpoint Management**: Automatic checkpoint saving/loading using Orbax
- **Configuration Management**: JSON-based config files for reproducibility
- **Weights & Biases Integration**: Automatic experiment tracking and logging
- **MPJPE Metrics**: Mean Per Joint Position Error evaluation
- **Multi-Stage Training**: Flexible stage-based training pipeline

## Quick Start

### Basic Training (All Stages)

```bash
python human_pose_pipeline/motion_prediction/train_motion_prediction_model.py \
    --data_path datasets/ \
    --wandb_project motion-prediction
```

This will:
- Auto-generate a run_id from wandb
- Train all 3 stages sequentially
- Save checkpoints to `human_pose_pipeline/models/motion_prediction/<run_id>/checkpoints/`
- Save config to `human_pose_pipeline/models/motion_prediction/<run_id>/dct_pose_transformer_args.json`

### Training a Specific Stage

```bash
# Start from Stage 1
python human_pose_pipeline/motion_prediction/train_motion_prediction_model.py \
    --stage 1 \
    --data_path datasets/

# Resume and continue from Stage 2
python human_pose_pipeline/motion_prediction/train_motion_prediction_model.py \
    --stage 2 \
    --resume \
    --data_path datasets/

# Resume and continue from Stage 3
python human_pose_pipeline/motion_prediction/train_motion_prediction_model.py \
    --stage 3 \
    --resume \
    --data_path datasets/
```

### Custom Run ID

```bash
python human_pose_pipeline/motion_prediction/train_motion_prediction_model.py \
    --run_id my_experiment_v1 \
    --data_path datasets/
```

### Training Without Weights & Biases

```bash
python human_pose_pipeline/motion_prediction/train_motion_prediction_model.py \
    --run_id local_test \
    --no_wandb \
    --data_path datasets/
```

## Configuration

### Default Configuration

A default configuration is provided in `configs/default_training_config.json`:

```json
{
  "d_model": 128,
  "nhead": 4,
  "num_layers": 2,
  "seq_len": 50,
  "seq_len_output": 10,
  "batch_size": 256,
  "learning_rate": 0.0001,
  "weight_decay": 1e-06,
  "max_grad_norm": 0.01,
  "stage1_epochs": 50,
  "stage2_epochs": 20,
  "stage3_epochs": 30,
  ...
}
```

### Customizing Hyperparameters

```bash
python human_pose_pipeline/motion_prediction/train_motion_prediction_model.py \
    --d_model 256 \
    --nhead 8 \
    --num_layers 4 \
    --batch_size 128 \
    --learning_rate 5e-5 \
    --stage1_epochs 100 \
    --stage2_epochs 30 \
    --stage3_epochs 50
```

## Command-Line Arguments

### Run Configuration
- `--run_id`: Unique run identifier (optional, defaults to wandb run id or timestamp)
- `--stage`: Training stage to start from (1, 2, or 3)
- `--resume`: Resume from latest checkpoint
- `--new_config`: Create new config even if one exists

### Model Hyperparameters
- `--d_model`: Model dimension (default: 128)
- `--nhead`: Number of attention heads (default: 4)
- `--num_layers`: Number of transformer layers (default: 2)
- `--seq_len`: Input sequence length (default: 50)
- `--seq_len_output`: Output sequence length (default: 10)

### Training Hyperparameters
- `--batch_size`: Batch size (default: 256)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-6)
- `--max_grad_norm`: Max gradient norm for clipping (default: 0.01)

### Stage Epochs
- `--stage1_epochs`: Epochs for Stage 1 (default: 50)
- `--stage2_epochs`: Epochs for Stage 2 (default: 20)
- `--stage2_lambda_decay_epochs`: Epochs for lambda decay in Stage 2 (default: 5)
- `--stage3_epochs`: Epochs for Stage 3 (default: 30)

### Data Settings
- `--data_path`: Path to datasets directory (default: ../datasets)
- `--seed`: Random seed (default: 420)

### Weights & Biases
- `--wandb_project`: W&B project name (default: motion-prediction)
- `--wandb_entity`: W&B entity/username
- `--use_wandb`: Enable W&B logging (default: True)
- `--no_wandb`: Disable W&B logging

## VS Code Debugging

Five debug configurations are available in `.vscode/launch.json`:

1. **Train Motion Prediction Model (Stage 1)**: Start fresh Stage 1 training
2. **Train Motion Prediction Model (Stage 2)**: Resume and train Stage 2
3. **Train Motion Prediction Model (Stage 3)**: Resume and train Stage 3
4. **Train Motion Prediction Model (Full Pipeline)**: Train all stages sequentially
5. **Train Motion Prediction Model (No W&B)**: Quick test without W&B

Press `F5` in VS Code and select the desired configuration from the dropdown.

## Logged Metrics

The following metrics are logged to Weights & Biases during training:

### Training Metrics
- `train/loss`: Total training loss
- `train/nll_loss`: Negative log-likelihood loss (Stages 2 & 3)
- `train/pose_loss`: Pose prediction MAE loss
- `train/lambda`: Lambda weight for Stage 2

### Evaluation Metrics
- `eval/val_nll_loss`: Validation negative log-likelihood
- `eval/val_pose_loss`: Validation pose prediction MAE
- `eval/val_mpjpe`: Validation Mean Per Joint Position Error (mm)
- `eval/val_mpjpe_std`: Standard deviation of MPJPE

### Test Metrics
- `test_val_nll_loss`: Final test NLL
- `test_val_pose_loss`: Final test pose loss
- `test_val_mpjpe`: Final test MPJPE
- `test_val_mpjpe_std`: Final test MPJPE std

## Checkpoints

Checkpoints are saved:
- Every 10 epochs during training
- At the end of each stage
- In `human_pose_pipeline/models/motion_prediction/<run_id>/checkpoints/`

Each checkpoint contains:
- Model parameters
- Optimizer state
- Training step information

## Directory Structure

After training, your directory structure will look like:

```
human_pose_pipeline/models/motion_prediction/<run_id>/
├── dct_pose_transformer_args.json    # Configuration file
└── checkpoints/
    ├── checkpoint_10/                # Epoch 10 checkpoint
    ├── checkpoint_20/                # Epoch 20 checkpoint
    ├── checkpoint_50/                # End of Stage 1
    ├── checkpoint_70/                # End of Stage 2
    └── checkpoint_100/               # End of Stage 3
```

## Training Strategy & Parameter Freezing

### Why Three Stages?

The three-stage training approach ensures stable convergence:

1. **Stage 1** first learns good pose predictions without worrying about uncertainty
2. **Stage 2** learns uncertainty estimates based on fixed (frozen) pose features
3. **Stage 3** fine-tunes everything together for optimal performance

### Parameter Freezing in Stage 2

**Critical:** In Stage 2, ALL parameters except `uncertainty_head` are frozen. This means:

✅ **Trainable in Stage 2:**
- `uncertainty_head.mlp_0`
- `uncertainty_head.mlp_1`
- `uncertainty_head.mlp_2`
- `uncertainty_head.unc_proc_0`
- `uncertainty_head.unc_proc_1`
- `uncertainty_head.uncertainty_weight`

❌ **Frozen in Stage 2:**
- All transformer blocks (`transformer_block_*`)
- Frequency decoders (`low_freq_decoder`, `high_freq_decoder`)
- Input embeddings (`input_embed_*`)
- Positional embeddings (`freq_pos_embed`)
- Uncertainty embedding module (if using input uncertainty)

This is implemented by zeroing out gradients for frozen parameters during the backward pass.

## Loss Functions

### Stage 1: Pose Only
```python
Loss = MAE(predicted_poses, target_poses)
```
**Trainable**: All parameters

### Stage 2: Uncertainty Head Only
```python
Loss = GaussianNLL(predicted_poses, target_poses, cholesky_L) + λ * MAE(predicted_poses, target_poses)
```
- λ linearly decays from 1 to 0 over the first 5 epochs
- **Trainable**: `uncertainty_head` ONLY
- **Frozen**: Everything else

### Stage 3: End-to-End
```python
Loss = GaussianNLL(predicted_poses, target_poses, cholesky_L) + MAE(predicted_poses, target_poses)
```
**Trainable**: All parameters

## Learning Rate Scheduling

**Important:** Each stage gets its own independent learning rate schedule!

- **Stage 1**: LR schedule over 50 epochs (default)
- **Stage 2**: New LR schedule over 20 epochs (default)
- **Stage 3**: New LR schedule over 30 epochs (default)

This means:
- Each stage starts with a fresh warmup period
- Each stage decays from the initial LR to the minimum LR
- The optimizer is recreated between stages with a new schedule

### Default Behavior (Cosine Annealing)
For each stage with N epochs:
1. **Warmup (epochs 0-5)**: LR increases from 0 → initial_lr
2. **Decay (epochs 5-N)**: LR decreases from initial_lr → (initial_lr × 0.01)

### Example
With `--learning_rate 1e-3`:
- **Stage 1** (50 epochs):
  - Epochs 0-5: Warmup 0 → 1e-3
  - Epochs 5-50: Cosine decay 1e-3 → 1e-5
- **Stage 2** (20 epochs):
  - Epochs 0-5: Warmup 0 → 1e-3 (fresh start!)
  - Epochs 5-20: Cosine decay 1e-3 → 1e-5
- **Stage 3** (30 epochs):
  - Epochs 0-5: Warmup 0 → 1e-3 (fresh start!)
  - Epochs 5-30: Cosine decay 1e-3 → 1e-5

## Example Workflows

### Workflow 1: Complete Training from Scratch
```bash
# Stage 1: Train pose prediction
python train_motion_prediction_model.py --stage 1 --data_path datasets/

# Stage 2: Train uncertainty head
python train_motion_prediction_model.py --stage 2 --resume --data_path datasets/

# Stage 3: End-to-end finetuning
python train_motion_prediction_model.py --stage 3 --resume --data_path datasets/
```

### Workflow 2: All-in-One Training
```bash
# Train all stages in one command
python train_motion_prediction_model.py --stage 1 --data_path datasets/
```

### Workflow 3: Quick Local Test
```bash
# Fast test run without W&B
python train_motion_prediction_model.py \
    --run_id test_run \
    --stage1_epochs 2 \
    --stage2_epochs 2 \
    --stage3_epochs 2 \
    --batch_size 64 \
    --no_wandb \
    --data_path datasets/
```

## Troubleshooting

### Issue: "No checkpoints found"
**Solution**: Make sure you're using the correct `--run_id` when resuming, or don't specify it to use the wandb run id.

### Issue: CUDA out of memory
**Solution**: Reduce batch size: `--batch_size 128` or `--batch_size 64`

### Issue: wandb login required
**Solution**: Run `wandb login` or use `--no_wandb` flag

### Issue: Config file already exists
**Solution**: Use `--new_config` flag to override existing configuration

## References

- Model architecture: `src/models/dct_pose_transformer.py`
- Loss functions: `src/models/dct_pose_transformer.py:pose_prediction_loss`, `gaussian_nll_from_cholesky`
- Dataset: Human3.6M motion prediction dataset
- Evaluation metrics: `human_pose_pipeline/utils/eval_utils.py:evaluate_pose_prediction_scores_jax`
