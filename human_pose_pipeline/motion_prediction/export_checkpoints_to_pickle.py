"""Helper script to export checkpoints from each stage to pickle files."""

import os
import pickle
import argparse
import json

import orbax.checkpoint

from train_motion_prediction_model import TrainingConfig

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def load_checkpoint_params(
    checkpoint_dir: str,
    stage: int,
):
    """Load only the parameters from the latest checkpoint in a stage.

    Args:
        checkpoint_dir: Base checkpoint directory
        stage: Stage number (1, 2, or 3)

    Returns:
        Model parameters
    """
    # Load checkpoint
    stage_dir = os.path.join(checkpoint_dir, f"stage_{stage}")
    if not os.path.exists(stage_dir):
        raise ValueError(f"Stage directory {stage_dir} does not exist")

    # Find latest checkpoint in this stage
    checkpoints = [d for d in os.listdir(stage_dir) if d.startswith("checkpoint_")]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {stage_dir}")

    steps = [int(c.split("_")[1]) for c in checkpoints]
    latest_step = max(steps)

    checkpoint_path = os.path.join(stage_dir, f"checkpoint_{latest_step}")
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # Restore just the checkpoint data
    checkpoint_data = checkpointer.restore(checkpoint_path)

    # Extract params from the checkpoint structure
    if 'params' in checkpoint_data:
        return checkpoint_data['params']
    else:
        raise ValueError(f"No 'params' key found in checkpoint at {checkpoint_path}")


def save_model_pickle(
    params,
    output_path: str,
    config: TrainingConfig,
):
    """Save model parameters as a pickle file with standard structure.

    Args:
        params: Model parameters to save
        output_path: Full path to output pickle file
        config: Training configuration
    """
    # Create model data with standard structure
    model_data = {
        'model': 'DCTPoseTransformer',
        'params': params,
        'config': {
            'input_dim': config.input_dim,
            'd_model': config.d_model,
            'nhead': config.nhead,
            'num_layers': config.num_layers,
            'seq_len': config.seq_len,
            'seq_len_output': config.seq_len_output,
        }
    }

    # Save as pickle
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Saved model pickle to {output_path}")

    # Save args.json file with required fields
    args_dict = {
        "dataset": "Human36mMotionDataset3D",
        "data_path": config.data_path,
        "model": "DCTPoseTransformer",
        "output_dim": config.seq_len_output * config.input_dim,
        "input_dim": config.input_dim,
        "d_model": config.d_model,
        "nhead": config.nhead,
        "num_layers": config.num_layers,
        "seq_len": config.seq_len,
        "seq_len_output": config.seq_len_output,
    }

    args_path = output_path.replace('.pickle', '_args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=2)

    print(f"Saved args file to {args_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export checkpoints from each stage to pickle files"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Training run ID (e.g., 004qx4td)"
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Which stages to export (default: 1 2 3)"
    )

    args = parser.parse_args()

    # Paths
    model_dir = os.path.join(
        root_dir, "human_pose_pipeline", "models", "motion_prediction", args.run_id
    )
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    config_path = os.path.join(model_dir, "dct_pose_transformer_args.json")

    # Load configuration
    if not os.path.exists(config_path):
        raise ValueError(f"Configuration file not found: {config_path}")

    print(f"Loading configuration from {config_path}")
    config = TrainingConfig.load(config_path)

    # Export each stage
    for stage in args.stages:
        print(f"\n{'='*60}")
        print(f"Exporting Stage {stage}")
        print(f"{'='*60}")

        try:
            # Load checkpoint params
            params = load_checkpoint_params(checkpoint_dir, stage)

            # Save as pickle
            stage_dir = os.path.join(checkpoint_dir, f"stage_{stage}")
            pickle_path = os.path.join(stage_dir, "dct_pose_transformer.pickle")
            save_model_pickle(params, pickle_path, config)

        except Exception as e:
            print(f"Error exporting stage {stage}: {e}")
            continue

    print(f"\n{'='*60}")
    print("Export complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
