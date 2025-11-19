"""This script trains the the DCT Pose Transformer model for human motion prediction.

The training has multiple steps, where the model is saved in between steps and we can
use the --stage {1, 2, ...} argument to define from which stage we want to start.

Stages:
  1. Pose only: train the DCT pose transformer to only perform the human motion prediction,
      without any uncertainty estimation. Uses ADAM with initial learning rate 1e-4 and weight decay of 1e-6.
      N_epochs = 50. Clip max grad norm = 0.01.
      Uses pose_prediction_loss as loss function.
      (Freeze the uncertainty head or not needed as loss doesn't affect it?)
  2. Uncertainty head only: train the uncertainty head while keeping the transformer weights fixed.
      Detach the transformer output features from the computational graph before passing them
      to the uncertainty head, effectively treating these features as constant inputs.
      Loss = gaussian_nll_from_cholesky + lambda * pose_prediction_loss, where
      lambda starts at 1 and is gradually decreased to 0 within M=5 epochs.
  3. End-to-end finetuning: train the whole model end-to-end, with both the pose prediction loss and
      the uncertainty loss.
      Loss = gaussian_nll_from_cholesky + pose_prediction_loss.
"""