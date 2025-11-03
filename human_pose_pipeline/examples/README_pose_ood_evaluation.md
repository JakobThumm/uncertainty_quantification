# 2D Pose Estimation with OOD Detection - Evaluation Script

## Overview

The `pose_estimation_2d_with_ood.py` script performs comprehensive evaluation of pose estimation with OOD (Out-of-Distribution) detection using the Sketched Lanczos Uncertainty (SLU) method.

## Features

1. **Loads pre-computed OOD score functions** from cache (avoiding expensive recomputation)
2. **Evaluates pose estimation** on both ID (H36M) and OOD (TigerPose) datasets
3. **Generates scatter plots**:
   - Pose prediction accuracy (MPJPE) vs OOD score
   - Mean predicted uncertainty vs OOD score
4. **Evaluates calibration**: Percentage of datapoints within 1σ, 2σ, 3σ, and 4σ confidence intervals for samples classified as ID vs OOD

## Prerequisites

Before running this script, you must first compute and cache the OOD score functions using `score_model.py`:

```bash
python score_model.py \
  --ID_dataset H36M \
  --OOD_dataset tiger-pose \
  --data_path datasets/ \
  --model_save_path human_pose_pipeline/models/pose_estimation \
  --model RegressFlow \
  --run_name finetuned_h36m_regressflow_pred \
  --subsample_trainset 10000 \
  --subsample_testset 640 \
  --lanczos_hm_iter 0 \
  --lanczos_lm_iter 81 \
  --test_batch_size 64 \
  --train_batch_size 64 \
  --serialize_ggn_on_batches \
  --sketch srft \
  --sketch_size 100000 \
  --cache_dir cache/ \
  --load_score_functions  # Or run without this flag to compute from scratch
```

This will create cache files in `cache/` directory with names like:
- `H36M_RegressFlow_n9000_*_score_functions.cloudpickle`
- `H36M_RegressFlow_n9000_*_ggn.cloudpickle`
- `H36M_RegressFlow_n9000_*_sketch.cloudpickle`
- `H36M_RegressFlow_n9000_*_eigenpairs.cloudpickle`

## Usage

### Basic Usage

```bash
python human_pose_pipeline/examples/pose_estimation_2d_with_ood.py
```

This will:
- Load the cached OOD score functions
- Evaluate 50 samples from H36M (ID) dataset
- Evaluate 50 samples from TigerPose (OOD) dataset
- Generate plots in `results/pose_ood_evaluation/`

### Advanced Usage

```bash
python human_pose_pipeline/examples/pose_estimation_2d_with_ood.py \
  --cache_dir cache/ \
  --data_path datasets/ \
  --model_save_path human_pose_pipeline/models/pose_estimation \
  --run_name finetuned_h36m_regressflow_with_unc \
  --ood_threshold 0.5 \
  --max_samples_h36m 100 \
  --max_samples_tiger 100 \
  --output_dir results/my_evaluation
```

### Arguments

- `--cache_dir`: Directory containing cached score functions (default: `cache/`)
- `--data_path`: Path to datasets directory (default: `datasets/`)
- `--model_save_path`: Path to saved models (default: `human_pose_pipeline/models/pose_estimation`)
- `--run_name`: Model run name (default: `finetuned_h36m_regressflow_with_unc`)
- `--ood_threshold`: OOD classification threshold (default: auto-determined from eigenvalues)
- `--max_samples_h36m`: Maximum samples to evaluate from H36M (default: 50)
- `--max_samples_tiger`: Maximum samples to evaluate from TigerPose (default: 50)
- `--output_dir`: Directory for output plots (default: `results/pose_ood_evaluation`)

## Output

The script generates three main plots in the output directory:

### 1. `pose_accuracy_vs_ood_score.png`
Scatter plots showing the relationship between pose prediction accuracy (MPJPE in pixels) and OOD score for both H36M and TigerPose datasets. Points are colored by dataset (blue for ID, red for OOD), with a vertical line indicating the OOD threshold.

### 2. `mean_uncertainty_vs_ood_score.png`
Scatter plots showing the relationship between mean predicted uncertainty and OOD score for both datasets. This helps evaluate whether the model's uncertainty estimates correlate with the OOD scores.

### 3. `sigma_evaluation_id_vs_ood.png`
A 2×2 grid of bar charts showing the percentage of joints falling within 1σ, 2σ, 3σ, and 4σ confidence intervals for:
- **Top-left**: H36M samples classified as ID
- **Top-right**: H36M samples classified as OOD
- **Bottom-left**: TigerPose samples classified as ID
- **Bottom-right**: TigerPose samples classified as OOD

This evaluation reveals:
- **Calibration quality**: Well-calibrated uncertainties should have ~68% within 1σ, ~95% within 2σ, etc.
- **ID vs OOD behavior**: How the model's uncertainty estimates differ between ID and OOD classifications

## Expected Results

For well-calibrated uncertainty estimates with effective OOD detection:

1. **H36M (ID) samples classified as ID**: Should show good calibration (close to theoretical percentages)
2. **H36M (ID) samples classified as OOD**: Likely misclassifications or genuinely unusual poses
3. **TigerPose (OOD) samples classified as ID**: False negatives - the model is overconfident
4. **TigerPose (OOD) samples classified as OOD**: Correct detections - should show degraded calibration

## Implementation Details

### OOD Scoring

The script uses the pre-computed score function from the Sketched Lanczos method:

```python
score_fn, eigenval, approx_quadratic_form, quadratic_form = load_score_functions(
    cache_dir, base_key
)
```

The score function is applied to bounding box images extracted from each frame:

```python
ood_score = score_fn(preprocessed_image)
is_ood = ood_score > ood_threshold
```

### Mahalanobis Distance

For each joint prediction, we compute the Mahalanobis distance considering the full covariance:

```python
mahalanobis = (inv_sigma_xx * (delta_x ** 2) +
               inv_sigma_yy * (delta_y ** 2) +
               2 * inv_sigma_xy * (delta_x * delta_y))
```

This measures how many standard deviations the prediction is from the ground truth, accounting for the predicted uncertainty.

### Confidence Intervals

We use chi-squared thresholds for 2 degrees of freedom (x and y coordinates):
- 1σ: 68% probability → χ²(0.68, df=2)
- 2σ: 95% probability → χ²(0.95, df=2)
- 3σ: 99.73% probability → χ²(0.9973, df=2)
- 4σ: 99.994% probability → χ²(0.99994, df=2)

## Troubleshooting

### Cache Key Mismatch

If you get a `FileNotFoundError` about missing cache files, ensure that:
1. You've run `score_model.py` with the correct parameters
2. The `--subsample_trainset` and other parameters match between scoring and evaluation
3. The cache files exist in the specified `--cache_dir`

You can check available cache files with:
```bash
ls -lh cache/H36M_RegressFlow_*_score_functions.cloudpickle
```

### Model Loading Issues

If the model fails to load, verify that:
1. The model directory exists: `human_pose_pipeline/models/pose_estimation/H36M/RegressFlow/seed_420/`
2. The checkpoint files exist: `finetuned_h36m_regressflow_with_unc_params.pickle` and `_args.json`

### CUDA/GPU Issues

The script uses GPU for YOLO human detection and JAX model inference. If you encounter CUDA errors:
- Ensure CUDA is available and properly configured
- Check available GPU memory with `nvidia-smi`
- Reduce batch sizes or max_samples if needed

## References

- **Sketched Lanczos Uncertainty**: See `src/ood_scores/lm_lanczos.py` for implementation
- **Pose Estimation Pipeline**: See `human_pose_pipeline/pose_estimation/inference_helper.py`
- **Original pose estimation script**: `pose_estimation_2D.py`
