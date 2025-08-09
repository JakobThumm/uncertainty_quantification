# Neural Network Explainability

This module provides attribution methods for explaining neural network predictions, with a focus explaining out of distribution predictions. It includes both standard explainability techniques and novel approaches for explaining out-of-distribution (OOD) detection.

## Motivation

Understanding **why** a neural network makes specific predictions is crucial for:

1. **Trust and Interpretability**: Knowing which input features drive predictions
2. **Debugging Models**: Identifying when models focus on spurious correlations  
3. **OOD Analysis**: Understanding what makes samples appear out-of-distribution
4. **Method Comparison**: Comparing different uncertainty quantification approaches

While standard explainability methods (like Integrated Gradients) explain classification predictions, this module extends these techniques to explain **uncertainty estimates** and **OOD scores**.

## Methods

### Integrated Gradients (Baseline Method)

**Integrated Gradients (IG)** is our baseline explainability method for classification tasks. It computes pixel-level attribution scores by integrating gradients along a straight path from a baseline image to the input image.

**Formula:**
```
IntegratedGradients_i(x) = (x_i - x'_i) × ∫[α=0 to 1] ∂F(x' + α×(x-x')) / ∂x_i dα
```

Where:
- `x` = input image
- `x'` = baseline image (typically black/zero image)
- `F(·)` = classification model
- `α` = interpolation parameter

**Key Properties:**
- **Sensitivity**: Features that affect output get non-zero attribution
- **Implementation Invariance**: Equivalent models give identical attributions  
- **Completeness**: Sum of attributions equals output difference

### Integrated OOD Scores (Our Method)

**Integrated OOD Scores** adapts the IG methodology to explain uncertainty quantification methods. Instead of explaining classification logits, we explain OOD scores from methods like:

- **SLU (Sketched Lanczos Uncertainty)**: Our main uncertainty method
- **SCOD**: Spectral-normalized Confidence-based Out-of-Distribution detection
- **SWAG**: SWA-Gaussian for uncertainty estimation  
- **Deep Ensemble**: Ensemble-based uncertainty
- **Local Ensemble**: Lanczos-based local ensemble method

This helps answer: *"Which pixels make this image appear out-of-distribution?"*

### Baseline Image Strategies

Both methods support multiple baseline image types:

1. **Zero Baseline**: All-black image (standard approach)
2. **Mean Baseline**: Average training image (represents "typical" in-distribution)
3. **Noise Baseline**: Random noise (control condition)
4. **Adaptive Baseline**: A point between the zero baseline and the test image.

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment (note: 'unc' not 'virtualenv')
source ./unc/bin/activate
```

### 2. Train a Model

Train a LeNet model on MNIST for our example:

```bash
# Activate environment first
source ./unc/bin/activate

# Train LeNet on MNIST (single seed)
python train_model.py --run_name good --dataset MNIST --model LeNet \
    --activation_fun tanh --likelihood classification --n_epochs 50 \
    --batch_size 128 --optimizer adam --learning_rate 1e-3 --seed 1

# Test the trained model
python test_model.py --run_name good --dataset MNIST --model LeNet \
    --batch_size 128 --seed 1
```

**Expected Results**: 99.1% validation accuracy, 99.0% test accuracy

For multiple seeds (more robust):
```bash
# Use the provided training script (creates models with seeds 1-10)
bash bash/training/mnist_lenet.sh
```

This creates a model at: `../models/MNIST/LeNet/seed_1/good_params.pickle`

### 3. Explain Model Predictions

Use the visualization script to generate explanations:

```bash
# Basic IG explanation on MNIST LeNet
python visualize_explainer.py \
    --dataset MNIST \
    --model LeNet \
    --explainer ig \
    --model_seed 1 \
    --num_images 3

# With different baseline strategies
python visualize_explainer.py \
    --dataset MNIST \
    --model LeNet \
    --explainer ig \
    --baseline_type noise \
    --steps 100

# With completeness verification and benchmarking  
python visualize_explainer.py \
    --dataset MNIST \
    --model LeNet \
    --explainer ig \
    --model_seed 1 \
    --verify_completeness \
    --benchmark
```

**Expected Output**:
- Model predictions: [7, 2, 1] with >99% confidence
- Completeness error: 4.39% (with 100 steps)
- Processing time: ~0.8 seconds per image
- Visualizations saved to `./visualizations/`

### 4. View Results

Visualizations are saved in `./visualizations/` showing:
- **Input image**: Original image with predictions
- **Attribution magnitude**: Absolute importance of each pixel
- **Attribution polarity**: Positive/negative contributions

## Advanced Usage

### Multiple Datasets and Models

The framework supports various dataset/model combinations:

```bash
# Fashion-MNIST with LeNet
bash bash/training/fmnist_lenet.sh
python visualize_explainer.py --dataset FMNIST --model LeNet

# CIFAR-10 with ResNet  
bash bash/training/cifar_resnet.sh
python visualize_explainer.py --dataset CIFAR-10 --model ResNet

# MNIST with simple MLP
bash bash/training/mnist_mlp.sh  
python visualize_explainer.py --dataset MNIST --model MLP_depth1_hidden20
```

### Programmatic Usage

```python
from src.explainability import IntegratedGradients, create_zero_baseline
from src.models import pretrained_model_from_string

# Load model
model, params_dict, _ = pretrained_model_from_string(
    dataset_name="MNIST",
    model_name="LeNet", 
    run_name="good",
    seed=1
)

# Initialize explainer
ig_explainer = IntegratedGradients(model, params_dict['params'])

# Compute attributions
baseline = create_zero_baseline(image.shape)
attributions = ig_explainer.explain(
    input_image=image,
    baseline_image=baseline,
    target_class=predicted_class,
    steps=50
)

# Verify completeness axiom
attr_sum, output_diff, error = ig_explainer.verify_completeness(
    image, baseline, attributions, predicted_class
)
print(f"Completeness error: {error:.4f}")
```

### Integration Steps Analysis

Test different numbers of integration steps:

```bash
# Few steps (faster, less accurate)
python visualize_explainer.py --steps 20

# Many steps (slower, more accurate)  
python visualize_explainer.py --steps 200 --benchmark
```

## File Structure

```
src/explainability/
├── __init__.py                    # Module exports
├── integrated_gradients.py       # Plain IG baseline method  
├── integrated_ood_scores.py      # Our OOD explanation method (TODO)
├── interpolation.py               # Path interpolation utilities
├── gradient_utils.py              # JAX gradient computation  
├── baseline_images.py             # Baseline image creation
├── examples/                      # Test results and examples
│   ├── ig_MNIST_image_0.png      # Digit 7 explanation
│   ├── ig_MNIST_image_1.png      # Digit 2 explanation
│   └── ig_MNIST_image_2.png      # Digit 1 explanation
├── TESTING_RESULTS.md             # Validation results and benchmarks
└── README.md                      # This file

visualize_explainer.py             # Main visualization script
```

## Validation and Testing

See [`TESTING_RESULTS.md`](TESTING_RESULTS.md) for comprehensive validation results and benchmarks.

### Completeness Axiom

Verify that attributions sum to the output difference:

```bash
python visualize_explainer.py --verify_completeness
```

Good implementations should have relative error < 5%. Our implementation achieves 4.39% with 100 steps.

### Performance Benchmarking

Measure computation time for different step counts:

```bash
python visualize_explainer.py --benchmark
```

Typical results on MNIST LeNet (validated):
- 20 steps: ~0.79 seconds  
- 50 steps: ~0.84 seconds
- 100 steps: ~0.81 seconds

**Note**: Times include JIT compilation overhead. 100+ steps recommended for completeness axiom compliance.

## Extending the Framework

### Adding New Explainer Methods

1. Create new explainer class following the `IntegratedGradients` pattern
2. Add to `initialize_explainer()` in `visualize_explainer.py`
3. Implement required methods: `explain()`, `batch_explain()`

### Adding New Baseline Strategies  

1. Add function to `baseline_images.py`
2. Update `create_baseline_image()` in `visualize_explainer.py`
3. Add command-line option

### Adding New Datasets

The framework automatically supports any dataset in `src/datasets/`. Just ensure you have a trained model:

```bash
python train_model.py --dataset NEW_DATASET --model ARCHITECTURE --default_hyperparams
python visualize_explainer.py --dataset NEW_DATASET --model ARCHITECTURE
```

## Troubleshooting

### Model Loading Issues

**Error**: `FileNotFoundError: Model not found`

**Solution**: Ensure model is trained and path is correct:
```bash
# Check if model exists
ls ../models/MNIST/LeNet/seed_1/good_params.pickle

# Train if missing
python train_model.py --dataset MNIST --model LeNet --seed 1 --run_name good --default_hyperparams
```

### Memory Issues

**Error**: Out of memory during explanation

**Solutions**:
- Reduce number of integration steps: `--steps 20`
- Reduce batch size: `--batch_size 1`  
- Use smaller images or fewer visualization images: `--num_images 1`

### Completeness Axiom Violations

**Issue**: High relative error (>5%)

**Solutions**:
- Use more integration steps: `--steps 100` (achieves 4.39% error)
- Check model interface compatibility (use `apply_test` not `apply`)
- Verify baseline image creation

### JIT Compilation Errors

**Issue**: `TracerBoolConversionError` in validation functions

**Solution**: Validation checks are disabled in JIT context (implemented automatically)

## Next Steps

- **Integrated OOD Scores**: Implementation of our main contribution
- **Batch OOD Scoring**: Explaining OOD methods like SLU, SCOD
- **Comparative Analysis**: Side-by-side comparison of IG vs OOD explanations
- **Advanced Baselines**: Adaptive and learned baseline strategies

## References

Sundararajan et al. "Axiomatic Attribution for Deep Networks" (2017)