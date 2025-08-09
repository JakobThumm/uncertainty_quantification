# Integrated Gradients Implementation Plan

## Overview

This document details the implementation strategy for the **plain Integrated Gradients baseline method** that will serve as a comparison point for our Integrated OOD Scores method. We implement standard IG for classification predictions to establish a solid foundation and validate our implementation approach.

## Core Algorithm Implementation

### Mathematical Foundation

The Integrated Gradients algorithm computes feature attributions using:

```
IntegratedGradients_i(x) = (x_i - x'_i) × ∫[α=0 to 1] ∂F(x' + α×(x-x')) / ∂x_i dα
```

Where:
- `x` = input image
- `x'` = baseline image (typically black/zero image)
- `α` = interpolation parameter (0 to 1)
- `F` = model function (classification logits/probabilities)
- `i` = feature dimension (pixel/channel)

### JAX Implementation Strategy

#### 1. Path Interpolation
```python
def interpolate_images(baseline_image, input_image, steps=50):
    """Generate interpolated images along straight line path."""
    alphas = jnp.linspace(0.0, 1.0, steps + 1)  # Include endpoints
    # Shape: (steps+1, *image_shape)
    interpolated = baseline_image[None, ...] + alphas[:, None, None, None] * (input_image - baseline_image)[None, ...]
    return interpolated, alphas
```

#### 2. Gradient Computation
```python
def compute_gradients_wrt_inputs(model, params, images, target_class):
    """Compute gradients of model output w.r.t. input images."""
    def model_fn(images):
        logits = model.apply(params, images)
        if target_class is not None:
            return logits[:, target_class]  # Target specific class
        else:
            return jnp.max(logits, axis=1)  # Max logit
    
    grad_fn = jax.grad(lambda img: jnp.sum(model_fn(img[None, ...])))
    return jax.vmap(grad_fn)(images)
```

#### 3. Integration Approximation
```python
def integrate_gradients(gradients, input_image, baseline_image):
    """Approximate integral using trapezoidal rule."""
    # Remove endpoints for trapezoidal rule
    gradients = gradients[1:-1]  # Shape: (steps-1, *image_shape)
    
    # Trapezoidal rule: average adjacent points
    integrated = jnp.mean(gradients, axis=0)  # Average across interpolation steps
    
    # Scale by path difference
    attribution = (input_image - baseline_image) * integrated
    return attribution
```

## Integration with Existing Codebase

### Model Loading
Leverage existing infrastructure from `src/models/wrapper.py`:

```python
from src.models import pretrained_model_from_string

def load_pretrained_model(dataset, model_name, run_name, seed):
    """Load pretrained model using existing infrastructure."""
    model, params_dict, model_args = pretrained_model_from_string(
        dataset_name=dataset,
        model_name=model_name, 
        run_name=run_name,
        seed=seed,
        save_path="../models"
    )
    return model, params_dict['params'], model_args
```

### Dataset Integration
Use existing dataset loading patterns from `src/datasets/`:

```python
from src.datasets import dataloader_from_string

def load_test_data(dataset_name):
    """Load test dataset using existing infrastructure."""
    _, _, test_loader = dataloader_from_string(
        dataset_name,
        batch_size=1,  # Process one image at a time for IG
        shuffle=False,
        seed=0
    )
    return test_loader
```

## Baseline Image Strategies

### 1. Zero/Black Baseline
```python
def create_zero_baseline(image_shape):
    """Standard IG baseline - all zeros."""
    return jnp.zeros(image_shape)
```

### 2. Mean Training Baseline  
```python
def create_mean_baseline(train_loader):
    """Compute mean image from training data."""
    all_images = []
    for batch in train_loader:
        images = jnp.array(batch[0].numpy())
        all_images.append(images)
    
    all_images = jnp.concatenate(all_images, axis=0)
    return jnp.mean(all_images, axis=0)
```

### 3. Random Noise Baseline
```python
def create_noise_baseline(image_shape, key):
    """Random baseline for control experiments."""
    return jax.random.uniform(key, image_shape, minval=0.0, maxval=1.0)
```

## File Structure

```
src/explainability/
├── __init__.py
├── integrated_gradients.py           # Main IG implementation
├── baseline_images.py                # Baseline image creation utilities
├── gradient_utils.py                 # JAX gradient computation helpers
└── interpolation.py                  # Path interpolation utilities
```

### Core Implementation Files

#### `src/explainability/integrated_gradients.py`
```python
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple
from .baseline_images import create_zero_baseline, create_mean_baseline
from .interpolation import interpolate_images
from .gradient_utils import compute_model_gradients

class IntegratedGradients:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        
    def explain(self, 
                input_image: jnp.ndarray,
                baseline_image: jnp.ndarray,
                target_class: Optional[int] = None,
                steps: int = 50) -> jnp.ndarray:
        """Compute IG attributions for input image."""
        # Implementation here
        pass
        
    def batch_explain(self, 
                     images: jnp.ndarray,
                     baseline_image: jnp.ndarray,
                     target_classes: Optional[jnp.ndarray] = None,
                     steps: int = 50) -> jnp.ndarray:
        """Compute IG for batch of images efficiently."""
        # Implementation here
        pass
```

#### `src/explainability/gradient_utils.py`
```python
import jax
import jax.numpy as jnp

def compute_model_gradients(model, params, images, target_class=None):
    """Compute gradients of model predictions w.r.t. inputs."""
    # Implementation here
    pass

def model_prediction_fn(model, params, target_class=None):
    """Create prediction function for gradient computation."""
    # Implementation here  
    pass
```

## Testing & Validation Strategy

### 1. Axiom Verification
Implement tests to verify IG axioms:

```python
def test_sensitivity_axiom(ig_explainer, baseline, input1, input2):
    """Test that differing features get non-zero attribution."""
    # If inputs differ in one feature and have different outputs,
    # that feature should have non-zero attribution
    pass

def test_implementation_invariance(ig_explainer1, ig_explainer2, input_image):
    """Test that equivalent models give same attributions."""
    pass

def test_completeness_axiom(ig_explainer, input_image, baseline):
    """Test that attributions sum to output difference."""
    # sum(attributions) ≈ F(input) - F(baseline)
    pass
```

### 2. Sanity Checks
```python
def test_attribution_sanity():
    """Basic sanity checks for attribution quality."""
    # 1. Attributions should be same shape as input
    # 2. Attributions should highlight relevant features
    # 3. Different targets should give different attributions
    pass
```

### 3. Performance Benchmarks
```python
def benchmark_performance():
    """Measure computation time and memory usage."""
    # Test on different:
    # - Image sizes (MNIST 28x28 vs CIFAR 32x32)
    # - Number of interpolation steps (20, 50, 100, 200)
    # - Batch sizes
    pass
```

## Integration Points with OOD Explainer

### Shared Components
1. **Interpolation utilities** - Same path generation logic
2. **Baseline image creation** - Same baseline strategies  
3. **Visualization tools** - Same attribution plotting
4. **Gradient computation patterns** - Similar JAX autodiff usage

### Key Differences
1. **Target function**: Classification logits vs OOD scores
2. **Output interpretation**: Class probability vs uncertainty measure
3. **Baseline semantics**: "Neutral" classification vs "In-distribution" for OOD

## Implementation Phases

### Phase 1: Core Algorithm (Week 1)
1. Implement basic IG with zero baseline
2. Test on simple MNIST classification
3. Verify completeness axiom numerically
4. Basic visualization of attributions

### Phase 2: Baseline Strategies (Week 1-2)
1. Implement multiple baseline image types
2. Compare baselines on MNIST/CIFAR datasets  
3. Analyze attribution differences across baselines
4. Document baseline selection guidelines

### Phase 3: Optimization & Testing (Week 2)
1. Optimize for batch processing
2. Implement comprehensive test suite
3. Performance benchmarking
4. Memory usage optimization

### Phase 4: Integration (Week 2-3)
1. Create unified interface with OOD explainer
2. Shared visualization tools
3. Comparative analysis framework
4. Documentation and examples

## Success Criteria

1. **Correctness**: Pass all axiom tests (sensitivity, completeness, implementation invariance)
2. **Performance**: Handle CIFAR-10 batch (50 images) in <30 seconds
3. **Usability**: Simple API matching sklearn/torchvision patterns
4. **Integration**: Clean interface for comparison with OOD methods
5. **Documentation**: Complete examples and tutorials

This implementation will serve as the solid foundation for our Integrated OOD Scores method while providing a validated baseline for comparison.