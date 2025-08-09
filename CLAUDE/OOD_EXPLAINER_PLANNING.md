# OOD Explainer Project Planning

## Project Overview

We aim to build an **Out-of-Distribution (OOD) Explainer** that identifies which features (pixels) in an image have the highest impact on the OOD score. This will help answer questions like:
- Which pixels make an image look "out-of-distribution"?  
- What visual patterns does the OOD detector focus on?
- How can we debug/improve OOD detection methods?

## Integrated Gradients Methodology Summary

Based on the paper "Axiomatic Attribution for Deep Networks" and TensorFlow guide:

### Core Concept
Integrated Gradients (IG) explains model predictions by attributing importance to input features through path integration of gradients from a baseline to the input.

### Key Formula
```
IntegratedGradients_i(x) = (x_i - x'_i) × ∫[α=0 to 1] ∂F(x' + α×(x-x')) / ∂x_i dα
```

Where:
- `x` = input image
- `x'` = baseline image (typically black/zero image)  
- `α` = interpolation parameter (0 to 1)
- `F` = model function
- `i` = feature dimension (pixel)

### Three-Step Process

1. **Path Interpolation**: Generate interpolated images along straight line from baseline to input
   ```
   interpolated_image = baseline + α × (input - baseline)
   ```

2. **Gradient Computation**: Calculate gradients at each interpolation step
   ```
   gradients = ∇F(interpolated_image) w.r.t. input_features
   ```

3. **Integration Approximation**: Accumulate gradients using Riemann sum
   ```  
   IG ≈ (input - baseline) × Σ[k=1 to m] gradients(k) × (1/m)
   ```

### Key Properties (Axioms)

1. **Sensitivity**: If inputs differ in one feature and have different outputs, that feature gets non-zero attribution
2. **Implementation Invariance**: Functionally equivalent networks give identical attributions  
3. **Completeness**: Attributions sum to difference between output at input and baseline
4. **Symmetry-Preserving**: Symmetric features with identical values get identical attributions

### Practical Implementation

- **Baseline Selection**: Black image (all zeros) for image models
- **Steps**: 20-300 interpolation steps (more for higher accuracy)
- **Numerical Integration**: Trapezoidal rule for approximating integral
- **Visualization**: Sum absolute values across color channels for attribution mask

## Adapting IG for OOD Explanation

### High-Level Approach

Instead of explaining **classification predictions**, we want to explain **OOD scores** using **Integrated OOD Scores**:

```python
# Traditional IG for classification (baseline method)
F(x) = model.predict_class_probabilities(x)

# Our Integrated OOD Scores method
F(x) = ood_score_function(x)  # e.g., SLU, SCOD, ensemble uncertainty
```

### OOD Score Functions to Explain

From the existing codebase, we can explain any of these OOD methods:

1. **SLU (Sketched Lanczos Uncertainty)**: `src/ood_scores/lm_lanczos.py`
2. **SCOD**: `src/ood_scores/scod.py`  
3. **SWAG**: `src/ood_scores/swag.py`
4. **Deep Ensemble**: `src/ood_scores/ensemble.py`
5. **Laplace Approximations**: `src/ood_scores/diagonal_lla.py`
6. **Local Ensemble**: `src/ood_scores/hm_lanczos.py`

### Key Questions to Address

1. **What baseline image makes sense for OOD?**
   - Black image (absence of signal)?
   - Average training image (typical in-distribution example)?
   - Random noise?

2. **Which OOD score to focus on first?**
   - Start with SLU since it's the main contribution of this codebase?
   - Or simpler methods like max logit for validation?

3. **How to handle different score ranges/scales?**
   - OOD scores may have different ranges than classification probabilities
   - May need normalization for meaningful gradients

4. **Integration with existing infrastructure:**
   - Leverage existing model loading, scoring infrastructure
   - Build on JAX/Flax framework already used

## Implementation Plan

### Phase 1: Foundation
1. **Implement plain Integrated Gradients** as baseline method for comparison with classification predictions
2. **Create Integrated OOD Scores base class** that can wrap any OOD scoring function  
3. **Implement core Integrated OOD Scores algorithm** adapted for OOD scores
4. **Choose initial OOD method** implement SLU first
5. **Basic visualization** showing pixel attributions for OOD scores

### Phase 2: Validation & Comparison  
1. **Test on simple cases** where we know what should be important (e.g., adversarial perturbations)
2. **Compare different baseline images** (black, average image, noise)
3. **Validate across multiple OOD methods** (SLU, SCOD, ensemble)
4. **Compare Integrated OOD Scores with baseline method (plain IG)**: Do they highlight different regions?
5. **Sanity checks**: Do attributions highlight expected regions?

### Phase 3: Analysis & Applications
1. **Systematic evaluation** on datasets from the benchmark (MNIST, CIFAR, etc.)
2. **Compare OOD explanations across methods**: Do different UQ methods focus on same pixels?
3. **Use cases**: 
   - Debug OOD detection failures
   - Understand what makes images OOD
   - Improve OOD detection by understanding failure modes

### Phase 4: Advanced Features
1. **Multiple baseline images simultaneously** (ensemble of explanations)
2. **Path integration variants** (non-straight line paths)
3. **Feature interactions** (beyond individual pixel importance)
4. **Integration with existing benchmark pipeline**

## Technical Considerations

### JAX/TensorFlow Integration
- Existing codebase uses JAX, but IG examples are TensorFlow
- Need to implement IG using JAX's grad() and vmap() functions
- Leverage existing model loading and batch processing infrastructure

### Memory/Compute Efficiency  
- OOD methods like SLU are already optimized for memory
- IG requires multiple forward passes (one per interpolation step)
- May need batching strategies for large images or many interpolation steps

### Baseline Image Selection Strategy
Start with multiple baseline images and empirically compare:
1. **Zero/Black baseline image**: Standard IG approach
2. **Mean training baseline image**: Represents "typical" in-distribution
3. **Random baseline image**: Control condition  
4. **Blurred input baseline image**: Removes high-frequency details while preserving structure

## Success Metrics

1. **Qualitative**: Do explanations align with human intuition about what makes images OOD?
2. **Quantitative**: Do important pixels identified by IG correlate with adversarial perturbations?
3. **Consistency**: Do different OOD methods highlight similar regions for same images?
4. **Debugging utility**: Can explanations help identify OOD detection failures?

## File Structure Plan

```
src/explainability/
├── __init__.py
├── integrated_gradients.py       # Plain IG baseline method for classification
├── integrated_ood_scores.py      # Core integrated OOD scores implementation
├── baseline_images.py            # Different baseline image strategies  
├── visualizations.py             # Attribution plotting/analysis
└── ood_explainer.py              # Main interface class

experiments/
├── ood_explanation_demo.py      # Demo script showing OOD explanations
├── baseline_image_comparison.py # Compare different baseline image strategies
└── method_comparison.py         # Compare Integrated OOD Scores vs baseline methods
```

This planning document establishes the foundation for implementing an OOD explainer using **Integrated OOD Scores**, building on the existing uncertainty quantification infrastructure while adapting IG's proven methodology to the OOD detection domain. We will implement plain Integrated Gradients as a baseline method for comparison, and carefully distinguish between baseline images (for path integration) and baseline methods (for algorithmic comparison).