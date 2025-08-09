"""
Integrated Gradients implementation for neural network explainability.

This module provides a JAX-based implementation of the Integrated Gradients
attribution method for explaining neural network predictions.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Callable
import time

from .interpolation import interpolate_images, validate_image_shapes
from .gradient_utils import compute_gradients_at_interpolated_points, create_prediction_fn
from .baseline_images import create_zero_baseline, validate_baseline_image


class IntegratedGradients:
    """
    Integrated Gradients attribution method for neural network explanations.
    
    This implementation follows the original paper "Axiomatic Attribution for Deep Networks"
    by Sundararajan et al. (2017), providing pixel-level attribution scores for 
    classification predictions.
    """
    
    def __init__(self, model, params_dict):
        """
        Initialize IntegratedGradients explainer.
        
        Args:
            model: JAX/Flax model (wrapped Model class)
            params_dict: Model parameters dictionary
        """
        self.model = model
        self.params_dict = params_dict
        
    def explain(self, 
                input_image: jnp.ndarray,
                baseline_image: Optional[jnp.ndarray] = None,
                target_class: Optional[int] = None,
                steps: int = 50) -> jnp.ndarray:
        """
        Compute Integrated Gradients attributions for a single image.
        
        Args:
            input_image: Input image to explain, shape (H, W, C)
            baseline_image: Baseline image (None for zero baseline), shape (H, W, C)
            target_class: Target class index (None for max logit)
            steps: Number of interpolation steps
            
        Returns:
            attributions: Attribution scores, same shape as input_image
        """
        # Create zero baseline if none provided
        if baseline_image is None:
            baseline_image = create_zero_baseline(input_image.shape)
        
        # Validate inputs
        validate_image_shapes(baseline_image, input_image)
        # Note: Skip validate_baseline_image to avoid tracer issues in JIT compilation
        
        # Generate interpolated images along path
        interpolated_images, alphas = interpolate_images(
            baseline_image, input_image, steps
        )
        
        # Compute gradients at each interpolation point
        gradients = compute_gradients_at_interpolated_points(
            self.model, self.params_dict, interpolated_images, target_class
        )
        
        # Integrate gradients using trapezoidal rule
        attributions = self._integrate_gradients(
            gradients, input_image, baseline_image
        )
        
        return attributions
    
    def batch_explain(self, 
                     input_images: jnp.ndarray,
                     baseline_images: Optional[jnp.ndarray] = None,
                     target_classes: Optional[jnp.ndarray] = None,
                     steps: int = 50) -> jnp.ndarray:
        """
        Compute Integrated Gradients attributions for a batch of images.
        
        Args:
            input_images: Batch of input images, shape (batch_size, H, W, C)
            baseline_images: Batch of baseline images (None for zero baselines)
            target_classes: Target class indices (None for max logits)
            steps: Number of interpolation steps
            
        Returns:
            attributions: Attribution scores, same shape as input_images
        """
        batch_size = input_images.shape[0]
        
        # Create zero baselines if none provided
        if baseline_images is None:
            baseline_images = create_zero_baseline(input_images.shape)
        
        # Handle single baseline for entire batch
        if len(baseline_images.shape) == 3:  # (H, W, C)
            baseline_images = jnp.broadcast_to(
                baseline_images[None, ...], input_images.shape
            )
        
        # Vectorize explanation over batch
        if target_classes is None:
            # No target classes - use None for each image
            explain_fn = jax.vmap(
                lambda img, baseline: self.explain(img, baseline, None, steps),
                in_axes=(0, 0)
            )
            return explain_fn(input_images, baseline_images)
        else:
            # Different target class for each image
            explain_fn = jax.vmap(
                lambda img, baseline, target: self.explain(img, baseline, target, steps),
                in_axes=(0, 0, 0)
            )
            return explain_fn(input_images, baseline_images, target_classes)
    
    def _integrate_gradients(self, 
                           gradients: jnp.ndarray,
                           input_image: jnp.ndarray,
                           baseline_image: jnp.ndarray) -> jnp.ndarray:
        """
        Integrate gradients using trapezoidal rule approximation.
        
        Args:
            gradients: Gradients at interpolation points, shape (steps+1, H, W, C)
            input_image: Original input image
            baseline_image: Baseline image
            
        Returns:
            attributions: Integrated attribution scores
        """
        # Use trapezoidal rule: average of gradients (excluding endpoints)
        # This approximates the integral ∫[0,1] ∇F(baseline + α*(input-baseline)) dα
        avg_gradients = jnp.mean(gradients, axis=0)
        
        # Scale by path difference: (input - baseline) * ∫gradients
        attributions = (input_image - baseline_image) * avg_gradients
        
        return attributions
    
    def verify_completeness(self,
                           input_image: jnp.ndarray,
                           baseline_image: jnp.ndarray,
                           attributions: jnp.ndarray,
                           target_class: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Verify the completeness axiom of Integrated Gradients.
        
        The completeness axiom states that the sum of attributions should equal
        the difference between the model output at input and baseline.
        
        Args:
            input_image: Input image
            baseline_image: Baseline image  
            attributions: Computed attributions
            target_class: Target class for verification
            
        Returns:
            tuple: (attribution_sum, output_diff, relative_error)
        """
        # Create prediction function
        predict_fn = create_prediction_fn(self.model, self.params_dict, target_class)
        
        # Compute model outputs
        input_output = predict_fn(input_image[None, ...])[0]
        baseline_output = predict_fn(baseline_image[None, ...])[0]
        output_diff = input_output - baseline_output
        
        # Sum attributions
        attribution_sum = jnp.sum(attributions)
        
        # Compute relative error
        relative_error = jnp.abs(attribution_sum - output_diff) / (jnp.abs(output_diff) + 1e-8)
        
        return float(attribution_sum), float(output_diff), float(relative_error)
    
    def benchmark_performance(self,
                            image_shape: Tuple[int, ...],
                            steps_list: list = [20, 50, 100],
                            num_trials: int = 3) -> dict:
        """
        Benchmark performance for different numbers of integration steps.
        
        Args:
            image_shape: Shape of test images
            steps_list: List of step counts to test
            num_trials: Number of trials per configuration
            
        Returns:
            dict: Performance results with timing information
        """
        # Create dummy input and baseline
        key = jax.random.PRNGKey(42)
        input_image = jax.random.uniform(key, image_shape)
        baseline_image = create_zero_baseline(image_shape)
        
        results = {}
        
        for steps in steps_list:
            times = []
            
            for _ in range(num_trials):
                start_time = time.time()
                _ = self.explain(input_image, baseline_image, steps=steps)
                end_time = time.time()
                times.append(end_time - start_time)
            
            results[steps] = {
                'mean_time': jnp.mean(jnp.array(times)),
                'std_time': jnp.std(jnp.array(times)),
                'times': times
            }
        
        return results
    
    def compute_attribution_stats(self, attributions: jnp.ndarray) -> dict:
        """
        Compute statistical summary of attribution values.
        
        Args:
            attributions: Attribution array
            
        Returns:
            dict: Statistical summary
        """
        return {
            'mean': float(jnp.mean(attributions)),
            'std': float(jnp.std(attributions)),
            'min': float(jnp.min(attributions)),
            'max': float(jnp.max(attributions)),
            'abs_mean': float(jnp.mean(jnp.abs(attributions))),
            'positive_ratio': float(jnp.mean(attributions > 0)),
            'zero_ratio': float(jnp.mean(attributions == 0))
        }