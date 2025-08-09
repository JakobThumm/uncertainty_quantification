"""
Gradient computation utilities for Integrated OOD scores.

This module provides JAX-based functions for computing gradients of model
predictions with respect to input images.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Optional


def create_prediction_fn(model, params_dict, target_class: Optional[int] = None) -> Callable:
    """
    Create a prediction function for gradient computation.
    
    Args:
        model: JAX/Flax model (wrapped Model class)
        params_dict: Model parameters dictionary (with 'params' and possibly 'batch_stats')
        target_class: Specific class to target (None for max logit)
        
    Returns:
        Function that takes images and returns scalar predictions
    """
    def predict_fn(images: jnp.ndarray) -> jnp.ndarray:
        # Apply model to get logits using appropriate interface
        if model.has_batch_stats:
            logits = model.apply_test(params_dict['params'], params_dict['batch_stats'], images)
        else:
            logits = model.apply_test(params_dict['params'], images)
        
        if target_class is not None:
            # Return logit for specific target class
            if len(logits.shape) == 1:  # Single image
                return logits[target_class]
            else:  # Batch of images
                return logits[:, target_class]
        else:
            # Return max logit (most confident prediction)
            return jnp.max(logits, axis=-1)
    
    return predict_fn


def compute_model_gradients(model, 
                          params_dict, 
                          images: jnp.ndarray,
                          target_class: Optional[int] = None) -> jnp.ndarray:
    """
    Compute gradients of model predictions with respect to input images.
    
    Args:
        model: JAX/Flax model (wrapped Model class)
        params_dict: Model parameters dictionary
        images: Input images, shape (batch_size, H, W, C)
        target_class: Specific class to target (None for max logit)
        
    Returns:
        gradients: Gradients w.r.t. inputs, same shape as images
    """
    # Create prediction function
    predict_fn = create_prediction_fn(model, params_dict, target_class)
    
    # Define function to compute gradient for single image
    def single_image_grad(image: jnp.ndarray) -> jnp.ndarray:
        # Add batch dimension, compute prediction, take gradient
        def scalar_fn(img):
            return predict_fn(img[None, ...])[0]  # Remove batch dim from output
        
        return jax.grad(scalar_fn)(image)
    
    # Vectorize over batch dimension
    batch_grad_fn = jax.vmap(single_image_grad)
    return batch_grad_fn(images)


def compute_gradients_at_interpolated_points(model,
                                           params_dict, 
                                           interpolated_images: jnp.ndarray,
                                           target_class: Optional[int] = None) -> jnp.ndarray:
    """
    Compute gradients at each interpolated point along the path.
    
    Args:
        model: JAX/Flax model (wrapped Model class)
        params_dict: Model parameters dictionary
        interpolated_images: Shape (steps+1, H, W, C) or (batch_size, steps+1, H, W, C)
        target_class: Specific class to target (None for max logit)
        
    Returns:
        gradients: Gradients at each interpolation point, same shape as interpolated_images
    """
    if len(interpolated_images.shape) == 4:
        # Single sequence of interpolated images: (steps+1, H, W, C)
        return compute_model_gradients(model, params_dict, interpolated_images, target_class)
    
    elif len(interpolated_images.shape) == 5:
        # Batch of interpolated sequences: (batch_size, steps+1, H, W, C)
        batch_size, steps_plus_one = interpolated_images.shape[:2]
        
        # Reshape to (batch_size * (steps+1), H, W, C)
        reshaped = interpolated_images.reshape((-1,) + interpolated_images.shape[2:])
        
        # Compute gradients for all images
        gradients = compute_model_gradients(model, params_dict, reshaped, target_class)
        
        # Reshape back to (batch_size, steps+1, H, W, C)  
        return gradients.reshape(interpolated_images.shape)
    
    else:
        raise ValueError(f"Unexpected interpolated_images shape: {interpolated_images.shape}")


def verify_gradient_computation(model, params_dict, image: jnp.ndarray, 
                              target_class: Optional[int] = None,
                              epsilon: float = 1e-5) -> bool:
    """
    Verify gradient computation using finite differences (for debugging).
    
    Args:
        model: JAX/Flax model (wrapped Model class)
        params_dict: Model parameters dictionary
        image: Single input image
        target_class: Target class for gradient computation
        epsilon: Finite difference step size
        
    Returns:
        bool: True if gradients are approximately correct
    """
    # Compute analytical gradients
    analytical_grad = compute_model_gradients(model, params_dict, image[None, ...], target_class)[0]
    
    # Create prediction function
    predict_fn = create_prediction_fn(model, params_dict, target_class)
    
    # Compute finite difference gradient for first pixel as sanity check
    original_pred = predict_fn(image[None, ...])[0]
    
    # Perturb first pixel
    perturbed_image = image.at[0, 0, 0].set(image[0, 0, 0] + epsilon)
    perturbed_pred = predict_fn(perturbed_image[None, ...])[0]
    
    # Finite difference approximation
    finite_diff = (perturbed_pred - original_pred) / epsilon
    analytical_val = analytical_grad[0, 0, 0]
    
    # Check if they're close
    relative_error = jnp.abs(finite_diff - analytical_val) / (jnp.abs(analytical_val) + 1e-8)
    return relative_error < 1e-2  # Allow 1% relative error