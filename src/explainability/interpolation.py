"""
Path interpolation utilities for Integrated Gradients.

This module provides functions for generating interpolated images along 
straight-line paths from baseline images to input images.
"""

import jax.numpy as jnp
from typing import Tuple


def interpolate_images(baseline_image: jnp.ndarray, 
                      input_image: jnp.ndarray, 
                      steps: int = 50) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate interpolated images along straight line path from baseline to input.
    
    Args:
        baseline_image: Baseline image, shape (..., H, W, C)
        input_image: Target input image, shape (..., H, W, C) 
        steps: Number of interpolation steps (default: 50)
        
    Returns:
        interpolated_images: Array of interpolated images, shape (steps+1, ..., H, W, C)
        alphas: Interpolation coefficients, shape (steps+1,)
        
    Note:
        - Includes both endpoints (α=0 and α=1)
        - Uses linear interpolation: baseline + α * (input - baseline)
        - Works with batched inputs (multiple images simultaneously)
    """
    # Create interpolation coefficients from 0 to 1
    alphas = jnp.linspace(0.0, 1.0, steps + 1)
    
    # Compute difference vector
    diff = input_image - baseline_image
    
    # Generate interpolated images: baseline + α * (input - baseline)
    # Broadcasting: (steps+1, 1, 1, 1) * (H, W, C) + (H, W, C)
    alpha_shape = [steps + 1] + [1] * len(baseline_image.shape)
    alphas_reshaped = alphas.reshape(alpha_shape)
    
    interpolated_images = baseline_image[None, ...] + alphas_reshaped * diff[None, ...]
    
    return interpolated_images, alphas


def interpolate_batch(baseline_images: jnp.ndarray,
                     input_images: jnp.ndarray,
                     steps: int = 50) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate interpolated images for a batch of image pairs.
    
    Args:
        baseline_images: Batch of baseline images, shape (batch_size, H, W, C)
        input_images: Batch of input images, shape (batch_size, H, W, C)
        steps: Number of interpolation steps
        
    Returns:
        interpolated_images: Shape (batch_size, steps+1, H, W, C)
        alphas: Interpolation coefficients, shape (steps+1,)
    """
    # Vectorize over batch dimension
    batch_interpolate = jax.vmap(interpolate_images, in_axes=(0, 0, None))
    interpolated_batch, alphas = batch_interpolate(baseline_images, input_images, steps)
    
    return interpolated_batch, alphas


def validate_image_shapes(baseline_image: jnp.ndarray, input_image: jnp.ndarray) -> None:
    """
    Validate that baseline and input images have compatible shapes.
    
    Args:
        baseline_image: Baseline image array
        input_image: Input image array
        
    Raises:
        ValueError: If shapes are incompatible
    """
    if baseline_image.shape != input_image.shape:
        raise ValueError(
            f"Baseline image shape {baseline_image.shape} does not match "
            f"input image shape {input_image.shape}"
        )