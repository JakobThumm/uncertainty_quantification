"""
Baseline image creation utilities for Integrated Gradients.

This module provides functions for creating different types of baseline images
used as starting points for path integration in Integrated Gradients.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional


def create_zero_baseline(image_shape: Tuple[int, ...]) -> jnp.ndarray:
    """
    Create zero/black baseline image (standard IG approach).
    
    Args:
        image_shape: Shape of the target image (H, W, C) or (batch_size, H, W, C)
        
    Returns:
        baseline_image: All-zero image of specified shape
    """
    return jnp.zeros(image_shape)


def create_mean_baseline(train_loader, max_samples: Optional[int] = None) -> jnp.ndarray:
    """
    Create baseline image as mean of training data.
    
    Args:
        train_loader: Training data loader
        max_samples: Maximum number of samples to use (None for all)
        
    Returns:
        baseline_image: Mean training image, shape (H, W, C)
    """
    all_images = []
    samples_used = 0
    
    for batch in train_loader:
        # Convert to JAX array
        images = jnp.array(batch[0].numpy())
        all_images.append(images)
        
        samples_used += images.shape[0]
        if max_samples is not None and samples_used >= max_samples:
            break
    
    # Concatenate all images and compute mean
    all_images = jnp.concatenate(all_images, axis=0)
    if max_samples is not None:
        all_images = all_images[:max_samples]
        
    return jnp.mean(all_images, axis=0)


def create_noise_baseline(image_shape: Tuple[int, ...], 
                         key: jax.random.PRNGKey,
                         noise_type: str = "uniform",
                         **kwargs) -> jnp.ndarray:
    """
    Create random noise baseline image for control experiments.
    
    Args:
        image_shape: Shape of the target image
        key: JAX random key
        noise_type: Type of noise ("uniform", "normal")  
        **kwargs: Additional arguments for noise generation
        
    Returns:
        baseline_image: Random noise image
    """
    if noise_type == "uniform":
        minval = kwargs.get("minval", 0.0)
        maxval = kwargs.get("maxval", 1.0)
        return jax.random.uniform(key, image_shape, minval=minval, maxval=maxval)
    
    elif noise_type == "normal":
        mean = kwargs.get("mean", 0.0)
        std = kwargs.get("std", 1.0)
        return mean + std * jax.random.normal(key, image_shape)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def create_blurred_baseline(input_image: jnp.ndarray, 
                           blur_radius: float = 5.0) -> jnp.ndarray:
    """
    Create blurred version of input image as baseline.
    
    Note: This is a simplified implementation. For production use,
    consider implementing proper Gaussian blur with convolution.
    
    Args:
        input_image: Input image to blur
        blur_radius: Radius of blur effect
        
    Returns:
        baseline_image: Blurred version of input
    """
    # Simple approximation: downsample and upsample
    # This removes high-frequency details while preserving structure
    original_shape = input_image.shape
    
    # Downsample by factor related to blur radius
    downsample_factor = max(1, int(blur_radius // 2))
    
    if len(original_shape) == 3:  # (H, W, C)
        h, w, c = original_shape
        downsampled = input_image[::downsample_factor, ::downsample_factor, :]
        
        # Simple upsampling by repeating pixels
        upsampled = jnp.repeat(
            jnp.repeat(downsampled, downsample_factor, axis=0),
            downsample_factor, axis=1
        )
        
        # Crop to original size if needed
        upsampled = upsampled[:h, :w, :]
        
    else:
        # For batch processing, apply to each image
        upsampled = jax.vmap(lambda img: create_blurred_baseline(img, blur_radius))(input_image)
    
    return upsampled


def create_baseline_from_statistics(train_loader,
                                  baseline_type: str = "mean",
                                  percentile: Optional[float] = None) -> jnp.ndarray:
    """
    Create baseline image using various statistical measures of training data.
    
    Args:
        train_loader: Training data loader
        baseline_type: Type of baseline ("mean", "median", "percentile")
        percentile: Percentile to use if baseline_type="percentile" (0-100)
        
    Returns:
        baseline_image: Statistical baseline image
    """
    all_images = []
    
    for batch in train_loader:
        images = jnp.array(batch[0].numpy())
        all_images.append(images)
    
    all_images = jnp.concatenate(all_images, axis=0)
    
    if baseline_type == "mean":
        return jnp.mean(all_images, axis=0)
    
    elif baseline_type == "median":
        return jnp.median(all_images, axis=0)
    
    elif baseline_type == "percentile":
        if percentile is None:
            raise ValueError("percentile must be specified for percentile baseline")
        return jnp.percentile(all_images, percentile, axis=0)
    
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


def validate_baseline_image(baseline_image: jnp.ndarray) -> None:
    """
    Validate that baseline image has correct properties.
    
    Args:
        baseline_image: Baseline image to validate
        
    Raises:
        ValueError: If validation fails
    """
    # Check for reasonable value ranges (assuming normalized images)
    if jnp.any(baseline_image < -2.0) or jnp.any(baseline_image > 2.0):
        print(f"Warning: Baseline image values outside expected range [-2, 2]. "
              f"Min: {jnp.min(baseline_image):.3f}, Max: {jnp.max(baseline_image):.3f}")


def create_adaptive_baseline(input_image: jnp.ndarray,
                            adaptation_strength: float = 0.1) -> jnp.ndarray:
    """
    Create baseline that adapts to input image characteristics.
    
    This creates a baseline that preserves some low-level statistics
    of the input image while removing high-level semantic content.
    
    Args:
        input_image: Input image to adapt to
        adaptation_strength: How much to adapt (0.0 = zero baseline, 1.0 = original image)
        
    Returns:
        baseline_image: Adapted baseline image
    """
    # Start with zero baseline
    baseline = jnp.zeros_like(input_image)
    
    # Add small fraction of original image to preserve basic statistics
    adapted_baseline = baseline + adaptation_strength * input_image
    
    return adapted_baseline