"""
Explainability module for uncertainty quantification methods.

This module provides implementations of attribution methods including:
- Integrated Gradients (baseline method for classification)
- Integrated OOD Scores (our method for OOD explanations)
"""

from .integrated_gradients import IntegratedGradients
from .baseline_images import create_zero_baseline, create_mean_baseline, create_noise_baseline
from .interpolation import interpolate_images
from .gradient_utils import compute_model_gradients

__all__ = [
    "IntegratedGradients",
    "create_zero_baseline", 
    "create_mean_baseline",
    "create_noise_baseline",
    "interpolate_images",
    "compute_model_gradients"
]