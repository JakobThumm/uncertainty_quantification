#!/usr/bin/env python3
"""
Flexible visualization script for explainability methods.

This script can test different explainers (IG, Integrated OOD Scores, etc.)
on different datasets (MNIST, CIFAR-10, etc.) with various baseline strategies.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import argparse
import os
from typing import Optional, Tuple, Dict, Any

from src.models import pretrained_model_from_string
from src.datasets import dataloader_from_string
from src.explainability import IntegratedGradients, create_zero_baseline, create_mean_baseline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize explainability methods')
    
    # Dataset and model arguments
    parser.add_argument('--dataset', type=str, default='MNIST',
                       choices=['MNIST', 'FMNIST', 'CIFAR-10', 'SVHN'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default='MLP_depth1_hidden20',
                       help='Model architecture')
    parser.add_argument('--model_seed', type=int, default=1,
                       help='Model seed')
    parser.add_argument('--run_name', type=str, default='good',
                       help='Model run name')
    parser.add_argument('--model_path', type=str, default='../models',
                       help='Path to saved models')
    
    # Explainer arguments
    parser.add_argument('--explainer', type=str, default='ig',
                       choices=['ig', 'integrated_ood'],  # Will add more later
                       help='Explainer method to use')
    parser.add_argument('--baseline_type', type=str, default='zero',
                       choices=['zero', 'mean', 'noise'],
                       help='Type of baseline image')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of integration steps')
    parser.add_argument('--target_class', type=int, default=None,
                       help='Target class (None for predicted class)')
    
    # Visualization arguments
    parser.add_argument('--num_images', type=int, default=3,
                       help='Number of images to visualize')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for data loading')
    
    # Testing arguments
    parser.add_argument('--verify_completeness', action='store_true',
                       help='Verify completeness axiom')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    return parser.parse_args()


def load_model_and_data(args) -> Tuple[Any, Any, Any, Any]:
    """Load pretrained model and dataset."""
    print(f"Loading {args.model} trained on {args.dataset}...")
    
    try:
        model, params_dict, model_args = pretrained_model_from_string(
            dataset_name=args.dataset,
            model_name=args.model,
            run_name=args.run_name,
            seed=args.model_seed,
            save_path=args.model_path
        )
        print(f"✓ Model loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print(f"Expected path: {args.model_path}/{args.dataset}/{args.model}/seed_{args.model_seed}/{args.run_name}_params.pickle")
        return None, None, None, None
    
    # Load test data
    _, _, test_loader = dataloader_from_string(
        args.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=42
    )
    
    print(f"✓ Dataset {args.dataset} loaded successfully")
    
    return model, params_dict, model_args, test_loader


def create_baseline_image(baseline_type: str, 
                         image_shape: Tuple[int, ...],
                         train_loader: Optional[Any] = None) -> jnp.ndarray:
    """Create baseline image based on specified type."""
    if baseline_type == 'zero':
        return create_zero_baseline(image_shape)
    
    elif baseline_type == 'mean':
        if train_loader is None:
            print("Warning: No training data provided for mean baseline, using zero baseline")
            return create_zero_baseline(image_shape)
        return create_mean_baseline(train_loader)
    
    elif baseline_type == 'noise':
        key = jax.random.PRNGKey(42)
        return jax.random.uniform(key, image_shape, minval=0.0, maxval=1.0)
    
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


def initialize_explainer(explainer_type: str, model, params_dict):
    """Initialize the specified explainer."""
    if explainer_type == 'ig':
        return IntegratedGradients(model, params_dict)
    
    elif explainer_type == 'integrated_ood':
        # TODO: Implement when IntegratedOODScores is ready
        raise NotImplementedError("Integrated OOD Scores not yet implemented")
    
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")


def explain_images(explainer, images: jnp.ndarray, baseline_image: jnp.ndarray,
                  target_classes: Optional[jnp.ndarray] = None,
                  steps: int = 50) -> jnp.ndarray:
    """Compute explanations for batch of images."""
    print(f"Computing explanations with {steps} steps...")
    
    if len(images.shape) == 3:  # Single image
        if target_classes is not None:
            target_class = int(target_classes) if target_classes.ndim == 0 else int(target_classes[0])
        else:
            target_class = None
            
        return explainer.explain(images, baseline_image, target_class, steps)
    
    else:  # Batch of images
        return explainer.batch_explain(images, baseline_image, target_classes, steps)


def visualize_single_explanation(image: jnp.ndarray, 
                                attribution: jnp.ndarray,
                                predicted_class: int,
                                true_label: int,
                                confidence: float,
                                save_path: str):
    """Visualize explanation for a single image."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Handle different image formats
    if len(image.shape) == 3 and image.shape[2] == 1:  # Grayscale
        img_display = image[:, :, 0]
        attr_display = attribution[:, :, 0]
        cmap_img = 'gray'
    elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
        img_display = image
        attr_display = jnp.mean(attribution, axis=2)  # Average across channels
        cmap_img = None
    else:
        img_display = image
        attr_display = attribution
        cmap_img = 'gray'
    
    # Original image
    axes[0].imshow(img_display, cmap=cmap_img)
    axes[0].set_title(f'Input Image\nTrue: {true_label}, Pred: {predicted_class}\nConf: {confidence:.3f}')
    axes[0].axis('off')
    
    # Attribution magnitude
    abs_attr = jnp.abs(attr_display)
    im1 = axes[1].imshow(abs_attr, cmap='hot')
    axes[1].set_title('Attribution Magnitude')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Positive/negative attributions
    max_val = jnp.max(jnp.abs(attr_display))
    im2 = axes[2].imshow(attr_display, cmap='RdBu_r', vmin=-max_val, vmax=max_val)
    axes[2].set_title('Attribution (+/-)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_completeness_verification(explainer, images: jnp.ndarray, 
                                baseline_image: jnp.ndarray,
                                attributions: jnp.ndarray,
                                target_classes: Optional[jnp.ndarray] = None):
    """Verify completeness axiom for batch of images."""
    print("\nVerifying completeness axiom...")
    
    errors = []
    for i in range(len(images)):
        target_class = int(target_classes[i]) if target_classes is not None else None
        
        attr_sum, output_diff, rel_error = explainer.verify_completeness(
            images[i], baseline_image, attributions[i], target_class
        )
        
        errors.append(rel_error)
        if i < 3:  # Show details for first 3 images
            print(f"  Image {i}: attr_sum={attr_sum:.6f}, output_diff={output_diff:.6f}, error={rel_error:.4f}")
    
    mean_error = jnp.mean(jnp.array(errors))
    print(f"Mean relative error: {mean_error:.4f}")
    
    if mean_error < 0.05:
        print("✓ Completeness axiom satisfied (mean error < 5%)")
    else:
        print("✗ Completeness axiom violated (mean error >= 5%)")
    
    return errors


def run_benchmark(explainer, image_shape: Tuple[int, ...]):
    """Run performance benchmark."""
    print("\nRunning performance benchmark...")
    
    results = explainer.benchmark_performance(
        image_shape, 
        steps_list=[20, 50, 100],
        num_trials=3
    )
    
    for steps, timing in results.items():
        print(f"  {steps:3d} steps: {timing['mean_time']:.3f} ± {timing['std_time']:.3f} seconds")
    
    return results


def main():
    """Main function."""
    args = parse_arguments()
    
    print("=" * 60)
    print(f"Explainer Visualization: {args.explainer.upper()} on {args.dataset}")
    print("=" * 60)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model and data
    model, params_dict, model_args, test_loader = load_model_and_data(args)
    if model is None:
        return
    
    # Get test images
    first_batch = next(iter(test_loader))
    images = jnp.array(first_batch[0].numpy())
    labels = jnp.array(first_batch[1].numpy())
    
    # Select subset for visualization
    num_viz = min(args.num_images, len(images))
    viz_images = images[:num_viz]
    viz_labels = labels[:num_viz]
    
    # Convert one-hot labels to class indices if needed
    if len(viz_labels.shape) > 1 and viz_labels.shape[1] > 1:
        viz_labels = jnp.argmax(viz_labels, axis=1)
    
    print(f"\nProcessing {num_viz} images with shape {viz_images.shape[1:]}")
    
    # Create baseline image
    baseline_image = create_baseline_image(args.baseline_type, viz_images.shape[1:])
    print(f"Using {args.baseline_type} baseline: range [{jnp.min(baseline_image):.3f}, {jnp.max(baseline_image):.3f}]")
    
    # Initialize explainer
    explainer = initialize_explainer(args.explainer, model, params_dict)
    
    # Get model predictions
    if model.has_batch_stats:
        logits = model.apply_test(params_dict['params'], params_dict['batch_stats'], viz_images)
    else:
        logits = model.apply_test(params_dict['params'], viz_images)
    predicted_classes = jnp.argmax(logits, axis=1)
    confidences = jnp.max(jax.nn.softmax(logits), axis=1)
    
    target_classes = predicted_classes if args.target_class is None else jnp.full_like(predicted_classes, args.target_class)
    
    print(f"Predictions: {predicted_classes} (confidences: {confidences})")
    print(f"True labels: {viz_labels}")
    
    # Compute explanations
    attributions = explain_images(explainer, viz_images, baseline_image, target_classes, args.steps)
    
    # Create visualizations
    print(f"\nCreating visualizations in {args.save_dir}/...")
    for i in range(num_viz):
        save_path = f"{args.save_dir}/{args.explainer}_{args.dataset}_image_{i}.png"
        visualize_single_explanation(
            viz_images[i], attributions[i], 
            int(predicted_classes[i]), int(viz_labels[i]), float(confidences[i]),
            save_path
        )
    
    print(f"✓ Saved {num_viz} visualizations")
    
    # Optional completeness verification
    if args.verify_completeness and hasattr(explainer, 'verify_completeness'):
        run_completeness_verification(explainer, viz_images, baseline_image, attributions, target_classes)
    
    # Optional benchmark
    if args.benchmark and hasattr(explainer, 'benchmark_performance'):
        run_benchmark(explainer, viz_images.shape[1:])
    
    print(f"\n✓ All processing completed successfully!")


if __name__ == "__main__":
    main()