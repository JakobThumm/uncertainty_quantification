#!/usr/bin/env python3
"""
Extended Phase 3: Test Integrated Gradients on challenging OOD scenarios.

This script tests IG robustness on increasingly challenging OOD data:
1. Cross-domain: FMNIST model → MNIST data  
2. Rotated images: FMNIST model → 90° rotated MNIST
3. Heavily rotated: FMNIST model → 180° rotated MNIST
4. Analysis of attribution patterns under severe distribution shift
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from src.models import pretrained_model_from_string
from src.datasets.mnist import get_mnist, get_rotated_mnist
from src.explainability import IntegratedGradients, create_zero_baseline


def test_challenging_ood_scenarios():
    """Test IG on progressively challenging OOD scenarios."""
    
    print("="*80)
    print("EXTENDED PHASE 3: CHALLENGING OOD SCENARIOS FOR INTEGRATED GRADIENTS")
    print("="*80)
    
    # Load FMNIST-trained model
    print("Loading FMNIST-trained LeNet model...")
    model, params_dict, model_args = pretrained_model_from_string(
        dataset_name='FMNIST',
        model_name='LeNet', 
        seed=1,
        run_name='good',
        save_path='../models'
    )
    print("✓ Model loaded successfully\n")
    
    # Initialize IG explainer
    explainer = IntegratedGradients(model, params_dict)
    baseline = create_zero_baseline((28, 28, 1))
    
    # Define test scenarios
    scenarios = [
        {
            'name': 'Standard MNIST (moderate OOD)',
            'description': 'FMNIST model on normal MNIST digits',
            'data_fn': lambda: get_mnist(batch_size=3, shuffle=False, download=True, data_path='../datasets')[2]
        },
        {
            'name': '90° Rotated MNIST (challenging OOD)', 
            'description': 'FMNIST model on 90-degree rotated MNIST digits',
            'data_fn': lambda: get_rotated_mnist(angle=90, batch_size=3, shuffle=False, download=True, data_path='../datasets')[2]
        },
        {
            'name': '180° Rotated MNIST (extreme OOD)',
            'description': 'FMNIST model on 180-degree rotated (upside-down) MNIST digits', 
            'data_fn': lambda: get_rotated_mnist(angle=180, batch_size=3, shuffle=False, download=True, data_path='../datasets')[2]
        }
    ]
    
    results = []
    
    # Test each scenario
    for i, scenario in enumerate(scenarios):
        print(f"\n{'-'*60}")
        print(f"SCENARIO {i+1}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'-'*60}")
        
        # Load test data
        test_loader = scenario['data_fn']()
        test_batch = next(iter(test_loader))
        images = jnp.array(test_batch[0].numpy())[:3]  # First 3 images
        labels = jnp.array(test_batch[1].numpy())[:3]
        true_labels = jnp.argmax(labels, axis=1) if labels.ndim > 1 else labels
        
        print(f"Processing {len(images)} test images...")
        
        # Get model predictions
        if model.has_batch_stats and 'batch_stats' in params_dict:
            logits = model.apply_test(params_dict['params'], params_dict['batch_stats'], images)
        else:
            logits = model.apply_test(params_dict['params'], images)
        predictions = jnp.argmax(logits, axis=1)
        confidences = jnp.max(jax.nn.softmax(logits, axis=1), axis=1)
        
        print(f"True MNIST labels: {true_labels}")
        print(f"FMNIST model predictions: {predictions}")
        print(f"Confidence scores: {confidences}")
        print(f"Accuracy: {jnp.mean(predictions == true_labels):.1%}")
        
        # Compute explanations
        print("Computing explanations with 100 integration steps...")
        explanations = []
        for j, image in enumerate(images):
            explanation = explainer.explain(image, baseline, target_class=None, steps=100)
            explanations.append(explanation)
            print(f"  Image {j+1}/3: attribution range [{explanation.min():.6f}, {explanation.max():.6f}]")
        
        explanations = jnp.array(explanations)
        
        # Store results
        scenario_results = {
            'name': scenario['name'],
            'images': images,
            'true_labels': true_labels,
            'predictions': predictions,
            'confidences': confidences,
            'explanations': explanations,
            'accuracy': float(jnp.mean(predictions == true_labels))
        }
        results.append(scenario_results)
        
        print(f"✓ Scenario {i+1} completed")
    
    # Create comprehensive visualization
    print(f"\n{'='*60}")
    print("CREATING COMPARATIVE VISUALIZATION")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(len(scenarios), 9, figsize=(20, 4*len(scenarios)))
    if len(scenarios) == 1:
        axes = axes[None, :]
    
    for i, result in enumerate(results):
        # Scenario title
        fig.text(0.05, 1 - (i+0.5)/len(scenarios), f"SCENARIO {i+1}: {result['name']}", 
                rotation=90, va='center', ha='center', fontsize=12, weight='bold')
        
        for j in range(3):  # 3 images per scenario
            # Original image
            axes[i, j*3].imshow(result['images'][j].squeeze(), cmap='gray')
            axes[i, j*3].set_title(f"Image {j+1}\\nTrue: {result['true_labels'][j]}\\n"
                                 f"Pred: {result['predictions'][j]} ({result['confidences'][j]:.2%})")
            axes[i, j*3].axis('off')
            
            # Attribution magnitude
            attr_mag = jnp.abs(result['explanations'][j]).squeeze()
            im1 = axes[i, j*3+1].imshow(attr_mag, cmap='hot', vmin=0, vmax=attr_mag.max())
            axes[i, j*3+1].set_title("Attribution\\nMagnitude")
            axes[i, j*3+1].axis('off')
            plt.colorbar(im1, ax=axes[i, j*3+1], fraction=0.046)
            
            # Attribution polarity
            attr_raw = result['explanations'][j].squeeze()
            vmax = max(abs(attr_raw.min()), abs(attr_raw.max()))
            im2 = axes[i, j*3+2].imshow(attr_raw, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i, j*3+2].set_title("Attribution\\nPolarity")
            axes[i, j*3+2].axis('off')
            plt.colorbar(im2, ax=axes[i, j*3+2], fraction=0.046)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1)  # Make room for scenario labels
    plt.savefig('challenging_ood_scenarios_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comprehensive visualization to 'challenging_ood_scenarios_comparison.png'")
    
    # Analysis summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for i, result in enumerate(results):
        print(f"\nScenario {i+1}: {result['name']}")
        print(f"  Model accuracy: {result['accuracy']:.1%}")
        print(f"  Mean confidence: {jnp.mean(result['confidences']):.1%}")
        print(f"  Attribution statistics:")
        attrs = result['explanations']
        print(f"    - Total attribution: {jnp.sum(jnp.abs(attrs)):.2f}")
        print(f"    - Max magnitude: {jnp.max(jnp.abs(attrs)):.6f}")
        print(f"    - Attribution sparsity: {jnp.mean(jnp.abs(attrs) > 0.001):.1%} of pixels active")
    
    print(f"\n{'='*80}")
    print("EXTENDED PHASE 3 COMPLETED SUCCESSFULLY!")
    print("✓ Integrated Gradients tested on progressively challenging OOD scenarios")
    print("✓ Attribution patterns analyzed under severe distribution shift")
    print("✓ Framework established for comparing IG vs Integrated OOD Scores")
    print(f"{'='*80}")
    

if __name__ == '__main__':
    test_challenging_ood_scenarios()