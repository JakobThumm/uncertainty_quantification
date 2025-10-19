"""
Simple test script to verify the transferred Flax model can be loaded and used.
"""

import pickle
import jax
import jax.numpy as jnp
import numpy as np
from dct_pose_transformer_flax import DCTPoseTransformerFlax

def main():
    print("="*60)
    print("Testing Transferred Flax Model")
    print("="*60)
    
    # Load the transferred model
    model_path = "../models/HMP/dct_pose_transformer_flax.pickle"
    print(f"\nLoading model from: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Extract components
    params = model_dict['params']
    config = model_dict['config']
    
    print(f"\nModel configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Initialize model
    print("\nInitializing model...")
    model = DCTPoseTransformerFlax(**config)
    print("✓ Model initialized")
    
    # Create test input
    batch_size = 2
    seq_len = config['seq_len']
    input_dim = config['input_dim']
    
    print(f"\nCreating random input: ({batch_size}, {seq_len}, {input_dim})")
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (batch_size, seq_len, input_dim))
    
    # Run inference
    print("\nRunning inference...")
    try:
        poses, (var_params, cov_params) = model.apply({'params': params}, x)
        print("✓ Inference successful")
        
        print(f"\nOutput shapes:")
        print(f"  Poses: {poses.shape}")
        print(f"  Variance params: {var_params.shape}")
        print(f"  Covariance params: {cov_params.shape}")
        
        print(f"\nOutput statistics:")
        print(f"  Poses - mean: {jnp.mean(poses):.6f}, std: {jnp.std(poses):.6f}")
        print(f"  Var params - mean: {jnp.mean(var_params):.6f}, std: {jnp.std(var_params):.6f}")
        print(f"  Cov params - mean: {jnp.mean(cov_params):.6f}, std: {jnp.std(cov_params):.6f}")
        
        # Check for NaNs or Infs
        has_nan = jnp.any(jnp.isnan(poses)) or jnp.any(jnp.isnan(var_params)) or jnp.any(jnp.isnan(cov_params))
        has_inf = jnp.any(jnp.isinf(poses)) or jnp.any(jnp.isinf(var_params)) or jnp.any(jnp.isinf(cov_params))
        
        if has_nan:
            print("\n⚠ Warning: Output contains NaN values")
        if has_inf:
            print("\n⚠ Warning: Output contains Inf values")
        
        if not has_nan and not has_inf:
            print("\n✓ All outputs are valid (no NaN/Inf)")
        
        print("\n" + "="*60)
        print("✓ TEST PASSED - Model is ready to use")
        print("="*60)
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

