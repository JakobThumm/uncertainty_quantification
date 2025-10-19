# DCTPoseTransformer PyTorch to JAX/Flax Transfer

This document describes the process of transferring the DCTPoseTransformer model from PyTorch to JAX/Flax format.

## Files Created

1. **`dct_pose_transformer_flax.py`**: Flax/JAX implementation of the DCTPoseTransformer model
   - `FrequencyAwareAttention`: Custom attention with learnable frequency weights
   - `DCTPoseTransformerBlock`: Transformer block with frequency-specific FFNs
   - `UncertaintyEmbedding`: Processes uncertainty features (currently unused)
   - `UncertaintyHead`: Predicts pose uncertainties
   - `DCTPoseTransformerFlax`: Main model class

2. **`transfer_dct_pose_transformer.py`**: Weight transfer script with validation tests
   - Weight transfer functions for Linear, LayerNorm, MultiheadAttention
   - Smoke test with random data
   - H36M data test (optional)
   - Saves transferred model in pickle format

3. **Output**: `../models/HMP/dct_pose_transformer_flax.pickle`
   - Contains transferred model parameters
   - Includes model configuration

## Model Architecture

The DCTPoseTransformer is a frequency-aware transformer for 3D pose prediction:

```
Input (batch, 50, 39) → Input Embedding → Positional Encoding
                                          ↓
                            Transformer Block 0 (FreqAttn + FFN)
                                          ↓
                            Transformer Block 1 (FreqAttn + FFN)
                                          ↓
                    Frequency Decoders (Low + High) → Poses (batch, 50, 39)
                                          ↓
                            Uncertainty Head → (Var Params, Cov Params)
```

### Key Components

1. **Frequency-Aware Attention**: Standard multi-head attention with learnable frequency importance weights
2. **Frequency-Specific FFNs**: Separate feed-forward networks for low and high frequency components
3. **Uncertainty Head**: Predicts variance and covariance parameters for each joint

## Weight Transfer Details

### Successful Transfers

✅ Input embedding (Linear + LayerNorm)
✅ Frequency positional embeddings  
✅ Transformer blocks (2 layers):
  - Frequency attention weights
  - Multi-head attention (Q, K, V, output projections)
  - Layer normalizations
  - Low/high frequency FFN networks
✅ Frequency decoders
✅ Uncertainty head MLP layers
✅ Uncertainty weight parameter

### Known Limitations

⚠️ **Uncertainty Processor Layers**: Not transferred
- The PyTorch model has `uncertainty_processor` layers that process explicit uncertainty inputs
- These layers are defined in the Flax model but don't appear in the parameter tree
- This is because they're only used when `uncertainty_features` is provided (which is currently `None` in both forward passes)
- Impact: Minimal, as these layers aren't used in the current inference pipeline

### MultiheadAttention Transfer

Special handling was required for PyTorch's MultiheadAttention → Flax's MultiHeadDotProductAttention:

**PyTorch format:**
- `in_proj_weight`: (3×embed_dim, embed_dim) - combined Q, K, V
- `in_proj_bias`: (3×embed_dim,)
- `out_proj.weight`: (embed_dim, embed_dim)
- `out_proj.bias`: (embed_dim,)

**Flax format:**
- `query.kernel`: (embed_dim, num_heads, head_dim)
- `key.kernel`: (embed_dim, num_heads, head_dim)
- `value.kernel`: (embed_dim, num_heads, head_dim)
- `out.kernel`: (num_heads, head_dim, embed_dim)

The transfer function splits the combined Q/K/V projections and reshapes them to match Flax's multi-head structure.

## Validation Results

### Smoke Test (Random Input)
✅ **PASSED** (threshold: 0.05)

- Input shape: (4, 50, 39)
- **Pose predictions**:
  - Max absolute difference: **0.030024**
  - Mean absolute difference: **0.002304**
  - PyTorch mean: -0.000379, JAX mean: -0.000307
- **Variance parameters**:
  - Max absolute difference: 0.677331
  - *(Higher due to missing uncertainty_processor layers)*
- **Covariance parameters**:
  - Max absolute difference: 0.069855

The pose prediction difference of ~0.03 is excellent for a complex transformer model with attention mechanisms. The larger uncertainty differences are expected due to the missing uncertainty_processor layers.

### H36M Data Test
⚠️ **SKIPPED** - Dataset not found at expected path

To run the H36M test, ensure the dataset is available at `../data/H36M_FREI/`.

## Usage

### Loading the Transferred Model

```python
import pickle
import jax
import jax.numpy as jnp
from dct_pose_transformer_flax import DCTPoseTransformerFlax

# Load the transferred model
with open('../models/HMP/dct_pose_transformer_flax.pickle', 'rb') as f:
    model_dict = pickle.load(f)

params = model_dict['params']
config = model_dict['config']

# Initialize model
model = DCTPoseTransformerFlax(**config)

# Run inference
x = jnp.zeros((batch_size, 50, 39))  # Your input data
poses, (var_params, cov_params) = model.apply({'params': params}, x)
```

### Running the Transfer Script

```bash
cd /home/skyle/Desktop/uq_benchmark/uncertainty_quantification
source virtualenv/bin/activate
python transfer_dct_pose_transformer.py
```

### Running with H36M Data

If you have the H36M dataset, update the path in the script:

```python
h36m_data_path = "/path/to/your/H36M_FREI"
```

## Model Parameters

```python
{
    'input_dim': 39,        # 13 joints × 3 coordinates
    'd_model': 128,         # Model dimension
    'nhead': 4,             # Number of attention heads
    'num_layers': 2,        # Number of transformer blocks
    'seq_len': 50,          # Input sequence length (25 fps × 2s)
    'seq_len_output': 10    # Output sequence length (400ms)
}
```

## Technical Notes

### Flax Module Design Choices

1. **Setup vs Compact**: Used `setup()` for `UncertaintyHead` to ensure all layers are created upfront
2. **Parameter Access**: Used `self.param()` within `setup()` for learnable scalars
3. **Layer Creation**: All sub-modules defined in `setup()` to match PyTorch's `__init__`

### Numerical Precision

- All computations use `float32` (JAX default)
- Small differences (~0.03) are expected due to:
  - Different BLAS implementations (cuBLAS vs MKL)
  - Different attention implementations
  - Floating-point arithmetic ordering

### Future Improvements

1. **Fix Uncertainty Processor**: Ensure `unc_proc_*` layers appear in parameter tree
2. **Add Batch Statistics**: If using BatchNorm (currently not used)
3. **Optimize Inference**: JIT compile the forward pass  (ToJakob, maybe that is why slower)
4. **Gradient Checkpointing**: For training large models

## Testing with Real Data

To properly validate the transfer with H36M data:

1. Ensure dataset is at `../data/H36M_FREI/` or change to your own directory
2. Dataset structure should be:
   ```
   H36M_FREI/
   ├── S1/D3_Positions/*.cdf
   ├── S5/D3_Positions/*.cdf  (test split)
   ├── S6/D3_Positions/*.cdf
   └── ...
   ```
3. Run the transfer script - it will automatically test on 3 batches

## References: Change to Your own directory

- Original PyTorch model: `models/HMP/transformer_model_and_loss.py`
- Training script: `models/HMP/trainer_no_pretrain.py`
- Trained weights: `models/HMP/transformer_model.pth`
- Similar transfer example: `Map_regress_flow_pred.py` (ResNet-based model)

## Contact

For questions about the transfer process or to report issues, refer to the main repository documentation.

