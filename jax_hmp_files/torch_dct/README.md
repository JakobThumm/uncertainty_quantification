## Description of SCOD for Transformer & SCOD on Attention

### Overview
This package provides the implementation of SCOD for Transformer model, focusing on uncertainty quantification in human motion prediction.

### Main Experiment Files

1. **config_general.yaml**
   - Contains hyperparameters for SCOD for Transformer (the general model), including paths, SCOD-specific parameters, and dataset settings.

2. **config_attention.yaml**
   - Contains hyperparameters for both SCOD for Transformer (the general model) and SCOD on Attention (the adapted version), including paths, SCOD-specific parameters, and dataset settings.

3. **transformer_scod_general.py**
   - The primary experiment script of SCOD for Transformer.
   - Runs experiments on different modules and layers in the frequency transformer model.
   - Follows the same pipeline as SCOD for ResNet (implemented in the PoseEstimationResNet package). Refer to that package’s README.md for additional details.

4. **transformer_scod_attention.py**
   - An extended version of `transformer_scod_general.py`.
   - Supports both the general SCOD pipeline (`--mode scod`) and the SCOD on Attention pipeline (`--mode attention_scod`).
   - Provides a visualization mode (`--mode vis_attention`) to analyze In-Distribution (ID) and Out-of-Distribution (OOD) pose sequences in both time and frequency domains (DCT matrix), as well as attention maps for ID and OOD pose sequences.

5. **qualitative_analysis.py**
   - After running the SCOD experiment, this script visualizes the uncertainty distribution for both the original H36M dataset and OOD datasets, including Sequence Substitution, Random Frame Replacement, and Joint Shift.

### Experiment Results

1. **SCODResult**
   - Stores results from the general SCOD algorithm for the transformer model.
   - `settings(layer/module name).json` record sexperimental settings and hyperparameters, including:
     - The layers/modules where SCOD is applied.
     - SCOD-specific hyperparameters.
     - OOD generation parameters (e.g., shifted joints, noise levels, and perturbed pose indices).
   - `transformer_KL_div(layer/module name).json` files:
     - Each entry includes:
       1. Uncertainty per joint (variance) and per motion prediction (KL divergence).
       2. Uncertainty values for ID and various OOD perturbations (temporal sequence substitution, random frame replacement, joint shift).
       3. OOD data generation follows a perturbation pipeline, where each original ID data point produces three corresponding OOD data points.

2. **SCODonAttentionResult**
   - Stores results for SCOD on Attention applied to the transformer model.
   - Follows the same data structure as `SCODResult`, with unique identifiers appended to filenames for each experiment.

3. **OOD_pose_sequence_visualization**
   - Contains visualizations of ID and OOD pose sequences.

### Running Commands

#### Train Frequency Transformer Model
```sh
# No pretraining
python trainer_no_pretrain.py

# With pretraining
python trainer.py
```

#### Run SCOD Experiment
```sh
python transformer_scod.py
```

or

```sh
python transformer_scod_attention.py --mode scod
```

#### Run SCOD on Attention
```sh
python transformer_scod_attention.py --mode attention_scod
```

### Visualizing SCOD Results

#### General SCOD
1. Move the generated JSON file to `SCODResult`.
2. Update the filename in `qualitative_analysis.py`.
3. Run:
```sh
python qualitative_analysis.py
```

#### SCOD on Attention
1. Move the generated JSON file to `SCODonAttentionResult`.
2. Update the filename in `qualitative_analysis.py`.
3. Run:
```sh
python qualitative_analysis.py
```

