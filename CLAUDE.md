# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a benchmark for Bayesian neural network methods for uncertainty quantification, with a focus on the **Sketched Lanczos Uncertainty (SLU)** method introduced in the paper "Sketched Lanczos uncertainty score: a low-memory summary of the Fisher information". The project evaluates various uncertainty quantification (UQ) methods on multiple datasets and model architectures, focusing on out-of-distribution (OOD) detection performance in low-memory regimes.

## Environment Setup

The project uses a Python virtual environment with JAX/Flax and PyTorch dependencies:

```bash
# Initial setup (run once)
bash bash/setup.sh

# Activate environment for subsequent sessions  
source unc/bin/activate
```

Note: The setup script creates environment in `unc/` directory, not `virtualenv/`.

## Core Commands

### Training Models
```bash
python train_model.py --dataset DATASET --model MODEL --likelihood LIKELIHOOD --default_hyperparams
```

Example training commands:
- `python train_model.py --dataset MNIST --likelihood classification --model MLP --default_hyperparams`
- `python train_model.py --dataset FMNIST --likelihood classification --model LeNet --default_hyperparams`
- `python train_model.py --dataset CIFAR-10 --likelihood classification --model ResNet --default_hyperparams`

### Computing OOD Scores
```bash
python score_model.py --ID_dataset DATASET --OOD_datasets OOD1 OOD2 --model MODEL --score SCORE_METHOD [options]
```

Example scoring commands:
- `python score_model.py --ID_dataset FMNIST --OOD_datasets MNIST FMNIST-R --model LeNet --score local_ensemble --lanczos_hm_iter 3 --lanczos_lm_iter 0 --subsample_trainset 60000`
- `python score_model.py --ID_dataset CIFAR-10 --OOD_datasets SVHN CIFAR-10-C --model ResNet --score scod`

### Testing Models
```bash
python test_model.py --dataset DATASET --model MODEL --run_name RUN_NAME --seed SEED
```

### Batch Operations
Use bash scripts for running experiments across multiple seeds and configurations:
- Training: `bash/training/mnist_mlp.sh`, `bash/training/cifar_resnet.sh`, etc.
- Scoring: `bash/scoring/mnist_mlp/local_ensemble.sh`, etc.

## Detailed Folder Structure and Functionality

### Core Source Code (`src/`)

#### **`src/autodiff/`** - Automatic Differentiation Utilities
- `ggn.py`: Generalized Gauss-Newton (GGN) matrix computations
- `hessian.py`: Hessian matrix computations  
- `jacobian.py`: Jacobian computations for neural networks
- `ntk.py`: Neural Tangent Kernel calculations
- `projection.py`: Matrix projection utilities

#### **`src/datasets/`** - Data Loading and Processing
- Individual dataset loaders: `mnist.py`, `cifar10.py`, `celeba.py`, `imagenet.py`, etc.
- `wrapper.py`: Unified dataset interface
- `utils.py`: Common dataset utilities

#### **`src/models/`** - Neural Network Architectures  
- Architecture implementations: `mlp.py`, `lenet.py`, `resnet.py`, `swin.py`, `van.py`
- `linearized.py`: Linearized model variants for uncertainty quantification
- `wrapper.py`: Unified model interface
- `utils.py`: Model utilities and helper functions

#### **`src/training/`** - Training Infrastructure
- `trainer.py`, `trainer_fancy.py`: Main training loops
- `losses.py`: Loss functions (cross-entropy, binary classification, etc.)
- `minimizer_with_reg.py`: Optimizers with regularization support
- `regularizer.py`: Regularization methods (log determinant GGN/NTK)

#### **`src/ood_scores/`** - Uncertainty Quantification Methods
- `ensemble.py`: Deep ensemble methods
- `diagonal_lla.py`: Diagonal Laplace approximation  
- `scod.py`: SCOD (Sketching Curvature for OoD Detection)
- `swag.py`: SWAG (SWA-Gaussian) method
- `hm_lanczos.py`: High-memory Lanczos implementation
- `lm_lanczos.py`: **Low-memory Lanczos implementation (core of SLU)**
- `projected_ensemble.py`: Projected ensemble methods
- `max_logit.py`: Maximum logit baseline

#### **`src/lanczos/`** - Lanczos Algorithm Implementations
- `high_memory.py`: Standard Lanczos requiring O(kp) memory
- `low_memory.py`: **Memory-efficient Lanczos requiring O(k²ε⁻²) memory**

#### **`src/sketches/`** - Dimensionality Reduction Techniques
- `srft.py`: **Subsampled Random Fourier Transform (core sketching method)**
- `srft_torch.py`: PyTorch implementation of SRFT
- `dense.py`: Dense sketching matrices
- `no_sketch.py`: Identity sketching (no reduction)

#### **`src/estimators/`** - Matrix Property Estimators
- `hutchinson.py`: Hutchinson trace estimator
- `determinant.py`: Log-determinant estimation
- `frobenius.py`: Frobenius norm estimation

### Scripts and Results

#### **`bash/`** - Experiment Scripts
- `setup.sh`: Environment setup
- `training/`: Model training scripts for each dataset/architecture combination
- `scoring/`: OOD scoring scripts with various UQ methods
- `plot_figure_*.sh`: Scripts to reproduce paper figures

#### **Main Scripts**
- `train_model.py`: Main training script with extensive hyperparameter support
- `score_model.py`: **Main scoring script implementing SLU and baseline methods**  
- `test_model.py`: Model evaluation and testing
- `plot_data.py`, `plot_fixed_memory_budget.py`: Results visualization

## Sketched Lanczos Uncertainty (SLU) Algorithm

Based on the paper analysis, here's the step-by-step algorithm and corresponding code locations:

### Algorithm Overview
The SLU method combines Lanczos algorithm with sketching to compute uncertainty scores using only O(k²ε⁻²) memory instead of O(kp) for traditional methods.

### Step-by-Step Implementation

#### Step 1: GGN Matrix-Vector Product Setup
**Location**: `src/autodiff/ggn.py`
- Computes Generalized Gauss-Newton matrix G = J^T H J without explicitly forming G
- Efficient matrix-vector products G·v using automatic differentiation

#### Step 2: SRFT Sketch Matrix Initialization  
**Location**: `src/sketches/srft.py`
- Creates Subsampled Random Fourier Transform matrix S ∈ ℝ^(s×p)
- Sketch size s = Ω(kε⁻²log p log(k/δ)) for error bound ε with probability 1-δ
- Memory: O(p + s) instead of O(p²)

#### Step 3: Low-Memory Lanczos Iterations
**Location**: `src/lanczos/low_memory.py`  
- Runs k iterations of Lanczos algorithm on GGN matrix
- At each iteration i: computes v_i, sketches it as v_i^S = S·v_i, stores only sketched version
- Memory: O(3p) working space + O(sk) for sketched vectors

#### Step 4: Post-hoc Orthogonalization of Sketched Vectors
**Location**: `src/ood_scores/lm_lanczos.py` (Algorithm 1 in paper)
- Constructs matrix V_S = [v_1^S, ..., v_k^S] ∈ ℝ^(s×k)
- Orthogonalizes columns of V_S to get U_S ∈ ℝ^(s×k)
- **Key theoretical insight**: Sketching and orthogonalization approximately commute

#### Step 5: Uncertainty Score Computation
**Location**: `src/ood_scores/lm_lanczos.py` (low_memory_lanczos_score_fun)
- For test point x, computes Jacobian J_θ*(x)
- Sketches Jacobian: SJ_θ*(x)  
- **SLU score**: ||J_θ*(x)||_F² - ||U_S^T(SJ_θ*(x)^T)||_F²
- This approximates ||J_θ*(x)||_F² - ||U^T J_θ*(x)^T||_F² where U would be full eigenvectors

#### Step 6: Preconditioning (Optional Enhancement)
**Location**: `src/ood_scores/hm_lanczos.py` (smart_lanczos_score_fun)
- Runs high-memory Lanczos for k₀ steps to get stable top eigenspace  
- Continues with sketched Lanczos on preconditioned matrix G̃ = G - U₀Λ₀U₀^T
- Trades some memory (k₀·p) for numerical stability

### Memory Complexity Comparison
- **Traditional Lanczos**: O(kp) memory  
- **SLU**: O(k²ε⁻²) memory (independent of p up to log terms)
- **Practical benefit**: For large p, can use much higher rank k within same memory budget

### Supported Configurations
- **Datasets**: MNIST, FMNIST, SVHN, CIFAR-10/100, CelebA, ImageNet, UCI, Sinusoidal
- **Models**: MLP, LeNet, GoogleNet, ConvNeXt variants, ResNet variants, VAN variants, SWIN variants  
- **UQ Methods**: SLU, SCOD, SWAG, ensemble variants, Laplace approximations, traditional Lanczos
- **Likelihoods**: regression, classification, binary_multiclassification

## Known Issues

### CuDNN Version Mismatch
JAX compilation may require CuDNN 9.8.0 or higher. Install locally if system version is outdated:
```bash
# Download from NVIDIA, then:
tar -xf cudnn-linux-x86_64-9.8.0.*_cuda12-archive.tar.xz
mv cudnn-linux-x86_64-9.8.0.*_cuda12-archive ~/cuda
echo 'export LD_LIBRARY_PATH=~/cuda/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### JAX Version Compatibility  
Tested working versions for avoiding `EvalTrace` errors:
- jax==0.7.0, jaxlib==0.7.0, flax==0.8.4, optax==0.2.2

## Development Notes

- Models are saved in `../models/` by default (relative to repo root)
- Results are saved in `results/` directory  
- The codebase supports both high-memory and low-memory Lanczos iterations
- SRFT sketching can be configured with different sketch sizes for memory/accuracy tradeoffs
- Batch processing scripts use LSF job scheduler syntax but can be adapted for other systems
- The SLU method scales logarithmically with model parameters, making it suitable for very large models