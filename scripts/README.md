# Scripts Directory

This directory contains organized scripts for testing and development of the explainability methods.

## Directory Structure

```
scripts/
├── testing/          # Test scripts for validation and evaluation
├── training/         # Model training scripts (if needed)
└── analysis/         # Data analysis and visualization scripts
```

## Testing Scripts

### `testing/test_challenging_ood.py`
Extended Phase 3 testing script that evaluates Integrated Gradients on progressively challenging OOD scenarios:
- Cross-domain: FMNIST model → MNIST data
- 90° rotated MNIST 
- 180° rotated MNIST
- Comprehensive visualization and analysis

Usage:
```bash
cd scripts/testing
python test_challenging_ood.py
```

## Usage Notes

- All scripts should be run from the repository root directory or adjusted for proper import paths
- Scripts assume the virtual environment is activated: `source unc/bin/activate`
- Output files (visualizations, results) are saved to the repository root by default