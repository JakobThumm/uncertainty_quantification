# Uncertainty Quantification

This repo is a benchmark for Bayesian neural network methods for quantifying uncertainty.

It supports several datasets ("MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100", "CelebA", "ImageNet") and several model architectutres ("MLP", "LeNet", "GoogleNet", "ConvNeXt", "ResNet", "VAN", "SWIN"). Virtually any combination is possible. After a model is trained we can compute several Uncertainty Quantification scores ("scod", "swag", "ensemble", "local_ensemble", "sketched_local_ensemble", "low_rank_lla", "diagonal_lla") against a series of OoD datasets.

## Setup 
Clone the repository using the command `Git: Clone` and the URL of the repo.
- Open the terminal and run the following command:
```bash
bash/setup.sh
```
This will create a virtual environment and install all the required packages. After the first use, you will only need to activate the environment by calling

```bash
source unc/bin/activate
```

## Reproduce results

#### Train model

Train any combination of model and dataset with (for example)
```
source ./virtualenv/bin/activate;
CUDA_VISIBLE_DEVICES=0;

python train_model.py --dataset MNIST --likelihood classification --model MLP --default_hyperparams 

python train_model.py --dataset FMNIST --likelihood classification --model LeNet --default_hyperparams

python train_model.py --dataset CIFAR-10 --likelihood classification --model ResNet --default_hyperparams

python train_model.py --dataset CelebA --likelihood binary_multiclassification --model VAN_tiny --default_hyperparams

python train_model.py --dataset ImageNet --likelihood classification --model SWIN_large --default_hyperparams
```

#### Compute OOD metric

Compute the scores for a single model with (for example)
```
source ./virtualenv/bin/activate;
CUDA_VISIBLE_DEVICES=0;

python score_model.py --ID_dataset FMNIST --OOD_dataset MNIST FMNIST-R --model LeNet --score local_ensemble --run_name started_[DATE] --lanczos_hm_iter 3  --lanczos_lm_iter 0 --lanczos_seed 0 --subsample_trainset 60000 --test_batch_size 128

python score_model.py --ID_dataset CIFAR-10 --OOD_dataset SVHN CIFAR-10-C --model ResNet --score scod

python score_model.py --ID_dataset CelebA --OOD_dataset FOOD101 CelebA-Mustache CelebA-Bald CelebA-Eyeglasses --model VAN_tiny --subsample_trainset 10000 --lanczos_hm_iter 3 --lanczos_lm_iter 0 --test_batch_size 8 --train_batch_size 32 --serialize_ggn_on_batches

python score_model.py --ID_dataset ImageNet --OOD_datasets SVHN-256 FOOD101-256 ImageNet-classout --model VAN_large --subsample_trainset 100000 --lanczos_hm_iter 0 --lanczos_lm_iter 10 --test_batch_size 8 --train_batch_size 32 --serialize_ggn_on_batches --sketch srft --sketch_size 10000000
```

# Known Issues

## CuDNN Version Mismatch Error

### Problem
When importing JAX, you may encounter an error like:
```
E external/xla/xla/stream_executor/cuda/cuda_dnn.cc:466] Loaded runtime CuDNN library: 9.5.1 but source was compiled with: 9.8.0. CuDNN library needs to have matching major version and equal or higher minor version.
```

This indicates that JAX was compiled against a newer version of CuDNN than what's installed on your system.

### Solution: Upgrade CuDNN locally (without sudo)

1. Download the CuDNN tar.xz archive from [NVIDIA's developer site](https://developer.nvidia.com/cudnn) (requires free account)
2. Extract and set up locally:
   ```bash
   tar -xf cudnn-linux-x86_64-9.8.0.*_cuda12-archive.tar.xz
   mv cudnn-linux-x86_64-9.8.0.*_cuda12-archive ~/cuda
   echo 'export LD_LIBRARY_PATH=~/cuda/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

### Verification
Test that the issue is resolved:
```python
import jax.numpy as jnp
import numpy as np
result = jnp.array(np.array([0.0]))
print("JAX working correctly:", result)
```

## Jax version mismatch errors
### Problem description
Errors like:
```
AttributeError: 'EvalTrace' object has no attribute 'level'
```
might indicate a version mismatch between Jax, Flax, and Optax.

### Solution
```
pip install --upgrade jax jaxlib flax optax
```
The tested working versions currently are:
```
$ pip list | grep -E "(jax|flax|optax)"
flax                     0.8.4
jax                      0.7.0
jax-cuda12-pjrt          0.7.0
jax-cuda12-plugin        0.7.0
jaxlib                   0.7.0
optax                    0.2.2
```
So you might want to downgrade to these versions if the packages have mismatches or bugs at the moment.