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

# Human Pose Estimation

The repository includes a human pose estimation pipeline for uncertainty quantification on pose prediction tasks.

## Data Preprocessing

Preprocess the H36M dataset for pose estimation:
Performs the following steps:
  1. Transform the input image (1000, 1000) to YOLO 11 image size of (512, 640) (width, height).
  2. Find human bounding boxes in the image using YOLO 11 with variable size. 
  3. Change the box size to have the correct aspect ratio of 3/4.
  4. Crop the image to the box size.
  5. Transform the cropped image to the input size of the pose estimation network (192, 256).
  6. Normalize the RGB values by dividing by 255 and adding the offset -0.406, -0.457, -0.480 (not sure where this comes from. I assume average over images in H36M dataset)

**Single-frame preprocessing:**
```bash
python human_pose_pipeline/pose_estimation/preprocess_h36m_bbox.py \
    --input_dir datasets/H36M/extracted \
    --output_dir datasets/H36M/pre_processed \
    --splits train \
    --num_frames 1
```

**GPU-accelerated batch preprocessing:**
```bash
python human_pose_pipeline/pose_estimation/preprocess_h36m_bbox_gpu.py \
    --dataset_dir datasets/H36M/extracted \
    --output_dir datasets/H36M/pre_processed \
    --batch_size 128 \
    --device cuda
```

## Running Pose Estimation

**2D Pose Estimation:**
Performs the following steps:
  1. Transform the input image (1000, 1000) to YOLO 11 image size of (512, 640) (width, height).
  2. Find human bounding boxes in the image using YOLO 11 with variable size. 
  3. Change the box size to have the correct aspect ratio of 3/4.
  4. Crop the image to the box size.
  5. Transform the cropped image to the input size of the pose estimation network (192, 256).
  6. Normalize the RGB values by dividing by 255 and adding the offset -0.406, -0.457, -0.480 (not sure where this comes from. I assume average over images in H36M dataset)
  7. Estimate human pose.
  8. Transform human pose back into original image frame.

```bash
python human_pose_pipeline/examples/pose_estimation_2D.py
```

**2D Pose Estimation on Preprocessed Data:**
Performs the following steps:
  1. Load the preprocessed dataset.
  2. Estimate human pose.
  3. Transform human pose back into original image frame.
```bash
python human_pose_pipeline/examples/evaluate_preprocessed_h36m.py \
    --preprocessed_dir datasets/H36M/pre_processed \
    --checkpoint models_tianle/H36M/RegressFlow/seed_420 \
    --split validation \
    --num_samples 100 \
    --visualize \
    --save_dir results/preprocessed_eval_vis
```

**3D Pose Estimation:**
Additionally execute 3D triangulation:
  7. Perform human pose estimation on two images of different camera frames.
  8. Transform human pose back into original image frame.
  9. Perform 3D triangulation based on camera transforms.
  10. Estimate 3D uncertainty.

```bash
python human_pose_pipeline/examples/pose_estimation_3D.py
```

**ID vs. OOD Prediction:**
Evaluate the ID vs. OOD performance by executing the following steps:
  1. Perform steps 1-8 of 2D Pose Estimation.
  2. Preprocess tiger dataset: 
    - Rotate image by 90° to also have approximately 3/4 aspect ratio
    - Scale image to the input size of the pose estimation network (192, 256).
    - Normalize the RGB values by dividing by 255 and adding the offset -0.406, -0.457, -0.480.
  3. Predict tiger pose.
  4. Compare performance on human pose vs. tiger pose prediction.
```bash
python human_pose_pipeline/examples/id_vs_ood_pose_prediction.py
```

## Debugging and Visualization

**Debug single 2D pose:**
The 2D Pose Estimation just with a single image and visualization.
```bash
python human_pose_pipeline/examples/debug_pose_visualization.py
```

**Debug preprocessed pose:**
The 2D Pose Estimation on Preprocessed Data just with a single image and visualization.
```bash
python human_pose_pipeline/examples/debug_preprocessed_pose.py \
    --preprocessed_dir datasets/H36M/pre_processed \
    --checkpoint models_tianle/H36M/RegressFlow/seed_420 \
    --split train \
    --sample_idx 0 \
    --save_path results/debug_pose.png
```

**Debug 3D pose:**
The 3D Pose Estimation just with a single image and visualization.
```bash
python human_pose_pipeline/examples/debug_3d_pose_visualization.py
```

## OOD Detection Pose Prediction

**Run the Score Model Function on Pose Estimation**
```bash
python score_model.py --ID_dataset H36M --OOD_dataset tiger-pose --data_path datasets/ --model_save_path models_tianle --model RegressFlow --run_name finetuned_h36m_regressflow_pred --subsample_trainset 10000 --lanczos_hm_iter 0 --lanczos_lm_iter 10 --test_batch_size 128 --train_batch_size 128 --serialize_ggn_on_batches --sketch srft --sketch_size 100000
```

### Caching Intermediate Computations

The Lanczos algorithm involves several expensive computations (20+ minutes for large models). You can cache intermediate results at different stages to speed up parameter tuning:

**Granular Caching Levels:**

1. **GGN Vector Product** (~20 min JIT compilation)
2. **Sketch Operator** (fast, but needs to match GGN)
3. **Eigenpairs** (expensive Lanczos iterations)

**First run - compute everything and save to cache:**
```bash
python score_model.py --ID_dataset H36M --OOD_dataset tiger-pose \
  --data_path datasets/ --model_save_path models_tianle \
  --model RegressFlow --run_name finetuned_h36m_regressflow_pred \
  --subsample_trainset 10000 --lanczos_hm_iter 0 --lanczos_lm_iter 81 \
  --test_batch_size 256 --train_batch_size 256 --serialize_ggn_on_batches \
  --sketch srft --sketch_size 10000 \
  --cache_dir cache
```

**Load GGN only** (skip ~20 min JIT compilation):
```bash
python score_model.py [same args as above] \
  --cache_dir cache --load_ggn_vector_product
```

**Load GGN + Sketch** (skip GGN + sketch creation):
```bash
python score_model.py [same args as above] \
  --cache_dir cache --load_ggn_vector_product --load_sketch_op
```

**Load everything** (skip GGN + sketch + Lanczos, fastest!):
```bash
python score_model.py [same args as above] \
  --cache_dir cache --load_ggn_vector_product --load_sketch_op --load_eigenpairs
```

**Notes:**
- `--cache_dir DIR`: If set, automatically saves newly computed elements to DIR
- Loading has dependencies: sketch requires GGN, eigenpairs requires both
- Cache files are named based on dataset, model, run_name, and key parameters
- Cache files use `.cloudpickle` extension
- Separate files for each stage: `*_ggn.cloudpickle`, `*_sketch.cloudpickle`, `*_eigenpairs.cloudpickle`

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