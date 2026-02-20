# Results Documentation for Human Pose Prediction Pipeline

## 2D Human Pose Estimation

=====> The yolov5s network seems to be much better at human detection.
        Maybe use that model instead of yolov11n.

### Marian Pytorch on 3 validation files (yolo threshold = 0.8) (Model: estimation_model_finetuned_on_h36m.pth)

Run with:
```
{
  "name": "Marian Experiment2 2D Pose Estimation",
  "type": "debugpy",
  "request": "launch",
  "cwd": "marian_code/Experiment2/",
  "program": "2D_Pose_Estimation.py",
  "console": "integratedTerminal"
},
```
Results
```
    Total frames processed: 4881
    Total joints evaluated: 63453
    Average MPJPE: 7.64 pixels
    Average percentage of keypoints within 1 std: 73.14%
    Average percentage of keypoints within 2 std: 91.45%
    Average percentage of keypoints within 3 std: 97.50%
    Average percentage of keypoints within 4 std: 99.17%
```
### Jax on 3 validation files (yolo threshold = 0.3) (Model: jax_resnet50_regressflow)

Run with:
```
{
  "name": "Pose Estimation 2D",
  "type": "debugpy",
  "request": "launch",
  "program": "human_pose_pipeline/examples/pose_estimation_2D.py",
  "console": "integratedTerminal"
},
```
Results:
```
    Total frames processed: 4988
    Total joints evaluated: 64844
    Average MPJPE: 7.83
    Average percentage of keypoints within 1 std: 73.08%
    Average percentage of keypoints within 2 std: 92.45%
    Average percentage of keypoints within 3 std: 97.51%
    Average percentage of keypoints within 4 std: 99.14%
```
==> The models estimation_model_finetuned_on_h36m.pth and jax_resnet50_regressflow seem to match.

### Jax on 3 validation files (yolo threshold = 0.3) (Model: finetuned_h36m_regressflow_with_unc)

Run with:
```
{
  "name": "Pose Estimation 2D",
  "type": "debugpy",
  "request": "launch",
  "program": "human_pose_pipeline/examples/pose_estimation_2D.py",
  "console": "integratedTerminal"
},
# Used Model
models_dir = os.path.join(root_dir, "human_pose_pipeline/models/pose_estimation", "H36M", "RegressFlow", "seed_420")
checkpoint_path_jax = os.path.join(models_dir, "finetuned_h36m_regressflow_with_unc")
```
Results:
```
    Total frames processed: 4783
    Total joints evaluated: 62179
    Average MPJPE: 7.70
    Average percentage of keypoints within 1 std: 70.91%
    Average percentage of keypoints within 2 std: 90.88%
    Average percentage of keypoints within 3 std: 96.87%
    Average percentage of keypoints within 4 std: 98.85%
```
==> Here, the accuracy of the jax_resnet50_regressflow and finetuned_h36m_regressflow_with_unc roughly match.

### YOLOv26 with sigma prediction (not fine-tuned on H36M)

Run with:
```
{
  "name": "Pose Estimation 2D YOLO",
  "type": "debugpy",
  "request": "launch",
  "program": "human_pose_pipeline/examples/pose_estimation_2D_yolo.py",
  "console": "integratedTerminal"
},
```
Results:
```
  Frames processed:        4998
  Joints evaluated:        64974
  Average MPJPE:           13.98 px
  Within 1 std (68%):      97.28%
  Within 2 std (95%):      99.41%
  Within 3 std (99.7%):    99.88%
  Within 4 std (99.99%):   100.00%
```
==> Prediction accuracy is worse than our previous model but that is expected as not fine-tuned on H36M data. Uncertainty is a bit too high, but useable.

## 3D Pose Estimation

### Marian Pytorch on 10 validation files with 1000 max_frames (yolo threshold = 0.8) (Model: estimation_model_finetuned_on_h36m.pth)
Run with
```
{
  "name": "Marian Experiment2 3D Pose Estimation",
  "type": "debugpy",
  "request": "launch",
  "program": "marian_code/Experiment2/3D_Pose_Estimation.py",
  "console": "integratedTerminal"
},
```
Results:
```
Actions: ['Discussion 1', 'Sitting 1', 'SittingDown 1', 'Posing 1', 'Eating', 'SittingDown', 'Smoking 2', 'Directions', 'Purchases 1', 'Waiting']
    Total frames processed: 10000
    Total joints evaluated: 130000
    Average MPJPE: 201.17 mm
    Average Pixel MPJPE: 11.56 pixels
    Average percentage of keypoints within 1 std: 43.36%
    Average percentage of keypoints within 2 std: 69.78%
    Average percentage of keypoints within 3 std: 83.08%
    Average percentage of keypoints within 4 std: 88.10%
    Average percentage of 2D keypoints within 1 std: 53.26%
    Average percentage of 2D keypoints within 2 std: 78.51%
    Average percentage of 2D keypoints within 3 std: 88.64%
    Average percentage of 2D keypoints within 4 std: 91.56%
```
### Jax on 3 validation files (yolo threshold = 0.3) (Model: finetuned_h36m_regressflow_with_unc)
Run with:
```
{
  "name": "Pose Estimation 3D Full Eval",
  "type": "debugpy",
  "request": "launch",
  "program": "human_pose_pipeline/examples/pose_estimation_3D_full_eval.py",
  "console": "integratedTerminal",
  "args": [
      "--max_sequences", "1",
      "--split", "validation"
  ]
},
```
Results:
```
Actions: ['Discussion 1', 'Sitting 1', 'SittingDown 1', 'Posing 1', 'Eating', 'SittingDown', 'Smoking 2', 'Directions', 'Purchases 1', 'Waiting']
    Total frames processed: 9959
    Total joints evaluated: 129467
    Average MPJPE: 31.79 mm
    Average percentage of keypoints within 1 std: 45.30%
    Average percentage of keypoints within 2 std: 73.19%
    Average percentage of keypoints within 3 std: 88.06%
    Average percentage of keypoints within 4 std: 94.34%
```
Findings:
 - Jax `finetuned_h36m_regressflow_with_unc` seems to be much better than Pytorch `estimation_model_finetuned_on_h36m`! The actions are the same.
 - This would require further investigation but our model is better, so I guess it is okay.

### RGB-D 3D Pose Estimation
- Emulating the RGB-D camera did not work with the H36M setup as the cameras were too far apart.
- Approach 1: `Human36mDatasetEmulatedRGBD` tries to perform stereo matching and triangulation but there is way too little usable overlap for accurate depth information.
- Approach 2: `Human36mDatasetGTPoseRGBD` tries to draw circles with the ground truth depth at the GT joint positions in the depth image. However, the circles are either too large, causing overlap and wrong depth information or too small causing detection of background. This would require a way more sophisticated method.
--> I would say we just use our normal triangulated 3D Pose Estimation for evaluation on H36M.

## Motion Prediction

### Pytorch all validation data (true pose input)
Run with:
```
{
  "name": "Marian Experiment1 Eval 3D Motion Prediction",
  "type": "debugpy",
  "request": "launch",
  "program": "marian_code/Experiment1/13_Joints/validate_model.py",
  "console": "integratedTerminal"
},
```
Results:
- **model_13_joints_with_uncert**
  ```
  Overall MPJPE: 23.79 mm
  Per-Time Errors:
  Time point 1 error =    6.37 mm
  Time point 2 error =    7.10 mm
  Time point 3 error =   10.23 mm
  Time point 4 error =   14.71 mm
  Time point 5 error =   19.68 mm
  Time point 6 error =   24.78 mm
  Time point 7 error =   30.14 mm
  Time point 8 error =   35.72 mm
  Time point 9 error =   41.48 mm
  Time point 10 error =   47.65 mm
  Overall Coverage:
    Level      Percentage   Expected    
    1σ          95.64%        68.00%
    2σ          98.59%        95.00%
    3σ          99.47%        99.73%
    4σ          99.78%        99.99%
  ```
 - **model_13_joints_calibrated_uncert.pth**
  ```
  Validation MPJPE: 33.31 mm
  Validation MPJPE per frame:
    Frame +1 (t+40ms): 9.71 mm
    Frame +2 (t+80ms): 13.49 mm
    Frame +3 (t+120ms): 18.15 mm
    Frame +4 (t+160ms): 23.80 mm
    Frame +5 (t+200ms): 29.53 mm
    Frame +6 (t+240ms): 35.52 mm
    Frame +7 (t+280ms): 41.50 mm
    Frame +8 (t+320ms): 47.59 mm
    Frame +9 (t+360ms): 53.71 mm
    Frame +10 (t+400ms): 60.08 mm
  Overall Coverage:
    Level      Percentage   Expected    
    1σ          95.50%        68.00%
    2σ          98.65%        95.00%
    3σ          99.55%        99.73%
    4σ          99.83%        99.99%
  ```

### Jax Model Trained from Scratch
Train with:
```
{
  "name": "Train Motion Prediction Model (Stage 1)",
  "type": "debugpy",
  "request": "launch",
  "program": "human_pose_pipeline/motion_prediction/train_motion_prediction_model.py",
  "console": "integratedTerminal",
  "args": [
      "--stage", "1",
      "--data_path", "datasets/",
      "--batch_size", "256",
      "--d_model", "128",
      "--nhead", "4",
      "--num_layers", "2",
      "--stage1_epochs", "50",
      "--stage2_epochs", "15",
      "--stage3_epochs", "15",
      "--learning_rate", "0.001",
      "--seed", "0",
      "--use_lr_schedule",
      "--lr_schedule_type", "cosine",
      "--lr_warmup_epochs", "3",
      "--lr_min_factor", "0.1",
      "--weight_decay", "0.000001",
      "--max_grad_norm", "0.6796845430167515",
      "--wandb_project", "motion-prediction",
      "--use_wandb"
  ]
},
```
Evaluate with:
```
{
  "name": "Motion Prediction Evaluation",
  "type": "debugpy",
  "request": "launch",
  "program": "human_pose_pipeline/examples/motion_prediction.py",
  "console": "integratedTerminal",
  "args": [
      "--data_path", "datasets/",
      "--split", "validation",
      "--model_save_path", "DELETED MODEL",
  ]
},
```
Results:
```
Overall MPJPE: 55.37 mm, Std: 57.21 mm
Per-Time Errors:
Time point 1 error =   41.71 mm
Time point 2 error =   41.50 mm
Time point 3 error =   42.04 mm
Time point 4 error =   44.46 mm
Time point 5 error =   48.67 mm
Time point 6 error =   54.36 mm
Time point 7 error =   60.29 mm
Time point 8 error =   67.02 mm
Time point 9 error =   72.76 mm
Time point 10 error =   80.86 mm
Per-Joint Errors:
Joint 1 error =   53.84 mm
Joint 2 error =   49.87 mm
Joint 3 error =   47.07 mm
Joint 4 error =   71.11 mm
Joint 5 error =   66.25 mm
Joint 6 error =   95.89 mm
Joint 7 error =   91.17 mm
Joint 8 error =   37.35 mm
Joint 9 error =   35.16 mm
Joint 10 error =   40.51 mm
Joint 11 error =   40.31 mm
Joint 12 error =   44.57 mm
Joint 13 error =   46.68 mm
Uncertainty Coverage Stats:
  Overall coverage within 1 std: 41.21%
  Overall coverage within 2 std: 56.98%
  Overall coverage within 3 std: 68.79%
  Overall coverage within 4 std: 77.41%
```

### Jax Model Trained Starting from Pytorch Weights
Train with:
```
{
  "name": "Train Motion Prediction Model from Transferred Weights",
  "type": "debugpy",
  "request": "launch",
  "program": "human_pose_pipeline/motion_prediction/train_motion_prediction_model.py",
  "console": "integratedTerminal",
  "args": [
      "--stage", "1",
      "--data_path", "datasets/",
      "--batch_size", "256",
      "--d_model", "128",
      "--nhead", "4",
      "--num_layers", "2",
      "--init_weights_path", "human_pose_pipeline/models/motion_prediction/dct_pose_transformer_transferred.pickle",
      "--stage1_epochs", "30",
      "--stage2_epochs", "15",
      "--stage3_epochs", "15",
      "--learning_rate", "0.0001",
      "--use_lr_schedule",
      "--lr_schedule_type", "cosine",
      "--lr_warmup_epochs", "3",
      "--lr_min_factor", "0.1",
      "--weight_decay", "0.000001",
      "--max_grad_norm", "0.6796845430167515",
      "--wandb_project", "motion-prediction",
      "--use_wandb"
  ]
},
```
Validate with:
```
{
  "name": "Motion Prediction Evaluation",
  "type": "debugpy",
  "request": "launch",
  "program": "human_pose_pipeline/examples/motion_prediction.py",
  "console": "integratedTerminal",
  "args": [
      "--data_path", "datasets/",
      "--dataset_name", "Human36mMotionDataset3DWithInputUncertainty",  // or "Human36mMotionDataset3D"
      "--split", "validation",
      "--model_save_path", "human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle",
      // "--enable_ood",
      "--motion_score_fn_path", "human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM1440of1600_sketch_srft_seed0_size20000.cloudpickle"
  ]
},
```
#### Results with true pose inputs (no uncertainty)
**Model: with scaled input uncertainty Stage 4.**

Results:
```
Overall MPJPE: 23.12 mm, Std: 32.05 mm

Per-Time Errors:
Time point 1 error =    7.75 mm
Time point 2 error =    7.72 mm
Time point 3 error =   10.00 mm
Time point 4 error =   14.09 mm
Time point 5 error =   18.65 mm
Time point 6 error =   23.61 mm
Time point 7 error =   28.83 mm
Time point 8 error =   34.35 mm
Time point 9 error =   40.06 mm
Time point 10 error =   46.17 mm

Per-Joint Errors:
Joint 1 error =   20.35 mm
Joint 2 error =   18.37 mm
Joint 3 error =   18.50 mm
Joint 4 error =   28.93 mm
Joint 5 error =   28.41 mm
Joint 6 error =   39.26 mm
Joint 7 error =   38.38 mm
Joint 8 error =   15.35 mm
Joint 9 error =   14.93 mm
Joint 10 error =   18.16 mm
Joint 11 error =   18.69 mm
Joint 12 error =   20.00 mm
Joint 13 error =   21.24 mm

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 0.01%
  Overall coverage within 2 std: 0.02%
  Overall coverage within 3 std: 0.04%
  Overall coverage within 4 std: 0.07%
```

**Model: with scaled input uncertainty Stage 2**
Results:
```
Overall MPJPE: 23.14 mm, Std: 32.17 mm

Per-Time Errors:
Time point 1 error =    7.76 mm
Time point 2 error =    7.66 mm
Time point 3 error =    9.93 mm
Time point 4 error =   14.04 mm
Time point 5 error =   18.62 mm
Time point 6 error =   23.68 mm
Time point 7 error =   28.91 mm
Time point 8 error =   34.43 mm
Time point 9 error =   40.16 mm
Time point 10 error =   46.26 mm

Per-Joint Errors:
Joint 1 error =   20.48 mm
Joint 2 error =   18.55 mm
Joint 3 error =   18.67 mm
Joint 4 error =   28.88 mm
Joint 5 error =   28.41 mm
Joint 6 error =   39.27 mm
Joint 7 error =   38.03 mm
Joint 8 error =   15.33 mm
Joint 9 error =   14.98 mm
Joint 10 error =   18.20 mm
Joint 11 error =   18.82 mm
Joint 12 error =   20.03 mm
Joint 13 error =   21.25 mm

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 71.77%
  Overall coverage within 2 std: 86.87%
  Overall coverage within 3 std: 92.99%
  Overall coverage within 4 std: 95.92%
```

**Model: with unscaled input uncertainty Stage 4.**

Results:
```
Overall MPJPE: 45.24 mm, Std: 45.53 mm

Per-Time Errors:
Time point 1 error =  101.58 mm
Time point 2 error =   48.93 mm
Time point 3 error =   28.28 mm
Time point 4 error =   23.64 mm
Time point 5 error =   28.06 mm
Time point 6 error =   31.77 mm
Time point 7 error =   38.15 mm
Time point 8 error =   44.45 mm
Time point 9 error =   50.98 mm
Time point 10 error =   56.53 mm

Per-Joint Errors:
Joint 1 error =   36.24 mm
Joint 2 error =   39.90 mm
Joint 3 error =   48.34 mm
Joint 4 error =   73.68 mm
Joint 5 error =   51.64 mm
Joint 6 error =   59.76 mm
Joint 7 error =   68.78 mm
Joint 8 error =   30.62 mm
Joint 9 error =   27.44 mm
Joint 10 error =   38.62 mm
Joint 11 error =   36.17 mm
Joint 12 error =   35.18 mm
Joint 13 error =   41.74 mm

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 0.07%
  Overall coverage within 2 std: 0.16%
  Overall coverage within 3 std: 0.29%
  Overall coverage within 4 std: 0.43%
```

Findings:
  - Model with scaled input uncertainty (`x = x + uncertainty_features / self.unit_conversion`) works better if input uncertainty is missing.
  - They perform similarly if input uncertainty is given. -> Use scaled input.
  - Uncertainty Coverage is completely wrong if uncertainty features are not given anymore. Use model of Stage 2 in this case. (`human_pose_pipeline/models/motion_prediction/final_training_run/checkpoints/stage_2/dct_pose_transformer.pickle`)


#### Results with Uncertain Inputs (predicted 3D poses from custom model)
```
Overall MPJPE: 23.38 mm, Std: 32.45 mm

Per-Time Errors:
Time point 1 error =    7.84 mm
Time point 2 error =    7.86 mm
Time point 3 error =   10.21 mm
Time point 4 error =   14.34 mm
Time point 5 error =   18.91 mm
Time point 6 error =   23.91 mm
Time point 7 error =   29.15 mm
Time point 8 error =   34.69 mm
Time point 9 error =   40.40 mm
Time point 10 error =   46.49 mm

Per-Joint Errors:
Joint 1 error =   20.64 mm
Joint 2 error =   18.61 mm
Joint 3 error =   18.74 mm
Joint 4 error =   29.19 mm
Joint 5 error =   28.71 mm
Joint 6 error =   39.55 mm
Joint 7 error =   38.70 mm
Joint 8 error =   15.56 mm
Joint 9 error =   15.18 mm
Joint 10 error =   18.40 mm
Joint 11 error =   18.85 mm
Joint 12 error =   20.30 mm
Joint 13 error =   21.50 mm

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 69.96%
  Overall coverage within 2 std: 86.11%
  Overall coverage within 3 std: 92.62%
  Overall coverage within 4 std: 95.68%
```

### Adjusted Covariance Evaluation (with uncertain inputs)
Tune with:
```
{
  "name": "Tune Motion Prediction Covariance Offset",
  "type": "debugpy",
  "request": "launch",
  "program": "human_pose_pipeline/motion_prediction/tune_covariance_offset.py",
  "console": "integratedTerminal",
  "args": [
      "--results_file", "results/motion_prediction/motion_prediction_results_train.cloudpickle",
  ]
},
```
Evaluate with:
```
{
  "name": "Evaluate Motion Prediction Covariance",
  "type": "debugpy",
  "request": "launch",
  "program": "human_pose_pipeline/motion_prediction/evaluate_covariance.py",
  "console": "integratedTerminal",
  "args": [
      "--results_file", "results/motion_prediction/motion_prediction_results_validation.cloudpickle",
  ]
},
```
Results (Model: with scaled input uncertainty Stage 4.):

**Tuned Predictions**
```
Loaded predictions shape: (54496, 10, 13, 3)
Loaded targets shape: (54496, 10, 13, 3)
Loaded covariance matrices shape: (54496, 10, 13, 3, 3)

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 89.81%
  Overall coverage within 2 std: 96.97%
  Overall coverage within 3 std: 98.88%
  Overall coverage within 4 std: 99.51%

Per-Time Coverage Stats:

  Overall coverage within 1 std:
    Frame 0: 80.67%
    Frame 1: 87.68%
    Frame 2: 90.38%
    Frame 3: 90.98%
    Frame 4: 91.35%
    Frame 5: 91.50%
    Frame 6: 91.51%
    Frame 7: 91.44%
    Frame 8: 91.36%
    Frame 9: 91.21%

  Overall coverage within 2 std:
    Frame 0: 95.42%
    Frame 1: 96.73%
    Frame 2: 97.04%
    Frame 3: 97.15%
    Frame 4: 97.28%
    Frame 5: 97.28%
    Frame 6: 97.24%
    Frame 7: 97.22%
    Frame 8: 97.18%
    Frame 9: 97.13%

  Overall coverage within 3 std:
    Frame 0: 98.73%
    Frame 1: 98.84%
    Frame 2: 98.84%
    Frame 3: 98.92%
    Frame 4: 98.96%
    Frame 5: 98.94%
    Frame 6: 98.92%
    Frame 7: 98.90%
    Frame 8: 98.87%
    Frame 9: 98.86%

  Overall coverage within 4 std:
    Frame 0: 99.54%
    Frame 1: 99.50%
    Frame 2: 99.48%
    Frame 3: 99.52%
    Frame 4: 99.54%
    Frame 5: 99.54%
    Frame 6: 99.52%
    Frame 7: 99.50%
    Frame 8: 99.49%
    Frame 9: 99.48%

Per-Joint Coverage Stats:

  Overall coverage within 1 std:
    Joint 0: 88.01%
    Joint 1: 89.78%
    Joint 2: 89.04%
    Joint 3: 87.16%
    Joint 4: 86.56%
    Joint 5: 92.68%
    Joint 6: 92.25%
    Joint 7: 90.12%
    Joint 8: 90.53%
    Joint 9: 89.39%
    Joint 10: 88.11%
    Joint 11: 92.87%
    Joint 12: 91.01%

  Overall coverage within 2 std:
    Joint 0: 96.64%
    Joint 1: 97.31%
    Joint 2: 97.16%
    Joint 3: 96.31%
    Joint 4: 95.58%
    Joint 5: 98.27%
    Joint 6: 97.87%
    Joint 7: 97.00%
    Joint 8: 97.40%
    Joint 9: 96.94%
    Joint 10: 96.12%
    Joint 11: 97.32%
    Joint 12: 96.67%

  Overall coverage within 3 std:
    Joint 0: 98.88%
    Joint 1: 99.04%
    Joint 2: 99.04%
    Joint 3: 98.79%
    Joint 4: 98.35%
    Joint 5: 99.52%
    Joint 6: 99.37%
    Joint 7: 98.78%
    Joint 8: 99.03%
    Joint 9: 98.88%
    Joint 10: 98.47%
    Joint 11: 98.73%
    Joint 12: 98.54%

  Overall coverage within 4 std:
    Joint 0: 99.57%
    Joint 1: 99.59%
    Joint 2: 99.62%
    Joint 3: 99.54%
    Joint 4: 99.33%
    Joint 5: 99.84%
    Joint 6: 99.79%
    Joint 7: 99.38%
    Joint 8: 99.55%
    Joint 9: 99.49%
    Joint 10: 99.35%
    Joint 11: 99.29%
    Joint 12: 99.31%
Predicted spherical reachable set coverage stats for 0.99 likelihood:
Overall coverage within set: 98.93%
Mean volume = 0.0035 m^3

Per-Time Coverage Stats:
    Frame 0: 98.96%
    Frame 1: 99.02%
    Frame 2: 98.92%
    Frame 3: 98.95%
    Frame 4: 98.95%
    Frame 5: 98.94%
    Frame 6: 98.92%
    Frame 7: 98.90%
    Frame 8: 98.87%
    Frame 9: 98.86%

Per-Time Volume [m^3]:
    Frame 0: 0.0000
    Frame 1: 0.0001
    Frame 2: 0.0003
    Frame 3: 0.0008
    Frame 4: 0.0019
    Frame 5: 0.0040
    Frame 6: 0.0073
    Frame 7: 0.0123
    Frame 8: 0.0192
    Frame 9: 0.0288

Per-Joint Coverage Stats:
    Joint 0: 98.96%
    Joint 1: 99.13%
    Joint 2: 99.09%
    Joint 3: 98.62%
    Joint 4: 98.15%
    Joint 5: 99.41%
    Joint 6: 99.19%
    Joint 7: 99.03%
    Joint 8: 99.24%
    Joint 9: 99.16%
    Joint 10: 98.85%
    Joint 11: 98.74%
    Joint 12: 98.51%

Per-Joint Volume [m^3]:
    Joint 0: 0.0017
    Joint 1: 0.0015
    Joint 2: 0.0014
    Joint 3: 0.0045
    Joint 4: 0.0039
    Joint 5: 0.0208
    Joint 6: 0.0179
    Joint 7: 0.0011
    Joint 8: 0.0010
    Joint 9: 0.0017
    Joint 10: 0.0019
    Joint 11: 0.0039
    Joint 12: 0.0049
```
**SARA simple velocity model coverage stats**
```
Overall coverage within set: 98.42%
Mean volume = 0.1827 m^3

Per-Time Coverage Stats:
    Frame 0: 98.15%
    Frame 1: 98.18%
    Frame 2: 98.23%
    Frame 3: 98.28%
    Frame 4: 98.35%
    Frame 5: 98.42%
    Frame 6: 98.51%
    Frame 7: 98.61%
    Frame 8: 98.70%
    Frame 9: 98.79%

Per-Time Volume [m^3]:
    Frame 0: 0.0011
    Frame 1: 0.0088
    Frame 2: 0.0296
    Frame 3: 0.0703
    Frame 4: 0.1373
    Frame 5: 0.2372
    Frame 6: 0.3766
    Frame 7: 0.5622
    Frame 8: 0.8005
    Frame 9: 1.0981

Per-Joint Coverage Stats:
    Joint 0: 99.76%
    Joint 1: 99.77%
    Joint 2: 99.79%
    Joint 3: 99.60%
    Joint 4: 99.80%
    Joint 5: 98.03%
    Joint 6: 99.26%
    Joint 7: 99.79%
    Joint 8: 99.85%
    Joint 9: 97.23%
    Joint 10: 98.39%
    Joint 11: 93.74%
    Joint 12: 94.47%

Per-Joint Volume [m^3]:
    Joint 0: 0.1827
    Joint 1: 0.1827
    Joint 2: 0.1827
    Joint 3: 0.1827
    Joint 4: 0.1827
    Joint 5: 0.1827
    Joint 6: 0.1827
    Joint 7: 0.1827
    Joint 8: 0.1827
    Joint 9: 0.1827
    Joint 10: 0.1827
    Joint 11: 0.1827
    Joint 12: 0.1827
```

## Full Evaluation Pipeline

### Regressflow and Motion Prediction
Run with:
```
{
  "name": "Eval Full Pipeline",
  "type": "debugpy",
  "request": "launch",
  "program": "human_pose_pipeline/examples/eval_full_pipeline.py",
  "console": "integratedTerminal",
  "args": [
    "--pose_model_save_path", "human_pose_pipeline/models/pose_estimation",
    "--pose_run_name", "jax_resnet50_regressflow",
    "--pose_base_key", "H36M_RegressFlowResNet18_3Joints_n9000_4998731f",
    "--motion_model_save_path", "human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle",
    "--motion_score_fn_path", "human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM1440of1600_sketch_srft_seed0_size20000.cloudpickle",
    "--split", "validation",
    "--action", "Directions",
    "--max_sequences", "1",
    "--enable_ood"
  ]
},
```
Results:
```
================================
Evaluating 3D pose estimation.
================================

Overall MPJPE: 19.68 mm

Per-Time Errors:
  Time point 1 error =   19.68 mm

Per-Joint Errors:
  Joint 1 error =   22.95 mm
  Joint 2 error =   21.05 mm
  Joint 3 error =   20.06 mm
  Joint 4 error =   26.09 mm
  Joint 5 error =   30.48 mm
  Joint 6 error =   31.00 mm
  Joint 7 error =   26.87 mm
  Joint 8 error =   17.42 mm
  Joint 9 error =   15.94 mm
  Joint 10 error =   11.00 mm
  Joint 11 error =    9.86 mm
  Joint 12 error =   12.12 mm
  Joint 13 error =   10.99 mm
Saved overall MPJPE results to results/motion_prediction/mpjpe_results_validation.csv
Saved per-time MPJPE results to results/motion_prediction/per_time_mpjpe_results_validation.csv
Saved per-joint MPJPE results to results/motion_prediction/per_joint_mpjpe_results_validation.csv

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 38.84%
  Overall coverage within 2 std: 70.07%
  Overall coverage within 3 std: 86.35%
  Overall coverage within 4 std: 93.93%

Per-Time Coverage Stats:

  Overall coverage within 1 std:
    Frame 0: 38.84%

  Overall coverage within 2 std:
    Frame 0: 70.07%

  Overall coverage within 3 std:
    Frame 0: 86.35%

  Overall coverage within 4 std:
    Frame 0: 93.93%

Per-Joint Coverage Stats:

  Overall coverage within 1 std:
    Joint 0: 0.00%
    Joint 1: 25.36%
    Joint 2: 45.85%
    Joint 3: 36.43%
    Joint 4: 19.27%
    Joint 5: 29.57%
    Joint 6: 56.37%
    Joint 7: 42.41%
    Joint 8: 22.37%
    Joint 9: 56.37%
    Joint 10: 53.93%
    Joint 11: 54.71%
    Joint 12: 62.35%

  Overall coverage within 2 std:
    Joint 0: 2.77%
    Joint 1: 57.48%
    Joint 2: 71.54%
    Joint 3: 74.75%
    Joint 4: 61.24%
    Joint 5: 72.31%
    Joint 6: 83.17%
    Joint 7: 72.65%
    Joint 8: 72.31%
    Joint 9: 86.60%
    Joint 10: 83.06%
    Joint 11: 82.06%
    Joint 12: 91.03%

  Overall coverage within 3 std:
    Joint 0: 20.71%
    Joint 1: 78.96%
    Joint 2: 84.61%
    Joint 3: 93.58%
    Joint 4: 88.26%
    Joint 5: 93.91%
    Joint 6: 92.80%
    Joint 7: 95.02%
    Joint 8: 95.79%
    Joint 9: 97.12%
    Joint 10: 93.02%
    Joint 11: 90.48%
    Joint 12: 98.34%

  Overall coverage within 4 std:
    Joint 0: 55.04%
    Joint 1: 90.59%
    Joint 2: 94.13%
    Joint 3: 98.89%
    Joint 4: 95.90%
    Joint 5: 97.90%
    Joint 6: 96.57%
    Joint 7: 99.00%
    Joint 8: 99.56%
    Joint 9: 100.00%
    Joint 10: 99.00%
    Joint 11: 94.57%
    Joint 12: 100.00%
Saved overall coverage results to results/motion_prediction/coverage_results_validation.csv
Saved per-time coverage results to results/motion_prediction/per_time_coverage_results_validation.csv
Saved per-joint coverage results to results/motion_prediction/per_joint_coverage_results_validation.csv
================================
Evaluating motion prediction.
================================
================================
Evaluating motion uncertainty prediction.
================================

Overall MPJPE: 41.90 mm

Per-Time Errors:
  Time point 1 error =   22.70 mm
  Time point 2 error =   25.02 mm
  Time point 3 error =   29.32 mm
  Time point 4 error =   34.12 mm
  Time point 5 error =   39.61 mm
  Time point 6 error =   44.33 mm
  Time point 7 error =   49.78 mm
  Time point 8 error =   53.99 mm
  Time point 9 error =   58.06 mm
  Time point 10 error =   62.09 mm

Per-Joint Errors:
  Joint 1 error =   41.17 mm
  Joint 2 error =   36.65 mm
  Joint 3 error =   37.37 mm
  Joint 4 error =   57.77 mm
  Joint 5 error =   59.59 mm
  Joint 6 error =   98.34 mm
  Joint 7 error =   70.94 mm
  Joint 8 error =   31.48 mm
  Joint 9 error =   31.79 mm
  Joint 10 error =   21.07 mm
  Joint 11 error =   21.29 mm
  Joint 12 error =   19.58 mm
  Joint 13 error =   17.69 mm
Saved overall MPJPE results to results/motion_prediction/mpjpe_results_validation.csv
Saved per-time MPJPE results to results/motion_prediction/per_time_mpjpe_results_validation.csv
Saved per-joint MPJPE results to results/motion_prediction/per_joint_mpjpe_results_validation.csv

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 64.44%
  Overall coverage within 2 std: 80.77%
  Overall coverage within 3 std: 88.86%
  Overall coverage within 4 std: 93.23%

Per-Time Coverage Stats:

  Overall coverage within 1 std:
    Frame 0: 26.08%
    Frame 1: 31.01%
    Frame 2: 44.08%
    Frame 3: 57.82%
    Frame 4: 68.37%
    Frame 5: 75.83%
    Frame 6: 80.34%
    Frame 7: 84.22%
    Frame 8: 87.25%
    Frame 9: 89.42%

  Overall coverage within 2 std:
    Frame 0: 46.67%
    Frame 1: 52.35%
    Frame 2: 67.08%
    Frame 3: 80.07%
    Frame 4: 86.94%
    Frame 5: 91.49%
    Frame 6: 93.79%
    Frame 7: 95.51%
    Frame 8: 96.56%
    Frame 9: 97.24%

  Overall coverage within 3 std:
    Frame 0: 61.75%
    Frame 1: 67.47%
    Frame 2: 81.07%
    Frame 3: 91.03%
    Frame 4: 94.91%
    Frame 5: 97.12%
    Frame 6: 98.04%
    Frame 7: 98.70%
    Frame 8: 99.14%
    Frame 9: 99.35%

  Overall coverage within 4 std:
    Frame 0: 72.91%
    Frame 1: 77.99%
    Frame 2: 89.44%
    Frame 3: 96.01%
    Frame 4: 98.06%
    Frame 5: 98.99%
    Frame 6: 99.46%
    Frame 7: 99.75%
    Frame 8: 99.82%
    Frame 9: 99.86%

Per-Joint Coverage Stats:

  Overall coverage within 1 std:
    Joint 0: 47.99%
    Joint 1: 55.90%
    Joint 2: 55.47%
    Joint 3: 53.70%
    Joint 4: 48.19%
    Joint 5: 53.99%
    Joint 6: 67.92%
    Joint 7: 55.84%
    Joint 8: 52.92%
    Joint 9: 79.59%
    Joint 10: 84.18%
    Joint 11: 86.92%
    Joint 12: 95.13%

  Overall coverage within 2 std:
    Joint 0: 67.66%
    Joint 1: 74.75%
    Joint 2: 75.91%
    Joint 3: 74.18%
    Joint 4: 68.09%
    Joint 5: 76.09%
    Joint 6: 83.33%
    Joint 7: 75.39%
    Joint 8: 74.25%
    Joint 9: 90.91%
    Joint 10: 94.78%
    Joint 11: 95.30%
    Joint 12: 99.37%

  Overall coverage within 3 std:
    Joint 0: 77.97%
    Joint 1: 84.48%
    Joint 2: 86.48%
    Joint 3: 84.58%
    Joint 4: 79.10%
    Joint 5: 88.47%
    Joint 6: 91.21%
    Joint 7: 85.12%
    Joint 8: 85.74%
    Joint 9: 95.60%
    Joint 10: 98.51%
    Joint 11: 97.94%
    Joint 12: 99.94%

  Overall coverage within 4 std:
    Joint 0: 84.99%
    Joint 1: 90.07%
    Joint 2: 91.92%
    Joint 3: 90.27%
    Joint 4: 86.39%
    Joint 5: 94.57%
    Joint 6: 95.27%
    Joint 7: 90.18%
    Joint 8: 91.79%
    Joint 9: 97.60%
    Joint 10: 99.56%
    Joint 11: 99.39%
    Joint 12: 100.00%
Saved overall coverage results to results/motion_prediction/coverage_results_validation.csv
Saved per-time coverage results to results/motion_prediction/per_time_coverage_results_validation.csv
Saved per-joint coverage results to results/motion_prediction/per_joint_coverage_results_validation.csv
Predicted spherical reachable set coverage stats for 0.99 likelihood:
Overall coverage within set: 76.63%
Mean volume = 0.0028 m^3

Per-Time Coverage Stats:
    Frame 0: 58.23%
    Frame 1: 63.88%
    Frame 2: 74.02%
    Frame 3: 81.18%
    Frame 4: 83.45%
    Frame 5: 84.33%
    Frame 6: 83.67%
    Frame 7: 82.69%
    Frame 8: 80.67%
    Frame 9: 74.13%

Per-Time Volume [m^3]:
    Frame 0: 0.0001
    Frame 1: 0.0002
    Frame 2: 0.0005
    Frame 3: 0.0010
    Frame 4: 0.0021
    Frame 5: 0.0036
    Frame 6: 0.0059
    Frame 7: 0.0088
    Frame 8: 0.0120
    Frame 9: 0.0128

Per-Joint Coverage Stats:
    Joint 0: 67.75%
    Joint 1: 72.76%
    Joint 2: 74.99%
    Joint 3: 71.32%
    Joint 4: 67.60%
    Joint 5: 77.93%
    Joint 6: 77.82%
    Joint 7: 73.74%
    Joint 8: 73.85%
    Joint 9: 82.85%
    Joint 10: 85.13%
    Joint 11: 84.36%
    Joint 12: 86.04%

Per-Joint Volume [m^3]:
    Joint 0: 0.0015
    Joint 1: 0.0012
    Joint 2: 0.0011
    Joint 3: 0.0038
    Joint 4: 0.0032
    Joint 5: 0.0177
    Joint 6: 0.0150
    Joint 7: 0.0008
    Joint 8: 0.0008
    Joint 9: 0.0012
    Joint 10: 0.0016
    Joint 11: 0.0022
    Joint 12: 0.0045
================================
Evaluating motion SARA uncertainty.
================================
SARA simple velocity model coverage stats:
Overall coverage within set: 99.38%
Mean volume = 0.1827 m^3

Per-Time Coverage Stats:
    Frame 0: 97.85%
    Frame 1: 99.19%
    Frame 2: 99.22%
    Frame 3: 99.22%
    Frame 4: 99.30%
    Frame 5: 99.35%
    Frame 6: 99.64%
    Frame 7: 100.00%
    Frame 8: 100.00%
    Frame 9: 100.00%

Per-Time Volume [m^3]:
    Frame 0: 0.0011
    Frame 1: 0.0088
    Frame 2: 0.0296
    Frame 3: 0.0703
    Frame 4: 0.1373
    Frame 5: 0.2372
    Frame 6: 0.3766
    Frame 7: 0.5622
    Frame 8: 0.8005
    Frame 9: 1.0981

Per-Joint Coverage Stats:
    Joint 0: 99.99%
    Joint 1: 100.00%
    Joint 2: 99.75%
    Joint 3: 99.81%
    Joint 4: 99.36%
    Joint 5: 93.61%
    Joint 6: 99.38%
    Joint 7: 100.00%
    Joint 8: 100.00%
    Joint 9: 100.00%
    Joint 10: 100.00%
    Joint 11: 100.00%
    Joint 12: 100.00%

Per-Joint Volume [m^3]:
    Joint 0: 0.1827
    Joint 1: 0.1827
    Joint 2: 0.1827
    Joint 3: 0.1827
    Joint 4: 0.1827
    Joint 5: 0.1827
    Joint 6: 0.1827
    Joint 7: 0.1827
    Joint 8: 0.1827
    Joint 9: 0.1827
    Joint 10: 0.1827
    Joint 11: 0.1827
    Joint 12: 0.1827
```

Findings:
 - MPJPE ~40 higher than full evaluation ~23
 - Test this sequence in motion_predcition.py only.
 - Coverage stats significantly under-performing.