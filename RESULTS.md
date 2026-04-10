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
      "--max_sequences", "10",
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
Overall MPJPE: 47.82 mm, Std: 45.06 mm

Per-Time Errors:
Time point 1 error =   33.12 mm
Time point 2 error =   33.77 mm
Time point 3 error =   36.20 mm
Time point 4 error =   40.04 mm
Time point 5 error =   43.95 mm
Time point 6 error =   48.37 mm
Time point 7 error =   53.03 mm
Time point 8 error =   57.94 mm
Time point 9 error =   63.13 mm
Time point 10 error =   68.61 mm

Per-Joint Errors:
Joint 1 error =   37.02 mm
Joint 2 error =   38.15 mm
Joint 3 error =   38.26 mm
Joint 4 error =   57.26 mm
Joint 5 error =   58.11 mm
Joint 6 error =   75.33 mm
Joint 7 error =   78.25 mm
Joint 8 error =   45.11 mm
Joint 9 error =   36.48 mm
Joint 10 error =   33.88 mm
Joint 11 error =   35.26 mm
Joint 12 error =   41.68 mm
Joint 13 error =   46.82 mm

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 55.07%
  Overall coverage within 2 std: 78.65%
  Overall coverage within 3 std: 89.36%
  Overall coverage within 4 std: 94.31%
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
Loaded predictions shape: (47040, 10, 13, 3)
Loaded targets shape: (47040, 10, 13, 3)
Loaded covariance matrices shape: (47040, 10, 13, 3, 3)

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 84.59%
  Overall coverage within 2 std: 95.86%
  Overall coverage within 3 std: 98.73%
  Overall coverage within 4 std: 99.55%

Per-Time Coverage Stats:

  Overall coverage within 1 std:
    Frame 0: 69.06%
    Frame 1: 78.43%
    Frame 2: 82.92%
    Frame 3: 85.35%
    Frame 4: 86.84%
    Frame 5: 87.83%
    Frame 6: 88.43%
    Frame 7: 88.87%
    Frame 8: 89.06%
    Frame 9: 89.14%

  Overall coverage within 2 std:
    Frame 0: 90.43%
    Frame 1: 94.38%
    Frame 2: 95.70%
    Frame 3: 96.36%
    Frame 4: 96.73%
    Frame 5: 96.92%
    Frame 6: 97.02%
    Frame 7: 97.04%
    Frame 8: 97.03%
    Frame 9: 96.99%

  Overall coverage within 3 std:
    Frame 0: 96.85%
    Frame 1: 98.30%
    Frame 2: 98.76%
    Frame 3: 99.01%
    Frame 4: 99.12%
    Frame 5: 99.11%
    Frame 6: 99.10%
    Frame 7: 99.06%
    Frame 8: 99.03%
    Frame 9: 98.99%

  Overall coverage within 4 std:
    Frame 0: 98.80%
    Frame 1: 99.42%
    Frame 2: 99.62%
    Frame 3: 99.69%
    Frame 4: 99.70%
    Frame 5: 99.69%
    Frame 6: 99.67%
    Frame 7: 99.65%
    Frame 8: 99.64%
    Frame 9: 99.61%

Per-Joint Coverage Stats:

  Overall coverage within 1 std:
    Joint 0: 85.17%
    Joint 1: 85.51%
    Joint 2: 85.29%
    Joint 3: 78.75%
    Joint 4: 75.94%
    Joint 5: 87.34%
    Joint 6: 84.09%
    Joint 7: 78.81%
    Joint 8: 87.93%
    Joint 9: 88.93%
    Joint 10: 86.42%
    Joint 11: 89.81%
    Joint 12: 85.69%

  Overall coverage within 2 std:
    Joint 0: 96.64%
    Joint 1: 97.01%
    Joint 2: 96.83%
    Joint 3: 94.38%
    Joint 4: 92.04%
    Joint 5: 97.37%
    Joint 6: 95.80%
    Joint 7: 93.19%
    Joint 8: 97.51%
    Joint 9: 97.31%
    Joint 10: 96.21%
    Joint 11: 96.91%
    Joint 12: 94.96%

  Overall coverage within 3 std:
    Joint 0: 99.10%
    Joint 1: 99.25%
    Joint 2: 99.19%
    Joint 3: 98.52%
    Joint 4: 97.22%
    Joint 5: 99.32%
    Joint 6: 98.81%
    Joint 7: 97.81%
    Joint 8: 99.44%
    Joint 9: 99.16%
    Joint 10: 98.80%
    Joint 11: 98.97%
    Joint 12: 97.91%

  Overall coverage within 4 std:
    Joint 0: 99.68%
    Joint 1: 99.80%
    Joint 2: 99.75%
    Joint 3: 99.58%
    Joint 4: 98.92%
    Joint 5: 99.78%
    Joint 6: 99.58%
    Joint 7: 99.25%
    Joint 8: 99.86%
    Joint 9: 99.71%
    Joint 10: 99.62%
    Joint 11: 99.59%
    Joint 12: 99.04%
Predicted spherical reachable set coverage stats for 0.99 likelihood:
Overall coverage within set: 98.93%
Mean volume = 0.0174 m^3

Per-Time Coverage Stats:
    Frame 0: 97.45%
    Frame 1: 98.66%
    Frame 2: 98.99%
    Frame 3: 99.13%
    Frame 4: 99.22%
    Frame 5: 99.21%
    Frame 6: 99.20%
    Frame 7: 99.17%
    Frame 8: 99.14%
    Frame 9: 99.11%

Per-Time Volume [m^3]:
    Frame 0: 0.0022
    Frame 1: 0.0036
    Frame 2: 0.0058
    Frame 3: 0.0090
    Frame 4: 0.0134
    Frame 5: 0.0194
    Frame 6: 0.0274
    Frame 7: 0.0378
    Frame 8: 0.0510
    Frame 9: 0.0673

Per-Joint Coverage Stats:
    Joint 0: 99.39%
    Joint 1: 99.53%
    Joint 2: 99.46%
    Joint 3: 98.76%
    Joint 4: 97.74%
    Joint 5: 99.23%
    Joint 6: 98.77%
    Joint 7: 98.08%
    Joint 8: 99.50%
    Joint 9: 99.44%
    Joint 10: 99.04%
    Joint 11: 99.06%
    Joint 12: 98.07%

Per-Joint Volume [m^3]:
    Joint 0: 0.0087
    Joint 1: 0.0092
    Joint 2: 0.0090
    Joint 3: 0.0206
    Joint 4: 0.0179
    Joint 5: 0.0690
    Joint 6: 0.0640
    Joint 7: 0.0105
    Joint 8: 0.0095
    Joint 9: 0.0094
    Joint 10: 0.0084
    Joint 11: 0.0227
    Joint 12: 0.0203
```
**SARA simple velocity model coverage stats**
```
SARA simple velocity model coverage stats:
Overall coverage within set: 97.58%
Mean volume = 0.1827 m^3

Per-Time Coverage Stats:
    Frame 0: 89.11%
    Frame 1: 96.85%
    Frame 2: 98.06%
    Frame 3: 98.44%
    Frame 4: 98.61%
    Frame 5: 98.75%
    Frame 6: 98.86%
    Frame 7: 98.96%
    Frame 8: 99.04%
    Frame 9: 99.12%

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
    Joint 0: 99.43%
    Joint 1: 99.08%
    Joint 2: 99.48%
    Joint 3: 98.27%
    Joint 4: 98.12%
    Joint 5: 96.09%
    Joint 6: 96.41%
    Joint 7: 98.15%
    Joint 8: 99.19%
    Joint 9: 97.36%
    Joint 10: 98.41%
    Joint 11: 94.41%
    Joint 12: 94.13%

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

### Tried Different Offset Strategy

```
        # Subtract only the head position (first 3 entries) from all joints
        offset = x[:, -1:, 0:3]
        offset = jnp.tile(offset, (1, 1, x.shape[-1] // 3))
```
Eval MPJPE = 55mm slightly worse!
-> Reverse Change.

## Full Evaluation Pipeline

### Action = Directions, 1 Sequence
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
  Joint 4 error =   26.10 mm
  Joint 5 error =   30.48 mm
  Joint 6 error =   30.99 mm
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
  Overall coverage within 1 std: 38.72%
  Overall coverage within 2 std: 70.13%
  Overall coverage within 3 std: 86.41%
  Overall coverage within 4 std: 93.93%

Per-Time Coverage Stats:

  Overall coverage within 1 std:
    Frame 0: 38.72%

  Overall coverage within 2 std:
    Frame 0: 70.13%

  Overall coverage within 3 std:
    Frame 0: 86.41%

  Overall coverage within 4 std:
    Frame 0: 93.93%

Per-Joint Coverage Stats:

  Overall coverage within 1 std:
    Joint 0: 0.00%
    Joint 1: 25.25%
    Joint 2: 45.74%
    Joint 3: 36.77%
    Joint 4: 17.50%
    Joint 5: 29.13%
    Joint 6: 56.48%
    Joint 7: 42.19%
    Joint 8: 22.70%
    Joint 9: 56.37%
    Joint 10: 53.60%
    Joint 11: 54.60%
    Joint 12: 63.01%

  Overall coverage within 2 std:
    Joint 0: 2.99%
    Joint 1: 58.25%
    Joint 2: 71.87%
    Joint 3: 73.98%
    Joint 4: 60.91%
    Joint 5: 72.76%
    Joint 6: 82.83%
    Joint 7: 73.53%
    Joint 8: 71.87%
    Joint 9: 87.04%
    Joint 10: 82.72%
    Joint 11: 81.95%
    Joint 12: 91.03%

  Overall coverage within 3 std:
    Joint 0: 20.38%
    Joint 1: 79.84%
    Joint 2: 84.16%
    Joint 3: 93.36%
    Joint 4: 88.37%
    Joint 5: 94.68%
    Joint 6: 92.47%
    Joint 7: 95.13%
    Joint 8: 95.68%
    Joint 9: 97.23%
    Joint 10: 93.02%
    Joint 11: 90.59%
    Joint 12: 98.45%

  Overall coverage within 4 std:
    Joint 0: 55.70%
    Joint 1: 91.36%
    Joint 2: 93.47%
    Joint 3: 98.45%
    Joint 4: 95.57%
    Joint 5: 98.34%
    Joint 6: 96.12%
    Joint 7: 98.89%
    Joint 8: 99.56%
    Joint 9: 100.00%
    Joint 10: 99.11%
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

Overall MPJPE: 34.66 mm

Per-Time Errors:
  Time point 1 error =   23.02 mm
  Time point 2 error =   24.22 mm
  Time point 3 error =   26.38 mm
  Time point 4 error =   29.60 mm
  Time point 5 error =   32.67 mm
  Time point 6 error =   36.04 mm
  Time point 7 error =   39.14 mm
  Time point 8 error =   42.32 mm
  Time point 9 error =   45.23 mm
  Time point 10 error =   48.01 mm

Per-Joint Errors:
  Joint 1 error =   27.44 mm
  Joint 2 error =   28.48 mm
  Joint 3 error =   25.88 mm
  Joint 4 error =   54.93 mm
  Joint 5 error =   46.46 mm
  Joint 6 error =   99.17 mm
  Joint 7 error =   64.15 mm
  Joint 8 error =   22.02 mm
  Joint 9 error =   17.91 mm
  Joint 10 error =   17.39 mm
  Joint 11 error =   16.08 mm
  Joint 12 error =   16.23 mm
  Joint 13 error =   14.49 mm
Saved overall MPJPE results to results/motion_prediction/mpjpe_results_validation.csv
Saved per-time MPJPE results to results/motion_prediction/per_time_mpjpe_results_validation.csv
Saved per-joint MPJPE results to results/motion_prediction/per_joint_mpjpe_results_validation.csv

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 91.63%
  Overall coverage within 2 std: 97.43%
  Overall coverage within 3 std: 99.16%
  Overall coverage within 4 std: 99.73%

Per-Time Coverage Stats:

  Overall coverage within 1 std:
    Frame 0: 81.65%
    Frame 1: 87.77%
    Frame 2: 90.46%
    Frame 3: 92.14%
    Frame 4: 93.06%
    Frame 5: 93.54%
    Frame 6: 93.91%
    Frame 7: 94.24%
    Frame 8: 94.61%
    Frame 9: 94.96%

  Overall coverage within 2 std:
    Frame 0: 95.01%
    Frame 1: 97.13%
    Frame 2: 97.60%
    Frame 3: 97.55%
    Frame 4: 97.60%
    Frame 5: 97.63%
    Frame 6: 97.72%
    Frame 7: 97.77%
    Frame 8: 98.07%
    Frame 9: 98.26%

  Overall coverage within 3 std:
    Frame 0: 99.14%
    Frame 1: 99.18%
    Frame 2: 99.11%
    Frame 3: 98.99%
    Frame 4: 98.96%
    Frame 5: 99.00%
    Frame 6: 99.08%
    Frame 7: 99.20%
    Frame 8: 99.42%
    Frame 9: 99.57%

  Overall coverage within 4 std:
    Frame 0: 99.77%
    Frame 1: 99.62%
    Frame 2: 99.59%
    Frame 3: 99.58%
    Frame 4: 99.58%
    Frame 5: 99.68%
    Frame 6: 99.84%
    Frame 7: 99.86%
    Frame 8: 99.90%
    Frame 9: 99.92%

Per-Joint Coverage Stats:

  Overall coverage within 1 std:
    Joint 0: 91.94%
    Joint 1: 90.66%
    Joint 2: 94.82%
    Joint 3: 74.67%
    Joint 4: 83.50%
    Joint 5: 73.15%
    Joint 6: 88.16%
    Joint 7: 97.32%
    Joint 8: 99.51%
    Joint 9: 99.50%
    Joint 10: 98.88%
    Joint 11: 99.78%
    Joint 12: 99.36%

  Overall coverage within 2 std:
    Joint 0: 98.88%
    Joint 1: 98.43%
    Joint 2: 99.19%
    Joint 3: 92.74%
    Joint 4: 94.47%
    Joint 5: 87.58%
    Joint 6: 95.68%
    Joint 7: 99.92%
    Joint 8: 100.00%
    Joint 9: 100.00%
    Joint 10: 99.89%
    Joint 11: 100.00%
    Joint 12: 99.85%

  Overall coverage within 3 std:
    Joint 0: 99.71%
    Joint 1: 99.92%
    Joint 2: 99.75%
    Joint 3: 98.21%
    Joint 4: 98.41%
    Joint 5: 94.44%
    Joint 6: 98.70%
    Joint 7: 100.00%
    Joint 8: 100.00%
    Joint 9: 100.00%
    Joint 10: 100.00%
    Joint 11: 100.00%
    Joint 12: 100.00%

  Overall coverage within 4 std:
    Joint 0: 99.92%
    Joint 1: 100.00%
    Joint 2: 99.85%
    Joint 3: 99.56%
    Joint 4: 99.59%
    Joint 5: 97.74%
    Joint 6: 99.88%
    Joint 7: 100.00%
    Joint 8: 100.00%
    Joint 9: 100.00%
    Joint 10: 100.00%
    Joint 11: 100.00%
    Joint 12: 100.00%
Saved overall coverage results to results/motion_prediction/coverage_results_validation.csv
Saved per-time coverage results to results/motion_prediction/per_time_coverage_results_validation.csv
Saved per-joint coverage results to results/motion_prediction/per_joint_coverage_results_validation.csv
Predicted spherical reachable set coverage stats for 0.99 likelihood:
Overall coverage within set: 99.18%
Mean volume = 0.0125 m^3

Per-Time Coverage Stats:
    Frame 0: 99.35%
    Frame 1: 99.39%
    Frame 2: 99.23%
    Frame 3: 99.11%
    Frame 4: 99.01%
    Frame 5: 98.92%
    Frame 6: 98.90%
    Frame 7: 99.08%
    Frame 8: 99.29%
    Frame 9: 99.51%

Per-Time Volume [m^3]:
    Frame 0: 0.0014
    Frame 1: 0.0024
    Frame 2: 0.0038
    Frame 3: 0.0060
    Frame 4: 0.0093
    Frame 5: 0.0138
    Frame 6: 0.0202
    Frame 7: 0.0284
    Frame 8: 0.0389
    Frame 9: 0.0519

Per-Joint Coverage Stats:
    Joint 0: 99.66%
    Joint 1: 99.93%
    Joint 2: 99.88%
    Joint 3: 98.81%
    Joint 4: 98.61%
    Joint 5: 94.00%
    Joint 6: 98.44%
    Joint 7: 100.00%
    Joint 8: 100.00%
    Joint 9: 100.00%
    Joint 10: 100.00%
    Joint 11: 100.00%
    Joint 12: 100.00%

Per-Joint Volume [m^3]:
    Joint 0: 0.0068
    Joint 1: 0.0071
    Joint 2: 0.0076
    Joint 3: 0.0172
    Joint 4: 0.0159
    Joint 5: 0.0616
    Joint 6: 0.0579
    Joint 7: 0.0064
    Joint 8: 0.0056
    Joint 9: 0.0058
    Joint 10: 0.0046
    Joint 11: 0.0125
    Joint 12: 0.0091
================================
Evaluating motion SARA uncertainty.
================================
SARA simple velocity model coverage stats:
Overall coverage within set: 99.18%
Mean volume = 0.1827 m^3

Per-Time Coverage Stats:
    Frame 0: 96.83%
    Frame 1: 98.85%
    Frame 2: 98.98%
    Frame 3: 99.11%
    Frame 4: 99.20%
    Frame 5: 99.27%
    Frame 6: 99.54%
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
    Joint 2: 99.74%
    Joint 3: 99.66%
    Joint 4: 99.24%
    Joint 5: 92.11%
    Joint 6: 98.57%
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
 - MPJPE in this simple task slightly better than average evaluation performance.
 - Everything is working now!

### Model Trained on Augmented Data

Results:
```
================================
Evaluating motion prediction.
================================
================================
Evaluating motion uncertainty prediction.
================================

Overall MPJPE: 34.59 mm

Per-Time Errors:
  Time point 1 error =   23.09 mm
  Time point 2 error =   24.09 mm
  Time point 3 error =   26.75 mm
  Time point 4 error =   29.71 mm
  Time point 5 error =   33.08 mm
  Time point 6 error =   36.02 mm
  Time point 7 error =   39.12 mm
  Time point 8 error =   41.99 mm
  Time point 9 error =   44.85 mm
  Time point 10 error =   47.23 mm

Per-Joint Errors:
  Joint 1 error =   27.29 mm
  Joint 2 error =   29.98 mm
  Joint 3 error =   26.82 mm
  Joint 4 error =   54.43 mm
  Joint 5 error =   48.96 mm
  Joint 6 error =   93.83 mm
  Joint 7 error =   62.41 mm
  Joint 8 error =   24.25 mm
  Joint 9 error =   20.55 mm
  Joint 10 error =   17.26 mm
  Joint 11 error =   15.34 mm
  Joint 12 error =   15.31 mm
  Joint 13 error =   13.27 mm
Saved overall MPJPE results to results/motion_prediction/mpjpe_results_validation.csv
Saved per-time MPJPE results to results/motion_prediction/per_time_mpjpe_results_validation.csv
Saved per-joint MPJPE results to results/motion_prediction/per_joint_mpjpe_results_validation.csv

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 90.91%
  Overall coverage within 2 std: 97.39%
  Overall coverage within 3 std: 99.13%
  Overall coverage within 4 std: 99.77%

Per-Time Coverage Stats:

  Overall coverage within 1 std:
    Frame 0: 81.47%
    Frame 1: 87.52%
    Frame 2: 89.45%
    Frame 3: 91.04%
    Frame 4: 91.88%
    Frame 5: 92.56%
    Frame 6: 93.17%
    Frame 7: 93.53%
    Frame 8: 94.06%
    Frame 9: 94.44%

  Overall coverage within 2 std:
    Frame 0: 95.32%
    Frame 1: 96.56%
    Frame 2: 97.25%
    Frame 3: 97.44%
    Frame 4: 97.45%
    Frame 5: 97.65%
    Frame 6: 97.83%
    Frame 7: 97.97%
    Frame 8: 98.11%
    Frame 9: 98.27%

  Overall coverage within 3 std:
    Frame 0: 98.97%
    Frame 1: 99.16%
    Frame 2: 99.23%
    Frame 3: 99.14%
    Frame 4: 99.08%
    Frame 5: 99.03%
    Frame 6: 99.00%
    Frame 7: 99.14%
    Frame 8: 99.23%
    Frame 9: 99.30%

  Overall coverage within 4 std:
    Frame 0: 99.77%
    Frame 1: 99.81%
    Frame 2: 99.77%
    Frame 3: 99.70%
    Frame 4: 99.69%
    Frame 5: 99.73%
    Frame 6: 99.76%
    Frame 7: 99.77%
    Frame 8: 99.82%
    Frame 9: 99.84%

Per-Joint Coverage Stats:

  Overall coverage within 1 std:
    Joint 0: 89.11%
    Joint 1: 88.74%
    Joint 2: 92.66%
    Joint 3: 76.04%
    Joint 4: 78.59%
    Joint 5: 74.64%
    Joint 6: 87.75%
    Joint 7: 96.73%
    Joint 8: 99.54%
    Joint 9: 99.18%
    Joint 10: 99.52%
    Joint 11: 99.81%
    Joint 12: 99.54%

  Overall coverage within 2 std:
    Joint 0: 98.08%
    Joint 1: 98.40%
    Joint 2: 98.72%
    Joint 3: 93.19%
    Joint 4: 92.18%
    Joint 5: 89.96%
    Joint 6: 95.55%
    Joint 7: 99.93%
    Joint 8: 100.00%
    Joint 9: 100.00%
    Joint 10: 100.00%
    Joint 11: 100.00%
    Joint 12: 100.00%

  Overall coverage within 3 std:
    Joint 0: 99.58%
    Joint 1: 99.95%
    Joint 2: 99.70%
    Joint 3: 97.81%
    Joint 4: 96.99%
    Joint 5: 96.52%
    Joint 6: 98.14%
    Joint 7: 100.00%
    Joint 8: 100.00%
    Joint 9: 100.00%
    Joint 10: 100.00%
    Joint 11: 100.00%
    Joint 12: 100.00%

  Overall coverage within 4 std:
    Joint 0: 99.94%
    Joint 1: 100.00%
    Joint 2: 99.84%
    Joint 3: 99.44%
    Joint 4: 98.86%
    Joint 5: 99.33%
    Joint 6: 99.56%
    Joint 7: 100.00%
    Joint 8: 100.00%
    Joint 9: 100.00%
    Joint 10: 100.00%
    Joint 11: 100.00%
    Joint 12: 100.00%
Saved overall coverage results to results/motion_prediction/coverage_results_validation.csv
Saved per-time coverage results to results/motion_prediction/per_time_coverage_results_validation.csv
Saved per-joint coverage results to results/motion_prediction/per_joint_coverage_results_validation.csv
Predicted spherical reachable set coverage stats for 0.99 likelihood:
Overall coverage within set: 98.82%
Mean volume = 0.0093 m^3

Per-Time Coverage Stats:
    Frame 0: 98.48%
    Frame 1: 98.75%
    Frame 2: 98.94%
    Frame 3: 98.78%
    Frame 4: 98.79%
    Frame 5: 98.77%
    Frame 6: 98.80%
    Frame 7: 98.85%
    Frame 8: 98.99%
    Frame 9: 99.03%

Per-Time Volume [m^3]:
    Frame 0: 0.0011
    Frame 1: 0.0018
    Frame 2: 0.0029
    Frame 3: 0.0045
    Frame 4: 0.0069
    Frame 5: 0.0103
    Frame 6: 0.0149
    Frame 7: 0.0208
    Frame 8: 0.0285
    Frame 9: 0.0379

Per-Joint Coverage Stats:
    Joint 0: 99.29%
    Joint 1: 99.78%
    Joint 2: 99.64%
    Joint 3: 97.05%
    Joint 4: 96.16%
    Joint 5: 95.05%
    Joint 6: 97.67%
    Joint 7: 100.00%
    Joint 8: 100.00%
    Joint 9: 100.00%
    Joint 10: 100.00%
    Joint 11: 100.00%
    Joint 12: 100.00%

Per-Joint Volume [m^3]:
    Joint 0: 0.0042
    Joint 1: 0.0048
    Joint 2: 0.0048
    Joint 3: 0.0115
    Joint 4: 0.0098
    Joint 5: 0.0525
    Joint 6: 0.0450
    Joint 7: 0.0053
    Joint 8: 0.0049
    Joint 9: 0.0042
    Joint 10: 0.0030
    Joint 11: 0.0113
    Joint 12: 0.0070
```

 - Very comparable results. 
 - Not much changed, however, probably good to have it.
 - Use model trained on augmented data `r42sn31c` as new final model from now on.

## Full Evaluation RGB-D YOLO Lab Recordings

Run with:
```
python human_pose_pipeline/examples/eval_full_pipeline_rgbd_yolo.py --enable_tracking --depth_uncertainty 0.05 --start_at 1656 
```

Results (Tuned on first 2000 data points only)
```
============================================================
Full Pipeline - JAX Implementation
============================================================

Initializing models...

Initializing YOLO pose model...
Model loaded on CUDA (GPU: NVIDIA GeForce RTX 5090)
Loading JAX model from /home/thumm/code/sketching_lanczos/uncertainty_quantification/human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle...
JAX model loaded successfully.
  - Model type: DCTPoseTransformer
  - Output dim: 1560
  - Has batch stats: False
Compiling JIT inference function...
JIT compilation complete!
Models initialized successfully!

Loading OOD score functions...

Loading RGB-D dataset...
Loaded 12404 paired RGBD samples from datasets/rgbd_test/
Dataset loaded with 12404 frames, from which we are using 10748 frames             with 1 subsampling.
Evaluating sequence:: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10748/10748 [07:32<00:00, 23.76it/s]
Full pipeline completed!
Saved motion prediction results to results/full_pipeline_rgbd_yolo/motion_prediction_results.cloudpickle
================================
Evaluating motion prediction.
================================
================================
Evaluating motion uncertainty prediction.
================================

Overall MPJPE: 83.91 mm

Per-Time Errors:
  Time point 1 error =   45.40 mm
  Time point 2 error =   52.84 mm
  Time point 3 error =   61.72 mm
  Time point 4 error =   71.30 mm
  Time point 5 error =   80.00 mm
  Time point 6 error =   89.04 mm
  Time point 7 error =   97.01 mm
  Time point 8 error =  105.93 mm
  Time point 9 error =  113.95 mm
  Time point 10 error =  122.10 mm

Per-Joint Errors:
  Joint 1 error =   68.26 mm
  Joint 2 error =   72.23 mm
  Joint 3 error =   60.43 mm
  Joint 4 error =  104.17 mm
  Joint 5 error =   94.54 mm
  Joint 6 error =  162.30 mm
  Joint 7 error =  140.70 mm
  Joint 8 error =   60.48 mm
  Joint 9 error =   60.42 mm
  Joint 10 error =   67.68 mm
  Joint 11 error =   65.11 mm
  Joint 12 error =   70.87 mm
  Joint 13 error =   63.65 mm
Saved overall MPJPE results to results/motion_prediction/mpjpe_results_test.csv
Saved per-time MPJPE results to results/motion_prediction/per_time_mpjpe_results_test.csv
Saved per-joint MPJPE results to results/motion_prediction/per_joint_mpjpe_results_test.csv

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 93.79%
  Overall coverage within 2 std: 98.12%
  Overall coverage within 3 std: 99.27%
  Overall coverage within 4 std: 99.65%

Per-Time Coverage Stats:

  Overall coverage within 1 std:
    Frame 0: 97.35%
    Frame 1: 95.95%
    Frame 2: 94.79%
    Frame 3: 93.97%
    Frame 4: 93.41%
    Frame 5: 92.91%
    Frame 6: 92.67%
    Frame 7: 92.32%
    Frame 8: 92.28%
    Frame 9: 92.22%

  Overall coverage within 2 std:
    Frame 0: 99.08%
    Frame 1: 98.57%
    Frame 2: 98.20%
    Frame 3: 97.95%
    Frame 4: 97.86%
    Frame 5: 97.84%
    Frame 6: 97.86%
    Frame 7: 97.84%
    Frame 8: 97.94%
    Frame 9: 98.02%

  Overall coverage within 3 std:
    Frame 0: 99.54%
    Frame 1: 99.30%
    Frame 2: 99.16%
    Frame 3: 99.10%
    Frame 4: 99.12%
    Frame 5: 99.14%
    Frame 6: 99.23%
    Frame 7: 99.27%
    Frame 8: 99.36%
    Frame 9: 99.43%

  Overall coverage within 4 std:
    Frame 0: 99.73%
    Frame 1: 99.61%
    Frame 2: 99.53%
    Frame 3: 99.53%
    Frame 4: 99.58%
    Frame 5: 99.62%
    Frame 6: 99.66%
    Frame 7: 99.71%
    Frame 8: 99.76%
    Frame 9: 99.77%

Per-Joint Coverage Stats:

  Overall coverage within 1 std:
    Joint 0: 94.79%
    Joint 1: 93.29%
    Joint 2: 95.86%
    Joint 3: 92.41%
    Joint 4: 94.31%
    Joint 5: 90.79%
    Joint 6: 94.32%
    Joint 7: 95.57%
    Joint 8: 95.23%
    Joint 9: 92.93%
    Joint 10: 93.61%
    Joint 11: 91.87%
    Joint 12: 94.26%

  Overall coverage within 2 std:
    Joint 0: 98.16%
    Joint 1: 97.75%
    Joint 2: 98.56%
    Joint 3: 97.71%
    Joint 4: 98.16%
    Joint 5: 96.59%
    Joint 6: 98.18%
    Joint 7: 98.66%
    Joint 8: 98.76%
    Joint 9: 98.46%
    Joint 10: 98.73%
    Joint 11: 97.49%
    Joint 12: 98.30%

  Overall coverage within 3 std:
    Joint 0: 99.13%
    Joint 1: 99.08%
    Joint 2: 99.41%
    Joint 3: 99.09%
    Joint 4: 99.17%
    Joint 5: 98.44%
    Joint 6: 99.12%
    Joint 7: 99.45%
    Joint 8: 99.60%
    Joint 9: 99.65%
    Joint 10: 99.75%
    Joint 11: 99.15%
    Joint 12: 99.40%

  Overall coverage within 4 std:
    Joint 0: 99.54%
    Joint 1: 99.49%
    Joint 2: 99.75%
    Joint 3: 99.55%
    Joint 4: 99.53%
    Joint 5: 99.22%
    Joint 6: 99.50%
    Joint 7: 99.79%
    Joint 8: 99.81%
    Joint 9: 99.93%
    Joint 10: 99.94%
    Joint 11: 99.63%
    Joint 12: 99.73%
Saved overall coverage results to results/motion_prediction/coverage_results_test.csv
Saved per-time coverage results to results/motion_prediction/per_time_coverage_results_test.csv
Saved per-joint coverage results to results/motion_prediction/per_joint_coverage_results_test.csv
Predicted spherical reachable set coverage stats for 0.99 likelihood:
Overall coverage within set: 98.41%
Mean volume = 0.1804 m^3

Per-Time Coverage Stats:
    Frame 0: 99.43%
    Frame 1: 99.11%
    Frame 2: 98.89%
    Frame 3: 98.75%
    Frame 4: 98.65%
    Frame 5: 98.47%
    Frame 6: 98.32%
    Frame 7: 97.87%
    Frame 8: 97.49%
    Frame 9: 97.10%

Per-Time Volume [m^3]:
    Frame 0: 0.0689
    Frame 1: 0.0808
    Frame 2: 0.0985
    Frame 3: 0.1234
    Frame 4: 0.1547
    Frame 5: 0.1936
    Frame 6: 0.2397
    Frame 7: 0.2902
    Frame 8: 0.3491
    Frame 9: 0.4159

Per-Joint Coverage Stats:
    Joint 0: 98.36%
    Joint 1: 98.29%
    Joint 2: 98.64%
    Joint 3: 98.20%
    Joint 4: 98.34%
    Joint 5: 97.43%
    Joint 6: 98.27%
    Joint 7: 98.65%
    Joint 8: 98.82%
    Joint 9: 98.77%
    Joint 10: 98.92%
    Joint 11: 98.14%
    Joint 12: 98.50%

Per-Joint Volume [m^3]:
    Joint 0: 0.1112
    Joint 1: 0.1095
    Joint 2: 0.1091
    Joint 3: 0.2754
    Joint 4: 0.2596
    Joint 5: 0.8775
    Joint 6: 0.8313
    Joint 7: 0.0895
    Joint 8: 0.0851
    Joint 9: 0.0833
    Joint 10: 0.0730
    Joint 11: 0.1265
    Joint 12: 0.1094
================================
Evaluating motion SARA uncertainty.
================================
SARA simple velocity model coverage stats:
Overall coverage within set: 99.27%
Mean volume = 0.6714 m^3

Per-Time Coverage Stats:
    Frame 0: 99.21%
    Frame 1: 99.11%
    Frame 2: 99.14%
    Frame 3: 99.16%
    Frame 4: 99.23%
    Frame 5: 99.28%
    Frame 6: 99.33%
    Frame 7: 99.36%
    Frame 8: 99.42%
    Frame 9: 99.47%

Per-Time Volume [m^3]:
    Frame 0: 0.1169
    Frame 1: 0.1901
    Frame 2: 0.2887
    Frame 3: 0.4166
    Frame 4: 0.5777
    Frame 5: 0.7757
    Frame 6: 1.0145
    Frame 7: 1.2978
    Frame 8: 1.6295
    Frame 9: 2.0134

Per-Joint Coverage Stats:
    Joint 0: 99.74%
    Joint 1: 99.66%
    Joint 2: 99.79%
    Joint 3: 99.26%
    Joint 4: 99.28%
    Joint 5: 96.04%
    Joint 6: 97.92%
    Joint 7: 99.87%
    Joint 8: 99.85%
    Joint 9: 99.87%
    Joint 10: 99.88%
    Joint 11: 99.67%
    Joint 12: 99.67%

Per-Joint Volume [m^3]:
    Joint 0: 0.6714
    Joint 1: 0.6714
    Joint 2: 0.6714
    Joint 3: 0.6714
    Joint 4: 0.6714
    Joint 5: 0.6714
    Joint 6: 0.6714
    Joint 7: 0.6714
    Joint 8: 0.6714
    Joint 9: 0.6714
    Joint 10: 0.6714
    Joint 11: 0.6714
    Joint 12: 0.6714
```

### Tuned Uncertainty Calibration Values
Run with:
```
python human_pose_pipeline/motion_prediction/evaluate_covariance.py --results_file results/full_pipeline_rgbd_yolo/motion_prediction_results.cloudpickle --config rgbd_yolo --fps 30.0 
```
Results (tuned on all data):
```
Loading results from /home/thumm/code/sketching_lanczos/uncertainty_quantification/results/full_pipeline_rgbd_yolo/motion_prediction_results.cloudpickle...
Loaded predictions shape: (10232, 10, 13, 3)
Loaded targets shape: (10232, 10, 13, 3)
Loaded covariance matrices shape: (10232, 10, 13, 3, 3)
====================================
Coverage Stats Before Calibration
====================================

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 55.94%
  Overall coverage within 2 std: 75.58%
  Overall coverage within 3 std: 85.39%
  Overall coverage within 4 std: 90.67%

Per-Time Coverage Stats:

  Overall coverage within 1 std:
    Frame 0: 75.80%
    Frame 1: 69.76%
    Frame 2: 64.04%
    Frame 3: 59.03%
    Frame 4: 55.17%
    Frame 5: 51.81%
    Frame 6: 49.43%
    Frame 7: 46.60%
    Frame 8: 44.66%
    Frame 9: 42.98%

  Overall coverage within 2 std:
    Frame 0: 90.43%
    Frame 1: 86.16%
    Frame 2: 82.09%
    Frame 3: 78.44%
    Frame 4: 75.38%
    Frame 5: 72.77%
    Frame 6: 70.65%
    Frame 7: 68.38%
    Frame 8: 66.55%
    Frame 9: 64.90%

  Overall coverage within 3 std:
    Frame 0: 95.33%
    Frame 1: 92.62%
    Frame 2: 89.97%
    Frame 3: 87.54%
    Frame 4: 85.54%
    Frame 5: 83.56%
    Frame 6: 82.00%
    Frame 7: 80.39%
    Frame 8: 79.01%
    Frame 9: 77.93%

  Overall coverage within 4 std:
    Frame 0: 97.26%
    Frame 1: 95.46%
    Frame 2: 93.76%
    Frame 3: 92.21%
    Frame 4: 90.85%
    Frame 5: 89.47%
    Frame 6: 88.45%
    Frame 7: 87.31%
    Frame 8: 86.40%
    Frame 9: 85.52%

Per-Joint Coverage Stats:

  Overall coverage within 1 std:
    Joint 0: 51.24%
    Joint 1: 57.30%
    Joint 2: 65.22%
    Joint 3: 49.41%
    Joint 4: 52.92%
    Joint 5: 37.10%
    Joint 6: 40.90%
    Joint 7: 65.19%
    Joint 8: 64.41%
    Joint 9: 55.90%
    Joint 10: 55.44%
    Joint 11: 64.70%
    Joint 12: 67.45%

  Overall coverage within 2 std:
    Joint 0: 76.34%
    Joint 1: 76.91%
    Joint 2: 83.18%
    Joint 3: 70.38%
    Joint 4: 74.14%
    Joint 5: 57.33%
    Joint 6: 62.35%
    Joint 7: 83.76%
    Joint 8: 82.82%
    Joint 9: 76.62%
    Joint 10: 77.21%
    Joint 11: 79.08%
    Joint 12: 82.44%

  Overall coverage within 3 std:
    Joint 0: 87.51%
    Joint 1: 86.22%
    Joint 2: 90.80%
    Joint 3: 81.44%
    Joint 4: 84.93%
    Joint 5: 70.48%
    Joint 6: 75.46%
    Joint 7: 91.66%
    Joint 8: 91.04%
    Joint 9: 86.93%
    Joint 10: 87.76%
    Joint 11: 86.42%
    Joint 12: 89.48%

  Overall coverage within 4 std:
    Joint 0: 92.56%
    Joint 1: 91.08%
    Joint 2: 94.45%
    Joint 3: 87.89%
    Joint 4: 90.65%
    Joint 5: 78.82%
    Joint 6: 83.49%
    Joint 7: 95.27%
    Joint 8: 94.89%
    Joint 9: 92.50%
    Joint 10: 93.25%
    Joint 11: 90.66%
    Joint 12: 93.23%
====================================
Coverage Stats After Calibration
====================================

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 94.09%
  Overall coverage within 2 std: 98.23%
  Overall coverage within 3 std: 99.30%
  Overall coverage within 4 std: 99.66%

Per-Time Coverage Stats:

  Overall coverage within 1 std:
    Frame 0: 96.68%
    Frame 1: 95.40%
    Frame 2: 94.41%
    Frame 3: 93.84%
    Frame 4: 93.55%
    Frame 5: 93.32%
    Frame 6: 93.32%
    Frame 7: 93.26%
    Frame 8: 93.46%
    Frame 9: 93.68%

  Overall coverage within 2 std:
    Frame 0: 98.86%
    Frame 1: 98.35%
    Frame 2: 98.05%
    Frame 3: 97.91%
    Frame 4: 97.91%
    Frame 5: 97.98%
    Frame 6: 98.11%
    Frame 7: 98.21%
    Frame 8: 98.37%
    Frame 9: 98.52%

  Overall coverage within 3 std:
    Frame 0: 99.44%
    Frame 1: 99.18%
    Frame 2: 99.09%
    Frame 3: 99.07%
    Frame 4: 99.15%
    Frame 5: 99.22%
    Frame 6: 99.33%
    Frame 7: 99.40%
    Frame 8: 99.51%
    Frame 9: 99.60%

  Overall coverage within 4 std:
    Frame 0: 99.66%
    Frame 1: 99.54%
    Frame 2: 99.49%
    Frame 3: 99.51%
    Frame 4: 99.59%
    Frame 5: 99.65%
    Frame 6: 99.72%
    Frame 7: 99.76%
    Frame 8: 99.80%
    Frame 9: 99.83%

Per-Joint Coverage Stats:

  Overall coverage within 1 std:
    Joint 0: 94.92%
    Joint 1: 93.63%
    Joint 2: 96.12%
    Joint 3: 92.81%
    Joint 4: 94.70%
    Joint 5: 91.24%
    Joint 6: 94.66%
    Joint 7: 95.80%
    Joint 8: 95.45%
    Joint 9: 93.29%
    Joint 10: 93.88%
    Joint 11: 92.17%
    Joint 12: 94.53%

  Overall coverage within 2 std:
    Joint 0: 98.22%
    Joint 1: 97.95%
    Joint 2: 98.65%
    Joint 3: 97.83%
    Joint 4: 98.27%
    Joint 5: 96.70%
    Joint 6: 98.25%
    Joint 7: 98.73%
    Joint 8: 98.83%
    Joint 9: 98.56%
    Joint 10: 98.83%
    Joint 11: 97.70%
    Joint 12: 98.44%

  Overall coverage within 3 std:
    Joint 0: 99.15%
    Joint 1: 99.13%
    Joint 2: 99.43%
    Joint 3: 99.14%
    Joint 4: 99.19%
    Joint 5: 98.47%
    Joint 6: 99.12%
    Joint 7: 99.48%
    Joint 8: 99.62%
    Joint 9: 99.69%
    Joint 10: 99.76%
    Joint 11: 99.22%
    Joint 12: 99.46%

  Overall coverage within 4 std:
    Joint 0: 99.54%
    Joint 1: 99.53%
    Joint 2: 99.74%
    Joint 3: 99.56%
    Joint 4: 99.54%
    Joint 5: 99.21%
    Joint 6: 99.51%
    Joint 7: 99.80%
    Joint 8: 99.81%
    Joint 9: 99.93%
    Joint 10: 99.95%
    Joint 11: 99.67%
    Joint 12: 99.74%
Predicted spherical reachable set coverage stats for 0.99 likelihood:
Overall coverage within set: 99.08%
Mean volume = 0.1943 m^3

Per-Time Coverage Stats:
    Frame 0: 99.34%
    Frame 1: 99.02%
    Frame 2: 98.85%
    Frame 3: 98.84%
    Frame 4: 98.88%
    Frame 5: 98.95%
    Frame 6: 99.08%
    Frame 7: 99.16%
    Frame 8: 99.31%
    Frame 9: 99.41%

Per-Time Volume [m^3]:
    Frame 0: 0.0537
    Frame 1: 0.0685
    Frame 2: 0.0896
    Frame 3: 0.1196
    Frame 4: 0.1587
    Frame 5: 0.2093
    Frame 6: 0.2723
    Frame 7: 0.3485
    Frame 8: 0.4403
    Frame 9: 0.5488

Per-Joint Coverage Stats:
    Joint 0: 99.01%
    Joint 1: 98.93%
    Joint 2: 99.29%
    Joint 3: 98.85%
    Joint 4: 99.02%
    Joint 5: 98.07%
    Joint 6: 98.92%
    Joint 7: 99.31%
    Joint 8: 99.48%
    Joint 9: 99.44%
    Joint 10: 99.60%
    Joint 11: 98.91%
    Joint 12: 99.25%

Per-Joint Volume [m^3]:
    Joint 0: 0.1205
    Joint 1: 0.1178
    Joint 2: 0.1173
    Joint 3: 0.2967
    Joint 4: 0.2797
    Joint 5: 0.9459
    Joint 6: 0.8963
    Joint 7: 0.0960
    Joint 8: 0.0914
    Joint 9: 0.0895
    Joint 10: 0.0787
    Joint 11: 0.1363
    Joint 12: 0.1178
====================================
SARA Coverage Stats
====================================
SARA simple velocity model coverage stats:
Overall coverage within set: 99.09%
Mean volume = 0.5025 m^3

Per-Time Coverage Stats:
    Frame 0: 98.96%
    Frame 1: 98.86%
    Frame 2: 98.90%
    Frame 3: 98.94%
    Frame 4: 99.01%
    Frame 5: 99.08%
    Frame 6: 99.19%
    Frame 7: 99.25%
    Frame 8: 99.31%
    Frame 9: 99.37%

Per-Time Volume [m^3]:
    Frame 0: 0.0681
    Frame 1: 0.1208
    Frame 2: 0.1954
    Frame 3: 0.2958
    Frame 4: 0.4257
    Frame 5: 0.5890
    Frame 6: 0.7894
    Frame 7: 1.0309
    Frame 8: 1.3171
    Frame 9: 1.6519

Per-Joint Coverage Stats:
    Joint 0: 99.67%
    Joint 1: 99.57%
    Joint 2: 99.71%
    Joint 3: 98.98%
    Joint 4: 99.10%
    Joint 5: 95.16%
    Joint 6: 97.47%
    Joint 7: 99.84%
    Joint 8: 99.81%
    Joint 9: 99.86%
    Joint 10: 99.86%
    Joint 11: 99.54%
    Joint 12: 99.57%

Per-Joint Volume [m^3]:
    Joint 0: 0.5025
    Joint 1: 0.5025
    Joint 2: 0.5025
    Joint 3: 0.5025
    Joint 4: 0.5025
    Joint 5: 0.5025
    Joint 6: 0.5025
    Joint 7: 0.5025
    Joint 8: 0.5025
    Joint 9: 0.5025
    Joint 10: 0.5025
    Joint 11: 0.5025
    Joint 12: 0.5025
```

### With OOD Detection Enabled
OOD Threshold = 600.000

```
  Overall coverage within 2 std:
    Frame 0: 98.88%
    Frame 1: 98.40%
    Frame 2: 98.09%
    Frame 3: 97.96%
    Frame 4: 97.96%
    Frame 5: 98.03%
    Frame 6: 98.16%
    Frame 7: 98.25%
    Frame 8: 98.40%
    Frame 9: 98.56%

  Overall coverage within 3 std:
    Frame 0: 99.45%
    Frame 1: 99.22%
    Frame 2: 99.12%
    Frame 3: 99.09%
    Frame 4: 99.17%
    Frame 5: 99.24%
    Frame 6: 99.35%
    Frame 7: 99.42%
    Frame 8: 99.53%
    Frame 9: 99.61%

  Overall coverage within 4 std:
    Frame 0: 99.66%
    Frame 1: 99.56%
    Frame 2: 99.51%
    Frame 3: 99.53%
    Frame 4: 99.61%
    Frame 5: 99.67%
    Frame 6: 99.72%
    Frame 7: 99.78%
    Frame 8: 99.81%
    Frame 9: 99.84%

Per-Joint Coverage Stats:

  Overall coverage within 1 std:
    Joint 0: 95.06%
    Joint 1: 93.74%
    Joint 2: 96.30%
    Joint 3: 92.86%
    Joint 4: 94.79%
    Joint 5: 91.25%
    Joint 6: 94.72%
    Joint 7: 95.85%
    Joint 8: 95.52%
    Joint 9: 93.29%
    Joint 10: 93.93%
    Joint 11: 92.16%
    Joint 12: 94.49%

  Overall coverage within 2 std:
    Joint 0: 98.25%
    Joint 1: 98.03%
    Joint 2: 98.75%
    Joint 3: 97.92%
    Joint 4: 98.33%
    Joint 5: 96.73%
    Joint 6: 98.30%
    Joint 7: 98.76%
    Joint 8: 98.89%
    Joint 9: 98.59%
    Joint 10: 98.86%
    Joint 11: 97.67%
    Joint 12: 98.42%

  Overall coverage within 3 std:
    Joint 0: 99.17%
    Joint 1: 99.18%
    Joint 2: 99.47%
    Joint 3: 99.19%
    Joint 4: 99.24%
    Joint 5: 98.50%
    Joint 6: 99.15%
    Joint 7: 99.50%
    Joint 8: 99.64%
    Joint 9: 99.69%
    Joint 10: 99.76%
    Joint 11: 99.20%
    Joint 12: 99.45%

  Overall coverage within 4 std:
    Joint 0: 99.56%
    Joint 1: 99.53%
    Joint 2: 99.75%
    Joint 3: 99.60%
    Joint 4: 99.57%
    Joint 5: 99.24%
    Joint 6: 99.53%
    Joint 7: 99.82%
    Joint 8: 99.82%
    Joint 9: 99.93%
    Joint 10: 99.95%
    Joint 11: 99.66%
    Joint 12: 99.73%
Saved overall coverage results to results/motion_prediction/coverage_results_test.csv
Saved per-time coverage results to results/motion_prediction/per_time_coverage_results_test.csv
Saved per-joint coverage results to results/motion_prediction/per_joint_coverage_results_test.csv
Predicted spherical reachable set coverage stats for 0.99 likelihood:
Overall coverage within set: 86.33%
Mean volume = 0.1182 m^3

Per-Time Coverage Stats:
    Frame 0: 92.18%
    Frame 1: 91.31%
    Frame 2: 90.50%
    Frame 3: 89.62%
    Frame 4: 88.63%
    Frame 5: 87.50%
    Frame 6: 85.93%
    Frame 7: 83.72%
    Frame 8: 80.60%
    Frame 9: 73.27%

Per-Time Volume [m^3]:
    Frame 0: 0.0468
    Frame 1: 0.0574
    Frame 2: 0.0719
    Frame 3: 0.0908
    Frame 4: 0.1134
    Frame 5: 0.1401
    Frame 6: 0.1674
    Frame 7: 0.1921
    Frame 8: 0.2096
    Frame 9: 0.1869

Per-Joint Coverage Stats:
    Joint 0: 86.27%
    Joint 1: 86.34%
    Joint 2: 86.48%
    Joint 3: 86.21%
    Joint 4: 86.41%
    Joint 5: 85.63%
    Joint 6: 86.27%
    Joint 7: 86.49%
    Joint 8: 86.60%
    Joint 9: 86.54%
    Joint 10: 86.67%
    Joint 11: 86.03%
    Joint 12: 86.34%

Per-Joint Volume [m^3]:
    Joint 0: 0.0729
    Joint 1: 0.0706
    Joint 2: 0.0704
    Joint 3: 0.1815
    Joint 4: 0.1716
    Joint 5: 0.5892
    Joint 6: 0.5555
    Joint 7: 0.0570
    Joint 8: 0.0543
    Joint 9: 0.0543
    Joint 10: 0.0475
    Joint 11: 0.0825
    Joint 12: 0.0713
```

### With OOD Detection Enabled and Reworked 2D to 3D function
Run with:
```
python human_pose_pipeline/examples/eval_full_pipeline_rgbd_yolo.py --motion_model_save_path human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle --enable_ood --motion_score_fn_path human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM720of800_sketch_srft_seed0_size10000.cloudpickle --enable_tracking --depth_uncertainty 0.05 --start_at 1656
```
Results:
```
Loading RGB-D dataset...
Loaded 12404 paired RGBD samples from datasets/rgbd_test/
Dataset loaded with 12404 frames, from which we are using 10748 frames             with 1 subsampling.
Evaluating sequence:: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 10748/10748 [09:05<00:00, 19.69it/s]
Full pipeline completed!
Saved motion prediction results to results/eval_full_pipeline_rgbd_yolo/motion_prediction_results.cloudpickle
Saved full pipeline results to results/eval_full_pipeline_rgbd_yolo/full_pipeline_results.cloudpickle
================================
Evaluating motion prediction.
================================
================================
Evaluating motion uncertainty prediction.
================================

Overall MPJPE: 75.15 mm

Per-Time Errors:
  Time point 1 error =   38.04 mm
  Time point 2 error =   44.77 mm
  Time point 3 error =   53.43 mm
  Time point 4 error =   62.51 mm
  Time point 5 error =   71.06 mm
  Time point 6 error =   79.97 mm
  Time point 7 error =   88.06 mm
  Time point 8 error =   96.72 mm
  Time point 9 error =  104.93 mm
  Time point 10 error =  113.12 mm

Per-Joint Errors:
  Joint 1 error =   61.17 mm
  Joint 2 error =   63.93 mm
  Joint 3 error =   53.12 mm
  Joint 4 error =   91.69 mm
  Joint 5 error =   82.85 mm
  Joint 6 error =  140.06 mm
  Joint 7 error =  127.14 mm
  Joint 8 error =   53.80 mm
  Joint 9 error =   54.64 mm
  Joint 10 error =   62.51 mm
  Joint 11 error =   59.71 mm
  Joint 12 error =   66.56 mm
  Joint 13 error =   59.70 mm
Saved overall MPJPE results to results/motion_prediction/mpjpe_results_test.csv
Saved per-time MPJPE results to results/motion_prediction/per_time_mpjpe_results_test.csv
Saved per-joint MPJPE results to results/motion_prediction/per_joint_mpjpe_results_test.csv

Uncertainty Coverage Stats:
  Overall coverage within 1 std: 94.96%
  Overall coverage within 2 std: 98.53%
  Overall coverage within 3 std: 99.42%
  Overall coverage within 4 std: 99.71%

Per-Time Coverage Stats:

  Overall coverage within 1 std:
    Frame 0: 96.97%
    Frame 1: 95.98%
    Frame 2: 95.21%
    Frame 3: 94.71%
    Frame 4: 94.46%
    Frame 5: 94.26%
    Frame 6: 94.29%
    Frame 7: 94.38%
    Frame 8: 94.56%
    Frame 9: 94.76%

  Overall coverage within 2 std:
    Frame 0: 99.04%
    Frame 1: 98.63%
    Frame 2: 98.32%
    Frame 3: 98.26%
    Frame 4: 98.24%
    Frame 5: 98.25%
    Frame 6: 98.42%
    Frame 7: 98.55%
    Frame 8: 98.74%
    Frame 9: 98.85%

  Overall coverage within 3 std:
    Frame 0: 99.51%
    Frame 1: 99.33%
    Frame 2: 99.20%
    Frame 3: 99.21%
    Frame 4: 99.27%
    Frame 5: 99.37%
    Frame 6: 99.45%
    Frame 7: 99.52%
    Frame 8: 99.62%
    Frame 9: 99.69%

  Overall coverage within 4 std:
    Frame 0: 99.69%
    Frame 1: 99.60%
    Frame 2: 99.57%
    Frame 3: 99.59%
    Frame 4: 99.65%
    Frame 5: 99.71%
    Frame 6: 99.76%
    Frame 7: 99.81%
    Frame 8: 99.86%
    Frame 9: 99.89%

Per-Joint Coverage Stats:

  Overall coverage within 1 std:
    Joint 0: 95.86%
    Joint 1: 94.72%
    Joint 2: 97.08%
    Joint 3: 94.16%
    Joint 4: 95.62%
    Joint 5: 93.26%
    Joint 6: 95.38%
    Joint 7: 96.65%
    Joint 8: 96.25%
    Joint 9: 94.07%
    Joint 10: 94.31%
    Joint 11: 92.39%
    Joint 12: 94.74%

  Overall coverage within 2 std:
    Joint 0: 98.85%
    Joint 1: 98.23%
    Joint 2: 99.02%
    Joint 3: 98.27%
    Joint 4: 98.69%
    Joint 5: 97.57%
    Joint 6: 98.32%
    Joint 7: 98.99%
    Joint 8: 99.16%
    Joint 9: 98.82%
    Joint 10: 98.76%
    Joint 11: 97.81%
    Joint 12: 98.41%

  Overall coverage within 3 std:
    Joint 0: 99.57%
    Joint 1: 99.20%
    Joint 2: 99.59%
    Joint 3: 99.28%
    Joint 4: 99.46%
    Joint 5: 98.87%
    Joint 6: 99.16%
    Joint 7: 99.64%
    Joint 8: 99.66%
    Joint 9: 99.73%
    Joint 10: 99.69%
    Joint 11: 99.29%
    Joint 12: 99.30%

  Overall coverage within 4 std:
    Joint 0: 99.77%
    Joint 1: 99.54%
    Joint 2: 99.82%
    Joint 3: 99.64%
    Joint 4: 99.71%
    Joint 5: 99.43%
    Joint 6: 99.54%
    Joint 7: 99.87%
    Joint 8: 99.81%
    Joint 9: 99.92%
    Joint 10: 99.89%
    Joint 11: 99.68%
    Joint 12: 99.62%
Saved overall coverage results to results/motion_prediction/coverage_results_test.csv
Saved per-time coverage results to results/motion_prediction/per_time_coverage_results_test.csv
Saved per-joint coverage results to results/motion_prediction/per_joint_coverage_results_test.csv
Predicted spherical reachable set coverage stats for 0.99 likelihood:
Overall coverage within set: 99.25%
Mean volume = 0.1651 m^3

Per-Time Coverage Stats:
    Frame 0: 99.43%
    Frame 1: 99.20%
    Frame 2: 99.02%
    Frame 3: 99.01%
    Frame 4: 99.08%
    Frame 5: 99.13%
    Frame 6: 99.27%
    Frame 7: 99.35%
    Frame 8: 99.45%
    Frame 9: 99.54%

Per-Time Volume [m^3]:
    Frame 0: 0.0330
    Frame 1: 0.0466
    Frame 2: 0.0660
    Frame 3: 0.0940
    Frame 4: 0.1317
    Frame 5: 0.1818
    Frame 6: 0.2456
    Frame 7: 0.3251
    Frame 8: 0.4226
    Frame 9: 0.5405

Per-Joint Coverage Stats:
    Joint 0: 99.53%
    Joint 1: 99.00%
    Joint 2: 99.49%
    Joint 3: 99.06%
    Joint 4: 99.35%
    Joint 5: 98.57%
    Joint 6: 98.99%
    Joint 7: 99.48%
    Joint 8: 99.58%
    Joint 9: 99.53%
    Joint 10: 99.53%
    Joint 11: 99.03%
    Joint 12: 99.10%

Per-Joint Volume [m^3]:
    Joint 0: 0.0994
    Joint 1: 0.0970
    Joint 2: 0.0967
    Joint 3: 0.2536
    Joint 4: 0.2407
    Joint 5: 0.8245
    Joint 6: 0.7821
    Joint 7: 0.0790
    Joint 8: 0.0751
    Joint 9: 0.0764
    Joint 10: 0.0669
    Joint 11: 0.1171
    Joint 12: 0.1010
================================
Evaluating motion SARA uncertainty.
================================
SARA simple velocity model coverage stats:
Overall coverage within set: 99.39%
Mean volume = 0.5008 m^3

Per-Time Coverage Stats:
    Frame 0: 99.39%
    Frame 1: 99.35%
    Frame 2: 99.33%
    Frame 3: 99.33%
    Frame 4: 99.32%
    Frame 5: 99.34%
    Frame 6: 99.40%
    Frame 7: 99.43%
    Frame 8: 99.49%
    Frame 9: 99.51%

Per-Time Volume [m^3]:
    Frame 0: 0.0681
    Frame 1: 0.1208
    Frame 2: 0.1954
    Frame 3: 0.2958
    Frame 4: 0.4257
    Frame 5: 0.5890
    Frame 6: 0.7894
    Frame 7: 1.0309
    Frame 8: 1.3171
    Frame 9: 1.6519

Per-Joint Coverage Stats:
    Joint 0: 99.85%
    Joint 1: 99.63%
    Joint 2: 99.81%
    Joint 3: 99.35%
    Joint 4: 99.44%
    Joint 5: 96.98%
    Joint 6: 98.21%
    Joint 7: 99.89%
    Joint 8: 99.86%
    Joint 9: 99.87%
    Joint 10: 99.86%
    Joint 11: 99.66%
    Joint 12: 99.63%

Per-Joint Volume [m^3]:
    Joint 0: 0.5008
    Joint 1: 0.5008
    Joint 2: 0.5008
    Joint 3: 0.5008
    Joint 4: 0.5008
    Joint 5: 0.5008
    Joint 6: 0.5008
    Joint 7: 0.5008
    Joint 8: 0.5008
    Joint 9: 0.5008
    Joint 10: 0.5008
    Joint 11: 0.5008
    Joint 12: 0.5008
```

## Conformal Prediction Sets

### RGB-D YOLO Results: Direct Sphere Method
Uses:
```
def convert_covariance_matrices_to_set(
    covariance_matrices: Union[np.ndarray, jnp.ndarray],
    likelihood: float
) -> np.ndarray:
    """Convert the covariance matrices to a spherical conformal prediction set X of likelihood confidence level.
    P(x \in X) >= likelihood.

    Args:
        covariance_matrices: Cov. matrices. Shape: [N, T, J, 3, 3]
        likelihood: Likelihood of points being in the set.
    Returns:
        Radius of the spherical reachable sets. Shape: [N, T, J]
    """
    from scipy.stats import chi2
    # largest eigenvalue
    # chi-square threshold for number of standard deviations in 3D
    chi_squared_val = chi2.ppf(likelihood, df=3)
    if isinstance(covariance_matrices, np.ndarray):
        lambda_max = np.max(np.linalg.eigvalsh(covariance_matrices), axis=-1)
        # sphere radius
        radius = np.sqrt(lambda_max * chi_squared_val)
    else:
        lambda_max = jnp.max(jnp.linalg.eigvalsh(covariance_matrices), axis=-1)
        # sphere radius
        radius = np.sqrt(lambda_max * chi_squared_val)

    return radius
```


Predicted spherical reachable set coverage stats for 0.99 likelihood:
Overall coverage within set: 99.25%
Mean volume = 0.1651 m^3

Per-Time Coverage Stats:
    Frame 0: 99.43%
    Frame 1: 99.20%
    Frame 2: 99.02%
    Frame 3: 99.01%
    Frame 4: 99.08%
    Frame 5: 99.13%
    Frame 6: 99.27%
    Frame 7: 99.35%
    Frame 8: 99.45%
    Frame 9: 99.54%

Per-Time Volume [m^3]:
    Frame 0: 0.0330
    Frame 1: 0.0466
    Frame 2: 0.0660
    Frame 3: 0.0940
    Frame 4: 0.1317
    Frame 5: 0.1818
    Frame 6: 0.2456
    Frame 7: 0.3251
    Frame 8: 0.4226
    Frame 9: 0.5405

Per-Joint Coverage Stats:
    Joint 0: 99.53%
    Joint 1: 99.00%
    Joint 2: 99.49%
    Joint 3: 99.06%
    Joint 4: 99.35%
    Joint 5: 98.57%
    Joint 6: 98.99%
    Joint 7: 99.48%
    Joint 8: 99.58%
    Joint 9: 99.53%
    Joint 10: 99.53%
    Joint 11: 99.03%
    Joint 12: 99.10%

Per-Joint Volume [m^3]:
    Joint 0: 0.0994
    Joint 1: 0.0970
    Joint 2: 0.0967
    Joint 3: 0.2536
    Joint 4: 0.2407
    Joint 5: 0.8245
    Joint 6: 0.7821
    Joint 7: 0.0790
    Joint 8: 0.0751
    Joint 9: 0.0764
    Joint 10: 0.0669
    Joint 11: 0.1171
    Joint 12: 0.1010
====================================
SARA Coverage Stats
====================================
SARA simple velocity model coverage stats:
Overall coverage within set: 99.39%
Mean volume = 0.5008 m^3

Per-Time Coverage Stats:
    Frame 0: 99.39%
    Frame 1: 99.35%
    Frame 2: 99.33%
    Frame 3: 99.33%
    Frame 4: 99.32%
    Frame 5: 99.34%
    Frame 6: 99.40%
    Frame 7: 99.43%
    Frame 8: 99.49%
    Frame 9: 99.51%

Per-Time Volume [m^3]:
    Frame 0: 0.0681
    Frame 1: 0.1208
    Frame 2: 0.1954
    Frame 3: 0.2958
    Frame 4: 0.4257
    Frame 5: 0.5890
    Frame 6: 0.7894
    Frame 7: 1.0309
    Frame 8: 1.3171
    Frame 9: 1.6519

Per-Joint Coverage Stats:
    Joint 0: 99.85%
    Joint 1: 99.63%
    Joint 2: 99.81%
    Joint 3: 99.35%
    Joint 4: 99.44%
    Joint 5: 96.98%
    Joint 6: 98.21%
    Joint 7: 99.89%
    Joint 8: 99.86%
    Joint 9: 99.87%
    Joint 10: 99.86%
    Joint 11: 99.66%
    Joint 12: 99.63%

Per-Joint Volume [m^3]:
    Joint 0: 0.5008
    Joint 1: 0.5008
    Joint 2: 0.5008
    Joint 3: 0.5008
    Joint 4: 0.5008
    Joint 5: 0.5008
    Joint 6: 0.5008
    Joint 7: 0.5008
    Joint 8: 0.5008
    Joint 9: 0.5008
    Joint 10: 0.5008
    Joint 11: 0.5008
    Joint 12: 0.5008

### RGB-D YOLO Results Ellipsoid to Sphere Approach