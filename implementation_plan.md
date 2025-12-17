# Plan to implement human pose estimation and motion prediction pipeline

The goal is to implement the human pose estimation and motion prediction pipeline into this JAX-based repository.
We want to use the sketching lanczos method to detect OOD cases.
The final pipeline should follow these steps:
 1. take in the images from two cameras
 2. detect all humans in the images
 3. create bounding boxes around the humans to get a cropped image
 4. perform a pose estimation on the cropped image with uncertainty information
 5. perform an OOD detection on the pose estimation
 6. use triangulation to get a 3D pose with uncertainty
 7. perform a motion prediction to estimate the human pose at t+50ms, +100ms, +150ms, and +200ms.
 8. perform an OOD detection on the motion prediction.
 9. return the predicted pose of the closest human and OOD labels

## What is already (partly) in place
We have two things:
 1. in marian_code/, there is code that performs steps 1., 3., 4., 6., 7., and 9. This code only works for a single human. The code is also written in pytorch and we would like to move to JAX to use the sketching lanczos approach.
 2. in this folder is the sketching lanczos OOD detection for step 5. and 7. We already converted the models to use the trained weights of Marian. See tianle_readme.md for a very short documentation. The sketching lanczos code should work on most models directly, but we need to test it.

## Implementation plan

 1. Download 
    - [x] Marians models from Tianles folder (now in models_tianle/H36M/RegressFlow)
    - [x] Move Marians code to the server
    - [x] Download the human 3.6M dataset with fetching script -> datasets/H36M
    - Download the close interactions dataset https://ci3d.imar.ro/chi3d -> Access granted.
    - Download the COCO dataset
    - [x] Download the Tiger pose dataset -> datasets/tiger-pose
 2. Test the pre-trained pose estimation models
    - [x] Write a folder structure for this project.
    - [x] Move existing file to the correct folders if neccessary.
    - [x] Check if `unc` python environment needs adaption for new models (packages in src/ViTPose/readMe.md might need to be installed.)
    - [x] Use the pre-trained models with converted weights into JAX.
    - [x] First, perform pose estimation on a few H3.6M examples.
    - [x] Then, perform "Experiment 2" from Marians folder -> Evaluation of the 3D human pose estimation with uncertainty quantification (steps 4 and 6 from above.)
      - [x] Test with real H36M examples - Load actual images and poses from the dataset
      - [x] Implement evaluation metrics - MPJPE, PCK metrics as in Marian's Experiment 2
      - [x] Add torch to the environment to get the bounding box estimation and switch to YOLO 11 -> We might want to change this later.
      - [x] Mirror Marian's code exactly
      - [x] Add uncertainty estimation
      - [x] Debug estimation and uncertainty
      - [x] Move utils, dataset functionality to correct folder
      - [x] Implement 3D triangulation - Convert 2D poses to 3D using camera parameters
    - [x] Create a dataset of preprocessed frames
    - [x] Test experiment 2 on preprocessed data 2D
    - [ ] Test experiment 2 on preprocessed data 3D
 3. Perform sketching lanczos OOD detection for pose estimation
    - Run the OOD detection with sketching lanczos on the 2D pose estimation model. ID data would be the human 3.6m dataset. For OOD data we can use tiger pose dataset.
      - [x] Write a tiger-pose dataset class
      - [x] Write an example script that predicts the poses for the ID data (h36m) and OOD data (tiger-pose) (similar to pose_estimation_2D.py) without uncertainty and compares the ID vs. OOD performance.
      - [x] Add tiger image transformation to the script
      - [x] Write a tiger dataset preprocessing
      - [x] Test the OOD detection with the score_model.py script and save the GNN matrix
      - [x] Write a script that works with the preprocessed data instead of the full images.
      - [x] Test score model function on regressflow model with low_memory_lanczos_score_fun
      - [x] Test with --OOD_dataset tiger-pose
      - [x] Add OOD classification to the evaluation script 
      - [x] plot a histogram over OOD scores with different colors for the two classes (ID vs OOD) -> Should already be implemented somewhere in this repo.
      - [x] plot a scatter plot with pose prediction accuracy over OOD score.
      - [x] perform an evaluation of how many datapoints are within 1, 2, 3, and 4 sigma for datapoints that were classified as ID vs OOD.
      - [x] Speed up pipeline: 
        - [x] Figure out which parts take the longest
        - [x] Maybe pre processing on GPU
        - [x] Definetly parallilze pose estimation and OOD scoring.
        - [x] Properly pre-compile the pose estimation -> speed up from 100ms to 4ms
        - [x] Reduce network output for OOD detection from 17x2 to 3x2 (hand, left hand, right hand) -> speed up from 160ms to 29ms (23ms expected)
        - [x] Write score fn that only scores certain layers to reduce the model size -> Prediction accuracy significantly lower.
        - [x] Pytorch the image processing pipeline -> from 9ms for image processing to 4ms.
        - [x] Use ResNet18 model from Marian
          - [x] Transfer weights
          - [x] Compute Lanczos
          - [x] Test model -> OOD takes ~12ms. Total frame time with OOD: 25ms, without OOD: 13ms
 4. Test the pre-trained pose prediction models
    - Use the pre-trained models with converted weights into JAX.
    - First, perform pose prediction on a few H3.6M examples.
    - Then, perform "Experiment 3" from Marians folder -> Evaluation of the 3D human pose prediction with uncertainty quantification (step 7 from above.)
 5. Perform OOD detection for motion prediction
    - Run the OOD detection with sketching lanczos on the 3D pose prediction model. ID data would be the human 3.6m dataset. OOD is the close interactions dataset or shuffled H36M dataset.
    - [x] Single motion prediction from Marians model
    - [x] Create motion prediction dataset
    - [x] Create small motion prediction model for OOD detection
    - [x] Create OOD dataset(s)
    - [x] Train and evaluate OOD scores
    - [x] Write eval script ID/OOD
    - [ ] Train motion prediction with uncertainty model
      - [x] Train DCT transformer model without uncertainty input
      - [x] Change dataset to Marians train/eval/test split and retrain.
      - [x] Retrain DCT transformer model without uncertainty input with Optuna.
      - [x] Create dataset with predicted uncertainty of pose estimation model
      - [x] Debugging: Test 3D pose estimation statistics on full pipeline
      - [x] Train DCT transformer model with uncertainty
      - [x] Create evaluation
 6. Debug prediction accuracies Pytorch vs. Jax
    - [x] Investigate prediction accuracy 2D Pose estimation in pytorch vs. Jax.
        =====> The yolov5s network seems to be much better at human detection.
                Maybe use that model instead of yolov11n.
        Results Marian Pytorch on 3 validation files (yolo threshold = 0.8) (Model: estimation_model_finetuned_on_h36m.pth):
            Total frames processed: 4881
            Total joints evaluated: 63453
            Average MPJPE: 7.64 pixels
            Average percentage of keypoints within 1 std: 73.14%
            Average percentage of keypoints within 2 std: 91.45%
            Average percentage of keypoints within 3 std: 97.50%
            Average percentage of keypoints within 4 std: 99.17%
        Results Jax on 3 validation files (yolo threshold = 0.3) (Model: jax_resnet50_regressflow)
            Total frames processed: 4988
            Total joints evaluated: 64844
            Average MPJPE: 7.83
            Average percentage of keypoints within 1 std: 73.08%
            Average percentage of keypoints within 2 std: 92.45%
            Average percentage of keypoints within 3 std: 97.51%
            Average percentage of keypoints within 4 std: 99.14%
        ==> The models estimation_model_finetuned_on_h36m.pth and jax_resnet50_regressflow seem to match.
        Results Jax on 3 validation files (yolo threshold = 0.3) (Model: finetuned_h36m_regressflow_with_unc)
            Total frames processed: 4783
            Total joints evaluated: 62179
            Average MPJPE: 7.70
            Average percentage of keypoints within 1 std: 70.91%
            Average percentage of keypoints within 2 std: 90.88%
            Average percentage of keypoints within 3 std: 96.87%
            Average percentage of keypoints within 4 std: 98.85%
        ==> Here, the accuracy of the jax_resnet50_regressflow and finetuned_h36m_regressflow_with_unc roughly match.
    - [x] Investigate the uncertainty coverage for the 2D Pose estimation in pytorch vs. Jax.
    - [x] Investigate prediction accuracy 3D Pose estimation in pytorch vs. Jax.
        Results Marian Pytorch on 10 validation files with 1000 max_frames (yolo threshold = 0.8) (Model: estimation_model_finetuned_on_h36m.pth):
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
        Results Jax on 3 validation files (yolo threshold = 0.3) (Model: finetuned_h36m_regressflow_with_unc)
        Actions: ['Discussion 1', 'Sitting 1', 'SittingDown 1', 'Posing 1', 'Eating', 'SittingDown', 'Smoking 2', 'Directions', 'Purchases 1', 'Waiting']
            Total frames processed: 9959
            Total joints evaluated: 129467
            Average MPJPE: 31.79 mm
            Average percentage of keypoints within 1 std: 45.30%
            Average percentage of keypoints within 2 std: 73.19%
            Average percentage of keypoints within 3 std: 88.06%
            Average percentage of keypoints within 4 std: 94.34%
        --> Findings: finetuned_h36m_regressflow_with_unc seems to be much better than estimation_model_finetuned_on_h36m! The actions are the same.
        --> This would require further investigation but our model is better, so I guess it is okay.
    - [x] Investigate the uncertainty coverage for the 3D Pose estimation in pytorch vs. Jax.
    - [x] Investigate prediction accuracy 3D motion prediction in pytorch vs. Jax.
        Pytorch all validation data, model = model_13_joints_with_uncert
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
        Model = model_13_joints_calibrated_uncert.pth
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
        
        Jax all validation data, trained model after stage 3:
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
        ==> Model did not learn correct covariance matrices.
    - [x] Investigate the uncertainty coverage for the 3D motion prediction in pytorch vs. Jax.
        ================================================================================
        JAX Model - VALIDATION RESULTS (human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle)
        ================================================================================

        Overall MPJPE: 23.38 mm, Std: 32.45 mm

        Per-Time Errors:
          Time point 1 error =    7.84 mm
          Time point 2 error =    7.86 mm
          Time point 3 error =   10.21 mm
          Time point 4 error =   14.34 mm
          Time point 5 error =   18.91 mm
          Time point 6 error =   23.91 mm
          Time point 7 error =   29.15 mm
          Time point 8 error =   34.68 mm
          Time point 9 error =   40.40 mm
          Time point 10 error =   46.48 mm

        Per-Joint Errors:
          Joint 1 error =   20.64 mm
          Joint 2 error =   18.61 mm
          Joint 3 error =   18.74 mm
          Joint 4 error =   29.19 mm
          Joint 5 error =   28.71 mm
          Joint 6 error =   39.55 mm
          Joint 7 error =   38.69 mm
          Joint 8 error =   15.56 mm
          Joint 9 error =   15.17 mm
          Joint 10 error =   18.40 mm
          Joint 11 error =   18.85 mm
          Joint 12 error =   20.29 mm
          Joint 13 error =   21.49 mm

        Uncertainty Coverage Stats:
          Overall coverage within 1 std: 69.96%
          Overall coverage within 2 std: 86.11%
          Overall coverage within 3 std: 92.62%
          Overall coverage within 4 std: 95.68%

        Running PyTorch model inference...
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 851/851 [00:01<00:00, 585.14it/s]

        ================================================================================
        PyTorch Model - VALIDATION RESULTS (marian_code/Experiment4/model_checkpoint_prediction_transformer_weights_from_end_to_end.pth)
        ================================================================================

        Overall MPJPE: 43.46 mm, Std: 48.34 mm

        Per-Time Errors:
          Time point 1 error =   23.69 mm
          Time point 2 error =   26.72 mm
          Time point 3 error =   28.96 mm
          Time point 4 error =   34.39 mm
          Time point 5 error =   39.21 mm
          Time point 6 error =   44.90 mm
          Time point 7 error =   50.34 mm
          Time point 8 error =   56.02 mm
          Time point 9 error =   61.96 mm
          Time point 10 error =   68.38 mm

        Per-Joint Errors:
          Joint 1 error =   44.19 mm
          Joint 2 error =   34.44 mm
          Joint 3 error =   34.71 mm
          Joint 4 error =   52.13 mm
          Joint 5 error =   49.53 mm
          Joint 6 error =   70.22 mm
          Joint 7 error =   64.09 mm
          Joint 8 error =   30.51 mm
          Joint 9 error =   31.78 mm
          Joint 10 error =   32.79 mm
          Joint 11 error =   36.32 mm
          Joint 12 error =   40.26 mm
          Joint 13 error =   43.98 mm

        Uncertainty Coverage Stats:
          Overall coverage within 1 std: 9.58%
          Overall coverage within 2 std: 16.08%
          Overall coverage within 3 std: 21.84%
          Overall coverage within 4 std: 26.88%
 7. Full single-human pipeline (finish set up)
    - First, find out how the current bounding box algorithm works in Marians code. E.g., Experiment 4. There, he did real-world tests, so it should include some bounding box algorithm.
    - Implement the full pipeline based on the code of Experiment 4 plus the new OOD detection. Everything in JAX.
    - [x] Train okay performing network in Jax (run id r24f9uig)
    - [x] Transfer weights from good working motion prediction model (model_13_joints_with_uncert)
    - [x] Train motion prediction network in Jax starting from pytorch weights (run id 17bg6nyk) -> good performance
    - [x] Use that network for OOD detection
    - [x] Hopefully: Marian finally provides correct pytorch network
    - [ ] Adapt human_pose_pipeline/examples/pose_estimation_3D_full_eval.py to perform the evaluation in a fully batched fashion.
      - [ ] 2D pose estimation, MPJPE, coverage
      - [ ] 2D OOD detection evaluation
      - [x] 3D pose estimation, MPJPE, coverage
    - [ ] Adapt human_pose_pipeline/examples/pose_estimation_3D_full_eval.py to create full 3D pipeline.
      - [x] Write base pipeline
      - [ ] Make everything Jax after YOLO
      - [ ] JIT compile parts of the pipeline?
      - [ ] Handle OOD cases
    - [ ] Adapt pipeline for RGBD camera
    - [ ] Write batched evaluation script that evaluates motion prediction (based on Motion Prediction Evaluation launch.json)
      - [ ] From ground truth measurements, MPJPE, coverage, per-action data
      - [ ] From estimated pose dataset, Write batched evaluation script that evaluates
      - [ ] OOD detection evaluation
    - [ ] Write bash script that runs all evaluations sequentially
    - [ ] Investigate calibration for covariances that might lead to better coverage
      - [ ] In Marian's thesis, adding a constant value of 10mm to the 3D pose estimation led to better coverage. Try this.
      - [ ] Add a constant value to the 3D motion prediction uncertainty as well
    - [ ] Unify the way models are saved and loaded. Direct paths.
 8. Extend to multi-human
    - Write code to detect all humans in the scene.
    - We want to 
      - (a) perform the pose estimation for all humans in the scene
      - (b) determine the closest body part of all humans to a given point in space, e.g. the camera frame.
      - (c) if more than one human is closer than a given threshold, return this as unsafe.
      - (d) determine the closest human.
      - (e) perform steps 5. - 9. with the closest human only.
 9. Integrate Lidar Sensor to validate YOLO human detection