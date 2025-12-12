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
            Results Marian Pytorch on 3 validation files (yolo threshold = 0.8):
              Total frames processed: 4881
              Total joints evaluated: 63453
              Average MPJPE: 7.64 pixels
              Average percentage of keypoints within 1 std: 73.14%
              Average percentage of keypoints within 2 std: 91.45%
              Average percentage of keypoints within 3 std: 97.50%
              Average percentage of keypoints within 4 std: 99.17%
            Results Jax on 3 validation files (yolo threshold = 0.3)
              Total frames processed: 4988
              Total joints evaluated: 64844
              Average MPJPE: 7.83
              Average percentage of keypoints within 1 std: 73.08%
              Average percentage of keypoints within 2 std: 92.45%
              Average percentage of keypoints within 3 std: 97.51%
              Average percentage of keypoints within 4 std: 99.14%
    - [x] Investigate the uncertainty coverage for the 2D Pose estimation in pytorch vs. Jax.
    - [ ] Investigate prediction accuracy 3D Pose estimation in pytorch vs. Jax.
            Results Marian Pytorch on 3 validation files (yolo threshold = 0.8):
              Total frames processed: 6545
              Total joints evaluated: 85085
              Average MPJPE: 82.64 mm (On GT 2D data -> 3D triangulation = 3.67 mm)
              Average percentage of keypoints within 1 std: 2.54%
              Average percentage of keypoints within 2 std: 16.95%
              Average percentage of keypoints within 3 std: 42.86%
              Average percentage of keypoints within 4 std: 61.47%
              ---
              Average percentage of keypoints within 1 std: 26.80%
              Average percentage of keypoints within 2 std: 53.92%
              Average percentage of keypoints within 3 std: 72.74%
              Average percentage of keypoints within 4 std: 82.48%
            Results Jax on 3 validation files (yolo threshold = 0.3)
              
    - [ ] Investigate the uncertainty coverage for the 3D Pose estimation in pytorch vs. Jax.
    - [ ] Investigate prediction accuracy 3D motion prediction in pytorch vs. Jax.
    - [ ] Investigate the uncertainty coverage for the 3D motion prediction in pytorch vs. Jax.
 7. Full single-human pipeline
    - First, find out how the current bounding box algorithm works in Marians code. E.g., Experiment 4. There, he did real-world tests, so it should include some bounding box algorithm.
    - Implement the full pipeline based on the code of Experiment 4 plus the new OOD detection. Everything in JAX.
    - [ ] Implement motion prediction model that takes uncertainty as input
    - [ ] Write script that performs pose estimation with uncertainty + motion prediction with uncertainty with input from pose estimation
 8. Extend to multi-human
    - Write code to detect all humans in the scene.
    - We want to 
      - (a) perform the pose estimation for all humans in the scene
      - (b) determine the closest body part of all humans to a given point in space, e.g. the camera frame.
      - (c) if more than one human is closer than a given threshold, return this as unsafe.
      - (d) determine the closest human.
      - (e) perform steps 5. - 9. with the closest human only.
 9. Replace pytorch YOLO v5 with something JAX-based
    - We can only run OOD on JAX models, so to perform OOD on the human detection, we need a JAX-based model.