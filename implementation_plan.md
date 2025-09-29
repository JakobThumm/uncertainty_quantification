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
    - [ ] Then, perform "Experiment 2" from Marians folder -> Evaluation of the 3D human pose estimation with uncertainty quantification (steps 4 and 6 from above.)
      - [x] Test with real H36M examples - Load actual images and poses from the dataset
      - [x] Implement evaluation metrics - MPJPE, PCK metrics as in Marian's Experiment 2
      - [x] Add torch to the environment to get the bounding box estimation and switch to YOLO 11 -> We might want to change this later.
      - [x] Mirror Marian's code exactly
      - [x] Add uncertainty estimation
      - [x] Debug estimation and uncertainty
      - [x] Move utils, dataset functionality to correct folder
      - [ ] Implement 3D triangulation - Convert 2D poses to 3D using camera parameters
 3. Perform sketching lanczos OOD detection for pose estimation
    - Run the OOD detection with sketching lanczos on the 2D pose estimation model. ID data would be the human 3.6m dataset. For OOD data we can use tiger pose dataset.
 4. Test the pre-trained pose prediction models
    - Use the pre-trained models with converted weights into JAX.
    - First, perform pose prediction on a few H3.6M examples.
    - Then, perform "Experiment 3" from Marians folder -> Evaluation of the 3D human pose prediction with uncertainty quantification (step 7 from above.)
 5. Perform OOD detection for motion prediction
    - Run the OOD detection with sketching lanczos on the 3D pose prediction model. ID data would be the human 3.6m dataset. OOD is the close interactions dataset.
 6. Implement the training pipeline in JAX
    - The training pipeline is based on Marians code in the Experiment 1 folder. 
    - We want to have the training pipeline, so that we could do fine tuning and adaptions if needed.
 7. Full single-human pipeline
    - First, find out how the current bounding box algorithm works in Marians code. E.g., Experiment 4. There, he did real-world tests, so it should include some bounding box algorithm.
    - Implement the full pipeline based on the code of Experiment 4 plus the new OOD detection. Everything in JAX.
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