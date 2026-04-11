# Performs the motion prediction from predicted 3D pose inputs with uncertainty without OOD detection
python human_pose_pipeline/examples/motion_prediction.py \
  --data_path datasets/ \
  --dataset_name Human36mMotionDataset3DWithInputUncertainty \
  --split test \
  --model_save_path human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle \
  --output_dir results/final/motion_prediction/no_uncertainty_no_ood
# Create the results table in 
python human_pose_pipeline/generate_plots/generate_conformal_prediction_set_results.py