#!/bin/bash
python human_pose_pipeline/examples/eval_full_pipeline.py \
    --pose_model_save_path "human_pose_pipeline/models/pose_estimation" \
    --pose_run_name "jax_resnet50_regressflow" \
    --pose_base_key "H36M_RegressFlowResNet18_3Joints_n9000_4998731f" \
    --motion_model_save_path "human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle" \
    --motion_score_fn_path "human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM720of800_sketch_srft_seed0_size10000.cloudpickle" \
    --split "test" \
    --enable_ood \
    --n_correct_poses_required 3 \
    --output_dir "results/final/full_pipeline/n_correct_poses_required_3/"

python human_pose_pipeline/examples/eval_full_pipeline.py \
    --pose_model_save_path "human_pose_pipeline/models/pose_estimation" \
    --pose_run_name "jax_resnet50_regressflow" \
    --pose_base_key "H36M_RegressFlowResNet18_3Joints_n9000_4998731f" \
    --motion_model_save_path "human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle" \
    --motion_score_fn_path "human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM720of800_sketch_srft_seed0_size10000.cloudpickle" \
    --split "test" \
    --enable_ood \
    --n_correct_poses_required 50 \
    --output_dir "results/final/full_pipeline/n_correct_poses_required_50/"

python human_pose_pipeline/examples/eval_full_pipeline.py \
    --pose_model_save_path "human_pose_pipeline/models/pose_estimation" \
    --pose_run_name "jax_resnet50_regressflow" \
    --pose_base_key "H36M_RegressFlowResNet18_3Joints_n9000_4998731f" \
    --motion_model_save_path "human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle" \
    --motion_score_fn_path "human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM720of800_sketch_srft_seed0_size10000.cloudpickle" \
    --split "test" \
    --enable_ood \
    --n_correct_poses_required 5 \
    --output_dir "results/final/full_pipeline/n_correct_poses_required_5/"

python human_pose_pipeline/examples/eval_full_pipeline.py \
    --pose_model_save_path "human_pose_pipeline/models/pose_estimation" \
    --pose_run_name "jax_resnet50_regressflow" \
    --pose_base_key "H36M_RegressFlowResNet18_3Joints_n9000_4998731f" \
    --motion_model_save_path "human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle" \
    --motion_score_fn_path "human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM720of800_sketch_srft_seed0_size10000.cloudpickle" \
    --split "test" \
    --enable_ood \
    --n_correct_poses_required 10 \
    --output_dir "results/final/full_pipeline/n_correct_poses_required_10/"

python human_pose_pipeline/generate_plots/generate_full_pipeline_results.py