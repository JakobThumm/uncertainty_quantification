## Filesystem

- /uncertainty_quantification
- /models
    /H36M/RegressFlow/seed_420
        finetuned_h36m_regressflow_pred_args.json
        finetuned_h36m_regressflow_pred_params.pickle
- /datasets
    /H36M_FREI
        /S1
        /S9
        /S11

## RegressFlow flax model for HPE
In /uncertainty_quantification/src/models
# regressflow.py : pose prediction only
# regressflow_wiz_alea.py(Marian's original): pose prediction + aleatoric unc pred
Two Models differ only in the head, regressflow outputs coordinates directly, regressflow_wiz_alea outputs a dict that contains coordinates, covariancem, variance...(unluckily, this is not complitable with this repo style)

Solution: Use regressflow for epistemic unc estimate, use regressflow_wiz_alea for aleatoric later


## score model command (TODO: test on server!)
TO run on the whole model first, test whether enough memory budget

# Sketched Lanczos(Run this!)
(same recepie as shown in the article for resnet, train and test batch can change)

python score_model.py --ID_dataset H36M --OOD_dataset H36M --model RegressFlow --run_name finetuned_h36m_regressflow_pred --subsample_trainset 100 --test_batch_size 4 --train_batch_size 4 --lanczos_hm_iter 0 --lanczos_lm_iter 81 --lanczos_seed 1 --sketch srft --sketch_size 10000

# The author say it is "smart", maybe interesting run this
python score_model.py --ID_dataset H36M --OOD_dataset H36M --model RegressFlow --run_name finetuned_h36m_regressflow_pred --subsample_trainset 100 --test_batch_size 4 --train_batch_size 4 --serialize_ggn_on_batches --lanczos_seed 1 --sketch srft --sketch_size 1000



<!-- # scod
python score_model.py --ID_dataset H36M --OOD_dataset H36M --model RegressFlow --score scod --run_name finetuned_h36m_regressflow_pred --subsample_trainset 100

# smart_lla
python score_model.py --ID_dataset H36M --OOD_dataset H36M --model RegressFlow --score smart_lla --run_name finetuned_h36m_regressflow_pred --subsample_trainset 100 --lanczos_hm_iter 0 -->



## Evaluation
If previous score model is successful, we can evaluate its AUROC score by evaluate.py