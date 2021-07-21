# same as in run.ipynb

# Pretraining
python pretrain_rotation.py dataset
python pretrain_jigsaw_puzzle.py dataset
python pretrain_contrastive_predictive_coding.py dataset
python pretrain_moco.py dataset
# Downstream with random weights
python downstream.py --fine-tune-last-layer True dataset none
python downstream.py dataset none
python downstream.py --whole-dataset True dataset none
# Downstream with weights from pretraining
python downstream.py --pretrain-path results/pretrain_rotation/lr0.0002_weight_decay0.0_bs256_epochs2_image_size64_resnetFalse_ --fine-tune-last-layer True dataset rotation
python downstream.py --pretrain-path results/pretrain_rotation/lr0.0002_weight_decay0.0_bs256_epochs2_image_size64_resnetFalse_ dataset rotation
python downstream.py --pretrain-path results/pretrain_jigsaw_puzzle/lr0.0002_weight_decay0.0_bs256_epochs2_image_size64_num_tiles_per_dim3_number_of_permutations64_resnetFalse_ --fine-tune-last-layer True dataset jigsaw_puzzle
python downstream.py --pretrain-path results/pretrain_jigsaw_puzzle/lr0.0002_weight_decay0.0_bs256_epochs2_image_size64_num_tiles_per_dim3_number_of_permutations64_resnetFalse_ dataset jigsaw_puzzle
python downstream.py --pretrain-path results/pretrain_contrastive_predictive_coding/lr0.0002_weight_decay0.0_bs128_epochs2_image_size64_num_patches_per_dim4_resnetFalse_ --fine-tune-last-layer True dataset cpc
python downstream.py --pretrain-path results/pretrain_contrastive_predictive_coding/lr0.0002_weight_decay0.0_bs128_epochs2_image_size64_num_patches_per_dim4_resnetFalse_ dataset cpc
python downstream.py --pretrain-path results/pretrain_moco/lr0.0002_weight_decay0.0_bs256_epochs2_image_size64_resnetFalse_ --fine-tune-last-layer True dataset moco
python downstream.py --pretrain-path results/pretrain_moco/lr0.0002_weight_decay0.0_bs256_epochs2_image_size64_resnetFalse_ dataset moco
# Eval downstream on testset
python eval_downstream.py --exp-suffix eval_none_e2e --downstream-path results/downstream/pretrain_tasknone_fine_tune_last_layerFalse_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs2_image_size64_resnetFalse_whole_datasetFalse_ dataset
python eval_downstream.py --exp-suffix eval_none_ftl --downstream-path results/downstream/pretrain_tasknone_fine_tune_last_layerTrue_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs2_image_size64_resnetFalse_whole_datasetFalse_ dataset
python eval_downstream.py --exp-suffix eval_none_e2e_whole_dataset --downstream-path results/downstream/pretrain_tasknone_fine_tune_last_layerFalse_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs20_image_size64_resnetFalse_whole_datasetTrue_ dataset
python eval_downstream.py --exp-suffix eval_rotation_ftl --downstream-path results/downstream/pretrain_taskrotation_fine_tune_last_layerTrue_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs2_image_size64_resnetFalse_whole_datasetFalse_ dataset
python eval_downstream.py --exp-suffix eval_rotation_e2e --downstream-path results/downstream/pretrain_taskrotation_fine_tune_last_layerFalse_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs2_image_size64_resnetFalse_whole_datasetFalse_ dataset
python eval_downstream.py --exp-suffix eval_jigsaw_puzzle_ftl --downstream-path results/downstream/pretrain_taskjigsaw_puzzle_fine_tune_last_layerTrue_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs2_image_size64_resnetFalse_whole_datasetFalse_ dataset
python eval_downstream.py --exp-suffix eval_jigsaw_puzzle_e2e --downstream-path results/downstream/pretrain_taskjigsaw_puzzle_fine_tune_last_layerFalse_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs2_image_size64_resnetFalse_whole_datasetFalse_ dataset
python eval_downstream.py --exp-suffix eval_cpc_ftl --downstream-path results/downstream/pretrain_taskcpc_fine_tune_last_layerTrue_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs2_image_size64_resnetFalse_whole_datasetFalse_ dataset
python eval_downstream.py --exp-suffix eval_cpc_e2e --downstream-path results/downstream/pretrain_taskcpc_fine_tune_last_layerFalse_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs2_image_size64_resnetFalse_whole_datasetFalse_ dataset
python eval_downstream.py --exp-suffix eval_moco_ftl --downstream-path results/downstream/pretrain_taskmoco_fine_tune_last_layerTrue_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs2_image_size64_resnetFalse_whole_datasetFalse_ dataset
python eval_downstream.py --exp-suffix eval_moco_e2e --downstream-path results/downstream/pretrain_taskmoco_fine_tune_last_layerFalse_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs2_image_size64_resnetFalse_whole_datasetFalse_ dataset
