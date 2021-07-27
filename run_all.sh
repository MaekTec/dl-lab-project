# same as in run.ipynb

# Pretraining
python3 pretrain_rotation.py dataset
python3 pretrain_jigsaw_puzzle.py dataset
python3 pretrain_contrastive_predictive_coding.py dataset
python3 pretrain_moco.py dataset
# Downstream with random weights
python3 downstream.py --fine-tune-last-layer True dataset none
python3 downstream.py dataset none
python3 downstream.py --whole-dataset True dataset none
# Downstream with weights from pretraining
python3 downstream.py --pretrain-path results/pretrain_rotation/lr0.0002_weight_decay0.0_bs256_epochs20_image_size64_resnetFalse_ --fine-tune-last-layer True dataset rotation
python3 downstream.py --pretrain-path results/pretrain_rotation/lr0.0002_weight_decay0.0_bs256_epochs20_image_size64_resnetFalse_ dataset rotation
python3 downstream.py --pretrain-path results/pretrain_jigsaw_puzzle/lr0.0002_weight_decay0.0_bs256_epochs20_image_size64_num_tiles_per_dim3_number_of_permutations64_resnetFalse_ --fine-tune-last-layer True dataset jigsaw_puzzle
python3 downstream.py --pretrain-path results/pretrain_jigsaw_puzzle/lr0.0002_weight_decay0.0_bs256_epochs20_image_size64_num_tiles_per_dim3_number_of_permutations64_resnetFalse_ dataset jigsaw_puzzle
python3 downstream.py --pretrain-path results/pretrain_contrastive_predictive_coding/lr0.0002_weight_decay0.0_bs128_epochs20_image_size64_num_patches_per_dim4_resnetFalse_ --fine-tune-last-layer True dataset cpc
python3 downstream.py --pretrain-path results/pretrain_contrastive_predictive_coding/lr0.0002_weight_decay0.0_bs128_epochs20_image_size64_num_patches_per_dim4_resnetFalse_ dataset cpc
python3 downstream.py --pretrain-path results/pretrain_moco/lr0.0002_weight_decay0.0_bs256_epochs30_image_size64_resnetFalse_ --fine-tune-last-layer True dataset moco
python3 downstream.py --pretrain-path results/pretrain_moco/lr0.0002_weight_decay0.0_bs256_epochs30_image_size64_resnetFalse_ dataset moco
# Eval downstream on testset
python3 eval_downstream.py --exp-suffix eval_none_e2e --downstream-path results/downstream/pretrain_tasknone_fine_tune_last_layerFalse_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs60_image_size64_resnetFalse_whole_datasetFalse_ dataset
python3 eval_downstream.py --exp-suffix eval_none_ftl --downstream-path results/downstream/pretrain_tasknone_fine_tune_last_layerTrue_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs60_image_size64_resnetFalse_whole_datasetFalse_ dataset
python3 eval_downstream.py --exp-suffix eval_none_e2e_whole_dataset --downstream-path results/downstream/pretrain_tasknone_fine_tune_last_layerFalse_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs60_image_size64_resnetFalse_whole_datasetTrue_ dataset
python3 eval_downstream.py --exp-suffix eval_rotation_ftl --downstream-path results/downstream/pretrain_taskrotation_fine_tune_last_layerTrue_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs60_image_size64_resnetFalse_whole_datasetFalse_ dataset
python3 eval_downstream.py --exp-suffix eval_rotation_e2e --downstream-path results/downstream/pretrain_taskrotation_fine_tune_last_layerFalse_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs60_image_size64_resnetFalse_whole_datasetFalse_ dataset
python3 eval_downstream.py --exp-suffix eval_jigsaw_puzzle_ftl --downstream-path results/downstream/pretrain_taskjigsaw_puzzle_fine_tune_last_layerTrue_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs60_image_size64_resnetFalse_whole_datasetFalse_ dataset
python3 eval_downstream.py --exp-suffix eval_jigsaw_puzzle_e2e --downstream-path results/downstream/pretrain_taskjigsaw_puzzle_fine_tune_last_layerFalse_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs60_image_size64_resnetFalse_whole_datasetFalse_ dataset
python3 eval_downstream.py --exp-suffix eval_cpc_ftl --downstream-path results/downstream/pretrain_taskcpc_fine_tune_last_layerTrue_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs60_image_size64_resnetFalse_whole_datasetFalse_ dataset
python3 eval_downstream.py --exp-suffix eval_cpc_e2e --downstream-path results/downstream/pretrain_taskcpc_fine_tune_last_layerFalse_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs60_image_size64_resnetFalse_whole_datasetFalse_ dataset
python3 eval_downstream.py --exp-suffix eval_moco_ftl --downstream-path results/downstream/pretrain_taskmoco_fine_tune_last_layerTrue_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs60_image_size64_resnetFalse_whole_datasetFalse_ dataset
python3 eval_downstream.py --exp-suffix eval_moco_e2e --downstream-path results/downstream/pretrain_taskmoco_fine_tune_last_layerFalse_lr_ftl0.0002_lr_e2e5e-05_weight_decay0.01_bs256_epochs60_image_size64_resnetFalse_whole_datasetFalse_ dataset
# Plots
python plots/plotter.py --csv_files plots/run-pretrain_rotation-tag-Accuracy_train.csv plots/run-pretrain_rotation-tag-Accuracy_val.csv plots/run-pretrain_jigsaw_puzzle-tag-Accuracy_train.csv plots/run-pretrain_jigsaw_puzzle-tag-Accuracy_val.csv plots/run-pretrain_contrastive_predictive_coding-tag-Accuracy_train.csv plots/run-pretrain_contrastive_predictive_coding-tag-Accuracy_val.csv plots/run-pretrain_moco-tag-Accuracy_train.csv plots/run-pretrain_moco-tag-Accuracy_val.csv --name pretrain_accuracy --xlabel Epoch --ylabel Accuracy --labels Rotation_Training Rotation_Validation Jigsaw_Puzzle_Training Jigsaw_Puzzle_Validation CPC_Training CPC_Validation MOCO_Training MOCO_Validation
python plots/plotter.py --csv_files plots/run-pretrain_rotation-tag-Loss_train.csv plots/run-pretrain_rotation-tag-Loss_val.csv plots/run-pretrain_jigsaw_puzzle-tag-Loss_train.csv plots/run-pretrain_jigsaw_puzzle-tag-Loss_val.csv plots/run-pretrain_contrastive_predictive_coding-tag-Loss_train.csv plots/run-pretrain_contrastive_predictive_coding-tag-Loss_val.csv plots/run-pretrain_moco-tag-Loss_train.csv plots/run-pretrain_moco-tag-Loss_val.csv --name pretrain_loss --xlabel Epoch --ylabel Loss --labels Rotation_Training Rotation_Validation Jigsaw_Puzzle_Training Jigsaw_Puzzle_Validation CPC_Training CPC_Validation MOCO_Training MOCO_Validation
python plots/plotter.py --csv_files plots/run-downstream_none_ftl-tag-Accuracy_val.csv plots/run-downstream_rotation_ftl-tag-Accuracy_val.csv plots/run-downstream_jigsaw_puzzle_ftl-tag-Accuracy_val.csv plots/run-downstream_cpc_ftl-tag-Accuracy_val.csv plots/run-downstream_moco_ftl-tag-Accuracy_val.csv --name downstream_ftl_accuracy_val --xlabel Epoch --ylabel Accuracy_Validation --labels No_pretraining Rotation Jigsaw_Puzzle CPC MOCO
python plots/plotter.py --csv_files plots/run-downstream_none_e2e_whole_dataset-tag-Accuracy_val.csv plots/run-downstream_none_e2e-tag-Accuracy_val.csv plots/run-downstream_rotation_e2e-tag-Accuracy_val.csv plots/run-downstream_jigsaw_puzzle_e2e-tag-Accuracy_val.csv plots/run-downstream_cpc_e2e-tag-Accuracy_val.csv plots/run-downstream_moco_e2e-tag-Accuracy_val.csv --name downstream_e2e_accuracy_val --xlabel Epoch --ylabel Accuracy_Validation --labels No_pretraining_whole_dataset No_pretraining Rotation Jigsaw_Puzzle CPC MOCO

