# raw_data: finetuning_set_smiles_plus_features_normalized_descriptors_mean0_std1 

nohup python src/main.py --splitting random --raw_data_path data/raw/finetuning_set_smiles_plus_features_normalized_descriptors_mean0_std1.csv --label normalized_expt_Raw_min_0_max_1 --gpu cuda:2 > finetuning_set_smiles_plus_features_normalized_descriptors_mean0_std1.log 2>&1 &