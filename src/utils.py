import numpy as np
import pandas as pd
import sys
import pickle as pkl
import random
import os
import pdb
from tqdm import tqdm
from threading import Thread, Lock
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import argparse

from unicore.modules import init_bert_params
from unicore.data import (
    Dictionary, NestedDictionaryDataset, TokenizeDataset, PrependTokenDataset,
    AppendTokenDataset, FromNumpyDataset, RightPadDataset, RightPadDataset2D,
    RawArrayDataset, RawLabelDataset,
)
from unimol.data import (
    KeyDataset, ConformerSampleDataset, AtomTypeDataset,
    RemoveHydrogenDataset, CroppingDataset, NormalizeDataset,
    DistanceDataset, EdgeTypeDataset, RightPadDatasetCoord, 
)
from unimol.models.transformer_encoder_with_pair import TransformerEncoderWithPair
from unimol.models.unimol import NonLinearHead, GaussianLayer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_predict, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from MolCLR_dataset import MolTestDatasetWrapper, MolTestDatasetWrapper_smiles
from gcn import GCN
from ginet import GINet
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.pipeline import Pipeline
import statistics
from unimol_encoder import UniMolEncoder


def construct_data_list(raw_data_path:str, label:str):
    data_df = pd.read_csv(raw_data_path)
    smiles_to_conformation_dict = pkl.load(open(f'data/intermediate/SMILES_accum_S2_smiles_to_conformation_dict.pkl', 'rb'))
    data_list = []
    for index, row in data_df.iterrows():
        smiles = row["SMILES"]
        if smiles in smiles_to_conformation_dict:
            data_item = {
                "atoms": smiles_to_conformation_dict[smiles]["atoms"],
                "coordinates": smiles_to_conformation_dict[smiles]["coordinates"],
                "smiles": smiles,
                "label": row[label],
            }    
            data_list.append(data_item)
    pkl.dump(data_list, open(f'data/intermediate/{os.path.basename(raw_data_path).split(".")[0]}_{label}_data_list.pkl', 'wb'))
    
def convert_whole_data_list_to_data_loader(remove_hydrogen, raw_data_path):
    def convert_data_list_to_dataset_(data_list):
        dictionary = Dictionary.load('data/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        smiles_dataset = KeyDataset(data_list, "smiles")
        label_dataset = KeyDataset(data_list, "label")
        dataset = ConformerSampleDataset(data_list, 1024, "atoms", "coordinates")
        dataset = AtomTypeDataset(data_list, dataset)
        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", remove_hydrogen, False)
        dataset = CroppingDataset(dataset, 1, "atoms", "coordinates", 256)
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, dictionary, max_seq_len=512)
        coord_dataset = KeyDataset(dataset, "coordinates")
        src_dataset = AppendTokenDataset(PrependTokenDataset(token_dataset, dictionary.bos()), dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = AppendTokenDataset(PrependTokenDataset(coord_dataset, 0.0), 0.0)
        distance_dataset = DistanceDataset(coord_dataset)
        return NestedDictionaryDataset({
            "input": {
                "src_tokens": RightPadDataset(src_dataset, pad_idx=dictionary.pad(),),
                "src_coord": RightPadDatasetCoord(coord_dataset, pad_idx=0,),
                "src_distance": RightPadDataset2D(distance_dataset, pad_idx=0,),
                "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0,),
                "smiles": RawArrayDataset(smiles_dataset),
            }, 
            "target": {
                "label": RawLabelDataset(label_dataset),
            }
        })
        
    batch_size = 1
    data_list = pkl.load(open(f'data/intermediate/{os.path.basename(raw_data_path).split(".")[0]}_Formal charge Class_data_list.pkl', 'rb'))
    
    data = [data_list[i] for i in range(len(data_list))]
    dataset = convert_data_list_to_dataset_(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collater)
    
    return data_loader

def generate_upsampled_raw_data(raw_data_path, target_column, minority_label):
    df = pd.read_csv(raw_data_path)
    df_majority = df[df[target_column] == (1 - minority_label)]
    df_minority = df[df[target_column] == minority_label]

    majority_count = df_majority.shape[0]
    minority_count = df_minority.shape[0]
    
    additional_samples_needed = majority_count - minority_count
    df_minority_additional = resample(df_minority,
                                      replace=True,      
                                      n_samples=additional_samples_needed,    
                                      random_state=123)  

    df_minority_upsampled = pd.concat([df_minority, df_minority_additional])

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    print(df_upsampled[target_column].value_counts())

    df_upsampled.to_csv(f"data/raw/{os.path.basename(raw_data_path).split('.')[0]}_upsampled_{target_column}.csv", index=False)
    

def save_UniMol_embedding(remove_hydrogen, raw_data_path, device):
    data_loader = convert_whole_data_list_to_data_loader(remove_hydrogen, raw_data_path)
    model = UniMolEncoder(device=device, remove_hydrogen=remove_hydrogen) # without shuffle
    model.to(device)
    if remove_hydrogen:
        surfix = 'no_h'
    else:
        surfix = 'all_h'
    
    save_path = f'data/embedding/{os.path.basename(raw_data_path).split(".")[0]}_embedding_{surfix}.pkl'
    embedding = {}
    
    for batch in tqdm(data_loader):
        smiles_list, molecule_representation = model(batch)
        assert len(smiles_list) == len(molecule_representation)
        for i in range(len(smiles_list)):
            embedding[smiles_list[i]] = molecule_representation[i]
    
    with open(save_path, "wb") as file:
        pkl.dump(embedding, file)
        
def save_MolCLR_embedding(raw_data_path, label, task, model_choice, device):
    dataset = MolTestDatasetWrapper(data_path=raw_data_path, target=label, task=task)
    dataloader = dataset.get_fulldata_loader()
    assert model_choice in ["gin", "gcn"]
    if model_choice == "gin":
        model = GINet()
        try:
            state_dict = torch.load("weight/gin.pth", map_location=device)
            model.load_my_state_dict(state_dict)
            model.to(device)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            raise FileNotFoundError
    elif model_choice == "gcn":
        model = GCN()
        try:
            state_dict = torch.load("weight/gcn.pth", map_location=device)
            model.load_my_state_dict(state_dict)
            model.to(device)
            print("Loaded pre-trained model with success.")       
        except FileNotFoundError:
            raise FileNotFoundError     
    
    if model_choice == "gin":
        surfix = "gin"
    elif model_choice == "gcn":
        surfix = "gcn"
    
    save_path = f'data/embedding/{os.path.basename(raw_data_path).split(".")[0]}_embedding_{surfix}.pkl'
    embedding = {}
    
    for batch in tqdm(dataloader):
        batch.to(device)
        smiles_list, molecule_representation = model(batch)
        assert len(smiles_list) == len(molecule_representation)
        for i in range(len(smiles_list)):
            embedding[smiles_list[i]] = molecule_representation[i]
    
    with open(save_path, "wb") as file:
        pkl.dump(embedding, file)

def rule_based_Accum_eval(raw_data_path:str, rule_based_embedding:list, rule_based_model:list):
    """
    e.g., rule_based_embedding = {"Formal charge Class":"gcn", "Q_vsa_Ppos Class":"gin", "vsa_don Class":"gin"}
    rule_based_model = {"Formal charge Class":"mlp", "Q_vsa_Ppos Class":"mlp", "vsa_don Class":"mlp"}
    """
    X = {}
    y = {}
    model = {}
    smiles_list = {}
    
    for label in ["Formal charge Class", "Q_vsa_Ppos Class", "vsa_don Class", "PA Accum Class"]:
        if rule_based_embedding[label] == "unimol_no_h":
            smiles_list[label], X[label], y[label] = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
        elif rule_based_embedding[label] == "unimol_all_h":
            smiles_list[label], X[label], y[label] = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
        elif rule_based_embedding[label] == "gin":
            smiles_list[label], X[label], y[label] = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
        elif rule_based_embedding[label] == "gcn":
            smiles_list[label], X[label], y[label] = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    
    assert len(X["Formal charge Class"]) == len(X["Q_vsa_Ppos Class"]) == len(X["vsa_don Class"])
    assert smiles_list["Formal charge Class"] == smiles_list["Q_vsa_Ppos Class"] == smiles_list["vsa_don Class"] == smiles_list["PA Accum Class"]
    
    for label in ["Formal charge Class", "Q_vsa_Ppos Class", "vsa_don Class"]:
        assert rule_based_model[label] == "mlp"
        with open(f"weight/{label}_{rule_based_embedding[label]}_{rule_based_model[label]}.pkl","rb") as file:
            trained_pipeline = pkl.load(file)
        hyperparameters = trained_pipeline.get_params()
        model[label] =  Pipeline([
            ('standardscaler', StandardScaler()),
            ('mlpclassifier', MLPClassifier(max_iter=10000, verbose=10, random_state=12))
        ]).set_params(**hyperparameters)
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accumulation_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'tp': [],
        'fp': [],
        'tn': [],
        'fn': []
    }

    for train_index, test_index in kf.split(X["Formal charge Class"]):

        # Train MLP classifiers using the extracted features and saved hyperparameters
        mlp_fc = model["Formal charge Class"]
        mlp_fc.fit(X["Formal charge Class"][train_index], y["Formal charge Class"][train_index])
        predictions_fc = mlp_fc.predict(X["Formal charge Class"][test_index])
        
        mlp_qvsa = model["Q_vsa_Ppos Class"]
        mlp_qvsa.fit(X["Q_vsa_Ppos Class"][train_index], y["Q_vsa_Ppos Class"][train_index])
        predictions_qvsa = mlp_qvsa.predict(X["Q_vsa_Ppos Class"][test_index])
        
        mlp_vsa = model["vsa_don Class"]
        mlp_vsa.fit(X["vsa_don Class"][train_index], y["vsa_don Class"][train_index])
        predictions_vsa = mlp_vsa.predict(X["vsa_don Class"][test_index])
        
        # Combine individual predictions using rule-based logic
        accumulation_predictions = predictions_vsa & (predictions_fc | predictions_qvsa)

        # Evaluate metrics
        accuracy = accuracy_score(y["PA Accum Class"][test_index], accumulation_predictions)
        precision = precision_score(y["PA Accum Class"][test_index], accumulation_predictions)
        recall = recall_score(y["PA Accum Class"][test_index], accumulation_predictions)
        f1 = f1_score(y["PA Accum Class"][test_index], accumulation_predictions)
        tn, fp, fn,tp = confusion_matrix(y["PA Accum Class"][test_index], accumulation_predictions).ravel()

        # Store metrics for this fold
        accumulation_metrics['accuracy'].append(accuracy)
        accumulation_metrics['precision'].append(precision)
        accumulation_metrics['recall'].append(recall)
        accumulation_metrics['f1'].append(f1)
        accumulation_metrics['tp'].append(tp)
        accumulation_metrics['fp'].append(fp)
        accumulation_metrics['tn'].append(tn)
        accumulation_metrics['fn'].append(fn)

    average_metrics = {metric: f"mean: {round(statistics.mean(values),4)}; std: {round(statistics.stdev(values),4)}" for metric, values in accumulation_metrics.items()}

    for metric, stat in average_metrics.items():
        print(f"{metric}          {stat}")
        
    return average_metrics

def get_UniMol_data_loader(raw_data_path, label, remove_hydrogen):
    with open(f"data/intermediate/{os.path.basename(raw_data_path).split('.')[0]}_{label}_data_list.pkl", "rb") as f:
        data_list = pkl.load(f)
    
    if f"upsampled_{label}" in raw_data_path:
        if remove_hydrogen:
            with open(f"data/embedding/{os.path.basename(raw_data_path).split('.')[0].replace(f'_upsampled_{label}','')}_embedding_no_h.pkl", "rb") as f:
                embedding = pkl.load(f)
        else:
            with open(f"data/embedding/{os.path.basename(raw_data_path).split('.')[0].replace(f'_upsampled_{label}','')}_embedding_all_h.pkl", "rb") as f:
                embedding = pkl.load(f)
    else:
        if remove_hydrogen:
            with open(f"data/embedding/{os.path.basename(raw_data_path).split('.')[0]}_embedding_no_h.pkl", "rb") as f:
                embedding = pkl.load(f)
        else:
            with open(f"data/embedding/{os.path.basename(raw_data_path).split('.')[0]}_embedding_all_h.pkl", "rb") as f:
                embedding = pkl.load(f)       
              
    X = []
    y = []
    # smiles_list = []
    
    # assert len(embedding) == len(data_list)
    for i in range(len(data_list)):
        # smiles_list.append(data_list[i]["smiles"])
        y.append(data_list[i]["label"])
        X.append(embedding[data_list[i]["smiles"]].cpu().detach().numpy())    
    
    # return smiles_list, np.array(X), np.array(y)
    return np.array(X), np.array(y)



def get_MolCLR_data_loader(raw_data_path, label, model):
    
    assert model in ["gin", "gcn"]
    with open(f"data/intermediate/{os.path.basename(raw_data_path).split('.')[0]}_{label}_data_list.pkl", "rb") as f:
        data_list = pkl.load(f)
    
    if f"upsampled_{label}" in raw_data_path:     
        with open(f"data/embedding/{os.path.basename(raw_data_path).split('.')[0].replace(f'_upsampled_{label}','')}_embedding_{model}.pkl", "rb") as f:
            embedding = pkl.load(f)
    else:     
        with open(f"data/embedding/{os.path.basename(raw_data_path).split('.')[0].replace(f'_upsampled_{label}','')}_embedding_{model}.pkl", "rb") as f:
            embedding = pkl.load(f)
    
    X = []
    y = []
    # smiles_list = []
    
    for i in range(len(data_list)):
        try:
            X.append(embedding[data_list[i]["smiles"]].cpu().detach().numpy())    
            # smiles_list.append(data_list[i]["smiles"])
            y.append(data_list[i]["label"])
        except:
            print(f"no MolCLR embedding {data_list[i]['smiles']}")
    
    # return smiles_list, np.array(X), np.array(y)
    return np.array(X), np.array(y)