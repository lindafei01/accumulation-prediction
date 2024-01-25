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

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[2]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[3]

scoring = {'accuracy': 'accuracy', 'precision': 'precision', 
           'recall': 'recall', 'f1': 'f1',
           'tp': make_scorer(tp), 'fp': make_scorer(fp), 
           'tn': make_scorer(tn), 'fn': make_scorer(fn)}

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

def set_random_seed(random_seed=1024):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

class UniMolModel(nn.Module):
    def __init__(self):
        super().__init__()
        dictionary = Dictionary.load('data/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), 512, self.padding_idx
        )
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=15,
            embed_dim=512,
            ffn_embed_dim=2048,
            attention_heads=64,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            max_seq_len=512,
            activation_fn='gelu',
            no_final_head_layer_norm=True,
        )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, 64, 'gelu'
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        self.apply(init_bert_params)

    def forward(
        self,
        sample,
    ):
        input = sample['input']
        src_tokens, src_distance, src_coord, src_edge_type \
            = input['src_tokens'], input['src_distance'], input['src_coord'], input['src_edge_type']
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (encoder_rep, encoder_pair_rep, delta_encoder_pair_rep, x_norm, delta_encoder_pair_rep_norm) \
            = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        output = {
            "molecule_representation": encoder_rep[:, 0, :],  # get cls token
            "smiles": sample['input']["smiles"],
        }
        return output

def calculate_3D_structure(raw_data_path):
    def get_smiles_list_():
        data_df = pd.read_csv(raw_data_path)
        smiles_list = data_df["SMILES"].tolist()
        smiles_list = list(set(smiles_list))
        print(len(smiles_list))
        return smiles_list

    def calculate_3D_structure_(smiles_list):
        n = len(smiles_list)
        global p
        index = 0
        while True:
            mutex.acquire()
            if p >= n:
                mutex.release()
                break
            index = p
            p += 1
            mutex.release()

            smiles = smiles_list[index]
            print(index, ':', round(index / n * 100, 2), '%', smiles)

            molecule = Chem.MolFromSmiles(smiles)
            try:
                molecule = AllChem.AddHs(molecule)
            except: 
                print("MolFromSmiles error", smiles)
                mutex.acquire()
                with open(f'data/result/{os.path.basename(raw_data_path).split(".")[0]}_invalid_smiles.txt', 'a') as f:
                    f.write('MolFromSmiles error' + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            coordinate_list = []
            result = AllChem.EmbedMolecule(molecule, randomSeed=42)
            if result != 0:
                print('EmbedMolecule failed', result, smiles)
                mutex.acquire()
                with open(f'data/result/{os.path.basename(raw_data_path).split(".")[0]}_invalid_smiles.txt', 'a') as f:
                    f.write('EmbedMolecule failed' + ' ' + str(result) + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            try:
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                print('MMFFOptimizeMolecule error', smiles)
                mutex.acquire()
                with open(f'data/result/{os.path.basename(raw_data_path).split(".")[0]}_invalid_smiles.txt', 'a') as f:
                    f.write('MMFFOptimizeMolecule error' + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            coordinates = molecule.GetConformer().GetPositions()
            
            assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)
            coordinate_list.append(coordinates.astype(np.float32))

            global smiles_to_conformation_dict
            mutex.acquire()
            smiles_to_conformation_dict[smiles] = {'smiles': smiles, 'atoms': atoms, 'coordinates': coordinate_list}
            mutex.release()  

    mutex = Lock()
    smiles_list = get_smiles_list_()
    global smiles_to_conformation_dict
    smiles_to_conformation_dict = {}
    global p
    p = 0
    thread_count = 16
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=calculate_3D_structure_, args=(smiles_list, )))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pkl.dump(smiles_to_conformation_dict, open(f'data/intermediate/{os.path.basename(raw_data_path).split(".")[0]}_smiles_to_conformation_dict.pkl', 'wb'))
    print('Valid smiles count:', len(smiles_to_conformation_dict))

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

def convert_smiles_list_to_data_loader(remove_hydrogen, smiles_list:list):
            
    def convert_data_list_to_dataset_(data_list):
        dictionary = Dictionary.load('data/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        smiles_dataset = KeyDataset(data_list, "smiles")
        # label_dataset = KeyDataset(data_list, "label")
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
            # "target": {
            #     "label": RawLabelDataset(label_dataset),
            # }
        })
    
    def calculate_3D_structure_(smiles_list):
        n = len(smiles_list)
        global p
        index = 0
        while True:
            mutex.acquire()
            if p >= n:
                mutex.release()
                break
            index = p
            p += 1
            mutex.release()

            smiles = smiles_list[index]
            print(index, ':', round(index / n * 100, 2), '%', smiles)

            molecule = Chem.MolFromSmiles(smiles)
            try:
                molecule = AllChem.AddHs(molecule)
            except: 
                print("MolFromSmiles error", smiles)
                mutex.acquire()
                mutex.release()
                continue
            atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            coordinate_list = []
            result = AllChem.EmbedMolecule(molecule, randomSeed=42)
            if result != 0:
                print('EmbedMolecule failed', result, smiles)
                mutex.acquire()
                mutex.release()
                continue
            try:
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                print('MMFFOptimizeMolecule error', smiles)
                mutex.acquire()
                mutex.release()
                continue
            coordinates = molecule.GetConformer().GetPositions()
            
            assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)
            coordinate_list.append(coordinates.astype(np.float32))

            global smiles_to_conformation_dict
            mutex.acquire()
            smiles_to_conformation_dict[smiles] = {'smiles': smiles, 'atoms': atoms, 'coordinates': coordinate_list}
            mutex.release()  

    mutex = Lock()
    global smiles_to_conformation_dict
    smiles_to_conformation_dict = {}
    global p
    p = 0
    thread_count = 16
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=calculate_3D_structure_, args=(smiles_list, )))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    batch_size = 1
    
    data_list = []
    for smiles in smiles_list:
        if smiles in smiles_to_conformation_dict:
            data_item = {
                "atoms": smiles_to_conformation_dict[smiles]["atoms"],
                "coordinates": smiles_to_conformation_dict[smiles]["coordinates"],
                "smiles": smiles,
            }    
            data_list.append(data_item)
    
    data = [data_list[i] for i in range(len(data_list))]
    
    dataset = convert_data_list_to_dataset_(data)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collater)
    
    return data_loader

class UniMolRegressor(nn.Module):
    def __init__(self, device, remove_hydrogen):
        super().__init__()
        self.encoder = UniMolModel()
        if remove_hydrogen:
            self.encoder.load_state_dict(torch.load('weight/mol_pre_no_h_220816.pt')['model'], strict=False)
        else:
            self.encoder.load_state_dict(torch.load('weight/mol_pre_all_h_220816.pt')['model'], strict=False)
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.device = device

    def move_batch_to_cuda(self, batch):
        batch['input'] = { k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch['input'].items() }
        batch['target'] = { k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch['target'].items() }
        return batch

    def forward(self, batch):
        batch = self.move_batch_to_cuda(batch)
        encoder_output = self.encoder(batch)
        molecule_representation = encoder_output['molecule_representation']
        smiles_list = encoder_output['smiles']
        x = self.mlp(molecule_representation)
        return x

class UniMolEncoder(nn.Module):
    def __init__(self, device, remove_hydrogen):
        super().__init__()
        self.encoder = UniMolModel()
        if remove_hydrogen:
            self.encoder.load_state_dict(torch.load('weight/mol_pre_no_h_220816.pt')['model'], strict=False)
        else:
            self.encoder.load_state_dict(torch.load('weight/mol_pre_all_h_220816.pt')['model'], strict=False)

        self.device = device

    def move_batch_to_cuda(self, batch):
        try:
            batch['input'] = { k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch['input'].items() }
            batch['target'] = { k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch['target'].items() }
        except:
            batch['input'] = { k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch['input'].items() }
        return batch

    def forward(self, batch):
        batch = self.move_batch_to_cuda(batch)
        encoder_output = self.encoder(batch)
        molecule_representation = encoder_output['molecule_representation']
        smiles_list = encoder_output['smiles']
        
        return smiles_list, molecule_representation
    
def evaluate(model, device, data_loader):
    model.eval()
    label_predict = torch.tensor([], dtype=torch.float32).to(device)
    label_true = torch.tensor([], dtype=torch.float32).to(device)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            label_predict_batch = model(batch)
            label_true_batch = batch['target']['label']

            label_predict = torch.cat((label_predict, label_predict_batch.detach()), dim=0)
            label_true = torch.cat((label_true, label_true_batch.detach()), dim=0)
    
    label_predict = label_predict.cpu().numpy()
    label_true = label_true.cpu().numpy()
    rmse = round(np.sqrt(mean_squared_error(label_true, label_predict)), 3)
    mae = round(mean_absolute_error(label_true, label_predict), 3)
    r2 = round(r2_score(label_true, label_predict), 3)
    corr = round(pearsonr(label_true, label_predict.squeeze())[0], 3)
    if abs(corr) > 1:
        print('')
    metric = {"rmse": rmse, 'mae': mae, 'r2': r2, "corr": corr}
    return metric

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

def train_svc(model_version, raw_data_path, label, args):
    
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplemented
    

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # pca = PCA(n_components=128)
    # X = pca.fit_transform(X_scaled)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # param_grid = {
    #     'C': [0.1, 1, 10, 100],  # Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
    #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Specifies the kernel type to be used in the algorithm.
    #     # 'degree': [2, 3, 4],  # Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
    #     'gamma': ["scale", "auto"],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    #     # 'coef0': [0.0, 0.1, 0.5],  # Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    #     # 'shrinking': [True, False],  # Whether to use the shrinking heuristic.
    #     # 'probability': [True, False],  # Whether to enable probability estimates.
    #     # 'tol': [1e-3, 1e-4],  # Tolerance for stopping criterion.
    #     # 'class_weight': [None, 'balanced'],  # Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one.
    #     # 'decision_function_shape': ['ovo', 'ovr'],  # Whether to return a one-vs-rest ('ovr') decision function of shape (n_samples, n_classes) or the original one-vs-one ('ovo') decision function.
    # }
    
    param_grid = {
        'C': [0.1, 1, 10],  # Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
        'kernel': ['linear', 'rbf', 'poly'],  # Specifies the kernel type to be used in the algorithm.
        'gamma': ["scale", "auto"],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    }


    svc = SVC()
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring=scoring, refit="accuracy", cv=kfold)

    grid_search.fit(X, y)   
    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    with open(f"log/{model_version}_{args.embedding}_svc.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'
            
            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]
            
            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")
    
    svc_best_model = grid_search.best_estimator_
    with open(f"weight/{model_version}_{args.embedding}_svc.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)

def train_mlp(model_version, raw_data_path, label, args):  
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplemented
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    param_grid = {
        'mlpclassifier__hidden_layer_sizes': [(128, 32,), (64, 64,)],
        'mlpclassifier__alpha': [0.0001, 0.001, 0.01],
        'mlpclassifier__solver': ['adam'],
        'mlpclassifier__learning_rate_init': [.005, .001]
    }
    
    mlp = MLPClassifier(max_iter=10000, verbose=10, random_state=12)
    
    pipeline = Pipeline([
        ('standardscaler', StandardScaler()),
        ('mlpclassifier', mlp)
    ])
    
    # pipeline = make_pipeline(StandardScaler(), mlp)
    
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring, refit="accuracy", cv=kfold, n_jobs=-1)
    
    grid_search.fit(X, y)
    
    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    with open(f"log/{model_version}_{args.embedding}_mlp.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'
            
            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]
            
            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")
    
    svc_best_model = grid_search.best_estimator_
    with open(f"weight/{model_version}_{args.embedding}_mlp.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)

def train_knn(model_version, raw_data_path, label, args):
    
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplementedError
    
    knn = KNeighborsClassifier()
    
    if args.resample:
        # smote = SMOTE(random_state=42)
        ros = RandomOverSampler(random_state=42)
        pipeline = IMBPipeline(steps=[('ros', ros), ('knn', knn)])
    else:
        # Create a pipeline with just KNN
        pipeline = Pipeline(steps=[('knn', knn)])
        
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],  
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']}
    
    param_grid = {f'knn__{key}': value for key, value in param_grid.items()}
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring=scoring, refit="accuracy", verbose=1)
    grid_search.fit(X, y)
    
    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    if args.resample:
        prefix = "resampled_"
    else:
        prefix = ""
    with open(f"log/{prefix}{model_version}_{args.embedding}_knn.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'
            
            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]
            
            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")
    
    svc_best_model = grid_search.best_estimator_
    with open(f"weight/{prefix}{model_version}_{args.embedding}_knn.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)

def train_LogisticRegression(model_version, raw_data_path, label, args):
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplemented
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {'C': np.logspace(-4, 4, 20), 'solver': ['liblinear', 'saga'], 'penalty': ['l1', 'l2']}

    logreg = LogisticRegression(max_iter=1000)
    grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, scoring=scoring, refit="accuracy", cv=kfold)
    grid_search.fit(X_scaled, y)

    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    with open(f"log/{model_version}_{args.embedding}_LogisticRegression.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'
            
            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]
            
            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")
    
    svc_best_model = grid_search.best_estimator_
    with open(f"weight/{model_version}_{args.embedding}_LogisticRegression.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)

def train_RidgeClassifier(model_version, raw_data_path, label, args):
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplemented
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    param_grid = {'alpha': np.logspace(-6, 6, 13)}

    ridge = RidgeClassifier()
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring=scoring, refit="accuracy", cv=kfold)
    grid_search.fit(X, y)

    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    with open(f"log/{model_version}_{args.embedding}_RidgeClassifier.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'
            
            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]
            
            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")
    
    svc_best_model = grid_search.best_estimator_
    with open(f"weight/{model_version}_{args.embedding}_RidgeClassifier.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)
    
def train_RandomForestClassifier(model_version, raw_data_path, label, args):
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplemented
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # param_grid = {
    # 'n_estimators': [50, 100, 200],  # 树的数量
    # 'max_features': ['auto', 'sqrt', 'log2'],  # 在分裂节点时考虑的特征数量
    # 'max_depth': [None, 10, 20, 30],  # 树的最大深度
    # 'min_samples_split': [2, 5, 10],  # 分裂内部节点所需的最小样本数
    # 'min_samples_leaf': [1, 2, 4],  # 在叶节点处需要的最小样本数
    # 'bootstrap': [True, False]  # 是否使用bootstrap采样
    # }
    
    param_grid = {
        'n_estimators': [50, 100],  # 树的数量
        'max_features': ['sqrt', 'log2'],  # 在分裂节点时考虑的特征数量
        'max_depth': [10, 20, 30],  # 树的最大深度
    }

    rf_classifier = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=kfold, scoring=scoring, refit="accuracy", n_jobs=-1)
    grid_search.fit(X, y)

    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    with open(f"log/{model_version}_{args.embedding}_RandomForestClassifier.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'
            
            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]
            
            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")
    
    svc_best_model = grid_search.best_estimator_

    with open(f"weight/{model_version}_{args.embedding}_RandomForestClassifier.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)

def train_GradientBoostingClassifier(model_version, raw_data_path, label, args):
    if args.embedding == "unimol_no_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=True)
    elif args.embedding == "unimol_all_h":
        X, y = get_UniMol_data_loader(raw_data_path=raw_data_path, label=label, remove_hydrogen=False)
    elif args.embedding == "gin":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gin")
    elif args.embedding == "gcn":
        X, y = get_MolCLR_data_loader(raw_data_path=raw_data_path, label=label, model="gcn")
    else:
        raise NotImplemented
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # param_grid = {
    #     'n_estimators': [100, 200, 300],  # 建立的弱学习器的数量
    #     'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    #     'max_depth': [3, 4, 5],  # 每个决策树的最大深度
    #     'min_samples_split': [2, 3, 4],  # 分裂内部节点所需的最小样本数
    #     'min_samples_leaf': [1, 2, 3],  # 在叶节点处需要的最小样本数
    #     'max_features': [None, 'sqrt', 'log2'],  # 寻找最佳分割时要考虑的特征数量
    #     'subsample': [0.8, 0.9, 1.0]  # 用于拟合各个基础学习器的样本比例
    # }
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
    }

    gb_classifier = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=gb_classifier, param_grid=param_grid, cv=kfold, scoring=scoring, refit="accuracy", n_jobs=-1)
    grid_search.fit(X, y)

    cv_results = grid_search.cv_results_

    best_index = grid_search.best_index_
    with open(f"log/{model_version}_{args.embedding}_GradientBoostingClassifier.log", "a") as file:
        file.write("Best Parameter Combination's Scores:\n")
        for scorer in scoring.keys():
            mean_metric_key = f'mean_test_{scorer}'
            std_metric_key = f'std_test_{scorer}'
            
            mean_score = cv_results[mean_metric_key][best_index]
            std_score = cv_results[std_metric_key][best_index]
            
            file.write(f"{scorer.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n")
    
    svc_best_model = grid_search.best_estimator_

    with open(f"weight/{model_version}_{args.embedding}_GradientBoostingClassifier.pkl", "wb") as file:
        pkl.dump(svc_best_model, file)
    
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

def get_UniMol_embedding(remove_hydrogen, smiles_list:list, device):
    data_loader = convert_smiles_list_to_data_loader(remove_hydrogen, smiles_list=smiles_list)
    assert remove_hydrogen is not None
    
    model = UniMolEncoder(device=device, remove_hydrogen=remove_hydrogen) # without shuffle
    model.to(device)
    
    representations = []
    smiles = []
    
    for batch in tqdm(data_loader):
        batch_smiles, batch_representations = model(batch)
        representations.append(batch_representations)
        smiles.extend(batch_smiles)
    
    return smiles, torch.cat(representations, dim=0)

def get_MolCLR_embedding(smiles_list:list, model_choice, device):
    dataset = MolTestDatasetWrapper_smiles(smiles_list=smiles_list)
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
    
    representations = []
    smiles = []
    for batch in tqdm(dataloader):
        batch.to(device)
        batch_smiles, batch_representations = model(batch)
        representations.append(batch_representations)
        smiles.extend(batch_smiles)
        
    return smiles, torch.cat(representations, dim=0)

def user_single_smiles_predict(smiles:str, device, embedding: dict, trained_model: dict):
    """
    get embedding, load trained_model, return prediction (direct & rule-based prediction of accum)
    e.g., embedding = {"Formal charge": "gin", "Q_vsa_Ppos": "gin", "vsa_don": "gin", "PA Accum": "gin"}
    trained_model = {"Formal charge": "knn", "Q_vsa_Ppos": "knn", "vsa_don": "knn", "PA Accum": "knn"}
    """
    assert (i in embedding for i in ["Formal charge", "Q_vsa_Ppos", "vsa_don", "PA Accum"])
    assert (i in trained_model for i in ["Formal charge", "Q_vsa_Ppos", "vsa_don", "PA Accum"])
    X = {}
    unembeddable_smiles = {}
    embeddable_smiles = {}
    
    for label, embed_type in embedding.items():
        if embed_type == "gin":
            _, X[label] = get_MolCLR_embedding(smiles_list=[smiles], model_choice="gin", device=device)
        elif embed_type == "gcn":
            _, X[label] = get_MolCLR_embedding(smiles_list=[smiles], model_choice="gcn", device=device)
        elif embed_type == "unimol_no_h":
            _, X[label] = get_UniMol_embedding(remove_hydrogen=True, smiles_list=[smiles], device=device)
        elif embed_type == "unimol_all_h":
            _, X[label] = get_UniMol_embedding(remove_hydrogen=False, smiles_list=[smiles], device=device)
        elif embedding == "mordred":
            raise NotImplementedError
    

    with open(f'weight/Formal charge Class_{embedding["Formal charge"]}_{trained_model["Formal charge"]}.pkl', 'rb') as Formal_charge_model_file:
        Formal_charge_model = pkl.load(Formal_charge_model_file)
    with open(f'weight/Q_vsa_Ppos Class_{embedding["Q_vsa_Ppos"]}_{trained_model["Q_vsa_Ppos"]}.pkl', 'rb') as Q_vsa_Ppos_model_file:
        Q_vsa_Ppos_model = pkl.load(Q_vsa_Ppos_model_file)   
    with open(f'weight/vsa_don Class_{embedding["vsa_don"]}_{trained_model["vsa_don"]}.pkl', 'rb') as vsa_don_model_file:
        vsa_don_model = pkl.load(vsa_don_model_file)
    with open(f'weight/PA Accum Class_{embedding["PA Accum"]}_{trained_model["PA Accum"]}.pkl', 'rb') as PA_Accum_model_file:
        PA_Accum_model = pkl.load(PA_Accum_model_file)
    
    Formal_charge_prediction = Formal_charge_model.predict(X["Formal charge"].cpu().detach().numpy()).item()
    Q_vsa_Ppos_prediction = Q_vsa_Ppos_model.predict(X["Q_vsa_Ppos"].cpu().detach().numpy()).item()
    vsa_don_prediction = vsa_don_model.predict(X["vsa_don"].cpu().detach().numpy()).item()
    rule_PA_Accum_prediction = Formal_charge_prediction and (Q_vsa_Ppos_prediction or vsa_don_prediction)
    direct_PA_Accum_prediction = PA_Accum_model.predict(X["PA Accum"].cpu().detach().numpy()).item()
    return Formal_charge_prediction, Q_vsa_Ppos_prediction, vsa_don_prediction, rule_PA_Accum_prediction, direct_PA_Accum_prediction

def user_smiles_list_predict(smiles_list_path:str, device, embedding: dict, trained_model: dict):
    Formal_charge_prediction = {}
    Q_vsa_Ppos_prediction = {}
    vsa_don_prediction = {}
    rule_PA_Accum_prediction = {}
    direct_PA_Accum_prediction = {}
    
    df = pd.read_csv(smiles_list_path)

    if "SMILES" in df.columns:
        smiles_list = df["SMILES"].tolist()
    else:
        print("The CSV file does not contain a 'smiles' column.")
        sys.exit()
    
    
    assert (i in embedding for i in ["Formal charge", "Q_vsa_Ppos", "vsa_don", "PA Accum"])
    assert (i in trained_model for i in ["Formal charge", "Q_vsa_Ppos", "vsa_don", "PA Accum"])
    X = {}
    
    for label, embed_type in embedding.items():
        if embed_type == "gin":
            _, X[label] = get_MolCLR_embedding(smiles_list=smiles_list, model_choice="gin", device=device)
        elif embed_type == "gcn":
            _, X[label] = get_MolCLR_embedding(smiles_list=smiles_list, model_choice="gcn", device=device)
        elif embed_type == "unimol_no_h":
            _, X[label] = get_UniMol_embedding(remove_hydrogen=True, smiles_list=smiles_list, device=device)
        elif embed_type == "unimol_all_h":
            _, X[label] = get_UniMol_embedding(remove_hydrogen=False, smiles_list=smiles_list, device=device)
        elif embedding == "mordred":
            raise NotImplementedError
    
    with open(f'weight/Formal charge Class_{embedding["Formal charge"]}_{trained_model["Formal charge"]}.pkl', 'rb') as Formal_charge_model_file:
        Formal_charge_model = pkl.load(Formal_charge_model_file)
    with open(f'weight/Q_vsa_Ppos Class_{embedding["Q_vsa_Ppos"]}_{trained_model["Q_vsa_Ppos"]}.pkl', 'rb') as Q_vsa_Ppos_model_file:
        Q_vsa_Ppos_model = pkl.load(Q_vsa_Ppos_model_file)   
    with open(f'weight/vsa_don Class_{embedding["vsa_don"]}_{trained_model["vsa_don"]}.pkl', 'rb') as vsa_don_model_file:
        vsa_don_model = pkl.load(vsa_don_model_file)
    with open(f'weight/PA Accum Class_{embedding["PA Accum"]}_{trained_model["PA Accum"]}.pkl', 'rb') as PA_Accum_model_file:
        PA_Accum_model = pkl.load(PA_Accum_model_file)
    
    Formal_charge_prediction = Formal_charge_model.predict(X["Formal charge"].cpu().detach().numpy())
    Q_vsa_Ppos_prediction = Q_vsa_Ppos_model.predict(X["Q_vsa_Ppos"].cpu().detach().numpy())
    vsa_don_prediction = vsa_don_model.predict(X["vsa_don"].cpu().detach().numpy())
    rule_PA_Accum_prediction = Formal_charge_prediction & (Q_vsa_Ppos_prediction | vsa_don_prediction)
    direct_PA_Accum_prediction = PA_Accum_model.predict(X["PA Accum"].cpu().detach().numpy())
    
    return smiles_list, Formal_charge_prediction, Q_vsa_Ppos_prediction, vsa_don_prediction, rule_PA_Accum_prediction, direct_PA_Accum_prediction
    
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="classification", type=str, choices=["regression", "classification"])
    parser.add_argument("--raw_data_path", default="data/raw/SMILES_accum_S2.csv", type=str, help="raw_data_path")
    parser.add_argument("--label", default="PA Accum Class", choices=["Formal charge Class", "Q_vsa_Ppos Class", "vsa_don Class", "PA Accum Class"], type=str, help="label")
    parser.add_argument("--gpu", default="cuda:4", type=str)
    # parser.add_argument("--remove_hydrogen", default=False, type=bool)
    parser.add_argument("--embedding", default="unimol_no_h", type=str, choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    # parser.add_argument("--molclr_model", default="gcn", type=str, choices=["gin", "gcn"])
    parser.add_argument("--resample", default=True, type=bool)
    parser.add_argument("--resample_method", default="", type=str, choices=["RandomOverSampler", "SMOTE"])
    parser.add_argument("--classifier", default="GradientBoostingClassifier", type=str, 
                        choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier", 
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    
    # args for user prediction
    parser.add_argument("--user_single_smiles_prediction", default=True, type=bool)
    parser.add_argument("--user_smiles_list_prediction", default=True, type=bool)
    parser.add_argument("--user_smiles_list", default="user_data/user_smiles_list.csv", type=str)
    parser.add_argument("--user_single_smiles", default="O=C1[C@@H](C)[C@H]2[C@@H](O1)c1c(C)c(C[NH3+])cc(C)c1CC2", type=str)
    parser.add_argument("--Formal_charge_embedding", default="gin", type=str, choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    parser.add_argument("--Q_vsa_Ppos_embedding", default="gcn", type=str, choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    parser.add_argument("--vsa_don_embedding", default="unimol_no_h", type=str, choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    parser.add_argument("--PA_Accum_embedding", default="unimol_all_h", type=str, choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    parser.add_argument("--Formal_charge_model", default="knn", type=str, choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier", 
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    parser.add_argument("--Q_vsa_Ppos_model", default="knn", type=str, choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier", 
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    parser.add_argument("--vsa_don_model", default="knn", type=str, choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier", 
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    parser.add_argument("--PA_Accum_model", default="knn", type=str, choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier", 
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    
    args = parser.parse_args()
    set_random_seed(1024)
    if torch.cuda.is_available() and args.gpu != "cpu":
        args.device = args.gpu
        torch.cuda.set_device(args.device)
    else:
        args.device = "cpu"
        
    model_version = args.label
    
    # ------------------------------------- featurization and embedding --------------------------------------
    # calculate_3D_structure(raw_data_path=args.raw_data_path)
    # generate_upsampled_raw_data(raw_data_path=args.raw_data_path, target_column="Formal charge Class", minority_label=0)
    # generate_upsampled_raw_data(raw_data_path=args.raw_data_path, target_column="Q_vsa_Ppos Class", minority_label=1)
    # generate_upsampled_raw_data(raw_data_path=args.raw_data_path, target_column="vsa_don Class", minority_label=0)
    # generate_upsampled_raw_data(raw_data_path=args.raw_data_path, target_column="PA Accum Class", minority_label=0)
    
    # construct_data_list(raw_data_path=args.raw_data_path, label=args.label)
    # save_UniMol_embedding(remove_hydrogen=args.remove_hydrogen, raw_data_path=args.raw_data_path, device=args.device)
    # save_MolCLR_embedding(raw_data_path=args.raw_data_path, label=args.label, task=args.task, 
                        #   model_choice=args.molclr_model, device=args.gpu)
    
    
    # # ---------------------------------------- training ----------------------------------------------
    # if args.classifier == "svc":
    #     train_svc(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args) # 0.698, with h; accum: 0.665 (unimol), 0.668 (gin)
    # elif args.classifier == "mlp":
    #     train_mlp(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args) # accum: 0.58 (unimol), 0.68 (gins), ★0.70 (gcn) 
    # elif args.classifier == "knn":
    #     train_knn(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args) # # 0.694; accum: 0.669 (unimol), 0.659 (gin)， ★0.701 (gcn)
    # elif args.classifier == "LogisticRegression":
    #     train_LogisticRegression(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args)
    # elif args.classifier == "RidgeClassifier":
    #     train_RidgeClassifier(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args)
    # elif args.classifier == "RandomForestClassifier":
    #     train_RandomForestClassifier(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args)
    # elif args.classifier == "GradientBoostingClassifier":
    #     train_GradientBoostingClassifier(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args)
    # print("training: All is well!")
    
    # evaluation of rule-based prediction of PA Accumulation
    rule_based_embedding = {"Formal charge Class":"gcn", "Q_vsa_Ppos Class":"gin", "vsa_don Class":"gin", "PA Accum Class":"gcn"}
    rule_based_model = {"Formal charge Class":"mlp", "Q_vsa_Ppos Class":"mlp", "vsa_don Class":"mlp", "PA Accum Class":"mlp"}
    rule_based_Accum_eval(raw_data_path=args.raw_data_path, rule_based_embedding=rule_based_embedding, rule_based_model=rule_based_model)
    
    
    # -------------------------------------- user single smiles prediction -----------------------------------------
    # embedding = {"Formal charge": args.Formal_charge_embedding, "Q_vsa_Ppos": args.Q_vsa_Ppos_embedding, 
    #              "vsa_don": args.vsa_don_embedding, "PA Accum": args.PA_Accum_embedding}
    # trained_model = {"Formal charge": args.Formal_charge_model, "Q_vsa_Ppos": args.Q_vsa_Ppos_model, 
    #                  "vsa_don": args.vsa_don_model, "PA Accum": args.PA_Accum_model}
    # (Formal_charge_prediction, Q_vsa_Ppos_prediction, vsa_don_prediction, 
    #  rule_PA_Accum_prediction, direct_PA_Accum_prediction) = user_single_smiles_predict(smiles=args.user_single_smiles, device=args.device, 
    #                                                                       embedding=embedding, trained_model=trained_model)
    # print(f"prediction of {args.user_smiles}:")
    # if Formal_charge_prediction:
    #     print("Formal charge >= 0.98")
    # else:
    #     print("Formal charge < 0.98")
    # if Q_vsa_Ppos_prediction:
    #     print("Q_vsa_Ppos >= 80")
    # else:
    #     print("Q_vsa_Ppos < 80")
    # if vsa_don_prediction:
    #     print("vsa_don >= 23")
    # else:
    #     print("vsa_don < 23")
    # if rule_PA_Accum_prediction:
    #     print("rule-base prediction of PA Accumulation [(vsa_don >= 23) && ((Q_vsa_Ppos >= 80) || (Formal Charge >= 0.98))]: Yes")
    # else:
    #     print("rule-base prediction of PA Accumulation [(vsa_don >= 23) && ((Q_vsa_Ppos >= 80) || (Formal Charge >= 0.98))]: No")
    # if direct_PA_Accum_prediction:
    #     print("direct prediction of PA Accumulation: Yes")
    # else:
    #     print("direct prediction of PA Accumulation: No")
    
    # ------------------------------------ user smiles list prediction ----------------------------------
    # embedding = {"Formal charge": args.Formal_charge_embedding, "Q_vsa_Ppos": args.Q_vsa_Ppos_embedding, 
    #              "vsa_don": args.vsa_don_embedding, "PA Accum": args.PA_Accum_embedding}
    # trained_model = {"Formal charge": args.Formal_charge_model, "Q_vsa_Ppos": args.Q_vsa_Ppos_model, 
    #                  "vsa_don": args.vsa_don_model, "PA Accum": args.PA_Accum_model}
    # (smiles_list, Formal_charge_prediction, Q_vsa_Ppos_prediction, vsa_don_prediction, 
    #  rule_PA_Accum_prediction, direct_PA_Accum_prediction) = user_smiles_list_predict(smiles_list_path=args.user_smiles_list, device=args.device, 
    #                                                                       embedding=embedding, trained_model=trained_model)
    
    
        
    
    
    
    
    

