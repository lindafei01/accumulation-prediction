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
from training import train_svc, train_mlp, train_knn, train_LogisticRegression, train_RidgeClassifier, train_RandomForestClassifier, train_GradientBoostingClassifier
from unimol_encoder import UniMolEncoder

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
    dataloader, embeddable = dataset.get_fulldata_loader()
    if not all(not value for value in embeddable.values()):
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
        
        return embeddable, smiles, torch.cat(representations, dim=0)

    else:
        return embeddable, None, None



def user_single_smiles_predict(smiles:str, device, embedding: dict, trained_model: dict):
    """
    get embedding, load trained_model, return prediction (direct & rule-based prediction of accum)
    e.g., embedding = {"Formal charge": "gin", "Q_vsa_Ppos": "gin", "vsa_don": "gin", "PA Accum": "gin"}
    trained_model = {"Formal charge": "knn", "Q_vsa_Ppos": "knn", "vsa_don": "knn", "PA Accum": "knn"}
    """
    assert (i in embedding for i in ["Formal charge", "Q_vsa_Ppos", "vsa_don", "PA Accum"])
    assert (i in trained_model for i in ["Formal charge", "Q_vsa_Ppos", "vsa_don", "PA Accum"])
    X = {}
    embeddable = True
    
    for label, embed_type in embedding.items():
        if embed_type == "gin":
            embeddable, _, X[label] = get_MolCLR_embedding(smiles_list=[smiles], model_choice="gin", device=device)
            if embeddable[smiles] == False:
                return None, None, None, None, None, None
        elif embed_type == "gcn":
            embeddable, _, X[label] = get_MolCLR_embedding(smiles_list=[smiles], model_choice="gcn", device=device)
            if embeddable[smiles] == False:
                return None, None, None, None, None, None
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
            embeddable, _, X[label] = get_MolCLR_embedding(smiles_list=smiles_list, model_choice="gin", device=device)
        elif embed_type == "gcn":
            embeddable, _, X[label] = get_MolCLR_embedding(smiles_list=smiles_list, model_choice="gcn", device=device)
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
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="classification", type=str, choices=["regression", "classification"])
    parser.add_argument("--raw_data_path", default="data/raw/SMILES_accum_S2.csv", type=str, help="raw_data_path")
    parser.add_argument("--label", default="vsa_don Class", choices=["Formal charge Class", "Q_vsa_Ppos Class", "vsa_don Class", "PA Accum Class"], type=str, help="label")
    parser.add_argument("--gpu", default="cuda:4", type=str)
    # parser.add_argument("--remove_hydrogen", default=False, type=bool)
    parser.add_argument("--embedding", default="unimol_no_h", type=str, choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    # parser.add_argument("--molclr_model", default="gcn", type=str, choices=["gin", "gcn"])
    parser.add_argument("--resample", default=True, type=bool)
    parser.add_argument("--resample_method", default="", type=str, choices=["RandomOverSampler", "SMOTE"])
    parser.add_argument("--classifier", default="RandomForestClassifier", type=str, 
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
    if args.classifier == "svc":
        train_svc(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args) # 0.698, with h; accum: 0.665 (unimol), 0.668 (gin)
    elif args.classifier == "mlp":
        train_mlp(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args) # accum: 0.58 (unimol), 0.68 (gins), ★0.70 (gcn) 
    elif args.classifier == "knn":
        train_knn(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args) # # 0.694; accum: 0.669 (unimol), 0.659 (gin)， ★0.701 (gcn)
    elif args.classifier == "LogisticRegression":
        train_LogisticRegression(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args)
    elif args.classifier == "RidgeClassifier":
        train_RidgeClassifier(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args)
    elif args.classifier == "RandomForestClassifier":
        train_RandomForestClassifier(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args)
    elif args.classifier == "GradientBoostingClassifier":
        train_GradientBoostingClassifier(model_version=model_version, raw_data_path=args.raw_data_path, label=args.label, args=args)
    print("training: All is well!")
    
    # evaluation of rule-based prediction of PA Accumulation
    # rule_based_embedding = {"Formal charge Class":"gcn", "Q_vsa_Ppos Class":"gin", "vsa_don Class":"gin", "PA Accum Class":"gcn"}
    # rule_based_model = {"Formal charge Class":"mlp", "Q_vsa_Ppos Class":"mlp", "vsa_don Class":"mlp", "PA Accum Class":"mlp"}
    # rule_based_Accum_eval(raw_data_path=args.raw_data_path, rule_based_embedding=rule_based_embedding, rule_based_model=rule_based_model)
    
    
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

def get_predicted_properties(smiles):
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_single_smiles_prediction", default=True, type=bool)
    # parser.add_argument("--user_smiles_list_prediction", default=False, type=bool)
    # parser.add_argument("--user_smiles_list", default="user_data/user_smiles_list.csv", type=str)
    parser.add_argument("--gpu", default="cuda:4", type=str)
    parser.add_argument("--user_single_smiles", type=str)
    parser.add_argument("--Formal_charge_embedding", default="gcn", type=str, choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    parser.add_argument("--Q_vsa_Ppos_embedding", default="gin", type=str, choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    parser.add_argument("--vsa_don_embedding", default="gin", type=str, choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    parser.add_argument("--PA_Accum_embedding", default="gcn", type=str, choices=["unimol_no_h", "unimol_all_h", "gin", "gcn", "mordred"])
    parser.add_argument("--Formal_charge_model", default="mlp", type=str, choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier", 
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    parser.add_argument("--Q_vsa_Ppos_model", default="mlp", type=str, choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier", 
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    parser.add_argument("--vsa_don_model", default="mlp", type=str, choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier", 
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    parser.add_argument("--PA_Accum_model", default="mlp", type=str, choices=["svc", "mlp", "knn", "LogisticRegression", "RidgeClassifier", 
                                 "RandomForestClassifier", "GradientBoostingClassifier"])
    
    args = parser.parse_args()
    set_random_seed(1024)
    if torch.cuda.is_available() and args.gpu != "cpu":
        args.device = args.gpu
        torch.cuda.set_device(args.device)
    else:
        args.device = "cpu"
        
        
    args.user_single_smiles = smiles
    embedding = {"Formal charge": args.Formal_charge_embedding, "Q_vsa_Ppos": args.Q_vsa_Ppos_embedding, 
                 "vsa_don": args.vsa_don_embedding, "PA Accum": args.PA_Accum_embedding}
    trained_model = {"Formal charge": args.Formal_charge_model, "Q_vsa_Ppos": args.Q_vsa_Ppos_model, 
                     "vsa_don": args.vsa_don_model, "PA Accum": args.PA_Accum_model}
    (Formal_charge_prediction, Q_vsa_Ppos_prediction, vsa_don_prediction, 
     rule_PA_Accum_prediction, direct_PA_Accum_prediction) = user_single_smiles_predict(smiles=args.user_single_smiles, device=args.device, 
                                                                          embedding=embedding, trained_model=trained_model)
    
    if Formal_charge_prediction is None:
        return {
            "Formal Charge": None,
            "Q_vsa_Ppos": None,
            "vsa_don": None,
            "PA Accumulation": None
        }
    
    else: 
        if Formal_charge_prediction:
            Foraml_charge_prompt = ">= 0.98"
        else:
            Foraml_charge_prompt = "< 0.98"
        if Q_vsa_Ppos_prediction:
            Q_vsa_Ppos_prompt = ">= 80"
        else:
            Q_vsa_Ppos_prompt = "< 80"
        if vsa_don_prediction:
            vsa_don_prompt = ">= 23"
        else:
            vsa_don_prompt = "< 23"
        if rule_PA_Accum_prediction:
            print("rule-base prediction of PA Accumulation [(vsa_don >= 23) && ((Q_vsa_Ppos >= 80) || (Formal Charge >= 0.98))]: Yes")
        else:
            print("rule-base prediction of PA Accumulation [(vsa_don >= 23) && ((Q_vsa_Ppos >= 80) || (Formal Charge >= 0.98))]: No")
        if direct_PA_Accum_prediction:
            direct_PA_Accum_prompt = "Accumalation: Yes"
        else:
            direct_PA_Accum_prompt = "Accumalation: No"
        
        return {
                "Formal Charge": Formal_charge_prediction,
                "Q_vsa_Ppos": Q_vsa_Ppos_prediction,
                "vsa_don": vsa_don_prediction,
                "PA Accumulation": direct_PA_Accum_prediction
            }
    
    
    
    
    

