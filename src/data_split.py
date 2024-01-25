import yaml
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from typing import List
from sklearn.model_selection import train_test_split

def _generate_scaffold(smile, include_chirality=False):
    mol = Chem.MolFromSmiles(smile)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def generate_scaffolds(data:pd.DataFrame, log_every_n=10):
    scaffolds = {}
    data_len = data.shape[0]

    print("About to generate scaffolds")
    for ind, smile in enumerate(data["smiles"]):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smile)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]
    
    print("Number of scaffold sets: %d" % len(scaffold_sets))
    for i, scaffold_set in enumerate(scaffold_sets):
        print("Scaffold set %d: %d molecules" % (i, len(scaffold_set)))

    return scaffold_sets


def scaffold_split(data:pd.DataFrame, valid_size, test_size):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(data)

    train_cutoff = train_size * data.shape[0]
    valid_cutoff = (train_size + valid_size) * data.shape[0]
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
            
    if "scaffold_dataset_type" in data.columns:
        data = data.drop("scaffold_dataset_type", axis=1)
    data.insert(loc=0, column="scaffold_dataset_type", value=None)
    
    for train_ind in train_inds:
        data.loc[train_ind, 'scaffold_dataset_type'] = 'train'
    for valid_ind in valid_inds:
        data.loc[valid_ind, 'scaffold_dataset_type'] = 'validate'    
    for test_ind in test_inds:
        data.loc[test_ind, 'scaffold_dataset_type'] = 'test'    
        
    return data

def random_split(data:pd.DataFrame, valid_size, test_size, random_seed=None):
    inds = list(range(data.shape[0]))
    train_size = 1 - valid_size - test_size
    train_inds, remaining_inds = train_test_split(inds, test_size=(1 - train_size), random_state=random_seed)
    test_ratio = test_size / (valid_size + test_size)
    valid_inds, test_inds = train_test_split(remaining_inds, test_size=test_ratio, random_state=random_seed)
    
    if "random_dataset_type" in data.columns:
        data = data.drop("random_dataset_type", axis=1)    
    data.insert(loc=0, column="random_dataset_type", value=None)
    
    for train_ind in train_inds:
        data.loc[train_ind, 'random_dataset_type'] = 'train'
    for valid_ind in valid_inds:
        data.loc[valid_ind, 'random_dataset_type'] = 'validate'    
    for test_ind in test_inds:
        data.loc[test_ind, 'random_dataset_type'] = 'test' 
        
    return data



if __name__ == "__main__":
    data_path = "data/raw/labeled-20240102_merged_descriptors_normalized_descriptors_mean0_std1.csv"
    valid_size = 0.15
    test_size = 0.15
    random_seed = 42
    data = pd.read_csv(data_path)
    data = scaffold_split(data, valid_size=valid_size, test_size=test_size)
    data = random_split(data, valid_size=valid_size, test_size=test_size, random_seed=random_seed)
        
    data.to_csv(data_path, index=False)