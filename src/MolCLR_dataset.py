# Copyright (c) 2023 Shihao Ma, Haotian Cui, WangLab @ U of T

# This source code is modified from https://github.com/yuyangw/MolCLR 
# under MIT License. The original license is included below:
# ========================================================================
# MIT License

# Copyright (c) 2021 Yuyang Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import codecs
import os
import io
import csv
from typing import List, Optional
from typing_extensions import Literal
import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")


ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    if isinstance(data_path, str):
        csv_file = open(data_path)
    elif isinstance(data_path, io.IOBase):
        csv_file = data_path
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    for i, row in enumerate(csv_reader):
        smiles = row["SMILES"]
        smiles = codecs.decode(smiles, "unicode_escape")
        label = row[target]
        mol = Chem.MolFromSmiles(smiles)
        if mol != None and label != "" and smiles != "":
            smiles_data.append(smiles)
            if task == "classification":
                labels.append(int(label))
            elif task == "regression":
                labels.append(float(label))
            else:
                ValueError("task must be either regression or classification")
    print(len(smiles_data))
    return smiles_data, labels


def read_cols(data_path, cols, return_np=True):
    """
    Reads a csv file and returns the columns specified in cols.

    Args:
        data_path: path to csv file
        cols: list of column names to return
        return_np: if True, returns a numpy array instead of a list
    """
    df = pd.read_csv(data_path)
    if return_np:
        return df[cols].to_numpy()
    else:
        return df[cols].values.tolist()


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task
        
        self.conversion = 1
        if "qm9" in data_path and target in ["homo", "lumo", "gap", "zpve", "u0"]:
            self.conversion = 27.211386246
            print(target, "Unit conversion needed!")

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum())) 
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag())) 
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )
            edge_feat.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        if self.task == "classification":
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1, -1)
        elif self.task == "regression":
            y = torch.tensor(
                self.labels[index] * self.conversion, dtype=torch.float
            ).view(1, -1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, smiles=self.smiles_data[index])
            
        return data

    def __len__(self):
        return len(self.smiles_data)
    
    def len(self):
        r"""Returns the number of graphs stored in the dataset."""
        return len(self.smiles_data)
    
    def get(self, idx: int):
        r"""Gets the data object at index :obj:`idx`."""
        print('implement abstract method get')
        
class MolTestDatasetWrapper(object):
    def __init__(
        self,
        data_path,
        target: str,
        task: Literal["classification", "regression"],
        batch_size=16,
        num_workers=4,
    ):
        """
        Args:
            feature_cols: list of column names for additional graph features
        """
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.task = task


    @property
    def dataset(self):
        if not hasattr(self, "_dataset"):
            self._dataset = MolTestDataset(
                    data_path=self.data_path, target=self.target, task=self.task
                )
        return self._dataset

    @property
    def all_smiles(self):
        return self.dataset.smiles_data


    def get_fulldata_loader(self):
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return loader

class MolTestDataset_smiles(Dataset):
    def __init__(self, smiles_list: list):
        super(Dataset, self).__init__()
        self.smiles_data = smiles_list

    def __getitem__(self, index):
        
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum())) 
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag())) 
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )
            edge_feat.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=self.smiles_data[index])
            
        return data

    def __len__(self):
        return len(self.smiles_data)
    
    def len(self):
        r"""Returns the number of graphs stored in the dataset."""
        return len(self.smiles_data)
    
    def get(self, idx: int):
        r"""Gets the data object at index :obj:`idx`."""
        print('implement abstract method get')
        
class MolTestDatasetWrapper_smiles(object):
    def __init__(
        self,
        smiles_list:list,
        batch_size=1,
        num_workers=1
    ):
        """
        Args:
            feature_cols: list of column names for additional graph features
        """
        super(object, self).__init__()
        embeddable = {}
        smiles_filtered = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                smiles_filtered.append(smiles)
                embeddable[smiles] = True
            except:
                embeddable[smiles] = False
                continue
                
        self.smiles_list = smiles_filtered
        self.embeddable = embeddable
        self.batch_size = batch_size
        self.num_workers=1

    @property
    def dataset(self):
        if not hasattr(self, "_dataset"):
            self._dataset = MolTestDataset_smiles(
                    smiles_list=self.smiles_list
                )
        return self._dataset

    @property
    def all_smiles(self):
        return self.dataset.smiles_data


    def get_fulldata_loader(self):
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return loader, self.embeddable

