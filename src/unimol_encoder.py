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
    
