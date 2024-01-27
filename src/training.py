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
from utils import get_MolCLR_data_loader, get_UniMol_data_loader


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[2]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[3]

scoring = {'accuracy': 'accuracy', 'precision': 'precision', 
           'recall': 'recall', 'f1': 'f1',
           'tp': make_scorer(tp), 'fp': make_scorer(fp), 
           'tn': make_scorer(tn), 'fn': make_scorer(fn)}

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