import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import ipdb
import json
from joblib import Parallel, delayed
from transformers import BertTokenizer, BertTokenizerFast
import os
import math
import pandas as pd
def mask_num_value(N_train,N_val,N_test, ratio):#dtype = 'float'
    assert N_train.shape[1] == N_val.shape[1] == N_test.shape[1]
    train_size = N_train.shape[0]
    val_size = N_val.shape[0]
    num_list = np.arange(N_train.shape[1])

    
    # if mask ratio == 0.0, then do not mask any data
    if ratio == 0.0:
        return N_train, N_val, N_test
    
    # generate mask matrix
    X = np.concatenate([N_train, N_val, N_test], axis=0)
    mask = np.random.rand(*X.shape)
    X[mask < ratio] = np.nan
    
    # mask data of X
    for num_id in num_list:
        mean = np.nanmean(X[:train_size, num_id])
        X[np.isnan(X[:, num_id]), num_id] = mean

    return X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]

def mask_cat_value(C_train,C_val,C_test, ratio):#dtype = 'object'
    assert C_train.shape[1] == C_val.shape[1] == C_test.shape[1]
    train_size = C_train.shape[0]
    val_size = C_val.shape[0]
    cat_list = np.arange(C_train.shape[1])
    
    # if mask ratio == 0.0, then do not mask any data
    if ratio == 0.0:
        return C_train, C_val, C_test
    
    # generate mask matrix
    X = np.concatenate([C_train, C_val, C_test], axis=0)
    #some of the data is None, so we need to replace it with 'nan'
    X[X == None ] = 'nan'
    mask = np.random.rand(*X.shape)
    X[mask < ratio] = 'nan'
    return X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]


def noise_value(train,val,test, ratio):
    assert train.shape[1] == val.shape[1] == test.shape[1]
    train_size = train.shape[0]
    val_size = val.shape[0]
    feature_list = np.arange(train.shape[1])
    
    if ratio == 0.0:
        return train, val, test
    
    X = np.concatenate([train, val, test], axis=0)
    mask = np.random.rand(*X.shape)
    mask[mask < ratio] = 0
    mask[mask >= ratio] = 1
    for feature_id in feature_list:
        distribution = X[:, feature_id]
        mask_index = mask[:, feature_id] == 0

        random_index = np.random.choice(np.arange(X.shape[0]), mask_index.sum())

        X[mask_index, feature_id] = distribution[random_index]
    return X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]

def nan_to_mean(N_train,N_val,N_test):
    for i in range(N_train.shape[1]):
        mean = np.nanmean(N_train[:,i])
        N_train[:,i] = np.nan_to_num(N_train[:,i],nan=mean)
        N_val[:,i] = np.nan_to_num(N_val[:,i],nan=mean)
        N_test[:,i] = np.nan_to_num(N_test[:,i],nan=mean)
    return N_train,N_val,N_test

def load(dataname, info, normalization, args):
    sample_ratio = args.sample_ratio
    ## loading pretrain word2vectory
    if os.path.exists('../tokenizer'):
        tokenizer = BertTokenizerFast.from_pretrained('../tokenizer')
    else:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained('../tokenizer')
    tokenizer.__dict__['model_max_length'] = 512

    task_type, n_num_features, n_cat_features, train_size, val_size, test_size = info.get('task_type'), info.get('n_num_features'), info.get('n_cat_features'), info.get('train_size'), info.get('val_size'), info.get('test_size')

    assert task_type in ['binclass', 'multiclass', 'regression']

    # preprocessing

    if sample_ratio == 1.0:
        selected_rows = np.arange(train_size)
        val_selected_row = np.arange(val_size)
    else:
        selected_rows = np.random.choice(train_size, int(train_size * sample_ratio), replace=False)
        val_selected_row = np.random.choice(val_size, int(val_size * sample_ratio), replace=False)
    train_size = int(train_size * sample_ratio)
    val_size = int(val_size * sample_ratio)
    ## numerical features
    N_train, N_val, N_test = np.load(f'../data/{dataname}/N_train.npy', allow_pickle=True).astype('float32'), np.load(f'../data/{dataname}/N_val.npy', allow_pickle=True).astype('float32'), np.load(f'../data/{dataname}/N_test.npy', allow_pickle=True).astype('float32')
    N_train = N_train[selected_rows, :]
    N_val = N_val[val_selected_row, :]

    # process missing value
    N_train,N_val,N_test = nan_to_mean(N_train,N_val,N_test)

    #process:noise value or mask value
    if args.process_type == 'mask':
        N_train,N_val,N_test = mask_num_value(N_train,N_val,N_test, args.process_ratio)
    elif args.process_type == 'noise':
        N_train,N_val,N_test = noise_value(N_train,N_val,N_test, args.process_ratio)


    N = np.concatenate([N_train, N_val, N_test], axis=0).astype('float32')

    
    ### feature-wise normalize
    if normalization == 'standard':
        preprocess = sklearn.preprocessing.StandardScaler().fit(N[:train_size])
    elif normalization == 'minmax':
        preprocess = sklearn.preprocessing.MinMaxScaler().fit(N[:train_size])
    elif normalization == 'quantile':
        preprocess = sklearn.preprocessing.QuantileTransformer(output_distribution='normal').fit(N[:train_size])
    elif normalization == 'none':
        preprocess = sklearn.preprocessing.FunctionTransformer().fit(N[:train_size])
    ### N: (bs, cols)
    N = preprocess.transform(N)


    ## catergorical features
    if n_cat_features != 0:
        C_train, C_val, C_test = np.load(f'../data/{dataname}/C_train.npy', allow_pickle=True), np.load(f'../data/{dataname}/C_val.npy', allow_pickle=True), np.load(f'../data/{dataname}/C_test.npy', allow_pickle=True)
        C_train = C_train[selected_rows, :]
        C_val = C_val[val_selected_row, :]

        # process:mask value or noise value
        if args.process_type == 'mask':
            C_train,C_val,C_test = mask_cat_value(C_train,C_val,C_test, args.process_ratio)
        elif args.process_type == 'noise':
            C_train,C_val,C_test = noise_value(C_train,C_val,C_test, args.process_ratio)

        C = np.concatenate([C_train, C_val, C_test], axis=0)
        C = [sklearn.preprocessing.LabelEncoder().fit_transform(C[:,i]).astype('int64').reshape(-1,1) for i in range(C.shape[1])]
        C = np.concatenate(C, axis=1)

    else:
        C = None


    ## label
    y_train, y_val, y_test = np.load(f'../data/{dataname}/y_train.npy', allow_pickle=True), np.load(f'../data/{dataname}/y_val.npy', allow_pickle=True), np.load(f'../data/{dataname}/y_test.npy', allow_pickle=True)
    y_train = y_train[selected_rows]
    y_val = y_val[val_selected_row]
    
    Y = np.concatenate([y_train, y_val, y_test], axis=0)
    ### regression
    if task_type == 'regression':
        Y = Y.astype('float32')
    ### classification
    else:
        Y = sklearn.preprocessing.LabelEncoder().fit_transform(Y).astype('int64')

    n_classes = int(max(Y)) + 1 if task_type == 'multiclass' else None
    ### !!! CRUCIAL for neural networks when solving regression problems !!!
    if task_type == 'regression':
        y_mean = Y[:train_size].mean().item()
        y_std = Y[:train_size].std().item()
        Y = (Y - y_mean) / y_std
    elif task_type == 'binclass':
        y_mean = y_std = None
        Y = Y.astype('float32')
    elif task_type == 'multiclass':
        y_mean = y_std = None
        Y = Y.astype('int64')

    # generate train, val, test
    X = {}
    y = {}

    if n_cat_features != 0:
        X_all = np.concatenate([N,C], axis=1)
        # categories = np.max(C, axis=0) + 1
        # leave one for masking with the last per cat
        categories = np.max(C, axis=0) + 2
    else:
        X_all = N
        categories = None

    X['train'], X['val'], X['test'] = X_all[:train_size], X_all[train_size:train_size+val_size], X_all[-test_size:]
    y['train'], y['val'], y['test'] = Y[:train_size], Y[train_size:train_size+val_size], Y[-test_size:]

    X = {k: torch.tensor(v, dtype=torch.float).cpu() for k, v in X.items()}
    y = {k: torch.tensor(v).cpu() for k, v in y.items()}

    return X, y, n_classes, y_mean, y_std, categories


# def load_balanced(dataname, info, normalization, ratio):

#     ## loading pretrain word2vectory
#     if os.path.exists('../tokenizer'):
#         tokenizer = BertTokenizerFast.from_pretrained('../tokenizer')
#     else:
#         tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
#         tokenizer.save_pretrained('../tokenizer')
#     tokenizer.__dict__['model_max_length'] = 512

#     task_type, n_num_features, n_cat_features, train_size, val_size, test_size = info.get('task_type'), info.get('n_num_features'), info.get('n_cat_features'), info.get('train_size'), info.get('val_size'), info.get('test_size')

#     assert task_type in ['binclass', 'multiclass']

#     # preprocessing

#     if ratio == 1.0:
#         selected_rows = np.arange(train_size)
#         val_selected_row = np.arange(val_size)
#     else:
#         selected_rows = np.random.choice(train_size, int(train_size * ratio), replace=False)
#         val_selected_row = np.random.choice(val_size, int(val_size * ratio), replace=False)
#     train_size = int(train_size * ratio)
#     val_size = int(val_size * ratio)

#     ## numerical features
#     N_train, N_val, N_test = np.load(f'../data/{dataname}/N_train.npy', allow_pickle=True), np.load(f'../data/{dataname}/N_val.npy', allow_pickle=True), np.load(f'../data/{dataname}/N_test.npy', allow_pickle=True)
#     N_train = N_train[selected_rows, :]
#     N_val = N_val[val_selected_row, :]
#     ## catergorical features
#     if n_cat_features != 0:
#         C_train, C_val, C_test = np.load(f'../data/{dataname}/C_train.npy', allow_pickle=True), np.load(f'../data/{dataname}/C_val.npy', allow_pickle=True), np.load(f'../data/{dataname}/C_test.npy', allow_pickle=True)
#         C_train = C_train[selected_rows, :]
#         C_val = C_val[val_selected_row, :]
    


#     ## label
#     y_train, y_val, y_test = np.load(f'../data/{dataname}/y_train.npy', allow_pickle=True), np.load(f'../data/{dataname}/y_val.npy', allow_pickle=True), np.load(f'../data/{dataname}/y_test.npy', allow_pickle=True)
#     y_train = y_train[selected_rows]
#     y_val = y_val[val_selected_row]
    
    


#     #balance data
#     N_train_val = np.concatenate([N_train, N_val], axis=0)
#     if n_cat_features != 0:
#         C_train_val = np.concatenate([C_train, C_val], axis=0)
#     Y_train_val = np.concatenate([y_train, y_val], axis=0)

#     unique_y ,count = np.unique(Y_train_val, return_counts=True)
#     min_count = count.min()
#     mask = np.zeros(Y_train_val.shape[0])
#     for i in range(len(unique_y)):
#         indices = np.where(Y_train_val==unique_y[i])[0]
#         mask[np.random.choice(indices,min_count,replace=False)] = 1


#     N_train_val = N_train_val[mask==1]
#     if n_cat_features != 0:
#         C_train_val = C_train_val[mask==1]
#     Y_train_val = Y_train_val[mask==1]


#     #generate balance test
#     unique_y ,count = np.unique(y_test, return_counts=True)
#     min_count = count.min()
#     mask = np.zeros(y_test.shape[0])
#     for i in range(len(unique_y)):
#         indices = np.where(y_test==unique_y[i])[0]
#         mask[np.random.choice(indices,min_count,replace=False)] = 1
    

#     N_balanced_test = N_test[mask==1]
#     if n_cat_features != 0:
#         C_balanced_test = C_test[mask==1]
#     y_balanced_test = y_test[mask==1]
#     y_balanced_test_size = y_balanced_test.shape[0]

        
#     train_ratio = train_size/(train_size+val_size)
#     val_ratio = val_size/(train_size+val_size)
#     train_size = int(train_ratio*len(Y_train_val))
#     val_size = int(val_ratio*len(Y_train_val))
    
#     N_train = N_train_val[:train_size]
#     N_val = N_train_val[train_size:train_size+val_size]
#     if n_cat_features != 0:
#         C_train = C_train_val[:train_size]
#         C_val = C_train_val[train_size:train_size+val_size]
#     y_train = Y_train_val[:train_size]
#     y_val = Y_train_val[train_size:train_size+val_size]


#     # preprocessing  X and y
#     N = np.concatenate([N_train, N_val, N_test,N_balanced_test], axis=0).astype('float32')
#     ### it has nan input for higgs_small
#     N = np.nan_to_num(N)
#     ### feature-wise normalize
#     if normalization == 'standard':
#         preprocess = sklearn.preprocessing.StandardScaler().fit(N[:train_size])
#     elif normalization == 'minmax':
#         preprocess = sklearn.preprocessing.MinMaxScaler().fit(N[:train_size])
#     elif normalization == 'quantile':
#         preprocess = sklearn.preprocessing.QuantileTransformer(output_distribution='normal').fit(N[:train_size])
#     elif normalization == 'none':
#         preprocess = sklearn.preprocessing.FunctionTransformer().fit(N[:train_size])
#     ### N: (bs, cols)
#     N = preprocess.transform(N)


#     ## catergorical features
    
#     if n_cat_features != 0:
#         C = np.concatenate([C_train, C_val, C_test,C_balanced_test], axis=0)

#         C = [sklearn.preprocessing.LabelEncoder().fit_transform(C[:,i]).astype('int64').reshape(-1,1) for i in range(C.shape[1])]
#         C = np.concatenate(C, axis=1)

#     else:
#         C = None

#     Y = np.concatenate([y_train, y_val, y_test,y_balanced_test], axis=0)
#     ### regression
#     if task_type == 'regression':
#         Y = Y.astype('float32')
#     ### classification
#     else:
#         Y = sklearn.preprocessing.LabelEncoder().fit_transform(Y).astype('int64')

#     n_classes = int(max(Y)) + 1 if task_type == 'multiclass' else None
#     ### !!! CRUCIAL for neural networks when solving regression problems !!!
#     if task_type == 'regression':
#         y_mean = Y[:train_size].mean().item()
#         y_std = Y[:train_size].std().item()
#         Y = (Y - y_mean) / y_std
#     elif task_type == 'binclass':
#         y_mean = y_std = None
#         Y = Y.astype('float32')
#     elif task_type == 'multiclass':
#         y_mean = y_std = None
#         Y = Y.astype('int64')

#     # generate train, val, test
#     X = {}
#     y = {}

#     if n_cat_features != 0:
#         X_all = np.concatenate([N,C], axis=1)
#         # categories = np.max(C, axis=0) + 1
#         # leave one for masking with the last per cat
#         categories = np.max(C, axis=0) + 2
#     else:
#         X_all = N
#         categories = None

#     X['train'], X['val'], X['test'], X['balance_test'] = X_all[:train_size], X_all[train_size:train_size+val_size], X_all[train_size+val_size:-y_balanced_test_size], X_all[-y_balanced_test_size:]
#     y['train'], y['val'], y['test'], y['balance_test'] = Y[:train_size], Y[train_size:train_size+val_size],Y[train_size+val_size:-y_balanced_test_size], Y[-y_balanced_test_size:]


#     X = {k: torch.tensor(v, dtype=torch.float).cuda() for k, v in X.items()}
#     y = {k: torch.tensor(v).cuda() for k, v in y.items()}
    
#     return X, y, n_classes, y_mean, y_std, categories


# def load_with_origin(dataname, info, normalization, ratio):
#     ## loading pretrain word2vectory
#     if os.path.exists('../tokenizer'):
#         tokenizer = BertTokenizerFast.from_pretrained('../tokenizer')
#     else:
#         tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
#         tokenizer.save_pretrained('../tokenizer')
#     tokenizer.__dict__['model_max_length'] = 512

#     task_type, n_num_features, n_cat_features, train_size, val_size, test_size = info.get('task_type'), info.get('n_num_features'), info.get('n_cat_features'), info.get('train_size'), info.get('val_size'), info.get('test_size')

#     assert task_type in ['binclass', 'multiclass', 'regression']

#     # preprocessing

#     if ratio == 1.0:
#         selected_rows = np.arange(train_size)
#         val_selected_row = np.arange(val_size)
#     else:
#         selected_rows = np.random.choice(train_size, int(train_size * ratio), replace=False)
#         val_selected_row = np.random.choice(val_size, int(val_size * ratio), replace=False)
#     train_size = int(train_size * ratio)
#     val_size = int(val_size * ratio)
#     ## numerical features
#     N_train, N_val, N_test = np.load(f'../data/{dataname}/N_train.npy', allow_pickle=True), np.load(f'../data/{dataname}/N_val.npy', allow_pickle=True), np.load(f'../data/{dataname}/N_test.npy', allow_pickle=True)
#     N_train = N_train[selected_rows, :]
#     N_val = N_val[val_selected_row, :]

#     N = np.concatenate([N_train, N_val, N_test], axis=0).astype('float32')
#     ### it has nan input for higgs_small
#     N = np.nan_to_num(N)
#     ### feature-wise normalize
#     if normalization == 'standard':
#         preprocess = sklearn.preprocessing.StandardScaler().fit(N[:train_size])
#     elif normalization == 'minmax':
#         preprocess = sklearn.preprocessing.MinMaxScaler().fit(N[:train_size])
#     elif normalization == 'quantile':
#         preprocess = sklearn.preprocessing.QuantileTransformer(output_distribution='normal').fit(N[:train_size])
#     elif normalization == 'none':
#         preprocess = sklearn.preprocessing.FunctionTransformer().fit(N[:train_size])
#     ### N: (bs, cols)
#     N_origin = N.copy()
#     N = preprocess.transform(N)


#     ## catergorical features
#     if n_cat_features != 0:
#         C_train, C_val, C_test = np.load(f'../data/{dataname}/C_train.npy', allow_pickle=True), np.load(f'../data/{dataname}/C_val.npy', allow_pickle=True), np.load(f'../data/{dataname}/C_test.npy', allow_pickle=True)
#         C_train = C_train[selected_rows, :]
#         C_val = C_val[val_selected_row, :]

#         C = np.concatenate([C_train, C_val, C_test], axis=0)

#         C = [sklearn.preprocessing.LabelEncoder().fit_transform(C[:,i]).astype('int64').reshape(-1,1) for i in range(C.shape[1])]
#         C = np.concatenate(C, axis=1)

#     else:
#         C = None


#     ## label
#     y_train, y_val, y_test = np.load(f'../data/{dataname}/y_train.npy', allow_pickle=True), np.load(f'../data/{dataname}/y_val.npy', allow_pickle=True), np.load(f'../data/{dataname}/y_test.npy', allow_pickle=True)
#     y_train = y_train[selected_rows]
#     y_val = y_val[val_selected_row]
    
#     Y = np.concatenate([y_train, y_val, y_test], axis=0)
#     ### regression
#     if task_type == 'regression':
#         Y = Y.astype('float32')
#     ### classification
#     else:
#         Y = sklearn.preprocessing.LabelEncoder().fit_transform(Y).astype('int64')

#     n_classes = int(max(Y)) + 1 if task_type == 'multiclass' else None
#     ### !!! CRUCIAL for neural networks when solving regression problems !!!
#     if task_type == 'regression':
    
#         y_mean = Y[:train_size].mean().item()
#         y_std = Y[:train_size].std().item()
#         Y = (Y - y_mean) / y_std
#     elif task_type == 'binclass':
#         y_mean = y_std = None
#         Y = Y.astype('float32')
#     elif task_type == 'multiclass':
#         y_mean = y_std = None
#         Y = Y.astype('int64')

#     # generate train, val, test
#     X = {}
#     y = {}
#     X_origin = {}
#     if n_cat_features != 0:
#         X_all = np.concatenate([N,C], axis=1)
#         # categories = np.max(C, axis=0) + 1
#         # leave one for masking with the last per cat
#         categories = np.max(C, axis=0) + 2
#     else:
#         X_all = N
#         categories = None
    
#     if n_cat_features != 0:
#         X_all_origin = np.concatenate([N_origin,C], axis=1)
#         # categories = np.max(C, axis=0) + 1
#         # leave one for masking with the last per cat
#         categories = np.max(C, axis=0) + 2
#     else:
#         X_all_origin = N_origin
#         categories = None

#     X['train'], X['val'], X['test'] = X_all[:train_size], X_all[train_size:train_size+val_size], X_all[-test_size:]
#     y['train'], y['val'], y['test'] = Y[:train_size], Y[train_size:train_size+val_size], Y[-test_size:]
#     X_origin['train'], X_origin['val'], X_origin['test'] = X_all_origin[:train_size], X_all_origin[train_size:train_size+val_size], X_all_origin[-test_size:]

#     X = {k: torch.tensor(v, dtype=torch.float).cuda() for k, v in X.items()}
#     y = {k: torch.tensor(v).cuda() for k, v in y.items()}
#     X_origin = {k: torch.tensor(v, dtype=torch.float).cuda() for k, v in X_origin.items()}

#     return X, y, X_origin, n_classes, y_mean, y_std, categories