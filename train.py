import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from data import load
from model import Model
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
import ipdb
import argparse
import pickle
import sklearn
import scipy
import gc
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from scipy.linalg import svd
import importlib
import toml
import os
import copy
import ot
import ast
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
def build_data_loader(dataset, batch_size=128, shuffle=False):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def get_score(pred, y, task_type, y_std):
    if task_type == 'binclass':
        pred = np.round(scipy.special.expit(pred))
        score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
    elif task_type == 'multiclass':
        pred = pred.argmax(1)
        score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
    elif task_type == 'regression':
        score = sklearn.metrics.mean_squared_error(y.reshape(-1,1), pred.reshape(-1,1)) ** 0.5 * y_std
    else:
        raise ValueError(f'task type {task_type} not supported')
    return score


def getPrefix_saveDir(args):
    phase2_prefix = f'{args.dataname}_{args.model_type}_{args.hyper}_{args.seed}_{args.sample_ratio}_{args.process_type}_{args.process_ratio}'
    phase1_prefix = phase2_prefix
    phase2_save_dir = f'results/scores'
    if args.denoise =='True':
        phase2_prefix = f'{phase2_prefix}_denoise_{args.pure_ratio}'
        phase2_save_dir = f'{phase2_save_dir}_denoise_{args.pure_ratio}'
    if args.prototype == 'True':
        phase2_prefix = f'{phase2_prefix}_prototype_{args.prototype_num}'
        phase2_save_dir = f'{phase2_save_dir}_prototype_{args.prototype_num}'
        if args.reg_type != 'none':
            assert args.reg_weight > 0.0
            phase2_prefix = f'{phase2_prefix}_{args.reg_type}_{args.reg_weight}'
            phase2_save_dir = f'{phase2_save_dir}_{args.reg_type}_{args.reg_weight}'
    phase1_save_dir =  f'results/scores'
    model_save_dir = f'results/models'
    return phase1_prefix, phase1_save_dir, phase2_prefix, phase2_save_dir, model_save_dir
def save_model(model, model_save_dir, prefix):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(model.state_dict(), f'{model_save_dir}/{prefix}_models.pth')
def save_score(score, save_dir, prefix):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(f'{save_dir}/{prefix}.npy', score)
def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_pred_forward_train(model,data_loader, device):
    pred = []
    for i, (num_data, cat_data, target) in enumerate(data_loader):
        num_data, cat_data, target = num_data.to(device), cat_data.to(device), target.to(device)
        pred.append(model.forward_train(num_data, cat_data).data.cpu().numpy())
    return np.concatenate(pred, axis=0)
def get_pred_forward_fintune(model,data_loader, device):
    pred = []
    for i, (num_data, cat_data, target) in enumerate(data_loader):
        num_data, cat_data, target = num_data.to(device), cat_data.to(device), target.to(device)
        pred.append(model.forward_fintune(num_data, cat_data).data.cpu().numpy())
    return np.concatenate(pred, axis=0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--model_type', type=str)

    parser.add_argument('--denoise', type=str, default='False')

    parser.add_argument('--prototype', type=str, default='False')
    parser.add_argument('--prototype_num', type=int, default=10)

    parser.add_argument('--reg_weight', type=float, default=0.1)
    parser.add_argument('--hyper', type=str, default='default')
    parser.add_argument('--reg_type', type=str, default='none',choices=['none','orth','cov'])#'none', 'orth', 'cov'
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--process_type', type=str, default='none', choices=['mask','noise','none'])
    parser.add_argument('--process_ratio', type=float, default=0.0)
    parser.add_argument('--pure_ratio', type=float, default=0.01)      
    args = parser.parse_args()
    
    config = toml.load(f'../hyper_{args.hyper}/{args.dataname}/{args.model_type}.toml')

    with open(f'../data/{args.dataname}/info.json') as f:
        info = json.load(f)
    if args.model_type in ['AutoInt', 'DCN2', 'FTTransformer','MLP','ResNet','SNN', 'Saint']:
        _set_seed(args.seed)
        gc.collect()
        torch.cuda.empty_cache()
        X, y, n_classes, y_mean, y_std, categories = load(args.dataname, info, config['data']['normalization'], args)
        task_type = info.get('task_type')
        print(task_type)

        n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')
        num_list = np.arange(n_num_features)
        cat_list = np.arange(n_num_features, n_num_features + n_cat_features) if n_cat_features!=None else None

        train_loader = build_data_loader(TensorDataset(X['train'][:,:n_num_features], X['train'][:,n_num_features:] if n_cat_features>0 else torch.empty(X['train'].shape[0], X['train'].shape[1]).cpu(), y['train']), config['training']['batch_size'], False)
        val_loader = build_data_loader(TensorDataset(X['val'][:,:n_num_features], X['val'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['val'].shape[0], X['val'].shape[1]).cpu(), y['val']), config['training']['batch_size'], False)
        test_loader = build_data_loader(TensorDataset(X['test'][:, :n_num_features], X['test'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['test'].shape[0], X['test'].shape[1]).cpu(), y['test']), config['training']['batch_size'], False)

        

        print(config)
        ## model initialization
        model = Model(n_num_features, args.model_type, n_classes if task_type == 'multiclass' else 1, 
        info=info, config = config, categories = categories, topic_num = args.prototype_num, args = args)

        model = model.cuda()
        device = torch.device('cuda')
        model.fit(train_loader, val_loader, device)

        # pred = model.forward_train(X['test'][:,:n_num_features].to(device), X['test'][:,n_num_features:].to(device) if n_cat_features>0 else torch.empty(X['test'].shape[0], X['test'].shape[1]).to(device)).data.cpu().numpy()
        pred = get_pred_forward_train(model, test_loader, device)
        test_y = y['test'].data.cpu().numpy()
        score1 = get_score(pred, test_y, task_type, y_std)
        print(f'phase1 test score: {score1}')
        
        phase1_prefix, phase1_save_dir, phase2_prefix, phase2_save_dir, model_save_dir = getPrefix_saveDir(args)
        save_score(score1, phase1_save_dir, phase1_prefix)

        model.fintune(train_loader, val_loader, device)
        # pred = model.forward_fintune(X['test'][:,:n_num_features].to(device), X['test'][:,n_num_features:].to(device) if n_cat_features>0 else torch.empty(X['test'].shape[0], X['test'].shape[1]).to(device)).data.cpu().numpy()
        pred = get_pred_forward_fintune(model, test_loader, device)
        test_y = y['test'].data.cpu().numpy()
        score2 = get_score(pred, test_y, task_type, y_std)
        print(f'phase2 test score: {score2}')
        save_model(model, model_save_dir, phase2_prefix)
        save_score(score2, phase2_save_dir, phase2_prefix)
    elif args.model_type in ['CatBoost']:
        assert args.denoise == 'False'
        assert args.prototype == 'False'
        assert args.reg_type == 'none'
        _set_seed(args.seed)

        X, y, n_classes, y_mean, y_std, categories = load(args.dataname, info, config['data']['normalization'], args)
        task_type = info.get('task_type')
        X = {k:v.numpy() for k,v in X.items()}

        y = {k:np.squeeze(v.numpy()) for k,v in y.items()}

        if task_type == 'regression':
            model = CatBoostRegressor(iterations=config['model']['iterations'], learning_rate=config['model']['learning_rate'], depth=config['model']['depth'], \
                                    l2_leaf_reg=config['model']['l2_leaf_reg'], bagging_temperature=config['model']['bagging_temperature'], leaf_estimation_iterations=config['model']['leaf_estimation_iterations'], \
                                        metric_period=config['model']['metric_period'], od_pval=config['model']['od_pval'], thread_count=config['model']['thread_count'], \
                                            early_stopping_rounds=config['model']['early_stopping_rounds'], task_type=config['model']['task_type'], random_seed = args.seed)
        else:
            if task_type == 'binclass':
                y = {k:v.astype('int32') for k,v in y.items()}
                loss_function = 'Logloss'
            elif task_type == 'multiclass':
                loss_function = 'MultiClass'
            model = CatBoostClassifier(iterations=config['model']['iterations'], learning_rate=config['model']['learning_rate'], depth=config['model']['depth'], \
                                    l2_leaf_reg=config['model']['l2_leaf_reg'], bagging_temperature=config['model']['bagging_temperature'], leaf_estimation_iterations=config['model']['leaf_estimation_iterations'], \
                                        metric_period=config['model']['metric_period'], od_pval=config['model']['od_pval'], thread_count=config['model']['thread_count'], \
                                            early_stopping_rounds=config['model']['early_stopping_rounds'], task_type=config['model']['task_type'], random_seed = args.seed, eval_metric='Accuracy', loss_function=loss_function)
        n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')
        cat_features = list(range(n_num_features, n_cat_features+n_num_features)) if n_cat_features!=None else None
        
        X = {k:pd.DataFrame(v) for k,v in X.items()}

        for column in cat_features:
            X['train'][column] = X['train'][column].astype('int32')
            X['val'][column] = X['val'][column].astype('int32')
            X['test'][column] = X['test'][column].astype('int32')

        model.fit(X['train'], y['train'],eval_set=(X['val'], y['val']), cat_features=cat_features,logging_level='Verbose')
        
        pred = model.predict(X['test'])
        pred = np.squeeze(pred)
        if task_type == 'regression':
            score = np.sqrt(np.mean((pred - y['test'])**2))*y_std
        else:
            score = np.mean(pred == y['test'])
        print(score)
        np.save(f'results/scores/{args.dataname}_{args.model_type}_{args.hyper}_{args.seed}_{args.sample_ratio}_{args.process_type}_{args.process_ratio}.npy', score)
        model.save_model(f'results/models/{args.dataname}_{args.model_type}_{args.hyper}_{args.seed}_{args.sample_ratio}_{args.process_type}_{args.process_ratio}.cbm')



    





