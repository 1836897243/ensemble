import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from data import load
from models_v5 import Model
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

def build_data_loader(dataset, batch_size=128, shuffle=False):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def get_loss_function(task_type):
    if task_type == 'regression': 
        loss_func = F.mse_loss
    elif task_type == 'binclass':
        loss_func = F.binary_cross_entropy_with_logits
    elif task_type == 'multiclass':
        loss_func = F.cross_entropy
    return loss_func

# def cosine_dist(a:torch.tensor,b:torch.tensor):
#     assert a.shape == b.shape
#     dot = torch.matmul(a,b.transpose(0,1))
#     a_ = torch.norm(a,p=2,dim=1,keepdim=True)
#     b_ = torch.norm(b,p=2,dim=1,keepdim=True)
#     return 1 - dot/(torch.matmul(a_,b_.T))
def check_args(args):
    assert args.prototype in ['True', 'False']
    assert args.reg_type in ['none', 'orth', 'cov']
    assert args.process_type in ['none', 'mask', 'noise']
    assert args.denoise in ['True', 'False']

# def get_prefix(args):
#     if args.prototype == 'False':
#         prefix = f'{args.dataname}_{args.model_type}_{args.prototype}_{args.sample_ratio}_{args.process_type}_{args.process_ratio}_{args.hyper}_{args.seed}'
#     elif args.prototype == 'True':
#         assert args.reg_type in ['none', 'orth', 'cov']
#         if args.reg_type == 'none':
#             assert args.reg_weight == 0.0
#             prefix = f'{args.dataname}_{args.model_type}_{args.prototype}_{args.prototype_num}_{args.sample_ratio}_{args.process_type}_{args.process_ratio}_{args.hyper}_{args.seed}'
#         else:
#             prefix = f'{args.dataname}_{args.model_type}_{args.prototype}_{args.prototype_num}_{args.reg_weight}_{args.reg_type}_{args.sample_ratio}_{args.process_type}_{args.process_ratio}_{args.hyper}_{args.seed}'
#         if args.denoise != 'False':
#             prefix = f'{prefix}_{args.denoise}'
#         if args.mask_type != 'none':
#             prefix = f'{prefix}_{args.mask_type}'
#     return prefix

def getPrefix_saveDir(args):
    prefix = f'{args.dataname}_{args.model_type}_{args.hyper}_{args.seed}_{args.sample_ratio}_{args.process_type}_{args.process_ratio}'
    save_dir = f'results/scores'
    if args.denoise =='True':
        prefix = f'{prefix}_denoise'
        save_dir = f'{save_dir}_denoise'
    if args.prototype == 'True':
        prefix = f'{prefix}_prototype_{args.prototype_num}'
        save_dir = f'{save_dir}_prototype_{args.prototype_num}'
        if args.reg_type != 'none':
            assert args.reg_weight > 0.0
            prefix = f'{prefix}_{args.reg_type}_{args.reg_weight}'
            save_dir = f'{save_dir}_{args.reg_type}_{args.reg_weight}'
    return prefix, save_dir
def run_one_epoch(model, data_loader, loss_func, optimizer=None):
   
    running_loss = 0.0
    for bid, (X_n, X_c, y) in enumerate(data_loader):
        '''
        to 
        '''
        X_n, X_c, y = X_n.cuda(), X_c.cuda(), y.cuda()
        encoder_hid = model.encoder(X_n, X_c)

        if model.args.denoise == 'True':
            encoder_hid = encoder_hid - model.denoise(encoder_hid)

        if model.args.prototype == 'False': 
            pred = model.head_1(encoder_hid)
        elif model.args.prototype == 'True':
            r = model.estimator_head(encoder_hid)
            r = torch.softmax(r, dim=1)
            hid = model.topic_linear(r)
            pred = model.head_2(hid)
        # pred loss
        if loss_func == F.cross_entropy:
            loss = loss_func(pred, y)
        else:
            loss = loss_func(pred, y.reshape(-1,1))

        # independence loss
        if model.args.prototype == 'True' and model.args.reg_type != 'none':
            assert model.args.reg_weight > 0.0
            if model.args.reg_type == 'cov':
                topics = model.topic_linear.weight.T  #shape=[10,256]：10 is the number of topics, 256 is the dimension of topic
                topics = topics - torch.mean(topics, dim=1, keepdim=True)
                covariance_matrix = torch.mm(topics, topics.T)/(topics.shape[1]-1)
                upper_triangular = torch.triu(covariance_matrix, diagonal=1)
                lower_triangular = torch.tril(covariance_matrix, diagonal=-1)
                loss_sparse = torch.sum(upper_triangular ** 2+lower_triangular ** 2)/topics.shape[0]
                loss = loss + loss_sparse*model.args.reg_weight


            if model.args.reg_type == 'orth':
                topics = model.topic_linear.weight.T
                r_1 = torch.sqrt(torch.sum(topics**2,dim=1,keepdim=True))
                topic_metrix = torch.mm(topics, topics.T) / torch.mm(r_1, r_1.T)
                topic_metrix = torch.clamp(topic_metrix.abs(), 0, 1)
                l1 = torch.sum(topic_metrix.abs())
                l2 = torch.sum(topic_metrix ** 2)
                loss_sparse = l1 / l2
                loss_constraint = torch.abs(l1 - topic_metrix.shape[0])
                r_loss = loss_sparse + 0.5*loss_constraint
                loss = loss + r_loss*model.args.reg_weight
        running_loss += loss.item()
        torch.cuda.empty_cache()
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return running_loss / len(data_loader)

def erase_data(X, y,args):
    if args.dataname != 'Brazilian_houses':
        return X,y
    index1 = y['train']> -60
    index2 = y['train']< 60
    index = index1 & index2
    X['train'] = X['train'][index]
    y['train'] = y['train'][index]

    index1 = y['val']> -60
    index2 = y['val']< 60
    index = index1 & index2
    X['val'] = X['val'][index]
    y['val'] = y['val'][index]

    index1 = y['test']> -60
    index2 = y['test']< 60
    index = index1 & index2
    X['test'] = X['test'][index]
    y['test'] = y['test'][index]


    return X,y



def fit(model, train_loader, val_loader, test_loader, loss_func, model_type, config, task_type, y_std, args):
    best_val_loss = 1e30
    best_model = None
    if args.prototype == 'False':
        param = list(model.encoder.parameters()) + list(model.head_1.parameters())
    elif args.prototype == 'True':
        param = list(list(model.head_2.parameters()) + list(model.estimator_head.parameters()) + list(model.topic_linear.parameters()))
    if args.denoise == 'True':
        param += list(model.denoise.parameters())
    optimizer = optim.AdamW(param, lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    early_stop = config['training']['patience']
    epochs = config['training']['n_epochs']

    patience = early_stop

    for eid in range(epochs):
        model.train()
        train_loss = run_one_epoch(
            model, train_loader, loss_func, optimizer
        )

        model.eval()
        val_loss = run_one_epoch(
            model, val_loader, loss_func
        )
        if eid % 10 == 0:
            task_type = 'regression'
        print(f'Epoch {eid}, train loss {train_loss}, val loss {val_loss}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.cpu()
            torch.cuda.empty_cache()
            best_model = copy.deepcopy(model)
            model.cuda()
            patience = early_stop
        else:
            patience = patience - 1

        if patience == 0:
            break
    del model
    torch.cuda.empty_cache()
    return best_model.cuda()

def test(model, test_loader, task_type, y_std, args, config,which_loader):
    model.eval()
    pred = []
    ground = []
    for bid, (X_n, X_c, y) in enumerate(test_loader):
        X_n, X_c, y = X_n.cuda(), X_c.cuda(), y
        ground.append(y.data.numpy())

        encoder_hid = model.encoder(X_n, X_c)
        if args.denoise == 'True':
            encoder_hid = encoder_hid - model.denoise(encoder_hid)

        if args.prototype == 'False':
            pred.append(model.head_1(encoder_hid).data.cpu().numpy())
        elif args.prototype == 'True':
            r = model.estimator_head(encoder_hid)
            r = torch.softmax(r, dim=1)
            hid = model.topic_linear(r)
            pred.append(model.head_2(hid).data.cpu().numpy())
        torch.cuda.empty_cache()
        
    pred = np.concatenate(pred, axis=0)
    y = np.concatenate(ground, axis=0)
    
    

    if task_type == 'binclass':
        pred = np.round(scipy.special.expit(pred))
        score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
    elif task_type == 'multiclass':
        pred = pred.argmax(1)
        score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(y.reshape(-1,1), pred.reshape(-1,1)) ** 0.5 * y_std

    print(f'{which_loader} result, {score.item()}')
    prefix, save_dir = getPrefix_saveDir(args)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'./results/models'):
        os.makedirs(f'./results/models')
    np.save(open(f'./{save_dir}/{prefix}.npy','wb'), score.item())
    torch.save(model.state_dict(), f'./results/models/{prefix}_model.pth')



def generate_topic(train_loader, val_loader, args, categories):
    if args.prototype == 'False':
        return None

    data = []
    for bid, (X_n, X_c, y) in enumerate(train_loader):
        if categories is None:
            data.append(X_n)
        else:
            data.append(torch.cat([X_n, X_c], dim=1))

    for bid, (X_n, X_c, y) in enumerate(val_loader):
        if categories is None:
            data.append(X_n)
        else:
            data.append(torch.cat([X_n, X_c], dim=1))

    data = torch.cat(data, dim=0)
    if args.model_type == 'MLP' or args.model_type == 'SNN':
        hidden_dim = config['model']['d_layers'][-1]
    elif args.model_type == 'FTTransformer':
        hidden_dim = config['model']['d_token']
    elif args.model_type == 'AutoInt':
        hidden_dim = config['model']['d_token'] * (n_num_features + n_cat_features)
    elif args.model_type == 'ResNet':
        hidden_dim = config['model']['d']
    elif args.model_type == 'DCN2':
        hidden_dim = config['model']['d'] if config['model']['stacked'] else 2 * config['model']['d']

    prototypes = np.random.randn(args.prototype_num, hidden_dim).astype('float32')
    return prototypes

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--model_type', type=str)

    parser.add_argument('--denoise', type=str, default='False')

    parser.add_argument('--prototype', type=str, default='False')
    parser.add_argument('--prototype_num', type=int, default=10)

    parser.add_argument('--reg_weight', type=float, default=0.1)
    parser.add_argument('--hyper', type=str, default='default')
    parser.add_argument('--reg_type', type=str, default='none')#'none', 'orth', 'cov','both'
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--process_type', type=str, default='none', help='mask, noise')
    parser.add_argument('--process_ratio', type=float, default=0.0)       
    args = parser.parse_args()
    check_args(args)
    _set_seed(args.seed)

    config = toml.load(f'../hypers_{args.hyper}/{args.dataname}/{args.model_type}.toml')

    with open(f'../data/{args.dataname}/info.json') as f:
        info = json.load(f)

    gc.collect()
    torch.cuda.empty_cache()

    # X, y, n_classes, y_mean, y_std, categories = load(args.dataname, info, config['data']['normalization'], args.sample_ratio)

    # task_type = info.get('task_type')
    # print(task_type)

    # n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')
    # num_list = np.arange(n_num_features)
    # cat_list = np.arange(n_num_features, n_num_features + n_cat_features) if n_cat_features!=None else None
   
    # train_loader = build_data_loader(TensorDataset(X['train'][:,:n_num_features], X['train'][:,n_num_features:] if n_cat_features>0 else torch.empty(X['train'].shape[0], X['train'].shape[1]).cuda(), y['train']), config['training']['batch_size'], False)
    # val_loader = build_data_loader(TensorDataset(X['val'][:,:n_num_features], X['val'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['val'].shape[0], X['val'].shape[1]).cuda(), y['val']), config['training']['batch_size'], False)
    # test_loader = build_data_loader(TensorDataset(X['test'][:, :n_num_features], X['test'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['test'].shape[0], X['test'].shape[1]).cuda(), y['test']), config['training']['batch_size'], False)
    X, y, n_classes, y_mean, y_std, categories = load(args.dataname, info, config['data']['normalization'], args)
    task_type = info.get('task_type')
    print(task_type)

    n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')
    num_list = np.arange(n_num_features)
    cat_list = np.arange(n_num_features, n_num_features + n_cat_features) if n_cat_features!=None else None
    X, y = erase_data(X, y,args)

    # to cpu
    '''
    for some datasets like yahoo, the data is too large to fit in GPU memory, so we need to move the data to CPU memory
    '''
    y = {k:v.cpu() for k,v in y.items()}
    X = {k:v.cpu() for k,v in X.items()}

    train_loader = build_data_loader(TensorDataset(X['train'][:,:n_num_features], X['train'][:,n_num_features:] if n_cat_features>0 else torch.empty(X['train'].shape[0], X['train'].shape[1]).cpu(), y['train']), config['training']['batch_size'], False)
    val_loader = build_data_loader(TensorDataset(X['val'][:,:n_num_features], X['val'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['val'].shape[0], X['val'].shape[1]).cpu(), y['val']), config['training']['batch_size'], False)
    test_loader = build_data_loader(TensorDataset(X['test'][:, :n_num_features], X['test'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['test'].shape[0], X['test'].shape[1]).cpu(), y['test']), config['training']['batch_size'], False)

    
    ## generating topcs
    topics = generate_topic(train_loader, val_loader, args, categories)

    ## model initialization
    model = Model(n_num_features, args.model_type, n_classes if task_type == 'multiclass' else 1, 
    info=info, config = config, categories = categories, topic_num = args.prototype_num, topics = topics, args = args)

    #model = copy.deepcopy(pretrain_model)

    model.cuda()
    loss_func = get_loss_function(task_type)

    

    phase1_args = copy.deepcopy(args)
    phase1_args.prototype = 'False'
    phase1_args.denoise = 'False'
    model.args = phase1_args
    #test_loader,val_loader = val_loader,test_loader
    best_model = fit(model, train_loader, val_loader,test_loader, loss_func, phase1_args.model_type, config, task_type, y_std, phase1_args)
    test(best_model, test_loader, task_type, y_std, phase1_args, config,'test loader')

 
    best_model.args = args
    best_model = fit(best_model, train_loader, val_loader,test_loader, loss_func, args.model_type, config, task_type, y_std, args)
    test(best_model, test_loader, task_type, y_std, args, config,'test loader')





