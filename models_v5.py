from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# from transformer import Net
import ipdb
from transformers import BertTokenizer, BertTokenizerFast
import os
import math
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch.nn.init as nn_init
import toml
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import json
from joblib import Parallel, delayed
import pandas as pd
from einops import rearrange, repeat
from sklearn.decomposition import PCA
import copy
import sys
import random
import Models
def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

class Model(nn.Module):
    def __init__(self, input_num, model_type, out_dim, info, config, categories, topic_num, topics, args) -> None:
        super().__init__()
        
        self.input_num = input_num ## number of numerical features
        self.out_dim = out_dim
        self.model_type = model_type
        self.info = info
        self.num_list = np.arange(info.get('n_num_features'))
        self.cat_list = np.arange(info.get('n_num_features'), info.get('n_num_features') + info.get('n_cat_features')) if info.get('n_cat_features')!=None else None
        self.categories = categories

        self.config = config

        self.topic_num = topic_num

        self.args = args
        self.build_model(args.seed,topics)



    def build_model(self,seed,topics):
        config = toml.load(f'../hypers_{self.args.hyper}/{self.args.dataname}/MLP.toml')
        # used for transforming raw data to proportion
        # if self.args.apply_attribute == 'True':
        #     self.estimator = Models.mlp.MLP(self.input_num, config['model']['d_layers']+[self.topic_num], config['model']['dropout'], self.out_dim, self.categories, config['model']['d_embedding'])
        # else:
        #     self.estimator = None
        
        _set_seed(seed)
        if self.model_type == 'MLP':
            self.encoder = Models.mlp.MLP(self.input_num, self.config['model']['d_layers'], self.config['model']['dropout'], self.out_dim, self.categories, self.config['model']['d_embedding'])
            _set_seed(seed)
            self.head_1 = nn.Linear(self.config['model']['d_layers'][-1], self.out_dim)
            self.head_2 = copy.deepcopy(self.head_1)
            self.head_input = self.config['model']['d_layers'][-1]
            hidden_dim = self.config['model']['d_layers'][-1]        


        elif self.model_type == 'SNN':
            self.encoder = Models.snn.SNN(self.input_num, self.config['model']['d_layers'], self.config['model']['dropout'], self.out_dim, self.categories, self.config['model']['d_embedding'])
            _set_seed(seed)
            self.head_1 = nn.Linear(self.config['model']['d_layers'][-1], self.out_dim)
            self.head_2 = copy.deepcopy(self.head_1)
            self.head_input = self.config['model']['d_layers'][-1]
            hidden_dim = self.config['model']['d_layers'][-1]    


        elif self.model_type == 'FTTransformer':
            self.encoder = Models.fttransformer.FTTransformer(self.input_num, self.categories, True, self.config['model']['n_layers'], self.config['model']['d_token'],
                            self.config['model']['n_heads'], self.config['model']['d_ffn_factor'], self.config['model']['attention_dropout'], self.config['model']['ffn_dropout'], self.config['model']['residual_dropout'],
                            self.config['model']['activation'], self.config['model']['prenormalization'], self.config['model']['initialization'], None, None)
            _set_seed(seed)
            self.head_1 = nn.Linear(self.config['model']['d_token'], self.out_dim)
            self.head_2 = copy.deepcopy(self.head_1)
            self.head_input = self.config['model']['d_token']
            hidden_dim = self.config['model']['d_token']
            

        elif self.model_type == 'ResNet':
            self.encoder = Models.resnet.ResNet(self.input_num, self.categories, self.config['model']['d_embedding'], self.config['model']['d'], self.config['model']['d_hidden_factor'], self.config['model']['n_layers'],
                            self.config['model']['activation'], self.config['model']['normalization'], self.config['model']['hidden_dropout'], self.config['model']['residual_dropout'])
            _set_seed(seed)
            self.head_1 = nn.Linear(self.config['model']['d'], self.out_dim)
            self.head_2 = copy.deepcopy(self.head_1)
            self.head_input = self.config['model']['d']
            hidden_dim = self.config['model']['d']
        
        
        elif self.model_type == 'DCN2':
            self.encoder = Models.dcn2.DCN2(self.input_num, self.config['model']['d'], self.config['model']['n_hidden_layers'], self.config['model']['n_cross_layers'],
                            self.config['model']['hidden_dropout'], self.config['model']['cross_dropout'], self.out_dim, self.config['model']['stacked'], self.categories, self.config['model']['d_embedding'])
            _set_seed(seed)
            self.head_1 = nn.Linear(self.config['model']['d'] if self.config['model']['stacked'] else 2 * self.config['model']['d'], self.out_dim)
            self.head_2 = copy.deepcopy(self.head_1)
            self.head_input = self.config['model']['d'] if self.config['model']['stacked'] else 2 * self.config['model']['d']
            hidden_dim = self.config['model']['d'] if self.config['model']['stacked'] else 2 * self.config['model']['d']


        elif self.model_type == 'AutoInt':
            self.encoder = Models.autoint.AutoInt(self.input_num, self.categories, self.config['model']['n_layers'], self.config['model']['d_token'], self.config['model']['n_heads'],
                            self.config['model']['attention_dropout'], self.config['model']['residual_dropout'], self.config['model']['activation'], self.config['model']['prenormalization'], self.config['model']['initialization'], 
                            None, None, self.out_dim)
            _set_seed(seed)
            self.head_1 = nn.Linear(self.config['model']['d_token'] * self.encoder.tokenizer.n_tokens, self.out_dim)
            self.head_2 = copy.deepcopy(self.head_1)
            
            self.head_input = self.config['model']['d_token'] * self.encoder.tokenizer.n_tokens
            hidden_dim = self.config['model']['d_token'] * self.encoder.tokenizer.n_tokens

        if self.args.prototype == 'True':
            _set_seed(seed)
            self.estimator_head = nn.Linear(hidden_dim, self.topic_num)
            self.topic_linear = nn.Linear(self.topic_num,hidden_dim,bias=False)
            if topics is not None:
                self.topic_linear.weight = nn.Parameter(torch.tensor(topics.T))
            else:
                tmp = self.topic_linear.weight.detach()*self.topic_num
                self.topic_linear.weight = nn.Parameter(tmp)
            
        if self.args.denoise == 'True':
            _set_seed(seed)
            self.denoise = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim))


    def forward(self, inputs_n, inputs_c):
        inputs_ = self.encoder(inputs_n, inputs_c)
        assert False
        return self.head_1(inputs_)
        