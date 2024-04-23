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
from sklearn.cluster import KMeans
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
from torch.nn.parallel import parallel_apply
def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_loss_func(task_type):
    if task_type == 'regression': 
        return F.mse_loss
    elif task_type == 'binclass':
        return F.binary_cross_entropy_with_logits
    elif task_type == 'multiclass':
        return F.cross_entropy
    else:
        raise ValueError(f'Unknown task type: {task_type}, should be one of [regression, binclass, multiclass]')
class Model(nn.Module):
    def __init__(self, input_num, model_type, out_dim, info, config, categories, topic_num, args) -> None:
        super().__init__()
        
        self.input_num = input_num ## number of numerical features
        self.out_dim = out_dim
        self.model_type = model_type
        self.info = info
        task_type = info.get('task_type')

        self.loss_func = get_loss_func(task_type)

        self.num_list = np.arange(info.get('n_num_features'))
        self.cat_list = np.arange(info.get('n_num_features'), info.get('n_num_features') + info.get('n_cat_features')) if info.get('n_cat_features')!=None else None
        self.categories = categories
        
        self.config = config
        self.topic_num = topic_num
        self.args = args
        self.build_model(args.seed)



    def build_model(self,seed):
        # set seed to make sure the same model initialization in experiments expecially for ablation study
        _set_seed(seed)
        if self.model_type == 'MLP':
            self.encoder = Models.mlp.MLP(self.input_num, self.config['model']['d_layers'], self.config['model']['dropout'], self.out_dim, self.categories, self.config['model']['d_embedding'])
            _set_seed(seed)
            self.head_train = nn.Linear(self.config['model']['d_layers'][-1], self.out_dim)
            self.head_fintune = copy.deepcopy(self.head_train)
            hidden_dim = self.config['model']['d_layers'][-1]        


        elif self.model_type == 'SNN':
            self.encoder = Models.snn.SNN(self.input_num, self.config['model']['d_layers'], self.config['model']['dropout'], self.out_dim, self.categories, self.config['model']['d_embedding'])
            _set_seed(seed)
            self.head_train = nn.Linear(self.config['model']['d_layers'][-1], self.out_dim)
            self.head_fintune = copy.deepcopy(self.head_train)
            hidden_dim = self.config['model']['d_layers'][-1]    


        elif self.model_type == 'FTTransformer':
            self.encoder = Models.fttransformer.FTTransformer(self.input_num, self.categories, True, self.config['model']['n_layers'], self.config['model']['d_token'],
                            self.config['model']['n_heads'], self.config['model']['d_ffn_factor'], self.config['model']['attention_dropout'], self.config['model']['ffn_dropout'], self.config['model']['residual_dropout'],
                            self.config['model']['activation'], self.config['model']['prenormalization'], self.config['model']['initialization'], None, None)
            _set_seed(seed)
            self.head_train = nn.Linear(self.config['model']['d_token'], self.out_dim)
            self.head_fintune = copy.deepcopy(self.head_train)
            hidden_dim = self.config['model']['d_token']
            

        elif self.model_type == 'ResNet':
            self.encoder = Models.resnet.ResNet(self.input_num, self.categories, self.config['model']['d_embedding'], self.config['model']['d'], self.config['model']['d_hidden_factor'], self.config['model']['n_layers'],
                            self.config['model']['activation'], self.config['model']['normalization'], self.config['model']['hidden_dropout'], self.config['model']['residual_dropout'])
            _set_seed(seed)
            self.head_train = nn.Linear(self.config['model']['d'], self.out_dim)
            self.head_fintune = copy.deepcopy(self.head_train)
            hidden_dim = self.config['model']['d']
        
        
        elif self.model_type == 'DCN2':
            self.encoder = Models.dcn2.DCN2(self.input_num, self.config['model']['d'], self.config['model']['n_hidden_layers'], self.config['model']['n_cross_layers'],
                            self.config['model']['hidden_dropout'], self.config['model']['cross_dropout'], self.out_dim, self.config['model']['stacked'], self.categories, self.config['model']['d_embedding'])
            _set_seed(seed)
            self.head_train = nn.Linear(self.config['model']['d'] if self.config['model']['stacked'] else 2 * self.config['model']['d'], self.out_dim)
            self.head_fintune = copy.deepcopy(self.head_train)
            hidden_dim = self.config['model']['d'] if self.config['model']['stacked'] else 2 * self.config['model']['d']


        elif self.model_type == 'AutoInt':
            self.encoder = Models.autoint.AutoInt(self.input_num, self.categories, self.config['model']['n_layers'], self.config['model']['d_token'], self.config['model']['n_heads'],
                            self.config['model']['attention_dropout'], self.config['model']['residual_dropout'], self.config['model']['activation'], self.config['model']['prenormalization'], self.config['model']['initialization'], 
                            None, None, self.out_dim)
            _set_seed(seed)
            self.head_train = nn.Linear(self.config['model']['d_token'] * self.encoder.tokenizer.n_tokens, self.out_dim)
            self.head_fintune = copy.deepcopy(self.head_train)
            hidden_dim = self.config['model']['d_token'] * self.encoder.tokenizer.n_tokens
            
        elif self.model_type == 'Saint':
            self.encoder = Models.saint.Saint(categories = self.categories, 
            num_continuous = self.input_num,                
            dim = self.config['model']['embedding_size'],                           
            dim_out = 1,                       
            depth = self.config['model']['transformer_depth'],                       
            heads = self.config['model']['attention_heads'],                         
            attn_dropout = self.config['model']['attention_dropout'],             
            ff_dropout = self.config['model']['ff_dropout'],                  
            mlp_hidden_mults = (4, 2),       
            cont_embeddings = self.config['model']['cont_embeddings'],
            attentiontype = self.config['model']['attentiontype'],
            final_mlp_style = self.config['model']['final_mlp_style'],
            representation_dim=self.config['model']['representation_dim'])
            _set_seed(seed)
            self.head_train = nn.Linear(self.config['model']['representation_dim'], self.out_dim)
            self.head_fintune = copy.deepcopy(self.head_train)
            hidden_dim = self.config['model']['representation_dim']
        if self.args.denoise == 'True':
            _set_seed(seed)
            self.denoise = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim))

        if self.args.prototype == 'True':
            _set_seed(seed)
            self.estimator_head = nn.Linear(hidden_dim, self.topic_num)
            self.topic_linear = nn.Linear(self.topic_num,hidden_dim,bias=False)
            topics = np.random.randn(self.args.prototype_num, hidden_dim).astype('float32')
            self.topic_linear.weight = nn.Parameter(torch.tensor(topics.T))     


    # for training
    def forward_train(self, inputs_n, inputs_c):
        encoder_hid = self.encoder(inputs_n, inputs_c)
        pred = self.head_train(encoder_hid)
        return pred
    
    def forward_fintune(self, inputs_n, inputs_c):
        encoder_hid = self.encoder(inputs_n, inputs_c)
        if self.args.denoise == 'True':
            encoder_hid = encoder_hid - self.denoise(encoder_hid)
        if self.args.prototype == 'False':
            pred = self.head_fintune(encoder_hid)
        elif self.args.prototype == 'True':
            r = self.estimator_head(encoder_hid)
            r = F.softmax(r,dim=-1)
            hid = self.topic_linear(r)
            pred = self.head_fintune(hid)
        return pred
    


    def _run_one_epoch_train(self, data_loader, optimizer, device):
        
        if optimizer is not None:
            self.train()
        else:
            self.eval()
        total_loss = 0
        
        # for every sample in the data_loader
        for i, (inputs_n, inputs_c, targets) in enumerate(data_loader):
            loss = 0
            inputs_n, inputs_c, targets = inputs_n.to(device), inputs_c.to(device), targets.to(device)
            
            pred = self.forward_train(inputs_n, inputs_c)

            if self.loss_func == F.cross_entropy:
                loss += self.loss_func(pred, targets)
            else:
                loss += self.loss_func(pred, targets.reshape(-1,1))
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data_loader)
    
    def fit(self, train_loader, val_loader, device):
        parameters = list(self.encoder.parameters())+list(self.head_train.parameters())
        optimizer = optim.AdamW(parameters, lr=self.config['training']['lr'], weight_decay=self.config['training']['weight_decay'])
        best_val_loss = float('inf')
        best_model = None
        best_epoch = 0
        n_epochs = self.config['training']['n_epochs']
        patience = self.config['training']['patience']
        for epoch in range(n_epochs):
            train_loss = self._run_one_epoch_train(train_loader, optimizer, device)
            val_loss = self._run_one_epoch_train(val_loader, None, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            print(f'Epoch {epoch} train loss: {train_loss}, val loss: {val_loss}')
            if epoch - best_epoch >= patience:
                break
        self.load_state_dict(best_model)
        return best_val_loss


    def get_grad_value(self):
        grad_norm = []
        for param in self.parameters():
            if param.grad is None:
                continue
            grad_norm.append(torch.abs(param.grad).flatten().clone().detach())
        grad_norm = torch.cat(grad_norm)
        grad_norm = torch.mean(grad_norm)
        return grad_norm
    
    def get_gradient(self, xn,xc,y):
        self.eval()
        grads = []
        for i in range(xn.shape[0]):
            self.zero_grad()
            if self.loss_func == F.cross_entropy:
                loss = self.loss_func(self.forward_train(xn[i].unsqueeze(0), xc[i].unsqueeze(0)), y[i].unsqueeze(0))
            else:
                loss = self.loss_func(self.forward_train(xn[i].unsqueeze(0), xc[i].unsqueeze(0)), y[i].unsqueeze(0).reshape(-1,1))
            loss.backward()
            grads.append(self.get_grad_value())
        grads = torch.tensor(grads)
        return grads
    # def get_gradient(self, X_n,X_c,y):
    #     def cal_gradient(X_n, X_c, y, model, config):
    #         optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    #         optimizer.zero_grad()
    #         if model.loss_func == F.cross_entropy:
    #             loss = model.loss_func(self.forward_train(X_n.unsqueeze(0), X_c.unsqueeze(0)), y.unsqueeze(0))
    #         else:
    #             loss = model.loss_func(self.forward_train(X_n.unsqueeze(0), X_c.unsqueeze(0)), y.unsqueeze(0).reshape(-1,1))
    #         print(loss.item())
    #         loss.backward(retain_graph=True)
    #         gradient_norm = []
    #         for name, param in model.named_parameters():
    #             if param.grad != None:
    #                 gradient_norm.append(torch.abs(param.grad).reshape(-1))
    #         print((f'gradient_norm:{gradient_norm}'))
    #         return torch.mean(torch.concat(gradient_norm)).item()
    #     tuples = [(X_n[i], X_c[i], y[i], copy.deepcopy(self), self.config) for i in range(X_n.shape[0])]  
    #     grads = parallel_apply([cal_gradient for k in range(X_n.shape[0])], tuples)
    #     return torch.tensor(grads)
    def _get_clean_representation(self,data_loader):
        self.eval()
        # get data from data_loader
        encoder_representation = []
        losses = []
        Xn_ = []
        Xc_ = []
        y_ = []
        for bid, (X_n, X_c, y) in enumerate(data_loader):
            X_n, X_c, y = X_n.cuda(), X_c.cuda(), y.cuda()
            encoder_hid = self.encoder(X_n, X_c)
            pred = self.head_train(encoder_hid)
            if self.loss_func == F.cross_entropy:
                loss = self.loss_func(pred, y, reduction='none')
            else:
                loss = self.loss_func(pred, y.reshape(-1,1), reduction='none')
            encoder_representation.append(encoder_hid.clone().detach())
            losses.append(loss)
            Xn_.append(X_n)
            Xc_.append(X_c)
            y_.append(y)


        encoder_representation = torch.cat(encoder_representation, dim=0)
        losses = torch.cat(losses, dim=0).squeeze()
        Xn_ = torch.cat(Xn_, dim=0)
        Xc_ = torch.cat(Xc_, dim=0)
        y_ = torch.cat(y_, dim=0)


        # get dimension-wise std
        std = torch.std(encoder_representation, dim=0)
        # sort by loss
        sort_arg = torch.argsort(losses)
        encoder_representation = encoder_representation[sort_arg]
        losses = losses[sort_arg]
        Xn_ = Xn_[sort_arg]
        Xc_ = Xc_[sort_arg]
        y_ = y_[sort_arg]


        # chose top 20% samples with lowest loss
        top20_representation_num = int(encoder_representation.shape[0]*0.2)
        losses = losses[:top20_representation_num]
        Xn_ = Xn_[:top20_representation_num]
        Xc_ = Xc_[:top20_representation_num]
        y_ = y_[:top20_representation_num]
        # chose samples with lowest gradient
        
        grads = self.get_gradient(Xn_,Xc_,y_)
        sort_arg = torch.argsort(grads)
        

        clean_representation_num = int(encoder_representation.shape[0]*self.args.pure_ratio)
        encoder_representation = encoder_representation[:top20_representation_num][sort_arg][:clean_representation_num]

        print(f'torch.mean(grads):{torch.mean(grads)}')
        print(f'torch.min(grads):{torch.min(grads)}')
        print(f'torch.max(grads):{torch.max(grads)}')
        print(f'torch.mean(selected grad):{torch.mean(grads[sort_arg][:clean_representation_num])}')
        print(f'torch.min(selected grad):{torch.min(grads[sort_arg][:clean_representation_num])}')
        print(f'torch.max(selected grad):{torch.max(grads[sort_arg][:clean_representation_num])}')
        return encoder_representation, std


    def _denoise(self,clean_representation, std,optimizer,device):
        repeat_time = 10
        clean_representation = clean_representation.repeat(repeat_time,1)

        batch_size = self.config['training']['batch_size']
        batch_num = clean_representation.shape[0]//batch_size
        batch_num = max(1,batch_num)
        total_loss = 0
        std = std.to(device)
        for i in range(batch_num):
            if i*batch_size>=clean_representation.shape[0]:
                break
            clean_representation_ = clean_representation[i*batch_size:(i+1)*batch_size].to(device)
            
            noise = torch.randn_like(clean_representation_)*std
            noise_representation_ = clean_representation_ + noise
            pred_noise = self.denoise(noise_representation_)
            loss = F.mse_loss(pred_noise, noise)

            pred_zero_noise = self.denoise(clean_representation_)
            loss_zero = F.mse_loss(pred_zero_noise, torch.zeros_like(pred_zero_noise).to(device))
            loss = loss + loss_zero
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss/batch_num
    
    def _run_one_epoch_fintune(self, data_loader, denoise_data, optimizer, device):
        if optimizer is not None:
            self.train()
        else:
            self.eval()
        total_loss = 0
        
        for i, (inputs_n, inputs_c, targets) in enumerate(data_loader):
            loss = 0
            inputs_n, inputs_c, targets = inputs_n.to(device), inputs_c.to(device), targets.to(device)
            
            pred = self.forward_fintune(inputs_n, inputs_c)
            if self.loss_func == F.cross_entropy:
                loss += self.loss_func(pred, targets)
            else:
                loss += self.loss_func(pred, targets.reshape(-1,1))
            # for prototype regularization
            if self.args.prototype == 'True' and self.args.reg_type != 'none':
                assert self.args.reg_weight > 0.0
                if self.args.reg_type == 'cov':
                    topics = self.topic_linear.weight.T  #shape=[10,256]：10 is the number of topics, 256 is the dimension of topic
                    topics = topics - torch.mean(topics, dim=1, keepdim=True)
                    covariance_matrix = torch.mm(topics, topics.T)/(topics.shape[1]-1)
                    upper_triangular = torch.triu(covariance_matrix, diagonal=1)
                    lower_triangular = torch.tril(covariance_matrix, diagonal=-1)
                    loss_sparse = torch.sum(upper_triangular ** 2+lower_triangular ** 2)/topics.shape[0]
                    loss = loss + loss_sparse*self.args.reg_weight

                if self.args.reg_type == 'orth':
                    topics = self.topic_linear.weight.T
                    r_1 = torch.sqrt(torch.sum(topics**2,dim=1,keepdim=True))
                    topic_metrix = torch.mm(topics, topics.T) / torch.mm(r_1, r_1.T)
                    topic_metrix = torch.clamp(topic_metrix.abs(), 0, 1)
                    l1 = torch.sum(topic_metrix.abs())
                    l2 = torch.sum(topic_metrix ** 2)
                    loss_sparse = l1 / l2
                    loss_constraint = torch.abs(l1 - topic_metrix.shape[0])
                    r_loss = loss_sparse + 0.5*loss_constraint
                    loss = loss + r_loss*self.args.reg_weight
            total_loss += loss.item()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
        if self.args.denoise == 'True' and optimizer is not None:
            clean_representation, std = denoise_data['clean_representation'], denoise_data['std']
            denoise_loss = self._denoise(clean_representation, std,optimizer,device)
            return total_loss / len(data_loader), denoise_loss
        return total_loss / len(data_loader)

    def fintune(self, train_loader, val_loader, device):
        parameters = list(self.head_fintune.parameters())
        if self.args.denoise == 'True':
            parameters += list(self.denoise.parameters())
        if self.args.prototype == 'True':
            parameters += list(self.estimator_head.parameters())+list(self.topic_linear.parameters())
        optimizer = optim.AdamW(parameters, lr=self.config['training']['lr'], weight_decay=self.config['training']['weight_decay'])
        best_val_loss = float('inf')
        best_model = None
        best_epoch = 0
        n_epochs = self.config['training']['n_epochs']
        patience = self.config['training']['patience']
        clean_representation, std = self._get_clean_representation(train_loader)
        denoise_data = {'clean_representation':clean_representation, 'std':std}
        for epoch in range(n_epochs):
            train_loss,denoise_loss = self._run_one_epoch_fintune(train_loader, denoise_data, optimizer, device)
            val_loss = self._run_one_epoch_fintune(val_loader, denoise_data, None, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.state_dict())
                best_epoch = epoch
            print(f'Epoch {epoch} train loss: {train_loss}, denoise_loss:{denoise_loss}, val loss: {val_loss}')
            if epoch - best_epoch >= patience:
                break
        self.load_state_dict(best_model)
        return best_val_loss
            
    
        
        