## Import libraries in python
import argparse
import time
import json
import logging
import sys
import os
import math
import pandas as pd
import numpy as np
from itertools import cycle

import random
import importlib
import glob
# import tensorflow as tf

from math import sqrt

import torch.utils.data.dataloader as Data


from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader

from utils.snip import *
from utils.grad_norm import *
from utils.synflow import *
from utils.predictors import *


from utils.pheno_generator import pheno_gen



def archt_val_pair (init_train_log_filepath, X_train,Y_train, n_samples, obj, ep, subdata, window_Size, bs, seed):

    epochs = ep

    max_rul = 125
    output_sequence_length = 1  
    time_step = window_Size+2  
    input_size = 14 

    # Data for training model-based predictor
    df_init = pd.read_csv(init_train_log_filepath)
    init_val_rmse = df_init["val_rmse"] 



    init_archt_genotype = []
    ind_grad_lst = []

    for idx, row in df_init.iterrows():

        train_dataset = TensorDataset(X_train,Y_train)
        train_loader = Data.DataLoader(dataset=train_dataset,batch_size = bs, shuffle=False)


        # genotype = [int(row['params_1']), int(row['params_2']), int(row['params_3']), int(row['params_4']), int(row['params_5']), int(row['params_6']), int(row['params_7']), int(row['params_8']), int(row['params_9']), int(row['params_10']), int(row['params_11']), int(row['params_12']), int(row['num_params']), row['train_rmse']]

        genotype = [int(row['params_1']), int(row['params_2']), int(row['params_3']), int(row['params_4']), int(row['params_5']), int(row['params_6']), int(row['params_7']), int(row['params_8']), int(row['params_9']), int(row['params_10']), int(row['params_11']), int(row['num_params'])]

        # genotype = [int(row['params_1']), int(row['params_2']), int(row['params_3']), int(row['params_4']), int(row['params_5']), int(row['params_6']), int(row['params_7']), int(row['params_8']), int(row['params_9']), int(row['params_10']), int(row['params_11']), int(row['params_12'])]


        model = pheno_gen(genotype, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length)

        criterion = nn.MSELoss()


        ######### calculate architecture score #########
        # Load data
        train_sample_array, train_label_array = next(iter(train_loader))



        grad_norm_arr = compute_snip_per_weight(model, train_sample_array, train_label_array, max_rul, criterion, split_data=1)
        # grad_norm_arr = get_grad_norm_arr(model, train_sample_array, train_label_array, max_rul, criterion, split_data=1, skip_grad=False)
        # grad_norm_arr = compute_synflow_per_weight(model, train_sample_array, train_label_array, mode = 'param')




        norm_lst = []
        for item in grad_norm_arr:
            temp = item.cpu().detach().numpy()        
            temp = np.sum(temp)
            norm_lst.append(temp)

        # print ("norm_lst", norm_lst)
        # grad_norm_value = grad_norm_arr[3].item()

        grad_norm_value = np.sum (norm_lst)
        # print ("grad_norm_value", idx, grad_norm_value)

        # Compute zero proxies
        genotype.append(grad_norm_value)

        init_archt_genotype.append(genotype)
        ind_grad_lst.append(grad_norm_value)




    model_trainx = init_archt_genotype
    model_trainy = init_val_rmse



    return  model_trainx, model_trainy


def init_geno_load (init_train_log_filepath, X_train,Y_train, n_samples, obj, ep, subdata, window_Size, bs, seed):

    epochs = ep

    max_rul = 125
    output_sequence_length = 1  
    time_step = window_Size+2  
    input_size = 14 

    # Data for training model-based predictor
    df_init = pd.read_csv(init_train_log_filepath)
    init_val_rmse = df_init["val_rmse"] 



    init_archt_genotype = []

    for idx, row in df_init.iterrows():



        # genotype = [int(row['params_1']), int(row['params_2']), int(row['params_3']), int(row['params_4']), int(row['params_5']), int(row['params_6']), int(row['params_7']), int(row['params_8']), int(row['params_9']), int(row['params_10']), int(row['params_11']), int(row['params_12']), int(row['num_params']), row['train_rmse']]

        genotype = [int(row['params_1']), int(row['params_2']), int(row['params_3']), int(row['params_4']), int(row['params_5']), int(row['params_6']), int(row['params_7']), int(row['params_8']), int(row['params_9']), int(row['params_10']), int(row['params_11'])]



        init_archt_genotype.append(genotype)






    return  init_archt_genotype



def geno2sample(genotype, X_train, Y_train,  ep, bs, window_Size):

    epochs = ep

    max_rul = 125
    output_sequence_length = 1  
    time_step = window_Size+2  
    input_size = 14 


    train_dataset = TensorDataset(X_train,Y_train)
    train_loader = Data.DataLoader(dataset=train_dataset,batch_size = bs,shuffle=False)


    model = pheno_gen(genotype, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length)

    criterion = nn.MSELoss()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    genotype.append(pytorch_total_params)


    ######### calculate architecture score #########
    # Load data
    train_sample_array, train_label_array = next(iter(train_loader))
    grad_norm_arr = compute_snip_per_weight(model, train_sample_array, train_label_array, max_rul, criterion, split_data=1)
    # grad_norm_arr = get_grad_norm_arr(model, train_sample_array, train_label_array, max_rul, criterion, split_data=1, skip_grad=False)
    # grad_norm_arr = compute_synflow_per_weight(model, train_sample_array, train_label_array, mode = 'param')

    norm_lst = []
    for item in grad_norm_arr:
        temp = item.cpu().detach().numpy()        
        temp = np.sum(temp)
        norm_lst.append(temp)

    # print ("norm_lst", norm_lst)
    # grad_norm_value = grad_norm_arr[3].item()

    grad_norm_value = np.sum (norm_lst)
    # print ("grad_norm_value", grad_norm_value)

    # Compute zero proxies
    genotype.append(grad_norm_value)





    return  genotype



def geno2snip(genotype, X_train, Y_train,  ep, bs, window_Size):

    epochs = ep

    max_rul = 125
    output_sequence_length = 1  
    time_step = window_Size+2  
    input_size = 14 


    train_dataset = TensorDataset(X_train,Y_train)
    train_loader = Data.DataLoader(dataset=train_dataset,batch_size = bs,shuffle=False)


    model = pheno_gen(genotype, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length)

    criterion = nn.MSELoss()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    genotype.append(pytorch_total_params)


    ######### calculate architecture score #########
    # Load data
    train_sample_array, train_label_array = next(iter(train_loader))
    grad_norm_arr = compute_snip_per_weight(model, train_sample_array, train_label_array, max_rul, criterion, split_data=1)
    # grad_norm_arr = get_grad_norm_arr(model, train_sample_array, train_label_array, max_rul, criterion, split_data=1, skip_grad=False)
    # grad_norm_arr = compute_synflow_per_weight(model, train_sample_array, train_label_array, mode = 'param')

    norm_lst = []
    for item in grad_norm_arr:
        temp = item.cpu().detach().numpy()        
        temp = np.sum(temp)
        norm_lst.append(temp)

    # print ("norm_lst", norm_lst)
    # grad_norm_value = grad_norm_arr[3].item()

    grad_norm_value = np.sum (norm_lst)




    return  grad_norm_value