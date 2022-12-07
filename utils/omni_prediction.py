'''
Created on April , 2021
@author:
'''

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

import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform
from scipy.stats import spearmanr
import glob
# import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt


import matplotlib.pyplot as plt
import scipy.io as sio
import torch.utils.data.dataloader as Data
import os

from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader

from utils.transformer_utils_gpu import *
from utils.transformer_net_gpu import *

from utils.snip import *
from utils.grad_norm import *
from utils.synflow import *
from utils.predictors import *

from utils.pheno_generator import pheno_gen

import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.backends.backend_agg as agg
import matplotlib.backends.backend_svg as svg

import scipy.stats as stats # https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.kendalltau.html


from naslib.predictors import (
    GCNPredictor,
    GPPredictor,
    LCEPredictor,
    LCEMPredictor,
    LGBoost,
    MLPPredictor,
    NGBoost,
    RandomForestPredictor,
    SparseGPPredictor,
    VarSparseGPPredictor,
    XGBoost,
    GPWLPredictor,
)

def omni_creator (X_train, Y_train, log_filepath, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length, pred_model= "NGB", bs=256):

    df_init = pd.read_csv(log_filepath)
    init_val_rmse = df_init["val_rmse"] 
    init_archt_genotype = []
    ind_grad_lst = []

    for idx, row in df_init.iterrows():
        train_dataset = TensorDataset(X_train,Y_train)
        train_loader = Data.DataLoader(dataset=train_dataset,batch_size = bs,shuffle=False)


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
        print ("grad_norm_value", idx, grad_norm_value)

        # Compute zero proxies
        genotype.append(grad_norm_value)

        init_archt_genotype.append(genotype)
        ind_grad_lst.append(grad_norm_value)


    # architecture, archt_genotype
    xtrain =  init_archt_genotype
    # validation results, val_rmse
    ytrain =  init_val_rmse

    if pred_model == "MLP":
        predictor = MLPPredictor(hpo_wrapper=False, hparams_from_file=False)
        predictor.fit(xtrain, ytrain, train_info=None, epochs=500, loss="mae", verbose=1)
    elif pred_model == "GP":
        predictor = GPPredictor()
        predictor.fit(xtrain, ytrain, train_info=None)
    elif pred_model == "VARGP":
        predictor = SparseGPPredictor(optimize_gp_hyper=True)
        predictor.fit(xtrain, ytrain, train_info=None)
    elif pred_model == "NGB":
        predictor = NGBoost(hpo_wrapper=False)
        predictor.fit(xtrain, ytrain)
    elif pred_model == "LGB":
        predictor = LGBoost(hpo_wrapper=False)
        predictor.fit(xtrain, ytrain)
    elif pred_model == "RF":
        predictor = RandomForestPredictor(hpo_wrapper=False)
        predictor.fit(xtrain, ytrain)


    return predictor



def omni_creator_manual (X_train, Y_train, pred_trainX, pred_trainY, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length, pred_model= "NGB", bs=256):



    # architecture, archt_genotype
    xtrain =  pred_trainX
    # validation results, val_rmse
    ytrain =  pred_trainY

    if pred_model == "MLP":
        predictor = MLPPredictor(hpo_wrapper=False, hparams_from_file=False)
        predictor.fit(xtrain, ytrain, train_info=None, epochs=1000, loss="mae", verbose=1)
    elif pred_model == "GP":
        predictor = GPPredictor()
        predictor.fit(xtrain, ytrain, train_info=None)
    elif pred_model == "VARGP":
        predictor = SparseGPPredictor(optimize_gp_hyper=True)
        predictor.fit(xtrain, ytrain, train_info=None)
    elif pred_model == "NGB":
        predictor = NGBoost(hpo_wrapper=False)
        predictor.fit(xtrain, ytrain)
    elif pred_model == "LGB":
        predictor = LGBoost(hpo_wrapper=False)
        predictor.fit(xtrain, ytrain)
    elif pred_model == "RF":
        predictor = RandomForestPredictor(hpo_wrapper=False)
        predictor.fit(xtrain, ytrain)


    return predictor