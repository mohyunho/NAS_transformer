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


current_dir = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(current_dir, 'Figures')
# Log file path of EA in csv
ea_log_path = os.path.join(current_dir, 'EA_log')
data_prep_dir = os.path.join(current_dir, 'data_prep')



def shuffle_array(sample_array, label_array):
    ind_list = list(range(len(sample_array)))
    ind_list = shuffle(ind_list)
    print("Shuffeling in progress")
    shuffle_sample = sample_array[ind_list, :, :]
    shuffle_label = label_array[ind_list,]
    return shuffle_sample, shuffle_label


def rank_corr_test(df_full, n_samples, obj, subdata, ep, seed, init='True', proxy_col='snip'):


    rmse_fulltrain = df_full["val_rmse"] 
    # rmse_fulltrain = df_full["train_rmse"] 
    rmse_test = df_full["test_rmse"]

    # num_params = df_full["num_params"]

    archt_score = df_full[proxy_col]


    # order = archt_score.argsort()
    # rank_archtscore = order.argsort() +1

    order = (-archt_score).argsort()
    rank_archtscore = order.argsort() +1

    order = rmse_fulltrain.argsort()
    rank_fulltrain = order.argsort() +1

    # order = num_params.argsort()
    # rank_numbp = order.argsort() +1

    order = rmse_test.argsort()
    rank_test = order.argsort() +1 

    df_full["rank_fulltrain"] = rank_fulltrain 
    df_full["rank_test"] = rank_test 
    df_full[proxy_col] = archt_score
    df_full["rank_score"] = rank_archtscore 

    if init=='True':
        df_full.to_csv(os.path.join(ea_log_path, 'rank_init_omni_%s_%s_%s_%s_%s.csv' % (n_samples, obj, subdata, ep, seed)))
    else:
        df_full.to_csv(os.path.join(ea_log_path, 'rank_query_omni_%s_%s_%s_%s_%s.csv' % (n_samples, obj, subdata, ep, seed)))



    tau, p_value = stats.kendalltau(rank_fulltrain, rank_archtscore)
    print ("tau", tau)
    print ("p_value", p_value)


    rho, p = spearmanr(df_full['rank_fulltrain'], df_full['rank_score'])
    print("rho", rho)
    print("p", p)

    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    # cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot

    # x_min = int(min(df_full['val_rmse'])) - 1
    # x_max = int(max(df_full['val_rmse'])) + 1

    x_min = 9
    x_max = 18


    y_min = int(min(df_full[proxy_col])) - 1000
    y_max = int(max(df_full[proxy_col])) + 1000

    if subdata == "FD002"or subdata == "FD004":
        x_min = int(min(df_full['val_rmse'])) - 1
        x_max = int(max(df_full['val_rmse'])) + 1


    ax.scatter(df_full['val_rmse'], df_full[proxy_col], facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )

    # ax.scatter(df_full['train_rmse'], df_full['test_rmse'], facecolor=(1.0, 1.0, 0.4),
    #            edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("tau %s and rho %s" %(round(tau,2),round(rho,2)), fontsize=15)
    ax.set_xlabel('Validation RMSE with GD', fontsize=12)
    # ax.set_xlabel('Train loss with GD', fontsize=12)
    ax.set_ylabel('Architecture %s' %proxy_col, fontsize=12)
    # ax.legend(fontsize=9)
    # Save figure
    # ax.set_rasterized(True)


    if init=='True':
        fig.savefig(os.path.join(pic_dir, 'corr_init_val_%s_%s_%s_%s_%s.png' % (n_samples, obj, subdata, ep, seed)),  bbox_inches='tight')
    else:
        fig.savefig(os.path.join(pic_dir, 'corr_query_val_%s_%s_%s_%s_%s.png' % (n_samples, obj, subdata, ep, seed)),  bbox_inches='tight')

    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')

    tau, p_value = stats.kendalltau(rank_fulltrain, rank_archtscore)
    print ("tau", tau)
    print ("p_value", p_value)


    rho, p = spearmanr(df_full['rank_test'], df_full['rank_score'])
    print("rho", rho)
    print("p", p)

    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    # cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot

    # x_min = int(min(df_full['test_rmse'])) - 1
    # x_max = int(max(df_full['test_rmse'])) + 1

    x_min = 9
    x_max = 18


    y_min = int(min(df_full[proxy_col])) - 1000
    y_max = int(max(df_full[proxy_col])) + 1000


    if subdata == "FD002"or subdata == "FD004":
        x_min = int(min(df_full['test_rmse'])) - 1
        x_max = int(max(df_full['test_rmse'])) + 1


    ax.scatter(df_full['test_rmse'], df_full[proxy_col], facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )

    # ax.scatter(df_full['train_rmse'], df_full['test_rmse'], facecolor=(1.0, 1.0, 0.4),
    #            edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("tau %s and rho %s" %(round(tau,2),round(rho,2)), fontsize=15)
    ax.set_xlabel('Test RMSE with GD', fontsize=12)
    # ax.set_xlabel('Train loss with GD', fontsize=12)
    ax.set_ylabel('Architecture %s' %proxy_col, fontsize=12)
    # ax.legend(fontsize=9)
    # Save figure
    # ax.set_rasterized(True)

    if init=='True':
        fig.savefig(os.path.join(pic_dir, 'corr_init_test_%s_%s_%s_%s_%s.png' % (n_samples, obj, subdata, ep, seed)),  bbox_inches='tight')
    else:
        fig.savefig(os.path.join(pic_dir, 'corr_query_test_%s_%s_%s_%s_%s.png' % (n_samples, obj, subdata, ep, seed)),  bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')





    # tau, p_value = stats.kendalltau(rank_fulltrain, rank_numbp)
    # print ("tau", tau)
    # print ("p_value", p_value)


    # rho, p = spearmanr(df_full['rank_fulltrain'], df_full['num_params'])
    # print("rho", rho)
    # print("p", p)

    # # Draw scatter plot
    # fig = matplotlib.figure.Figure(figsize=(3, 3))
    # agg.FigureCanvasAgg(fig)
    # # cmap = get_cmap(10)
    # ax = fig.add_subplot(1, 1, 1)
    # # Draw scatter plot

    # # x_min = int(min(df_full['val_rmse'])) - 1
    # # x_max = int(max(df_full['val_rmse'])) + 1

    # x_min = 9
    # x_max = 18


    # y_min = int(min(df_full['num_params'])) - 1000000
    # y_max = int(max(df_full['num_params'])) + 1000000



    # ax.scatter(df_full['val_rmse'], df_full['num_params'], facecolor=(1.0, 1.0, 0.4),
    #            edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )

    # # ax.scatter(df_full['train_rmse'], df_full['test_rmse'], facecolor=(1.0, 1.0, 0.4),
    # #            edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_title("tau %s and rho %s" %(round(tau,2),round(rho,2)), fontsize=15)
    # ax.set_xlabel('Validation RMSE with GD', fontsize=12)
    # # ax.set_xlabel('Train loss with GD', fontsize=12)
    # ax.set_ylabel('Numb params' , fontsize=12)
    # # ax.legend(fontsize=9)
    # # Save figure
    # # ax.set_rasterized(True)
    # if init=='True':
    #     fig.savefig(os.path.join(pic_dir, 'corr_init_numbp_%s_%s_%s_%s_%s.png' % (n_samples, obj, subdata, ep, seed)),  bbox_inches='tight')
    # else:
    #     fig.savefig(os.path.join(pic_dir, 'corr_query_numbp_%s_%s_%s_%s_%s.png' % (n_samples, obj, subdata, ep, seed)),  bbox_inches='tight')



    # ########
    # tau, p_value = stats.kendalltau(rank_test, rank_numbp)
    # print ("tau", tau)
    # print ("p_value", p_value)


    # rho, p = spearmanr(df_full['test_rmse'], df_full['num_params'])
    # print("rho", rho)
    # print("p", p)

    # # Draw scatter plot
    # fig = matplotlib.figure.Figure(figsize=(3, 3))
    # agg.FigureCanvasAgg(fig)
    # # cmap = get_cmap(10)
    # ax = fig.add_subplot(1, 1, 1)
    # # Draw scatter plot

    # # x_min = int(min(df_full['val_rmse'])) - 1
    # # x_max = int(max(df_full['val_rmse'])) + 1

    # x_min = 9
    # x_max = 18


    # y_min = int(min(df_full['num_params'])) - 1000000
    # y_max = int(max(df_full['num_params'])) + 1000000



    # ax.scatter(df_full['test_rmse'], df_full['num_params'], facecolor=(1.0, 1.0, 0.4),
    #            edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )

    # # ax.scatter(df_full['train_rmse'], df_full['test_rmse'], facecolor=(1.0, 1.0, 0.4),
    # #            edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_title("tau %s and rho %s" %(round(tau,2),round(rho,2)), fontsize=15)
    # ax.set_xlabel('Test RMSE with GD', fontsize=12)
    # # ax.set_xlabel('Train loss with GD', fontsize=12)
    # ax.set_ylabel('Numb params' , fontsize=12)
    # # ax.legend(fontsize=9)
    # # Save figure
    # # ax.set_rasterized(True)
    # if init=='True':
    #     fig.savefig(os.path.join(pic_dir, 'corr_init_test_numbp_%s_%s_%s_%s_%s.png' % (n_samples, obj, subdata, ep, seed)),  bbox_inches='tight')
    # else:
    #     fig.savefig(os.path.join(pic_dir, 'corr_query_test_numbp_%s_%s_%s_%s_%s.png' % (n_samples, obj, subdata, ep, seed)),  bbox_inches='tight')


    return



def pheno_gen(genotype, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length):



    dim_model = range(4, 4+(60)*4, 4)
    dim_k_s_lst = range(4, 4+(60)*4, 4)
    dim_v_s_lst = range(4, 4+(60)*4, 4)

    fc_s_lst = range(4, 4+(60)*4, 4)
    fc_t_lst = range(4, 4+(60)*4, 4)
    fc_d_lst = range(4, 4+(60)*4, 4)

    dim_m = dim_model[genotype[0]-1]
    dim_k_s = dim_k_s_lst[genotype[1]-1]
    dim_v_s = dim_v_s_lst[genotype[2]-1]
    dim_k_t = dim_k_s
    dim_v_t = dim_v_s
    dim_k_d = dim_k_s
    dim_v_d = dim_v_s
    fc1_s = fc_s_lst[genotype[3]-1]
    fc1_t = fc_t_lst[genotype[4]-1]
    fc1_d = fc_d_lst[genotype[5]-1]
    n_head_s = genotype[6]
    n_head_t = genotype[7]
    n_head_d = genotype[8]
    n_encoder_layers = genotype[9]
    n_decoder_layers = genotype[10]
    dec_seq_len = 4

    # print ("dim_m", dim_m)
    # print ("dim_k_s", dim_k_s)
    # print ("dim_v_s", dim_v_s)
    # print ("fc1_s", fc1_s)
    # print ("fc1_t", fc1_t)
    # print ("fc1_d", fc1_d)
    # print ("n_head_s", n_head_s)
    # print ("n_head_t", n_head_t)
    # print ("n_head_d", n_head_d)
    # print ("n_encoder_layers", n_encoder_layers)
    # print ("n_decoder_layers", n_decoder_layers)
    # print ("dec_seq_len", dec_seq_len)


    ### Phenotype network ########
    seed = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = TransFomer(dim_m, dim_k_s, dim_v_s, n_head_s, fc1_s, dim_k_t, dim_v_t, n_head_t, fc1_t, dim_k_d, dim_v_d, n_head_d, fc1_d, time_step, input_size, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers)

    return model




def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='NAS transformer')
    parser.add_argument('-w', type=int, default=40, help='sequence length', required=True)
    parser.add_argument('-t', type=int, default=0, required=False, help='seed')
    parser.add_argument('--pop', type=int, default=20, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=20, required=False, help='generations of evolution')
    parser.add_argument('--obj', type=str, default="soo", help='Use "soo" for single objective and "moo" for multiobjective')
    parser.add_argument('--subdata', type=str, default="001", help='subdataset of CMAPSS')
    parser.add_argument('--model', type=str, default="MLP", help='model based predictor')
    parser.add_argument('--base', type=str, default="val", help='baseline comparison')
    parser.add_argument('-ep', type=int, default=100, help='max epochs')
    parser.add_argument('-n', type=int, default=100, help='number of initial samples')
    parser.add_argument('-bs', type=int, default=256, help='batch size')
    parser.add_argument('--device', type=str, default="cuda", help='Use "basic" if GPU with cuda is not available')
    parser.add_argument('-n_samples', type=int, default=100, help='number of samples for initialization')
    parser.add_argument('--proxy', type=str, default="snip", help='proxy')
    parser.add_argument('-pt', type=int, default=10, help='patience')
    parser.add_argument('-n_val', type=int, default=20, help='number of samples for initialization')


    args = parser.parse_args()

    device = args.device

    seed = args.t
    obj = args.obj
    pop = args.pop
    gen = args.gen

    bs = args.bs
    pt = args.pt
    ep = args.ep

    subdata_idx = args.subdata
    subdata = "FD" + subdata_idx

    window_Size = args.w
    validation_numb = args.n_val
    pred_model = args.model


    base = args.base
    numb_N = args.n
    n_samples = args.n_samples


    proxy = args.proxy

    trial = args.t

    if proxy == "grad":
        proxy_col = "grad_norm"
    elif proxy == "score":
        proxy_col = "archt_score"
    elif proxy == "jacob":
        proxy_col = "jacob_cov"
    elif proxy == "synflow":
        proxy_col = "synflow"
    elif proxy == "snip":
        proxy_col = "snip"


    ################# Data load #######################
    X_train_path = os.path.join(data_prep_dir, '%s_%s_%s_trainX_new' %(subdata, window_Size, validation_numb))
    X_val_path = os.path.join(data_prep_dir, '%s_%s_%s_valX_new' %(subdata, window_Size, validation_numb))
    X_test_path = os.path.join(data_prep_dir, '%s_%s_testX_new' %(subdata, window_Size))
    Y_train_path = os.path.join(data_prep_dir, '%s_%s_%s_trainY' %(subdata, window_Size, validation_numb) ) 
    Y_val_path = os.path.join(data_prep_dir, '%s_%s_%s_valY' %(subdata, window_Size, validation_numb) ) 
    Y_test_path = os.path.join(data_prep_dir, '%s_%s_testY' %(subdata, window_Size))

    # Load preprocessed data
    X_train = sio.loadmat(X_train_path)  # load sliding window preprocessed and feature extracted (mean value and regression coefficient estimates feature) data

    X_train = X_train['train1X_new']

    print ("X_train.shape", X_train.shape)

    X_train = X_train.reshape(len(X_train),window_Size+2,14)
    Y_train = sio.loadmat(Y_train_path)
    Y_train = Y_train['train1Y']
    Y_train = Y_train.transpose()


    X_train, Y_train = shuffle_array(X_train, Y_train)


    X_val = sio.loadmat(X_val_path)  # load sliding window preprocessed and feature extracted (mean value and regression coefficient estimates feature) data

    X_val = X_val['val1X_new']

    print ("X_val.shape", X_val.shape)

    X_val = X_val.reshape(len(X_val),window_Size+2,14)
    Y_val = sio.loadmat(Y_val_path)
    Y_val = Y_val['val1Y']
    Y_val = Y_val.transpose()


    numb_valX =  X_val.shape[0]



    X_train = X_train.astype(np.float16)
    Y_train = Y_train.astype(np.float16)

    if torch.cuda.is_available():
        X_train = torch.Tensor(X_train).to(device)
        Y_train = torch.Tensor(Y_train).to(device)
    else:
        X_train = Variable(torch.Tensor(X_train).float())
        Y_train = Variable(torch.Tensor(Y_train).float())




    ## baseline (full training)
    # full_train_log_filepath = os.path.join(ea_log_path, 'mute_log_test_%s_%s_%s_%s_%s.csv' % (pop, gen, obj, subdata, seed))
    epochs = ep

    max_rul = 125
    output_sequence_length = 1  
    time_step = window_Size+2  
    input_size = 14 



    init_train_log_filepath = os.path.join(ea_log_path, 'initialization_rmse_%s_%s_%s_%s_%s_%s_%s.csv' % (validation_numb, pt, n_samples, obj, subdata, ep, seed))

 

    # Data for training model-based predictor
    df_init = pd.read_csv(init_train_log_filepath)


    if base == "val":
        init_val_rmse = df_init["val_rmse"] 
    elif base == "avg":
        init_val_rmse = df_init["avg"] 
    elif base == "test":
        init_val_rmse = df_init["test_rmse"] 

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







    df_init[proxy_col] = ind_grad_lst

    rank_corr_test(df_init, n_samples, obj, subdata, ep, seed, init='True', proxy_col='snip')



    ##################################### QUERY ##############################################

    query_train_log_filepath = os.path.join(ea_log_path, 'initialization_rmse_%s_%s_%s_%s_%s_%s_%s.csv' % (validation_numb, pt, n_samples, obj, subdata, ep, seed))
    df_query = pd.read_csv(query_train_log_filepath)

    if base == "val":
        query_val_rmse = df_query["val_rmse"] 

    elif base == "avg":
        query_val_rmse = df_query["avg"] 
    elif base == "test":
        query_val_rmse = df_query["val_rmse"] 


    ind_query_grad_lst = []
    query_archt_genotype = []
    for idx, row in df_query.iterrows():

        train_dataset = TensorDataset(X_train,Y_train)
        train_loader = Data.DataLoader(dataset=train_dataset,batch_size = bs,shuffle=False)



        # genotype = [int(row['params_1']), int(row['params_2']), int(row['params_3']), int(row['params_4']), int(row['params_5']), int(row['params_6']), int(row['params_7']), int(row['params_8']), int(row['params_9']), int(row['params_10']), int(row['params_11']), int(row['params_12']), int(row['num_params']), row['train_rmse']]

        genotype = [int(row['params_1']), int(row['params_2']), int(row['params_3']), int(row['params_4']), int(row['params_5']), int(row['params_6']), int(row['params_7']), int(row['params_8']), int(row['params_9']), int(row['params_10']), int(row['params_11']),  int(row['num_params'])]


        # genotype = [int(row['params_1']), int(row['params_2']), int(row['params_3']), int(row['params_4']), int(row['params_5']), int(row['params_6']), int(row['params_7']), int(row['params_8']), int(row['params_9']), int(row['params_10']), int(row['params_11']), int(row['params_12'])]


        model = pheno_gen(genotype, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length)
        
        criterion = nn.MSELoss()


        ######### calculate architecture score #########
        # Load data
        train_sample_array, train_label_array = next(iter(train_loader))


        grad_norm_arr = compute_snip_per_weight(model, train_sample_array, train_label_array, max_rul, criterion, split_data=1)


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


        query_archt_genotype.append(genotype)

        ind_query_grad_lst.append(grad_norm_value)



    df_query[proxy_col] = ind_query_grad_lst

    rank_corr_test(df_query, n_samples, obj, subdata, ep, seed, init='False', proxy_col='snip')




    ############################################## OMNI Predictor ##############################

    if numb_N == 0 :
        # architecture, archt_genotype
        xtrain =  init_archt_genotype
        # validation results, val_rmse
        ytrain =  init_val_rmse
    else:
        # architecture, archt_genotype
        xtrain = init_archt_genotype[:numb_N]
        # validation results, val_rmse
        ytrain = init_val_rmse[:numb_N]

    numb_N = len(xtrain)

    xval = query_archt_genotype
    yval = query_val_rmse

    # For any method that did not have an architecture encoding already defined (such as the tree-based methods, GP-based methods, 
    # and Bayesian Linear Regression), we use the standard adjacency matrix encoding, which consists of the adjacency matrix of 
    # the architecture along with a one-hot list of the operations

    # Select & load predictor

    start_train = time.time()
    if pred_model == "MLP":
        predictor = MLPPredictor(hpo_wrapper=False, hparams_from_file=False)
        trained_predictor, train_error = predictor.fit(xtrain, ytrain, train_info=None, epochs=1000, loss="mae", verbose=1)
    elif pred_model == "GP":
        predictor = GPPredictor()
        trained_predictor = predictor.fit(xtrain, ytrain, train_info=None)
    elif pred_model == "VARGP":
        predictor = SparseGPPredictor(optimize_gp_hyper=True)
        trained_predictor = predictor.fit(xtrain, ytrain, train_info=None)
    elif pred_model == "NGB":
        predictor = NGBoost(hpo_wrapper=False)
        trained_predictor = predictor.fit(xtrain, ytrain)
    elif pred_model == "LGB":
        predictor = LGBoost(hpo_wrapper=False)
        trained_predictor = predictor.fit(xtrain, ytrain)
    elif pred_model == "RF":
        predictor = RandomForestPredictor(hpo_wrapper=False)
        trained_predictor = predictor.fit(xtrain, ytrain)

    end_train = time.time()

    train_time = end_train - start_train
    train_time = round(train_time, 4)    
    print ("train_time", train_time)

    # print ("train_error", train_error)

    start_query = time.time()

    query_out = predictor.query(xval)


    end_query = time.time()
    query_time = end_query - start_query
    query_time = round(query_time, 4)    
    print ("query_time", query_time)


    # print ("query_out", query_out)
    # print ("ytrain", yval)


    order = (query_out).argsort()
    rank_queryout = order.argsort() +1


    order = (yval).argsort()
    rank_validation = order.argsort() +1


    df_eval = pd.DataFrame([])
    df_eval["val_rmse"] = yval
    df_eval["query_out"] = query_out
    df_eval["val_rmse_rank"] = rank_validation
    df_eval["query_out_rank"] = rank_queryout

    df_eval.to_csv(os.path.join(ea_log_path, 'rank_predictor_internal_%s_%s_%s_%s_%s_%s_%s_%s_%s_OMNI.csv' % (base, pred_model, pop, gen, obj, subdata, ep, seed, numb_N)))


    tau, p_value = stats.kendalltau(rank_queryout, rank_validation)
    print ("tau", tau)
    print ("p_value", p_value)

    rho, p = spearmanr(df_eval['query_out_rank'], df_eval['val_rmse_rank'])
    print("rho", rho)
    print("p", p)


    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    # cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot


    y_min = 9
    y_max = 16
    x_min = 9
    x_max = 16

    if subdata == "FD002"or subdata == "FD004":
        y_min = int(min(df_eval['val_rmse'])) - 1
        y_max = int(max(df_eval['val_rmse'])) + 1
        x_min = int(min(df_eval["query_out"])) - 1
        x_max = int(max(df_eval["query_out"])) + 1

    ax.scatter(df_eval['query_out'], df_eval["val_rmse"], facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("tau %s rho %s \n train %s query %s" %(round(tau,2),round(rho,2), train_time, query_time), fontsize=12)

    if base == "val":
        ax.set_ylabel('Validation RMSE with GD', fontsize=12)
    elif base == "test":
        ax.set_ylabel('Test RMSE with GD', fontsize=12)
    # ax.set_xlabel('Train loss with GD', fontsize=12)
    ax.set_xlabel('Predictor Query (%s)' %pred_model, fontsize=12)
    # ax.legend(fontsize=9)
    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'corr_predictor_internal_%s_%s_%s_%s_%s_%s_%s_%s_%s_OMNI.png' % (base, pred_model, pop, gen, obj, subdata, ep, seed, numb_N)),  bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')



if __name__ == '__main__':
    main()

