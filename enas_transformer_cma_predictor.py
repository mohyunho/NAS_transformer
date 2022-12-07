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
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform
import glob
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt

import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as Data
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader

from utils.transformer_utils_gpu import *
from utils.transformer_net_gpu import *

from utils.transformer_task_predictor import SimpleNeuroEvolutionTask
from utils.ea_predictor import GeneticAlgorithm

from utils.model_sample_creator import archt_val_pair, init_geno_load

# random seed predictable
jobs = 1

# path and directories

current_dir = os.path.dirname(os.path.abspath(__file__))

# data_process.py
data_prep_dir = os.path.join(current_dir, 'data_prep')

model_folder = os.path.join(current_dir, 'Models')


pic_dir = os.path.join(current_dir, 'Figures')
directory_path = os.path.join(current_dir, 'EA_log')

if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

if not os.path.exists(directory_path):
    os.makedirs(directory_path)    

if not os.path.exists(model_folder):
    os.makedirs(model_folder)  

if not os.path.exists(directory_path):
    os.makedirs(directory_path)  

model_temp_path = os.path.join(current_dir, 'Models', 'convELM_rep.h5')
torch_temp_path = os.path.join(current_dir, 'torch_model')

######################### Functions #########################
#Myscore function
def myScore(Target, Pred):
    tmp1 = 0
    tmp2 = 0
    for i in range(len(Target)):
        if Target[i] > Pred[i]:
            tmp1 = tmp1 + math.exp((-Pred[i] + Target[i]) / 13.0) - 1
        else:
            tmp2 = tmp2 + math.exp((Pred[i] - Target[i]) / 10.0) - 1
    tmp = tmp1 + tmp2
    return tmp


def shuffle_array(sample_array, label_array):
    ind_list = list(range(len(sample_array)))
    ind_list = shuffle(ind_list)
    print("Shuffeling in progress")
    shuffle_sample = sample_array[ind_list, :, :]
    shuffle_label = label_array[ind_list,]
    return shuffle_sample, shuffle_label

def release_list(a):
   del a[:]
   del a

def recursive_clean(directory_path):
    """clean the whole content of :directory_path:"""
    if os.path.isdir(directory_path) and os.path.exists(directory_path):
        files = glob.glob(directory_path + '*')
        for file_ in files:
            if os.path.isdir(file_):
                recursive_clean(file_ + '/')
            else:
                os.remove(file_)


def tensor_type_checker(tensor, device):
    if torch.cuda.is_available():
        tensor = tensor.to(device)
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")
    return tensor

##################################################

def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='enas transformer')
    parser.add_argument('--subdata', type=str, default="001", help='subdataset of CMAPSS')
    parser.add_argument('-w', type=int, default=40, help='sequence length', required=True)
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-ep', type=int, default=200, help='max epochs')

    parser.add_argument('-bs', type=int, default=256, help='batch size')
    parser.add_argument('-pt', type=int, default=10, help='patience')
    parser.add_argument('-vs', type=float, default=0.2, help='validation split')
    parser.add_argument('-lr', type=float, default=10**(-1*4), help='learning rate')


    parser.add_argument('--pop', type=int, default=20, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=20, required=False, help='generations of evolution')
    parser.add_argument('--device', type=str, default="cuda", help='Use "basic" if GPU with cuda is not available')
    parser.add_argument('--obj', type=str, default="soo", help='Use "soo" for single objective and "moo" for multiobjective')
    parser.add_argument('-t', type=int, default=0, help='trial')

    parser.add_argument('-n_samples', type=int, default=100, help='number of samples for initialization')
    parser.add_argument('--min', type=float, default=100, help='min query value')
    parser.add_argument('-n_val', type=int, default=20, help='number of samples for initialization')


    args = parser.parse_args()

    ############## Input arguments ##############
    window_Size = args.w
    win_stride = args.s
    device = args.device

    print(f"Using {device} device")

    lr = args.lr
    bs = args.bs
    ep = args.ep
    pt = args.pt
    vs = args.vs

    validation_numb = args.n_val

    subdata_idx = args.subdata
    subdata = "FD" + subdata_idx

    obj = args.obj
    trial = args.t
    n_samples = args.n_samples

    # random seed predictable
    jobs = 1
    seed = trial 
    np.random.seed(seed)
    random.seed(seed)

    min_query = args.min



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

    print ("X_train.shape", X_train.shape)
    print ("X_val.shape", X_val.shape)
    
    X_test = sio.loadmat(X_test_path)
    X_test = X_test['test1X_new']
    print ("X_test.shape", X_test.shape)
    X_test = X_test.reshape(len(X_test),window_Size+2,14)
    print ("X_test.shape", X_test.shape)

    Y_test = sio.loadmat(Y_test_path)
    Y_test = Y_test['test1Y']
    Y_test = Y_test.transpose()

    print(torch.cuda.is_available())


    X_train = X_train.astype(np.float16)
    Y_train = Y_train.astype(np.float16)
    X_val = X_val.astype(np.float16)
    Y_val = Y_val.astype(np.float16)
    X_test = X_test.astype(np.float16)
    Y_test = Y_test.astype(np.float16)



    if torch.cuda.is_available():
        X_train = torch.Tensor(X_train).to(device)
        Y_train = torch.Tensor(Y_train).to(device)
        X_val = torch.Tensor(X_val).to(device)
        Y_val = torch.Tensor(Y_val).to(device)
        X_test = torch.Tensor(X_test).to(device)    
        Y_test = torch.Tensor(Y_test).to(device)

    else:
        X_train = Variable(torch.Tensor(X_train).float())
        Y_train = Variable(torch.Tensor(Y_train).float())
        X_val = Variable(torch.Tensor(X_val).float())
        Y_val = Variable(torch.Tensor(Y_val).float())
        X_test = Variable(torch.Tensor(X_test).float())
        Y_test = Variable(torch.Tensor(Y_test).float())



    ######################## Parameters for the GA ###############################
    
    pop_size = args.pop
    n_generations = args.gen
    cx_prob = 0.5  # 0.25
    mut_prob = 0.5  # 0.7
    cx_op = "one_point"
    mut_op = "uniform"

    if obj == "soo":
        sel_op = "best"
        other_args = {
            'mut_gene_probability': 0.3  # 0.1
        }

        mutate_log_path = os.path.join(directory_path, 'mute_log_test_%s_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, subdata, trial))
        mutate_log_col = ['idx', 'params_1', 'params_2', 'params_3', 'params_4', 'params_5', 'params_6', 'params_7', 'params_8', 'params_9', 'params_10', 'params_11', 'fitness_1',
                          'gen']
        mutate_log_df = pd.DataFrame(columns=mutate_log_col, index=None)
        mutate_log_df.to_csv(mutate_log_path, index=False)

        def log_function(population, gen, hv=None, mutate_log_path=mutate_log_path):
            for i in range(len(population)):
                indiv = population[i]
                if indiv == []:
                    "non_mutated empty"
                    pass
                else:
                    # print ("i: ", i)
                    indiv.append(indiv.fitness.values[0])
                    indiv.append(gen)

            temp_df = pd.DataFrame(np.array(population), index=None)
            temp_df.to_csv(mutate_log_path, mode='a', header=None)
            print("population saved")
            return


    # elif obj == "moo":
    else:
        sel_op = "nsga2"
        other_args = {
            'mut_gene_probability': 0.4  # 0.1
        }
        mutate_log_path = os.path.join(directory_path, 'mute_log_test_%s_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, subdata, trial))
        mutate_log_col = ['idx', 'params_1', 'params_2', 'params_3', 'params_4', 'fitness_1',
                          'gen']
        mutate_log_df = pd.DataFrame(columns=mutate_log_col, index=None)
        mutate_log_df.to_csv(mutate_log_path, index=False)

        def log_function(population, gen, hv=None, mutate_log_path=mutate_log_path):
            for i in range(len(population)):
                indiv = population[i]
                if indiv == []:
                    "non_mutated empty"
                    pass
                else:
                    # print ("i: ", i)
                    indiv.append(indiv.fitness.values[0])
                    indiv.append(indiv.fitness.values[1])
                    # append val_rmse
                    indiv.append(hv)
                    indiv.append(gen)

            temp_df = pd.DataFrame(np.array(population), index=None)
            temp_df.to_csv(mutate_log_path, mode='a', header=None)
            print("population saved")
            return

    prft_path = os.path.join(directory_path, 'prft_out_ori_%s_%s_%s_%s.csv' % (pop_size, n_generations, subdata, trial))

    start = time.time()



    init_train_log_filepath = os.path.join(directory_path, 'initialization_rmse_%s_%s_%s_%s_%s_%s_%s.csv' % (validation_numb, pt, n_samples, obj, subdata, ep, seed))
    model_trainx, model_trainy = archt_val_pair (init_train_log_filepath, X_train,Y_train, n_samples, obj, ep, subdata, window_Size, bs, seed)
    individual_seed = init_geno_load (init_train_log_filepath, X_train,Y_train, n_samples, obj, ep, subdata, window_Size, bs, seed)





    # Assign & run EA
    task = SimpleNeuroEvolutionTask(
        train_sample_array = X_train,
        train_label_array = Y_train,
        val_sample_array = X_val,
        val_label_array = Y_val,
        pred_trainX = model_trainx,
        pred_trainY = model_trainy,
        lr = lr,
        epochs = ep,
        batch=bs,
        model_path = model_temp_path,
        device = device,
        obj = obj,
        trial = trial,
        window_Size = window_Size
    )

    # aic = task.evaluate(individual_seed)

    ga = GeneticAlgorithm(
        task=task,
        population_size=pop_size,
        n_generations=n_generations,
        cx_probability=cx_prob,
        mut_probability=mut_prob,
        min_query=min_query,
        crossover_operator=cx_op,
        mutation_operator=mut_op,
        selection_operator=sel_op,
        jobs=jobs,
        log_function=log_function,
        prft_path=prft_path,
        **other_args
    )

    pop, log, hof, prtf = ga.run()

    print("Best individual:")
    print(hof[0])
    print(prtf)

    # Save to the txt file
    # hof_filepath = tmp_path + "hof/best_params_fn-%s_ps-%s_ng-%s.txt" % (csv_filename, pop_size, n_generations)
    # with open(hof_filepath, 'w') as f:
    #     f.write(json.dumps(hof[0]))

    print("Best individual is saved")
    end = time.time()
    print("EA time: ", end - start)
    print ("####################  EA COMPLETE / HOF TEST   ##############################")


    ############### Load saved HOF train & test for obtaining test RMSE ###################





    results_lst = []
    prft_lst = []
    hv_trial_lst = []
    params_trial_lst = []
    prft_trial_lst = []
    ########################################


    for file in sorted(os.listdir(directory_path)):
        if file.startswith('mute_log_test_%s_%s_%s_%s' % (pop_size, n_generations, obj, subdata)):
            print ("path1: ", file)
            mute_log_df = pd.read_csv(os.path.join(directory_path, file))
            results_lst.append(mute_log_df)
        elif file.startswith("prft_out_28_30"):
            print("path2: ", file)
            prft_log_df = pd.read_csv(os.path.join(directory_path, file), header=0, names=["p1", 'p2', 'p3', 'p4'])
            prft_lst.append(prft_log_df)



    for loop_idx in range(len(results_lst)):
        print ("loop_idx", loop_idx)
        print ("file %s in progress..." %loop_idx)
        mute_log_df = results_lst[loop_idx]

        params_temp_lst =[]
        for idx, row in mute_log_df.iterrows():
            num_params = int(50*row["params_7"]) 
            params_temp_lst.append(num_params)

        mute_log_df["params"] = params_temp_lst
        ####################
        avgfit_lst = []
        avgparams_lst = []

        for i in mute_log_df['gen'].unique():
            hv_temp = mute_log_df.loc[mute_log_df['gen'] == i]['fitness_1'].values
            hv_value = sum(hv_temp) / len(hv_temp)
            avgfit_lst.append(hv_value)

            params_temp = mute_log_df.loc[mute_log_df['gen'] == i]['params'].values
            params_value = sum(params_temp) / len(params_temp)
            avgparams_lst.append(params_value)

        hv_trial_lst.append(avgfit_lst)
        # print(norm_hv)
        params_trial_lst.append(avgparams_lst)



    hv_gen = np.stack(hv_trial_lst)
    hv_gen_lst = []

    params_gen = np.stack(params_trial_lst)
    params_gen_lst = []


    for g in range(hv_gen.shape[1]):
        hv_temp =hv_gen[:,g]
        hv_gen_lst.append(hv_temp)

    for p_i in range(params_gen.shape[1]):
        pi_temp =params_gen[:,p_i]
        params_gen_lst.append(pi_temp)


    # print (hv_gen_lst)
    # print (len(hv_gen_lst))

    # fig_verify = plt.figure(figsize=(7, 5))

    fig_verify, ax1 = plt.subplots()
    fig_verify.set_figheight(7)
    fig_verify.set_figwidth(5)



    # ax2 = ax1.twinx()




    print ("hv_gen_lst", hv_gen_lst)
    mean_hv = np.array([np.mean(a) for a in hv_gen_lst])
    print ("mean_hv", mean_hv)
    mean_params = np.array([np.mean(a) for a in params_gen_lst])

    n_generations = len(mean_hv)

    x_ref = range(0, n_generations )
    plt.xticks(x_ref, fontsize=10, rotation=60)
    
    print ("len(mean_hv)", len(mean_hv))
    print ("len(x_ref)", len(x_ref))

    print ("mean_params", mean_params)

    print ("len(hv_trial_lst) ", len(hv_trial_lst) )

    if len(hv_trial_lst) == 1:    
        # plt.plot(x_ref, mean_hv, color='red', linewidth=1, label = 'Mean')
        ax1.plot(x_ref, mean_hv, color='red', linewidth=1, label = 'Validation RMSE')
        # ax2.plot(x_ref, mean_params, color='blue', linewidth=1, label = 'No. parameters')

    else:
        plt.plot(x_ref, mean_hv, color='red', linewidth=1, label = 'Mean')
        std_hv = np.array([np.std(a) for a in hv_gen_lst])
        plt.fill_between(x_ref, mean_hv-std_hv, mean_hv+std_hv,
            alpha=0.15, facecolor=(1.0, 0.8, 0.8))
        plt.plot(x_ref, mean_hv-std_hv, color='black', linewidth= 0.5, linestyle='dashed')
        plt.plot(x_ref, mean_hv+std_hv, color='black', linewidth= 0.5, linestyle='dashed', label = 'Std')




    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Fitness', color='red')
    # ax2.set_ylabel('No. parameters', color='blue')

    
    plt.yticks(fontsize=11)

    # plt.ylabel("Fitness", fontsize=16)
    # plt.xlabel("Generations", fontsize=16)

    # plt.legend(loc='upper right', fontsize=15)
    ax1.legend(loc=0)
    # ax2.legend(loc=0)
    fig_verify.savefig(os.path.join(pic_dir, 'fitness_plot_omni_%s_%s_%s.png' % (pop_size, n_generations, subdata)), dpi=1000,
                    bbox_inches='tight')
    fig_verify.savefig(os.path.join(pic_dir, 'fitness_plot_omni_%s_%s_%s.eps' % (pop_size, n_generations, subdata)), dpi=1000,
                    bbox_inches='tight')









if __name__ == '__main__':
    main()
