# -*- coding: utf-8 -*-
"""
@author: HD

"""
import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import torch.utils.data.dataloader as Data
import os
import copy
import ast


from tqdm import tqdm

from sklearn.utils import shuffle

from torch.autograd import Variable
from utils.transformer_utils_gpu import *
from utils.transformer_net_gpu import *
from torch.utils.data import TensorDataset,DataLoader

import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.backends.backend_agg as agg
import matplotlib.backends.backend_svg as svg
from matplotlib.ticker import FormatStrFormatter

from matplotlib import rc

from utils.pytorchtools import EarlyStopping

from utils.pheno_generator import pheno_gen
from utils.model_sample_creator import geno2sample
from utils.omni_prediction import *

from utils.snip import *
from utils.grad_norm import *
from utils.synflow import *
from utils.predictors import *



from utils.lr_finder import *
from scipy.stats import qmc

# torch.cuda.empty_cache()
# import gc
# gc.collect()

current_dir = os.path.dirname(os.path.abspath(__file__))
data_prep_dir = os.path.join(current_dir, 'data_prep')

model_folder = os.path.join(current_dir, 'Models')
log_dir_path = os.path.join(current_dir, 'EA_log')

pic_dir = os.path.join(current_dir, 'Figures')

if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)


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


def manual_gen():
    geno_lst = []
    geno_lst.append( [25, 25, 25, 25, 25, 25, 16, 16, 16, 3, 3] )

    return geno_lst

# [24, 24, 25, 32, 8, 8, 5, 20, 11, 5, 2]
# [28, 20, 31, 15, 19, 25, 15, 12, 5, 4, 1] 
#  ind_test_17_20_22_20_11_23_20_20_14_5_5
#  [16, 16, 16, 16, 16, 16, 4, 4, 4, 2, 1, 4]
# [25, 25, 25, 25, 25, 25, 16, 16, 16, 3, 3]


def lhc_topk(init_file_path, topk):
    init_log_df = pd.read_csv(init_file_path)
    init_log_df = init_log_df.sort_values(by=['val_rmse'])

    geno_lst = []
    for idx, row in init_log_df.iterrows():
        genotype = [int(row['params_1']), int(row['params_2']), int(row['params_3']), int(row['params_4']), int(row['params_5']), int(row['params_6']), int(row['params_7']), int(row['params_8']), int(row['params_9']), int(row['params_10']), int(row['params_11'])]
        geno_lst.append(genotype)

    return geno_lst[:topk]

def random_gen (topk, n_samples):

    # geno_lst = []

    # for k in range(topk):
    #     params_lst = []
    #     rnd_ubounds = [31, 31 ,31 ,31 ,31 ,31, 21 ,21 ,21 ,6, 6, 11]
    #     for ubound in rnd_ubounds:
    #         params_lst.append(np.random.randint(1, ubound))

    #     geno_lst.append(params_lst)

    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    n_parameters = 12
    sampler = qmc.LatinHypercube(d=n_parameters, seed = seed)

    l_bounds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    u_bounds = [30, 30 ,30 ,30 ,30 ,30, 20 ,20 ,20 ,5, 5, 10]


    geno_lst = sampler.integers(l_bounds=l_bounds, u_bounds=u_bounds, n=n_samples, endpoint=True, workers=1)
    

    return geno_lst[:topk]


## Threshold version
# def topk_query_gen (X_train, Y_train, search_space, topk, trained_predictor, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length, n_samples, bs, min_query):

#     # sp_archt_lst = []

#     # for k in range(search_space):
#     #     params_lst = []
#     #     rnd_ubounds = [31, 31 ,31 ,31 ,31 ,31, 21 ,21 ,21 ,6, 6, 11]
#     #     for ubound in rnd_ubounds:
#     #         params_lst.append(np.random.randint(1, ubound))

#     #     sp_archt_lst.append(params_lst)


#     seed = 0
#     np.random.seed(seed)
#     random.seed(seed)

#     n_parameters = 12
#     sampler = qmc.LatinHypercube(d=n_parameters, seed = seed)

#     l_bounds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
#     u_bounds = [30, 30 ,30 ,30 ,30 ,30, 20 ,20 ,20 ,5, 5, 10]


#     sp_archt_lst = sampler.integers(l_bounds=l_bounds, u_bounds=u_bounds, n=search_space, endpoint=True, workers=1)

#     # genotype_lst = copy.deepcopy(sp_archt_lst)

#     genotype_lst = []
#     # Create phenotype from genotype lst
#     # query by OMNI predictor
#     query_geno = []
#     for genotype in tqdm(sp_archt_lst):

#         genotype = genotype.tolist()
#         genotype_lst.append(genotype)
#         model = pheno_gen(genotype, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length)
#         query_in = geno2sample(genotype, X_train, Y_train,  epochs, bs, window_Size)
#         query_geno.append(query_in)

#         # query_output = trained_predictor.query(genotype)
#         # query_outputs.append(query_output)

#     print ("genotype_lst", genotype_lst)

#     query_outputs = trained_predictor.query(query_geno)
#     print ("query_outputs", query_outputs)
#     # ordering
#     df_temp = pd.DataFrame([])
#     df_temp['archt'] = genotype_lst
#     df_temp['output'] = query_outputs

#     print (df_temp)


#     query_thres = min_query*0.95
#     print ("query_thres", query_thres)
#     df_temp = df_temp.loc[df_temp['output'] > query_thres]

#     df_temp = df_temp.sort_values(by=['output'])

#     print (df_temp)


#     # return topk individual (genotype)
#     geno_lst = df_temp['archt'][:topk]

#     print (geno_lst)


#     return geno_lst


# Descending version
def topk_query_gen (X_train, Y_train, search_space, topk, trained_predictor, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length, n_samples, bs, min_query):

    # sp_archt_lst = []

    # for k in range(search_space):
    #     params_lst = []
    #     rnd_ubounds = [31, 31 ,31 ,31 ,31 ,31, 21 ,21 ,21 ,6, 6, 11]
    #     for ubound in rnd_ubounds:
    #         params_lst.append(np.random.randint(1, ubound))

    #     sp_archt_lst.append(params_lst)


    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    n_parameters = 12
    sampler = qmc.LatinHypercube(d=n_parameters, seed = seed)

    l_bounds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    u_bounds = [30, 30 ,30 ,30 ,30 ,30, 20 ,20 ,20 ,5, 5, 10]


    sp_archt_lst = sampler.integers(l_bounds=l_bounds, u_bounds=u_bounds, n=search_space, endpoint=True, workers=1)

    # genotype_lst = copy.deepcopy(sp_archt_lst)

    genotype_lst = []
    # Create phenotype from genotype lst
    # query by OMNI predictor
    query_geno = []
    for genotype in tqdm(sp_archt_lst):

        genotype = genotype.tolist()
        genotype_lst.append(genotype)
        model = pheno_gen(genotype, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length)
        query_in = geno2sample(genotype, X_train, Y_train,  epochs, bs, window_Size)
        query_geno.append(query_in)

        # query_output = trained_predictor.query(genotype)
        # query_outputs.append(query_output)

    print ("genotype_lst", genotype_lst)

    query_outputs = trained_predictor.query(query_geno)
    print ("query_outputs", query_outputs)
    # ordering
    df_temp = pd.DataFrame([])
    df_temp['archt'] = genotype_lst
    df_temp['output'] = query_outputs

    print (df_temp)

    df_temp = df_temp.loc[df_temp['output'] <= min_query]
    df_temp = df_temp.sort_values(by=['output'], ascending=False)

    print (df_temp)


    # return topk individual (genotype)
    geno_lst = df_temp['archt'][:topk]

    print (geno_lst)


    return geno_lst


# # descending
# def ga_omni_gen (ea_log_file, topk, gen, min_query):

#     # Check the ea log and select ind in the last generation

#     # mutate_filename = 'EA_log/mute_log_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, trial)
#     mute_log_df = pd.read_csv(ea_log_file)
#     last_gen = mute_log_df['gen'].values[-1]
#     print ("last_gen", last_gen)
#     last_gen_df = mute_log_df.loc[mute_log_df['gen'] == last_gen]

#     # drop duplicates
#     last_gen_df = last_gen_df.drop_duplicates(subset=['fitness_1'], keep='last')
#     print ("last_gen_df", last_gen_df)

#     # # ordering
#     # last_gen_df = last_gen_df.sort_values(by=['fitness_1'])
#     # print ("last_gen_df", last_gen_df)

#     last_gen_df = last_gen_df.loc[last_gen_df['fitness_1'] <= min_query]
#     last_gen_df = last_gen_df.sort_values(by=['fitness_1'], ascending=False)
#     print ("last_gen_df", last_gen_df)


#     # return topk individual (genotype)

#     geno_lst = []
#     for idx, row in last_gen_df.iterrows():
#         genotype = [int(row['params_1']), int(row['params_2']), int(row['params_3']), int(row['params_4']), int(row['params_5']), int(row['params_6']), int(row['params_7']), int(row['params_8']), int(row['params_9']), int(row['params_10']), int(row['params_11'])]
#         geno_lst.append(genotype)

#     return geno_lst[:topk]



# ascending
def ga_omni_gen (ea_log_file, topk, gen, min_query):

    # Check the ea log and select ind in the last generation

    # mutate_filename = 'EA_log/mute_log_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, trial)
    mute_log_df = pd.read_csv(ea_log_file)
    last_gen = mute_log_df['gen'].values[-1]
    print ("last_gen", last_gen)
    last_gen_df = mute_log_df.loc[mute_log_df['gen'] == last_gen]

    # drop duplicates
    last_gen_df = last_gen_df.drop_duplicates(subset=['fitness_1'], keep='last')
    print ("last_gen_df", last_gen_df)

    # ordering
    last_gen_df = last_gen_df.sort_values(by=['fitness_1'])
    print ("last_gen_df", last_gen_df)


    # return topk individual (genotype)

    geno_lst = []
    for idx, row in last_gen_df.iterrows():
        genotype = [int(row['params_1']), int(row['params_2']), int(row['params_3']), int(row['params_4']), int(row['params_5']), int(row['params_6']), int(row['params_7']), int(row['params_8']), int(row['params_9']), int(row['params_10']), int(row['params_11'])]
        geno_lst.append(genotype)

    return geno_lst[:topk]


# def ga_retrain (ea_log_file, topk, gen, min_query):

#     # Check the ea log and select ind in the last generation

#     # mutate_filename = 'EA_log/mute_log_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, trial)
#     mute_log_df = pd.read_csv(ea_log_file)
#     last_gen = mute_log_df['gen'].values[-1]

#     last_gen_df = mute_log_df.loc[mute_log_df['gen'] == last_gen]

#     # drop duplicates
#     last_gen_df = last_gen_df.drop_duplicates(subset=['fitness_1'], keep='first')


#     # ordering
#     last_gen_df = last_gen_df.sort_values(by=['fitness_1'])


#     first_gen_lst = []
#     # filtering
#     for idx, row in last_gen_df.iterrows():
#         overlap_df = mute_log_df.loc[(mute_log_df["params_1"] == row["params_1"]) & (mute_log_df["params_2"] == row["params_2"]) & (mute_log_df["params_3"] == row["params_3"]) & (mute_log_df["params_4"] == row["params_4"]) & (mute_log_df["params_5"] == row["params_5"]) & (mute_log_df["params_6"] == row["params_6"]) & (mute_log_df["params_7"] == row["params_7"]) & (mute_log_df["params_8"] == row["params_8"]) & (mute_log_df["params_9"] == row["params_9"]) & (mute_log_df["params_10"] == row["params_10"]) & (mute_log_df["params_11"] == row["params_11"])]

#         overlap_df = overlap_df.drop_duplicates(subset=['fitness_1'], keep='first')
#         # print ("overlap_df", overlap_df)
#         first_gen = overlap_df["gen"].values[0]
#         first_gen_lst.append(first_gen)

#     last_gen_df["gen"] = first_gen_lst


#     last_gen_df = last_gen_df[last_gen_df["gen"] != 10]
#     print ( last_gen_df)
#     # return topk individual (genotype)

#     geno_lst = []
#     for idx, row in last_gen_df.iterrows():
#         genotype = [int(row['params_1']), int(row['params_2']), int(row['params_3']), int(row['params_4']), int(row['params_5']), int(row['params_6']), int(row['params_7']), int(row['params_8']), int(row['params_9']), int(row['params_10']), int(row['params_11'])]
#         geno_lst.append(genotype)

#     return geno_lst[:topk]


def ga_retrain (retrain_log_file, topk, gen, min_query):

    # Check the ea log and select ind in the last generation

    # mutate_filename = 'EA_log/mute_log_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, trial)
    retrain_log_df = pd.read_csv(retrain_log_file)

    # ordering
    retrain_log_df = retrain_log_df.sort_values(by=['val'])
    

    # convert str to list 
    retrain_log_df['history'] = retrain_log_df['history'].apply(lambda x: ast.literal_eval(x))
    geno_lst = retrain_log_df["history"]
    print (geno_lst)
    print (type(geno_lst))

    return geno_lst[:topk]


def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='data preparation')
    parser.add_argument('--subdata', type=str, default="001", help='subdataset of CMAPSS')
    parser.add_argument('-w', type=int, default=40, help='sequence length', required=True)
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-ep_init', type=int, default=200, help='max epochs')
    parser.add_argument('-ep_train', type=int, default=200, help='max epochs')
    parser.add_argument('-t', type=int, default=0, help='trial')
    parser.add_argument('--device', type=str, default="cuda", help='Use "basic" if GPU with cuda is not available')
    # parser.add_argument('-t', type=int, required=True, help='trial')

    parser.add_argument('-bs', type=int, default=256, help='batch size')
    parser.add_argument('-pt', type=int, default=10, help='patience')
    parser.add_argument('-vs', type=float, default=0.2, help='validation split')
    parser.add_argument('-lr', type=float, default=10**(-1*5), help='learning rate')

    parser.add_argument('-topk', type=int, default=10, help='number of samples for evaluation')
    parser.add_argument('--model', type=str, default="NGB", help='model based predictor')
    parser.add_argument('-sp', type=int, default=10**3, help='size of search space')
    parser.add_argument('-n_samples', type=int, default=100, help='number of samples for initialization')
    parser.add_argument('-n_val', type=int, default=20, help='number of samples for initialization')


    parser.add_argument('--obj', type=str, default="soo", help='Use "soo" for single objective and "moo" for multiobjective')
    parser.add_argument('--sc', type=str, default="top_query", help='scenario')
    parser.add_argument('--pop', type=int, default=50, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=20, required=False, help='generations of evolution')
    parser.add_argument('--min', type=float, default=0, help='min query value')


    args = parser.parse_args()

    win_stride = args.s
    device = args.device

    print(f"Using {device} device")

    subdata_idx = args.subdata
    subdata = "FD" + subdata_idx
    window_Size = args.w

    lr = args.lr
    bs = args.bs
    ep_train = args.ep_train
    ep_init = args.ep_init
    pt = args.pt
    vs = args.vs

    validation_numb = args.n_val    

    pop_size = args.pop
    n_generations = args.gen
    obj = args.obj
    
    topk = args.topk

    subdata_idx = args.subdata
    subdata = "FD" + subdata_idx
    trial = args.t
    seed = args.t
    np.random.seed(seed)
    random.seed(seed)

    scenario = args.sc
    search_space = args.sp

    pred_model = args.model
    n_samples = args.n_samples

    min_query = args.min

    pic_ind_dir = os.path.join(pic_dir, 'pic_topk_%s_%s_%s' %(scenario, topk, subdata))
    if not os.path.exists(pic_ind_dir):
        os.makedirs(pic_ind_dir)

    # Reproducibility check



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


    #Dataloader 
    train_dataset = TensorDataset(X_train,Y_train)
    train_loader = Data.DataLoader(dataset=train_dataset,batch_size = bs,shuffle=False)
    val_dataset = TensorDataset(X_val,Y_val)
    val_loader = Data.DataLoader(dataset=val_dataset,batch_size = bs,shuffle=False)
    test_dataset = TensorDataset(X_test,Y_test)
    test_loader = Data.DataLoader(dataset=test_dataset,batch_size = bs,shuffle=False)



    # Prepare OMNI predictor model
    max_rul = 125
    epochs = ep_train
    output_sequence_length = 1  

    time_step = window_Size+2  
    input_size = 14 



    init_train_log_filepath = os.path.join(log_dir_path, 'initialization_rmse_%s_%s_%s_%s_%s_%s_%s.csv' % (validation_numb, pt, n_samples, obj, subdata, ep_init, seed))
    # init_train_log_filepath = os.path.join(ea_log_path, 'log_query_rmse_%s_%s_%s_%s_%s_%s.csv' % (pop, gen, obj, subdata, ep, trial))

    trained_predictor = omni_creator(X_train, Y_train, init_train_log_filepath, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length, pred_model= pred_model, bs=bs)




    seed = 0
    np.random.seed(seed)
    random.seed(seed)


    ea_log_file =  os.path.join(log_dir_path, 'mute_log_test_%s_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, subdata, trial))

    retrain_log_file =  os.path.join(log_dir_path, 'retrain_log_%s_%s_%s.csv' % (pop_size, n_generations, subdata))


    log_col = ['idx', 'params_1', 'params_2', 'params_3', 'params_4', 'params_5', 'params_6', 'params_7', 'params_8', 'params_9', 'params_10', 'params_11', 'fitness_1',
                        'gen']
    # log_df = pd.DataFrame(columns=log_col, index=None)


    ind_test_lst = []
    ind_test_score_lst = []
    ind_val_lst = []
    ind_train_lst = []
    ind_train_time = []
    num_params = []
    lr_lst = []
    ind_query_lst = []

    if scenario == "manual":
        geno_lst = manual_gen()
    elif scenario == "lhc":
        geno_lst = lhc_topk(init_train_log_filepath, topk)    
    elif scenario == "random":
        geno_lst = random_gen (topk, n_samples)
    elif scenario == "top_query":
        geno_lst = topk_query_gen (X_train, Y_train, search_space, topk, trained_predictor, window_Size, time_step, input_size, max_rul, epochs, output_sequence_length, n_samples, bs, min_query)
    elif scenario == "ga_omni":
        geno_lst = ga_omni_gen (ea_log_file, topk, n_generations, min_query)
    elif scenario == "ga_retrain":
        geno_lst = ga_retrain (retrain_log_file, topk, n_generations, min_query)


    elif scenario == "proposed":
        print ("")
    else:
        print ("invaid scenario")
        sys.exit()


    df_save = pd.DataFrame([])
    print ("geno_lst", geno_lst)
    print ("len(geno_lst)", len(geno_lst))


    if scenario == "random" :
        geno_lst_copy = []
    else:
        geno_lst_copy = copy.deepcopy(geno_lst)

    for i, s in enumerate(geno_lst):

        genotype = s
        print ("i", i)
        print ("genotype", genotype)

        if scenario == "random" :
            genotype = genotype.tolist()
            geno_lst_copy.append(genotype)


        query_input = geno2sample(genotype, X_train, Y_train,  epochs, bs, window_Size)

        query_output = trained_predictor.query([query_input])
        print ("query_output", query_output)
        ind_query_lst.append(query_output)

        # max_rul = 125
        # epochs = ep
        # output_sequence_length = 1  

        # time_step = window_Size+2  
        # input_size = 14 

        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []


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

        print ("dim_m", dim_m)
        print ("dim_k_s", dim_k_s)
        print ("dim_v_s", dim_v_s)
        print ("fc1_s", fc1_s)
        print ("fc1_t", fc1_t)
        print ("fc1_d", fc1_d)
        print ("n_head_s", n_head_s)
        print ("n_head_t", n_head_t)
        print ("n_head_d", n_head_d)
        print ("n_encoder_layers", n_encoder_layers)
        print ("n_decoder_layers", n_decoder_layers)
        print ("dec_seq_len", dec_seq_len)


        ### Phenotype network ########
        seed = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        early_stopping = EarlyStopping(patience=10, delta= 0.1, verbose=False)
        # genotype = [int(best_ind_log['params_1'].values[0]), int(best_ind_log['params_2'].values[0]), int(best_ind_log['params_3'].values[0]), int(best_ind_log['params_4'].values[0]), int(best_ind_log['params_5'].values[0])]
        


        # model = TransFomer(dim_m, dim_k_s, dim_v_s, n_head_s, fc1_s, dim_k_t, dim_v_t, n_head_t, fc1_t, dim_k_d, dim_v_d, n_head_d, fc1_d, time_step, input_size, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers)
        model = TransFomer(int(dim_m), int(dim_k_s), int(dim_v_s), int(n_head_s), int(fc1_s), int(dim_k_t), int(dim_v_t), int(n_head_t), int(fc1_t), int(dim_k_d), int(dim_v_d), int(n_head_d), int(fc1_d), int(time_step), int(input_size), int(dec_seq_len), int(output_sequence_length), int(n_decoder_layers), int(n_encoder_layers))





        criterion = nn.MSELoss()
        

        # lr_finder = LRFinder(model, optimizer, criterion, device="cuda")

        # # lr_finder.range_test(train_loader, end_lr=0.0001, num_iter=100)
        # lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=0.0001, num_iter=100, step_mode="linear")

        # found_lr = lr_finder.suggestedlr()
        # found_lr = float(found_lr)
        # print ("found_lr", found_lr)



        # fig, ax, selected_lr= lr_finder.plot() # to inspect the loss-learning rate graph
        # # fig.savefig(os.path.join(pic_dir, 'lr_finder_%s_%s_%s_%s_%s_%s_%s.png' % (pop_size, n_generations, window_Size, lr, ep, subdata, genotype)), dpi=1000,
        # #                 bbox_inches='tight')

        # lr_finder.reset() # to reset the model and optimizer to their initial state



        found_lr = lr


        lr_lst.append(found_lr)

        optimizer = torch.optim.Adam(model.parameters(), lr=found_lr)
        # optimizer = torch.optim.Adam(model.parameters(), lr=found_lr)

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400,4000,6000,8000], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        #Training  and testing 
        loss_list = []
        train_loss_list = []
        val_loss_list = []
        test_loss_list = []
        test_score_list = []
        train_time = []
        test_time = []
        model_loss = 1000

        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        # https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_params.append(pytorch_total_params)
        print ("pytorch_total_params", pytorch_total_params)

        start_training = time.time()

        for epoch in range(epochs):
            #training
            model.train()
            start1 = time.time()
            for i,(X, Y) in enumerate(train_loader):

                X = X.to(device)
                out = model(X)
                # print ("out",out )
                loss = torch.sqrt(criterion(out*max_rul, Y*max_rul))
                optimizer.zero_grad()   
                loss.backward()
                optimizer.step()
                # print ("loss", loss)
                loss_list.append(loss.item())
            end1 = time.time()
            train_time.append(end1 - start1)
            loss_eopch = np.mean(np.array(loss_list)) 
            train_loss_list.append(loss_eopch)
            print('epoch = ',epoch,
                'train_loss = ',loss_eopch.item())


            #validation
            with torch.no_grad():
                model.eval()
                prediction_list = []
                label_list = []
                for j ,(batch_x,batch_y) in enumerate(val_loader):
                    start2= time.time()
                    prediction = model(batch_x)
                    end2 = time.time()
                    test_time.append(end2 - start2)
                    prediction[prediction<0] = 0        
                    prediction_list.append(prediction)
                    label_list.append(batch_y)
                
                pred_cpu = torch.cat(prediction_list).cpu()
                label_cpu = torch.cat(label_list).cpu()
                # out_batch_pre = torch.cat(prediction_list).detach().numpy()
                out_batch_pre = pred_cpu.detach().numpy()
                prediction_tensor = torch.from_numpy(out_batch_pre)   

                label_batch_pre = label_cpu.detach().numpy()
                label_tensor = torch.from_numpy(label_batch_pre)         


                val_loss = torch.sqrt(criterion(prediction_tensor*125, label_tensor*125))
                val_loss_list.append(val_loss)    
                # Y_val_numpy = Y_val.cpu().detach().numpy()
                # test_score = myScore(label_batch_pre*125, out_batch_pre*125)
                print('val_loss = ', val_loss.item())


                valid_losses.append(val_loss.item())
                valid_loss = np.average(valid_losses)
                avg_valid_losses.append(valid_loss)
                print ("valid_loss", valid_loss)


            print ("val_loss.item()", val_loss.item())


            #testing
            with torch.no_grad():
                model.eval()
                prediction_list = []
                for j ,(batch_x,batch_y) in enumerate(test_loader):
                    start2= time.time()
                    prediction = model(batch_x)
                    end2 = time.time()
                    test_time.append(end2 - start2)
                    prediction[prediction<0] = 0        
                    prediction_list.append(prediction)
                    
                pred_cpu = torch.cat(prediction_list).cpu()
                # out_batch_pre = torch.cat(prediction_list).detach().numpy()
                out_batch_pre = pred_cpu.detach().numpy()
                prediction_tensor = torch.from_numpy(out_batch_pre)                
                test_loss = torch.sqrt(criterion(prediction_tensor*125, Y_test.cpu()*125))
                test_loss_list.append(test_loss)    
                Y_test_numpy = Y_test.cpu().detach().numpy()
                test_score = myScore(Y_test_numpy*125, out_batch_pre*125)
                test_score_list.append(test_score)
                print('test_loss = ', test_loss.item(),
                    'test_score = ', test_score)


            valid_losses = []

            scheduler.step()

            early_stopping(valid_loss, model)
                
            if early_stopping.early_stop:
                print("Early stopping at: ", epoch)
                last_epoch = epoch
                break

            # if epoch > 50:
            #     early_stopping(valid_loss, model)
                    
            #     if early_stopping.early_stop:
            #         print("Early stopping at: ", epoch)
            #         last_epoch = epoch
            #         break


        min_val = min(val_loss_list)
        min_val_idx = val_loss_list.index(min_val)
        min_val_value = min_val.item()
        min_test_value = test_loss_list[min_val_idx].item()
        min_train_value = train_loss_list[min_val_idx].item()
        min_test_score = test_score_list[min_val_idx]


        print ("min_val_value", min_val_value)
        print ("min_test_value", min_test_value)
        print ("min_train_value", min_train_value)
        print ("min_test_score", min_test_score)


        ind_val_lst.append(min_val_value)  
        ind_test_lst.append(min_test_value)
        ind_train_lst.append(min_train_value)
        ind_test_score_lst.append(min_test_score)

        # ind_test_lst.append(test_loss.item())
        # ind_val_lst.append(val_loss.item())        

        print ("val_loss.item()", val_loss.item())

        end_training = time.time()
        training_time =  end_training - start_training
        print ("training_time", training_time)
        ind_train_time.append(training_time)


        # plot loss trend
        fig_verify = plt.figure(figsize=(4, 3))

        x_ref = range(1, len(val_loss_list) + 1)
        x_ref_ticks = range(0, len(val_loss_list), 10)

        test_rmse = float(test_loss_list[-1].numpy())
        test_rmse = round(test_rmse,2)
        print ("test_rmse", test_rmse)
        print (type(test_rmse))
        plt.plot(x_ref, train_loss_list, color='black', linewidth=1, label = 'train_rmse')
        plt.plot(x_ref, val_loss_list, color='red', linewidth= 1, linestyle='dashed', label = 'val_rmse')
        plt.plot(x_ref, test_loss_list, color='green', linewidth= 1, linestyle='dashed', label = 'test_rmse')

        plt.text(min_val_idx+0.1, min_test_value+0.1, "%s" %min_test_value, color='green')
        plt.text(min_val_idx+0.5, min_val_value-3, "%s" %min_val_value,  color='red')
        plt.text(min_val_idx+1.0, min_train_value+2, "%s" %min_train_value)


        plt.xticks(x_ref_ticks, fontsize=10, rotation=60)
        # plt.yticks(np.arange(10, int(max(train_loss_list))+1, 1),fontsize=9)
        plt.ylim(8, 25)
        plt.yticks(np.arange(8, 25, 1),fontsize=9)


        plt.ylabel("RMSE", fontsize=12)
        plt.xlabel("Epochs", fontsize=12)
        plt.legend(loc='upper right', fontsize=12)
        fig_verify.savefig(os.path.join(pic_ind_dir, 'ind_test_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.png' % (genotype[0], genotype[1], genotype[2], genotype[3], genotype[4], genotype[5], genotype[6], genotype[7], genotype[8], genotype[9], genotype[10])), dpi=1000,
                        bbox_inches='tight')
        # fig_verify.savefig(os.path.join(pic_dir, 'loss_%s_%s.eps' % (window_Size,  subdata)), dpi=1500,
        #                 bbox_inches='tight')










        del model
        del optimizer

    df_save["genotype"] = geno_lst_copy
    df_save["test_rmse"] = ind_test_lst
    df_save["test_score"] = ind_test_score_lst
    df_save["val_rmse"] = ind_val_lst
    df_save["query_rmse"] = ind_query_lst
    df_save["train_rmse"] = ind_train_lst
    df_save["num_params"] = num_params
    df_save["train_time"] = ind_train_time
    df_save["found_lr"] = lr_lst


    print ("mean val rmse:", df_save["val_rmse"].mean() )
    print ("std val rmse:", df_save["val_rmse"].std() )
    print ("mean query_rmse rmse:", df_save["query_rmse"].mean() )
    print ("std query_rmse rmse:", df_save["query_rmse"].std() )
    print ("mean test rmse:", df_save["test_rmse"].mean() )
    print ("std test rmse:", df_save["test_rmse"].std() )
    print ("mean test score:", df_save["test_score"].mean() )
    print ("std test score:", df_save["test_score"].std() )

    # mutate_log_path = os.path.join(log_dir_path, 'mute_log_test_%s_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, subdata, trial))

    test_filename = os.path.join(log_dir_path, 'test_rmse_topk_%s_%s_%s.csv' %(scenario, topk, subdata))
    df_save.to_csv(test_filename, index=False)


if __name__ == '__main__':
    main()    






