#!/bin/python3
"""
This file contains the implementation of a Task, used to load the data and compute the fitness of an individual
"""
import pandas as pd
from abc import abstractmethod
import time

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as Data
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader
# from input_creator import input_gen
from utils.transformer_net_gpu import *

from utils.pytorchtools import EarlyStopping

from utils.pheno_generator import pheno_gen
from utils.model_sample_creator import geno2sample
from utils.omni_prediction import *

from utils.snip import *
from utils.grad_norm import *
from utils.synflow import *
from utils.predictors import *

import copy

# from utils.pseudoInverse import pseudoInverse


def release_list(lst):
   del lst[:]
   del lst

class Task:
    @abstractmethod
    def get_n_parameters(self):
        pass

    @abstractmethod
    def get_parameters_bounds(self):
        pass

    @abstractmethod
    def evaluate(self, genotype):
        pass


class SimpleNeuroEvolutionTask(Task):
    '''
    TODO: Consider hyperparameters of ELM instead of the number of neurons in hidden layers of MLPs.
    Class for EA Task
    '''
    def __init__(self, train_sample_array, train_label_array, val_sample_array, val_label_array, pred_trainX, pred_trainY, lr, batch, epochs, model_path, device, obj, trial, window_Size):
        self.train_sample_array = train_sample_array
        self.train_label_array = train_label_array
        self.val_sample_array = val_sample_array
        self.val_label_array = val_label_array
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.model_path = model_path
        self.device = device
        self.obj = obj
        self.trial = trial
        self.window_Size = window_Size

        self.pred_trainX = pred_trainX
        self.pred_trainY = pred_trainY

        #Dataloader 
        self.train_dataset = TensorDataset(self.train_sample_array,self.train_label_array)
        self.train_loader = Data.DataLoader(dataset=self.train_dataset,batch_size = self.batch,shuffle=False)
        self.val_dataset = TensorDataset(self.val_sample_array,self.val_label_array)
        self.val_loader = Data.DataLoader(dataset=self.val_dataset,batch_size = self.batch,shuffle=False)

        self.trained_predictor = omni_creator_manual (self.train_sample_array, self.train_label_array, self.pred_trainX, self.pred_trainY, self.window_Size, self.train_sample_array[0].shape[0], self.train_sample_array[0].shape[1], max_rul=125, epochs = self.epochs, output_sequence_length = 1, pred_model= "NGB", bs=256)
        

    def get_n_parameters(self):
        return 11

    def get_parameters_bounds(self):
        bounds = [
            (6, 25), #d_model
            (6, 25), #d_ks
            (6, 25), #d_vs
            (6, 25), #fc_s
            (6, 25), #fc_t
            (6, 25), #fc_d
            (1, 16), #head_s
            (1, 16), #head_t
            (1, 16), #head_d            
            (1, 3), #num_enc_layers
            (1, 3), #num_dec_layers
        ]
        return bounds

    def evaluate(self, genotype):
        '''
        Create input & generate NNs & calculate fitness (to evaluate fitness of each individual)
        :param genotype:
        :return:
        '''
        print ("######################################################################################")

        learning_rate = self.lr
        time_step = self.train_sample_array[0].shape[0]
        input_size = self.train_sample_array[0].shape[1]

        # print("learning_rate: " ,learning_rate)
        # print ("time_step", time_step)
        # print ("input_size", input_size)

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

        max_rul = 125
        epochs = self.epochs
        output_sequence_length = 1  


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
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        geno_copy = copy.deepcopy(genotype)

        query_in = geno2sample(geno_copy, self.train_sample_array, self.train_label_array,  epochs, self.batch, self.window_Size)
        query_geno = []
        query_geno.append(query_in)
        print ("query_in", query_in)

        query_output = self.trained_predictor.query(query_geno)
        # print ("query_output", query_output)

        if isinstance(query_output, np.ndarray):
            query_output = query_output.item()

        # if query_output <= 8:
        #     query_output = 15.0

        if self.obj == "soo":
            # fitness = (val_penalty,)
            fitness = (query_output,)
        elif self.obj == "nsga2":
            # fitness = (val_value, sum(num_neuron_lst))
            fitness = (query_output, num_neuron)                

        print("fitness: ", fitness)

        model = None
        del model

        return fitness





    def update_pred_sample(self, invalid_ind):








        # self.trained_predictor = omni_creator_manual (self.train_sample_array, self.train_label_array, self.pred_trainX, self.pred_trainY, self.window_Size, self.train_sample_array[0].shape[0], self.train_sample_array[0].shape[1], max_rul=125, epochs = self.epochs, output_sequence_length = 1, pred_model= "NGB", bs=256)
        

        print ("temp")
        # self.pred_trainX.append()
        # self.pred_trainY.append()


        return