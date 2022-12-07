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
    def __init__(self, train_sample_array, train_label_array, val_sample_array, val_label_array, lr, batch, epochs, model_path, device, obj, trial):
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

        #Dataloader 
        self.train_dataset = TensorDataset(self.train_sample_array,self.train_label_array)
        self.train_loader = Data.DataLoader(dataset=self.train_dataset,batch_size = self.batch,shuffle=False)
        self.val_dataset = TensorDataset(self.val_sample_array,self.val_label_array)
        self.val_loader = Data.DataLoader(dataset=self.val_dataset,batch_size = self.batch,shuffle=False)



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

        dim_model = range(4, 4+(30)*4, 4)
        dim_k_s_lst = range(4, 4+(30)*4, 4)
        dim_v_s_lst = range(4, 4+(30)*4, 4)

        fc_s_lst = range(4, 4+(30)*4, 4)
        fc_t_lst = range(4, 4+(30)*4, 4)
        fc_d_lst = range(4, 4+(30)*4, 4)

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


        model = TransFomer(dim_m, dim_k_s, dim_v_s, n_head_s, fc1_s, dim_k_t, dim_v_t, n_head_t, fc1_t, dim_k_d, dim_v_d, n_head_d, fc1_d, time_step, input_size, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        #Training  and testing 
        loss_list = []
        train_loss_list = []
        val_loss_list = []
        test_loss_list = []
        train_time = []
        test_time = []
        model_loss = 1000

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=10, delta= 0.1, verbose=False)
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = [] 

        ###### Train the network and evaluate individual ######
        for epoch in tqdm(range(epochs)):
            #training
            model.train()
            start1 = time.time()
            for i,(X, Y) in enumerate(self.train_loader):
                X = X.to(self.device)
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
            # print('epoch = ',epoch,
            #     'train_loss = ',loss_eopch.item())

            #validation
            with torch.no_grad():
                model.eval()
                prediction_list = []
                label_list = []
                for j ,(batch_x,batch_y) in enumerate(self.val_loader):
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
                # # Y_val_numpy = Y_val.cpu().detach().numpy()
                # test_score = myScore(label_batch_pre*125, out_batch_pre*125)
                # print('val_loss = ', val_loss.item())
                valid_losses.append(val_loss.item())
                valid_loss = np.average(valid_losses)
                avg_valid_losses.append(valid_loss)
                # print ("valid_loss", valid_loss)

            # clear lists to track next epoch
            valid_losses = []
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping at: ", epoch)
                last_epoch = epoch
                break

        print('val_loss = ', val_loss.item())


        ###### Return fitness
        # if last_epoch == epochs:
        #     if self.obj == "soo":
        #         # fitness = (val_penalty,)
        #         fitness = (val_loss.item(),)
        #     elif self.obj == "nsga2":
        #         # fitness = (val_value, sum(num_neuron_lst))
        #         fitness = (val_loss.item(), num_neuron)
        # else:
        #     if self.obj == "soo":
        #         # fitness = (val_penalty,)
        #         fitness = (val_loss_list[-10].item(),)
        #     elif self.obj == "nsga2":
        #         # fitness = (val_value, sum(num_neuron_lst))
        #         fitness = (val_loss_list[-10].item(), num_neuron)

        if self.obj == "soo":
            # fitness = (val_penalty,)
            fitness = (val_loss.item(),)
        elif self.obj == "nsga2":
            # fitness = (val_value, sum(num_neuron_lst))
            fitness = (val_loss.item(), num_neuron)                

        print("fitness: ", fitness)

        model = None
        del model

        return fitness


