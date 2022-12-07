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

from torch.autograd import Variable
from utils.transformer_utils_gpu import *
from utils.transformer_net_gpu import *
from torch.utils.data import TensorDataset,DataLoader

from utils.pytorchtools import EarlyStopping

from utils.lr_finder import *
from scipy.stats import qmc

# torch.cuda.empty_cache()
# import gc
# gc.collect()


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



def gd_full(genotype, query_out, train_loader, val_loader, epochs, window_Size, lr, device="cuda"):
    


    max_rul = 125
    output_sequence_length = 1  

    time_step = window_Size+2  
    input_size = 14 

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
    dec_seq_len =4

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

    early_stopping = EarlyStopping(patience=10, delta= 0.1, verbose=False)
    # genotype = [int(best_ind_log['params_1'].values[0]), int(best_ind_log['params_2'].values[0]), int(best_ind_log['params_3'].values[0]), int(best_ind_log['params_4'].values[0]), int(best_ind_log['params_5'].values[0])]
    


    model = TransFomer(dim_m, dim_k_s, dim_v_s, n_head_s, fc1_s, dim_k_t, dim_v_t, n_head_t, fc1_t, dim_k_d, dim_v_d, n_head_d, fc1_d, time_step, input_size, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=found_lr)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400,4000,6000,8000], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    #Training  and testing 
    loss_list = []
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    train_time = []
    test_time = []
    model_loss = 1000

    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print ("pytorch_total_params", pytorch_total_params)

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
            # print('val_loss = ', val_loss.item())


            valid_losses.append(val_loss.item())
            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)
            # print ("valid_loss", valid_loss)


        
        # print ("query_out: ", query_out)
        print ("val_loss", val_loss.item())

        valid_losses = []

        scheduler.step()

        early_stopping(valid_loss, model)
                
        if early_stopping.early_stop:
            print("Early stopping at: ", epoch)
            last_epoch = epoch
            break


    min_val = min(val_loss_list)
    min_val_idx = val_loss_list.index(min_val)
    min_val_value = min_val.item()
    min_train_value = train_loss_list[min_val_idx].item()

    print ("query_out: ", query_out)
    print ("min_val_value", min_val_value)
    print ("min_train_value", min_train_value)


    return min_val_value
