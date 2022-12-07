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
import random
import torch
import torch.nn as nn

from utils.transformer_utils_gpu import *
from utils.transformer_net_gpu import *



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