import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from utils.transformer_utils_gpu import *

torch.cuda.empty_cache()
import gc
gc.collect()

seed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

class Sensors_EncoderLayer(torch.nn.Module):
    def __init__(self, dim_m, dim_val, dim_attn, n_heads, fc1_s, drop_rate=0.2):
        super(Sensors_EncoderLayer, self).__init__()
        self.attn = Sensor_MultiHeadAttentionBlock(dim_m, dim_val, dim_attn , n_heads)
        self.fc1 = nn.Linear(fc1_s, dim_m, device="cuda")
        self.fc2 = nn.Linear(dim_m, fc1_s, device="cuda")   
        self.norm1 = nn.LayerNorm(dim_m, device="cuda")
        self.norm2 = nn.LayerNorm(dim_m, device="cuda")
        self.dropout1 = nn.Dropout(drop_rate)
        # self.dropout2 = nn.Dropout(drop_rate)
        self.dropout3 = nn.Dropout(drop_rate)
        self.elu1 = nn.ReLU()
        self.elu2 = nn.ReLU()
    
    def forward(self, x):
        a = self.attn(x)
        a = self.dropout1(a)
        x = self.norm1(x + a)  
        x = self.elu1(x)
        a = self.fc1(self.elu2(self.fc2(x))) 
        # a = self.fc1(self.dropout2(self.elu2(self.fc2(x))))
        a = self.dropout3(a)
        x = self.norm2(x + a)        
        return x  


class Time_step_EncoderLayer(torch.nn.Module):
    def __init__(self, dim_m, dim_val, dim_attn, n_heads, fc1_t, drop_rate=0.2):
        super(Time_step_EncoderLayer, self).__init__()
        self.attn = TimeStepMultiHeadAttentionBlock(dim_m, dim_val, dim_attn , n_heads)
        self.fc1 = nn.Linear(fc1_t, dim_m, device="cuda")
        self.fc2 = nn.Linear(dim_m, fc1_t, device="cuda")
        self.dropout1 = nn.Dropout(drop_rate)
        # self.dropout2 = nn.Dropout(drop_rate)
        self.dropout3 = nn.Dropout(drop_rate)
        self.norm1 = nn.LayerNorm(dim_m, device="cuda")
        self.norm2 = nn.LayerNorm(dim_m, device="cuda")
        self.elu1 = nn.ReLU()
        self.elu2 = nn.ReLU()
    
    def forward(self, x):
        a = self.attn(x)
        a = self.dropout1(a)
        x = self.norm1(x + a) 
        x = self.elu1(x)        
        a = self.fc1(self.elu2(self.fc2(x)))         
        # a = self.fc1(self.dropout2(self.elu2(self.fc2(x))) )
        a = self.dropout3(a)
        x = self.norm2(x + a)          
        return x  


class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_m, dim_val, dim_attn, n_heads, fc1_d, drop_rate=0.2):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_m, dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_m, dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(fc1_d, dim_m, device="cuda")
        self.fc2 = nn.Linear(dim_m, fc1_d, device="cuda")
        self.dropout1 = nn.Dropout(drop_rate)
        # self.dropout2 = nn.Dropout(drop_rate)
        self.dropout3 = nn.Dropout(drop_rate)
        self.norm1 = nn.LayerNorm(dim_m, device="cuda")
        self.norm2 = nn.LayerNorm(dim_m, device="cuda")
        self.norm3 = nn.LayerNorm(dim_m, device="cuda")
        self.elu1 = nn.ReLU()
        self.elu2 = nn.ReLU()
        
    def forward(self, x, enc):
        a = self.attn1(x) 
        a = self.dropout1(a)
        x = self.norm1(a + x)        
        a = self.attn2(x, kv = enc)
        x = self.norm2(a + x)
        x = self.elu1(x)    
        a = self.fc1(self.elu2(self.fc2(x)))      
        # a = self.fc1(self.dropout2(self.elu2(self.fc2(x))))   
        a = self.dropout3(a)
        x = self.norm3(x + a)
        return x  

class TransFomer(torch.nn.Module):
    def __init__(self, dim_m, dim_k_s, dim_v_s, n_head_s, fc1_s, dim_k_t, dim_v_t, n_head_t, fc1_t, dim_k_d, dim_v_d, n_head_d, fc1_d, time_step, 
                 input_size, dec_seq_len, out_seq_len, n_decoder_layers = 1, n_encoder_layers = 1,
                 drop_rate = 0.2):
        
        super(TransFomer, self).__init__()
        self.dec_seq_len = dec_seq_len
        # self.dropout1 = nn.Dropout(drop_rate)
      


        self.sensor_encoder = nn.ModuleList([Sensors_EncoderLayer(dim_m, dim_v_s, dim_k_s, n_head_s, fc1_s) for i in range(n_encoder_layers)])
        self.time_encoder = nn.ModuleList([Time_step_EncoderLayer(dim_m, dim_v_t, dim_k_t, n_head_t, fc1_t) for i in range(n_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(dim_m, dim_v_d, dim_k_d, n_head_d, fc1_d) for i in range(n_decoder_layers)])

        # #Initiate Sensors encoder
        # self.sensor_encoder = []
        # for i in range(n_encoder_layers):
        #     self.sensor_encoder.append(Sensors_EncoderLayer(dim_val_s, dim_attn_s, n_heads))
                
        # #Initiate Time_step encoder
        # self.time_encoder = []    
        # for i in range(n_encoder_layers):
        #     self.time_encoder.append(Time_step_EncoderLayer(dim_val_t, dim_attn_t, n_heads))        
                
        # #Initiate Decoder
        # self.decoder = []
        # for i in range(n_decoder_layers):
        #     self.decoder.append(DecoderLayer(dim_val, dim_attn, n_heads))
                    

        # self.pos_s = PositionalEncoding(dim_val_s)  
        self.pos_t = PositionalEncoding(dim_m)   
        self.pos_t2 = PositionalEncoding(dim_m)   
        self.timestep_enc_input_fc = nn.Linear(input_size, dim_m, device="cuda")
        self.sensor_enc_input_fc = nn.Linear(time_step, dim_m, device="cuda")        
        self.dec_input_fc = nn.Linear(input_size, dim_m, device="cuda")
        self.out_fc = nn.Linear(dec_seq_len * dim_m, out_seq_len, device="cuda")
        self.norm1 = nn.LayerNorm(dim_m, device="cuda")
        self.elu1 = nn.ReLU()
        # self.elu2 = nn.ReLU()
        self.elu3 = nn.ReLU()
        # self.elu4 = nn.ReLU()
        # self.elu5 = nn.ReLU()

        # self.s_enc1 = Sensors_EncoderLayer(dim_val_s, dim_attn_s, n_heads)


    def checker(self, x):
        sensor_x = x.transpose(1,2)
        #print(sensor_x.size())

        print ("self.sensor_enc_input_fc(sensor_x)", self.sensor_enc_input_fc(sensor_x))
    
    def forward(self, x):
        
        #input embedding and positional encoding

        sensor_x = x.transpose(1,2)

        #print(sensor_x.size())


        # e = self.s_enc1(self.sensor_enc_input_fc(sensor_x))

        for i, l in enumerate(self.sensor_encoder):
            if i == 0:
                e = l(self.sensor_enc_input_fc(sensor_x))
            else:
                e = l(e)

        for i, l in enumerate(self.time_encoder):
            if i == 0:
                o = l(self.pos_t(self.timestep_enc_input_fc(x)))
            else:
                o = l(o)

        # e = self.sensor_encoder[0](self.sensor_enc_input_fc(sensor_x)) #((batch_size,sensor,dim_val_s))
        # o = self.time_encoder[0](self.pos_t(self.timestep_enc_input_fc(x))) #((batch_size,timestep,dim_val_t))
        
        # # sensors encoder 
        # for sensor_enc in self.sensor_encoder[1:]:
        #     e = sensor_enc(e)
        
        # #time step encoder 
        # for time_enc in self.time_encoder[1:]:
        #     o = time_enc(o)
            
        #feature fusion
        p = torch.cat((e,o), dim = 1)  #((batch_size,timestep+sensor,dim_val))
        p = self.norm1(p)               
        # p = self.elu1(p)  


        
        #decoder receive the output of feature fusion layer.        

        # for i, l in enumerate(self.decoder):
        #     if i == 0:
        #         o = l(self.pos_t(self.timestep_enc_input_fc(x)))



        d = self.decoder[0](self.dec_input_fc(x[:,-self.dec_seq_len:]), p) 



        # #decoder layers 
        # for dec_layer in self.decoder[1:]:
        #     d = dec_layer(d)
            
        #output the RUL
        #x = self.out_fc(d.flatten(start_dim=1))
        # d = self.elu2(d)  
        x = self.out_fc(self.elu3(d.flatten(start_dim=1)))        
        return x
    
    
    


    