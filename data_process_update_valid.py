# -*- coding: utf-8 -*-
"""


"""
import argparse
import time
import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import linear_model
from scipy import interpolate
import scipy.io as sio
from numpy import *
from math import sqrt

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'cmapss')
data_prep_dir = os.path.join(current_dir, 'data_prep')

if not os.path.exists(data_prep_dir):
    os.makedirs(data_prep_dir)


min_max_scaler = preprocessing.MinMaxScaler()


def load_train_csv(data_path_list, columns_ts):
    '''
    :param data_path_list: path of csv file
    :param columns_ts: declared columns in csv
    :return: assigned pandas dataframe from csv
    '''
    # train_FD = pd.read_csv(data_path_list, sep=' ', header=None, names=columns_ts, index_col=False)
    train_FD = pd.read_table(data_path_list, delimiter=" ",  header=None, names=columns_ts)


    return train_FD



def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='data preparation')
    parser.add_argument('--subdata', type=str, default="001", help='subdataset of CMAPSS')
    parser.add_argument('-w', type=int, default=40, help='sequence length', required=True)
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('--device', type=str, default="cuda", help='Use "basic" if GPU with cuda is not available')
    parser.add_argument('--vs', type=int, default=20, help='the number of engines to be used for validation')
    # parser.add_argument('-t', type=int, required=True, help='trial')

    args = parser.parse_args()

    win_stride = args.s
    device = args.device
    print(f"Using {device} device")

    subdata_idx = args.subdata
    subdata = "FD" + subdata_idx

    #parameters of data process
    piecewise_lin_ref = 125.0  
    window_Size = args.w
    validation_numb = args.vs
    

    rul_path = os.path.join(data_dir, 'RUL_%s.txt' %subdata)
    train_path = os.path.join(data_dir, 'train_%s.txt' %subdata)
    test_path = os.path.join(data_dir, 'test_%s.txt' %subdata)


    # num_sensors = 26
    # cols = ['unit_nr', 'cycles', 'os_1', 'os_2', 'os_3']
    # cols += ['sensor_{0:02d}'.format(s + 1) for s in range(num_sensors)]
    # cols_non_sensor = ['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL']


    #Import dataset
    RUL_F001 = np.loadtxt(rul_path)
    train_F001 = np.loadtxt(train_path)
    test_F001 = np.loadtxt(test_path)
    train_F001[:, 2:] = min_max_scaler.fit_transform(train_F001[:, 2:])
    test_F001[:, 2:] = min_max_scaler.transform(test_F001[:, 2:])
    train_01_nor = train_F001
    test_01_nor = test_F001


    #Delete worthless sensors
    # train_01_nor = np.delete(train_01_nor, [5, 9, 10, 14, 20, 22, 23], axis=1) 
    # test_01_nor = np.delete(test_01_nor, [5, 9, 10, 14, 20, 22, 23], axis=1)  
    train_01_nor = np.delete(train_01_nor, [2,3,4,5, 9, 10, 14, 20, 22, 23], axis=1) 
    test_01_nor = np.delete(test_01_nor, [2,3,4,5, 9, 10, 14, 20, 22, 23], axis=1) 
    # train_01_nor = np.delete(train_01_nor, [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23], axis=1) 
    # test_01_nor = np.delete(test_01_nor, [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23], axis=1) 

    trainX = []
    trainY = []
    trainY_bu = []

    valX = []
    valY = []
    valY_bu = []

    testX = []
    testY = []
    testY_bu = []
    testInd = []
    testLen = []
    testX_all = []
    testY_all = []
    test_len = []

    print (train_01_nor)
    print ("train_01_nor.shape",  train_01_nor.shape)


    rnd.seed(0)
    engine_index_list = list(range(100))
    engine_index_list = [x+1 for x in engine_index_list]
    print ("engine_index_list", engine_index_list)
    val_engine_index = rnd.sample(engine_index_list, validation_numb) #generate 
    
    print ("val_engine_index", val_engine_index)

    #Training set sliding time window processing
    for i in range(1, int(np.max(train_01_nor[:, 0])) + 1):  

        if i in val_engine_index:
            print ("validation purpose", i )
            ind = np.where(train_01_nor[:, 0] == i)  
            ind = ind[0] # check unit number
            data_temp = train_01_nor[ind, :]  # data for each engine
            for j in range(len(data_temp) - window_Size + 1): 
                valX.append(data_temp[j:j + window_Size, 2:].tolist()) 
                val_RUL = len(data_temp) - window_Size - j  
                val_bu = piecewise_lin_ref - val_RUL
                if val_RUL > piecewise_lin_ref:
                    val_RUL = piecewise_lin_ref
                    val_bu = 0.0
                valY.append(val_RUL)
                valY_bu.append(val_bu)
            

        else:

            ind = np.where(train_01_nor[:, 0] == i)  
            ind = ind[0] # check unit number
            data_temp = train_01_nor[ind, :]  # data for each engine
            for j in range(len(data_temp) - window_Size + 1): 
                trainX.append(data_temp[j:j + window_Size, 2:].tolist()) 
                train_RUL = len(data_temp) - window_Size - j  
                train_bu = piecewise_lin_ref - train_RUL
                if train_RUL > piecewise_lin_ref:
                    train_RUL = piecewise_lin_ref
                    train_bu = 0.0
                trainY.append(train_RUL)
                trainY_bu.append(train_bu)
            
            
    #Test set sliding time window processing
    for i in range(1, int(np.max(test_01_nor[:, 0])) + 1): 
        ind = np.where(test_01_nor[:, 0] == i)
        ind = ind[0]
        testLen.append(float(len(ind)))
        data_temp = test_01_nor[ind, :] 
        testY_bu.append(data_temp[-1, 1])
        if len(data_temp) < window_Size:  
            data_temp_a = []
            for myi in range(data_temp.shape[1]):
                x1 = np.linspace(0, window_Size - 1, len(data_temp))
                x_new = np.linspace(0, window_Size - 1, window_Size)
                tck = interpolate.splrep(x1, data_temp[:, myi])
                a = interpolate.splev(x_new, tck)
                data_temp_a.append(a.tolist())
            data_temp_a = np.array(data_temp_a)
            data_temp = data_temp_a.T
            data_temp = data_temp[:, 2:]
        else:
            data_temp = data_temp[-window_Size:, 2:]  

        data_temp = np.reshape(data_temp, (1, data_temp.shape[0], data_temp.shape[1])) 
        
        if i == 1:
            testX = data_temp
        else:
            testX = np.concatenate((testX, data_temp), axis=0)
        if RUL_F001[i - 1] > piecewise_lin_ref:
            testY.append(piecewise_lin_ref)
            #testY_bu.append(0.0)
        else:
            testY.append(RUL_F001[i - 1])    
            
            
    #All data processing of test set
    for i in range(1, int(np.max(test_01_nor[:, 0])) + 1):
        ind = np.where(test_01_nor[:, 0] == i)
        ind = ind[0]
        data_temp = test_01_nor[ind, :]
        data_RUL = RUL_F001[i - 1] 
        test_len.append(len(data_temp) - window_Size + 1) 
        for j in range(len(data_temp) - window_Size + 1):
            testX_all.append(data_temp[j:j + window_Size, 2:].tolist())
            test_RUL = len(data_temp) + data_RUL - window_Size - j 
            if test_RUL > piecewise_lin_ref:
                test_RUL = piecewise_lin_ref
            testY_all.append(test_RUL)
            
                    
    trainX = np.array(trainX)
    trainY = np.array(trainY)/piecewise_lin_ref 
    trainY_bu = np.array(trainY_bu)/piecewise_lin_ref

    valX = np.array(valX)
    valY = np.array(valY)/piecewise_lin_ref 
    valY_bu = np.array(valY_bu)/piecewise_lin_ref


    testX = np.array(testX)
    testY = np.array(testY)/piecewise_lin_ref
    testY_bu = np.array(testY_bu)/piecewise_lin_ref


    testX_all = np.array(testX_all)
    testY_all = np.array(testY_all)


    print ("trainX.shape", trainX.shape)
    print ("valX.shape", valX.shape)


    sio.savemat(os.path.join(data_prep_dir, '%s_%s_%s_trainX.mat' %(subdata, window_Size, validation_numb)), {"train1X": trainX})
    sio.savemat(os.path.join(data_prep_dir, '%s_%s_%s_trainY.mat' %(subdata, window_Size, validation_numb)), {"train1Y": trainY})

    sio.savemat(os.path.join(data_prep_dir, '%s_%s_%s_valX.mat' %(subdata, window_Size, validation_numb)), {"val1X": valX})
    sio.savemat(os.path.join(data_prep_dir, '%s_%s_%s_valY.mat' %(subdata, window_Size, validation_numb)), {"val1Y": valY})
    

    sio.savemat(os.path.join(data_prep_dir, '%s_%s_testX.mat' %(subdata, window_Size)), {"test1X": testX})
    sio.savemat(os.path.join(data_prep_dir, '%s_%s_testY.mat' %(subdata, window_Size)) , {"test1Y": testY})

        
            
    # Statistical features process 

            
    regr = linear_model.LinearRegression()  # feature of linear coefficient

    def fea_extract1(data):  # feature 1
        fea = []
        x = np.array(range(data.shape[0]))
        for i in range(data.shape[1]):
            regr.fit(x.reshape(-1, 1), np.ravel(data[:, i]))
            fea = fea + list(regr.coef_)
        return fea

    def fea_extract2(data):  # feature 2
        fea = []
        for i in range(data.shape[1]):
            fea.append(np.mean(data[:, i]))
        return fea


    trainX = sio.loadmat(os.path.join(data_prep_dir, '%s_%s_%s_trainX.mat' %(subdata, window_Size, validation_numb)))
    valX = sio.loadmat(os.path.join(data_prep_dir, '%s_%s_%s_valX.mat' %(subdata, window_Size, validation_numb)))
    testX =  sio.loadmat(os.path.join(data_prep_dir, '%s_%s_testX.mat' %(subdata, window_Size)))

    trainX = trainX['train1X']
    valX = valX['val1X']
    testX = testX['test1X']

    trainX_fea1 = []
    valX_fea1 = []
    testX_fea1 = []
    trainX_fea2 = []
    valX_fea2 = []
    testX_fea2 = []
    window_size = window_Size 

    Feasize = 14  # the number of choosed sensors

    # print ("trainX", trainX)

    trainX = np.reshape(trainX, [trainX.shape[0], window_size, Feasize, 1])
    valX = np.reshape(valX, [valX.shape[0], window_size, Feasize, 1])
    testX = np.reshape(testX, [testX.shape[0], window_size, Feasize, 1])

    print ("trainX.shape", trainX.shape)
    print ("valX.shape", valX.shape)

    for i in range(len(trainX)): 
        data_temp = trainX[i]
        trainX_fea1.append(fea_extract1(data_temp))
        trainX_fea2.append(fea_extract2(data_temp))

    for i in range(len(valX)): 
        data_temp = valX[i]
        valX_fea1.append(fea_extract1(data_temp))
        valX_fea2.append(fea_extract2(data_temp))

    for i in range(len(testX)):
        data_temp = testX[i]
        testX_fea1.append(fea_extract1(data_temp))
        testX_fea2.append(fea_extract2(data_temp))

    scale1 = preprocessing.MinMaxScaler().fit(trainX_fea1)#归一化
    trainX_fea1 = scale1.transform(trainX_fea1)
    valX_fea1 = scale1.transform(valX_fea1)
    testX_fea1 = scale1.transform(testX_fea1)

    scale2 = preprocessing.MinMaxScaler().fit(trainX_fea2)
    trainX_fea2 = scale2.transform(trainX_fea2)
    valX_fea2 = scale2.transform(valX_fea2)
    testX_fea2 = scale2.transform(testX_fea2)


    trainX_new = []
    valX_new = []
    testX_new = []


    for i in range(len(trainX)):
        data_temp0 = trainX[i]
        data_temp1 = np.reshape(trainX_fea1[i], [1, Feasize, 1])  # regr.coef_
        data_temp2 = np.reshape(trainX_fea2[i], [1, Feasize, 1])  # mean_value
        data_temp = np.vstack((data_temp0, data_temp1, data_temp2))
        trainX_new.append(data_temp)
    trainX_new = np.array(trainX_new)    


    for i in range(len(valX)):
        data_temp0 = valX[i]
        data_temp1 = np.reshape(valX_fea1[i], [1, Feasize, 1])  # regr.coef_
        data_temp2 = np.reshape(valX_fea2[i], [1, Feasize, 1])  # mean_value
        data_temp = np.vstack((data_temp0, data_temp1, data_temp2))
        valX_new.append(data_temp)
    valX_new = np.array(valX_new)    


    for i in range(len(testX)):
        data_temp0 = testX[i]
        data_temp1 = np.reshape(testX_fea1[i], [1, Feasize, 1])  # regr.coef_
        data_temp2 = np.reshape(testX_fea2[i], [1, Feasize, 1])  # mean_value
        data_temp = np.vstack((data_temp0, data_temp1, data_temp2))
        testX_new.append(data_temp)
    testX_new = np.array(testX_new)
            
    sio.savemat(os.path.join(data_prep_dir, '%s_%s_%s_trainX_new.mat' %(subdata, window_Size, validation_numb)) , {"train1X_new": trainX_new})
    sio.savemat(os.path.join(data_prep_dir, '%s_%s_%s_valX_new.mat' %(subdata, window_Size, validation_numb)) , {"val1X_new": valX_new})
    sio.savemat(os.path.join(data_prep_dir, '%s_%s_testX_new.mat' %(subdata, window_Size)) , {"test1X_new": testX_new})   
            
            
            
            
            


if __name__ == '__main__':
    main()




