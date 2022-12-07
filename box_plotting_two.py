import pandas as pd
import matplotlib.pyplot as plt
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
import itertools
from statannot import add_stat_annotation
from statsmodels.stats.weightstats import ttest_ind
from scipy import stats
import matplotlib.cm as cm

def text_values(pd_series):

    max = round(pd_series.max(),2)
    med = round(pd_series.median(),2)
    min = round(pd_series.min(),2)

    return max, med, min


def draw_brace(ax, xspan, yy, text):
    """Draws an annotated brace on the axes."""
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1)

    ax.text((xmax+xmin)/2., yy+.07*yspan, text, ha='center', va='bottom', fontsize=16)


def to_sci_not(number):
    a, b = '{:.2E}'.format(number).split('E')
    return '{:.3f}E{:+03d}'.format(float(a)/10, int(b)+1)

def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='data preparation')
    parser.add_argument('--subdata', type=str, default="001", help='subdataset of CMAPSS')
    parser.add_argument('-ep', type=int, default=100, help='max epochs')
    parser.add_argument('-pt', type=int, default=10, help='patience')
    parser.add_argument('-topk', type=int, default=10, help='topk')
    parser.add_argument('-n_val', type=int, default=20, help='number of samples for initialization')
    parser.add_argument('-t', type=int, default=0, help='trial')
    parser.add_argument('-n_lhc', type=int, default=100, help='trial')




    args = parser.parse_args()

    ep = args.ep
    pt = args.pt
    n_val = args.n_val
    topk = args.topk
    trial = args.t
    n_lhc = args.n_lhc

    subdata_idx = args.subdata
    subdata = "FD" + subdata_idx





    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(current_dir, 'temporary')

    pic_ind_dir = os.path.join(temp_dir, 'plots')
    if not os.path.exists(pic_ind_dir):
        os.makedirs(pic_ind_dir)

    init_filepath = os.path.join(temp_dir, 'initialization_rmse_%s_%s_%s_soo_%s_%s_%s.csv' %(n_val, pt, n_lhc, subdata, ep, trial))
    test_filepath = os.path.join(temp_dir, 'test_rmse_topk_ga_retrain_%s_%s.csv' %(topk, subdata))


    init_df = pd.read_csv(init_filepath)
    result_df = pd.read_csv(test_filepath)


    # sort values by val_rmse
    init_df = init_df.sort_values(by=['val_rmse'])
    result_df = result_df.sort_values(by=['val_rmse'])

    # select topk
    init_df = init_df[:topk]
    result_df = result_df[:topk]

    # array
    init_val = init_df["val_rmse"]
    result_val = result_df["val_rmse"]
    init_val_max, init_val_avg, init_val_min = text_values(init_val)
    result_val_max, result_val_avg, result_val_min = text_values(result_val)

    if subdata == "FD002":
        pair_text_ref_val = 0.5
        pair_text_ref_test = 0.9
    else:
        pair_text_ref_val = 0.5
        pair_text_ref_test = 0.5

    columns = [init_val, result_val]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    fig.subplots_adjust(0, 0, 1, 1)

    # figsize=(4.5, 4)
    ax.boxplot(columns)

    res = stats.ttest_ind(init_val, result_val, 
                      equal_var=True)

    pval = res[1] 
    print ("val pval:" , pval)
    if pval < 0.05:
        text = "p-value: %s < %s" %(to_sci_not(pval), to_sci_not(0.05))
    else:
        text = "p-value: %s >= %s" %(to_sci_not(pval), to_sci_not(0.05))

    max_val = max(init_val.max(), result_val.max()) 
    ax.set_ylim(top = max_val+ pair_text_ref_val)
    draw_brace(ax, (1, 2), max_val+ 0.15, text)

    plt.xticks([1, 2], ["LHS", "EA + predictor"], fontsize=16)
    plt.yticks(fontsize=16)

    plt.text(x= 1.1, y= init_val_max,  s = "%s"%init_val_max, fontsize=16)
    plt.text(x= 1.1, y= init_val_avg, s = "%s"%init_val_avg, fontsize=16)
    plt.text(x= 1.1, y= init_val_min, s = "%s"%init_val_min, fontsize=16)

    if subdata == "FD003":

        plt.text(x= 2.1, y= result_val_max+0.05, s = "%s" %result_val_max, fontsize=16)
        plt.text(x= 2.1, y= result_val_avg-0.05, s =  "%s" %result_val_avg, fontsize=16)
        plt.text(x= 2.1, y= result_val_min, s =  "%s" %result_val_min, fontsize=16)

    elif subdata == "FD001":
        plt.text(x= 2.1, y= result_val_max+0.05, s = "%s" %result_val_max, fontsize=16)
        plt.text(x= 2.1, y= result_val_avg+0.05, s =  "%s" %result_val_avg, fontsize=16)
        plt.text(x= 2.1, y= result_val_min, s =  "%s" %result_val_min, fontsize=16)

    else:
        plt.text(x= 2.1, y= result_val_max+0.05, s = "%s" %result_val_max, fontsize=16)
        plt.text(x= 2.1, y= result_val_avg, s =  "%s" %result_val_avg, fontsize=16)
        plt.text(x= 2.1, y= result_val_min, s =  "%s" %result_val_min, fontsize=16)
    # plt.title("%s" %subdata)

    plt.ylabel("Validation RMSE", fontsize=16)
    plt.xlabel("Top10 validation RMSE", fontsize=16, labelpad=10)

    plt.tight_layout()

    fig.savefig(os.path.join(pic_ind_dir, 'boxplots_val_%s.png' %(subdata)), dpi=1000,
                    bbox_inches='tight')
    fig.savefig(os.path.join(pic_ind_dir, 'boxplots_val_%s.eps' %(subdata)), dpi=3000,
                    bbox_inches='tight')





    # plot test boxplots 
    init_test = init_df["test_rmse"]
    result_test = result_df["test_rmse"]
    init_test_max, init_test_avg, init_test_min = text_values(init_test)
    result_test_max, result_test_avg, result_test_min = text_values(result_test)

    columns = [init_test, result_test]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    fig.subplots_adjust(0, 0, 1, 1)

    ax.boxplot(columns)

    res = stats.ttest_ind(init_test, result_test, 
                      equal_var=True)

    pval = res[1] 
    print ("test pval:" , pval)
    if pval < 0.05:
        text = "p-value: %s < %s" %(to_sci_not(pval), to_sci_not(0.05))
    else:
        text = "p-value: %s >= %s" %(to_sci_not(pval), to_sci_not(0.05))

    max_val = max(init_test.max(), result_test.max()) 
    ax.set_ylim(top = max_val+ pair_text_ref_test)
    draw_brace(ax, (1, 2), max_val+ 0.15, text)

    plt.xticks([1, 2], ["LHS", "EA + predictor"], fontsize=16)
    plt.yticks(fontsize=16)


    if subdata == "FD002":

        plt.text(x= 1.1, y= init_test_max-0.1,  s = "%s"%init_test_max, fontsize=16)
        plt.text(x= 1.1, y= init_test_avg, s = "%s"%init_test_avg, fontsize=16)
        plt.text(x= 1.1, y= init_test_min, s = "%s"%init_test_min, fontsize=16)

    else:
        plt.text(x= 1.1, y= init_test_max,  s = "%s"%init_test_max, fontsize=16)
        plt.text(x= 1.1, y= init_test_avg, s = "%s"%init_test_avg, fontsize=16)
        plt.text(x= 1.1, y= init_test_min, s = "%s"%init_test_min, fontsize=16)

    plt.text(x= 2.1, y= result_test_max, s = "%s" %result_test_max, fontsize=16)
    plt.text(x= 2.1, y= result_test_avg, s =  "%s" %result_test_avg, fontsize=16)
    plt.text(x= 2.1, y= result_test_min, s =  "%s" %result_test_min, fontsize=16)

    plt.ylabel("Test RMSE", fontsize=16)
    plt.xlabel("Top10 test RMSE", fontsize=16, labelpad=10)

    plt.tight_layout()

    fig.savefig(os.path.join(pic_ind_dir, 'boxplots_test_%s.png' %(subdata)), dpi=1000,
                    bbox_inches='tight')
    fig.savefig(os.path.join(pic_ind_dir, 'boxplots_test_%s.eps' %(subdata)), dpi=3000,
                    bbox_inches='tight')



    # plot comparisons
    init_test = init_df["test_rmse"]
    result_test = result_df["test_rmse"]
    init_test_max, init_test_avg, init_test_min = text_values(init_test)
    result_test_max, result_test_avg, result_test_min = text_values(result_test)

    columns = [init_test, result_test]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.boxplot(columns)

    plt.xticks([1, 2], ["LHS, \n top10 test RMSE", "EA + predictor, \n top10 test RMSE"], rotation=30, fontsize=10)

    plt.text(x= 1.1, y= init_test_max,  s = "%s"%init_test_max, fontsize=10)
    plt.text(x= 1.1, y= init_test_avg, s = "%s"%init_test_avg, fontsize=10)
    plt.text(x= 1.1, y= init_test_min, s = "%s"%init_test_min, fontsize=10)

    plt.text(x= 2.1, y= result_test_max, s = "%s" %result_test_max, fontsize=10)
    plt.text(x= 2.1, y= result_test_avg, s =  "%s" %result_test_avg, fontsize=10)
    plt.text(x= 2.1, y= result_test_min, s =  "%s" %result_test_min, fontsize=10)




    if subdata == "FD001":
        plt.xlim([0, 6])  
        plt.ylim([11, 14])  
        dast = 12.66
        agcnn = 12.42
        cnnatt = 11.48
        rnnae = 13.58 
        cnnlstm = 12.19
        semi = 12.56
        dag = 11.96
        proposed = result_test_min

    elif subdata == "FD003":        
        plt.xlim([0, 6])  
        plt.ylim([10, 14])  
        dast = 12.13
        agcnn = 13.39
        cnnatt = 12.31
        # rnnae = 19.16
        cnnlstm = 12.85
        semi = 12.10
        dag = 12.46
        proposed = result_test_min
        
    elif subdata == "FD002":        
        plt.xlim([0, 6])  
        plt.ylim([15, 24])  
        dast = 19.19
        agcnn = 19.43
        cnnatt = 17.25
        rnnae = 19.59
        cnnlstm = 19.93
        semi = 22.73
        dag = 20.34
        proposed = result_test_min

    elif subdata == "FD004":        
        plt.xlim([0, 6])  
        plt.ylim([10, 14])  
        dast = 12.13
        agcnn = 13.39
        cnnatt = 12.31
        # rnnae = 19.16
        cnnlstm = 12.85
        semi = 12.10
        dag = 12.46
        proposed = result_test_min


        pass
    
    if subdata == "FD003":
        results_list = [dast, agcnn, cnnatt,   cnnlstm, semi, dag, proposed]
        results_str = ["DAST(2022)", "AGCNN (2020)", "CNN+ATTENTION (2021)",  "CNN-LSTM(2020)", "Semi-Supervised (2019)", "DAG (2019)", "Proposed method"]
    else:
        results_list = [dast, agcnn, cnnatt, rnnae,  cnnlstm, semi, dag, proposed]
        results_str = ["DAST(2022)", "AGCNN (2020)", "CNN+ATTENTION (2021)", "RNN+AE (2020)", "CNN-LSTM(2020)", "Semi-Supervised (2019)", "DAG (2019)", "Proposed method"]
    marker_list = itertools.cycle(("o", "s", "p", "h", "H", "X", "D", "d", "v", "^", "<", ">")) 
    colors = cm.rainbow(np.linspace(0, 1, len(results_list)))

    text_x = 2.9
    text_fontsize= 10
    for rmse, label, m, c in zip(results_list, results_str, marker_list, colors):
        plt.scatter(x= 2.8, y = rmse, s = 30, color=c, marker = m)
        if label == "Semi-Supervised (2019)" and subdata == "FD003":
            plt.text (x=text_x, y =rmse-0.3, s="%s, %s" %(rmse, label), fontsize=text_fontsize)
        elif label == "DAST(2022)" and subdata == "FD003":
            plt.text (x= text_x, y =rmse-0.15, s="%s, %s" %(rmse, label), fontsize=text_fontsize)
        elif label == "DAST(2022)" and subdata == "FD002":
            plt.text (x=text_x, y =rmse-0.4, s="%s, %s" %(rmse, label), fontsize=text_fontsize)
        elif label == "DAST(2022)" and subdata == "FD001":
            plt.text (x=text_x, y =rmse+0.05, s="%s, %s" %(rmse, label), fontsize=text_fontsize)
        elif label == "CNN+ATTENTION (2021)" and subdata == "FD003":
            plt.text (x= text_x, y =rmse-0.05, s="%s, %s" %(rmse, label), fontsize=text_fontsize)
        elif label == "AGCNN (2020)" and subdata == "FD002":
            plt.text (x= text_x, y =rmse-0.2, s="%s, %s" %(rmse, label), fontsize=text_fontsize)
        elif label == "CNN+ATTENTION (2021)" and subdata == "FD001":
            plt.text (x= text_x, y =rmse-0.1, s="%s, %s" %(rmse, label), fontsize=text_fontsize)
        elif label == "Proposed method" and subdata == "FD001":
            plt.text (x= text_x, y =rmse, s="%s, %s" %(rmse, label), fontsize=text_fontsize)
        else:
            plt.text (x= text_x, y =rmse, s="%s, %s" %(rmse, label), fontsize=text_fontsize)



    plt.ylabel("Test RMSE", fontsize=11)
    plt.yticks(fontsize=11)

    fig.savefig(os.path.join(pic_ind_dir, 'boxplots_comparison_%s.png' %(subdata)), dpi=1000,
                    bbox_inches='tight')
    fig.savefig(os.path.join(pic_ind_dir, 'boxplots_comparison_%s.eps' %(subdata)), dpi=3000,
                    bbox_inches='tight')







if __name__ == '__main__':
    main()    