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
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import torch.utils.data.dataloader as Data
import os
import itertools
# from statannot import add_stat_annotation
from statsmodels.stats.weightstats import ttest_ind
from scipy import stats
import matplotlib.cm as cm

import seaborn as sns
from statannotations.Annotator import Annotator
# https://github.com/trevismd/statannotations



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



    df = pd.DataFrame([])
    sns.set(style="whitegrid")
    x = "Methods"
    y = "Validation RMSE"
    order = ["LHS", "EA + predictor"]

    init_rmse = init_df["val_rmse"].values
    result_rmse = result_df["val_rmse"].values

    df[y] = np.concatenate([init_rmse, result_rmse])
    df[x] = ["LHS"]*topk + ["EA + predictor"]*topk



    font_size = 20
    
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=df, x=x, y=y, order=order)
    annot = Annotator(ax, [("LHS", "EA + predictor")], data=df, x=x, y=y, order=order)
    annot.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=20)
    annot.apply_test()
    ax, test_results = annot.annotate()
    plt.tight_layout()

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel(x, fontsize=font_size)
    plt.ylabel(y, fontsize=font_size)

    fig.savefig(os.path.join(pic_ind_dir, 'boxplots_val_%s.png' %(subdata)), dpi=1000,
                    bbox_inches='tight')
    fig.savefig(os.path.join(pic_ind_dir, 'boxplots_val_%s.eps' %(subdata)), dpi=3000,
                    bbox_inches='tight')





    df = pd.DataFrame([])
    sns.set(style="whitegrid")
    x = "Methods"
    y = "Test RMSE"
    order = ["LHS", "EA + predictor"]

    init_rmse = init_df["test_rmse"].values
    result_rmse = result_df["test_rmse"].values

    df[y] = np.concatenate([init_rmse, result_rmse])
    df[x] = ["LHS"]*topk + ["EA + predictor"]*topk



 
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=df, x=x, y=y, order=order)
    annot = Annotator(ax, [("LHS", "EA + predictor")], data=df, x=x, y=y, order=order)
    annot.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=20)
    annot.apply_test()
    ax, test_results = annot.annotate()
    plt.tight_layout()

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel(x, fontsize=font_size)
    plt.ylabel(y, fontsize=font_size)

    fig.savefig(os.path.join(pic_ind_dir, 'boxplots_test_%s.png' %(subdata)), dpi=1000,
                    bbox_inches='tight')
    fig.savefig(os.path.join(pic_ind_dir, 'boxplots_test_%s.eps' %(subdata)), dpi=3000,
                    bbox_inches='tight')



if __name__ == '__main__':
    main()    