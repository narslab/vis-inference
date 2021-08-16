#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append("../python/")
from helpers import *

import numpy as np
import pandas as pd

IMAGE_WIDTH_LIST = [189, 252, 336] #, 384]
SCENARIO_LIST = ["Pr_Im", "PrPo_Im", "Pr_PoIm", "Pr_Po_Im"]

def getOptHyperparamsSummary(scenario_list, width_list, rectangular=False, orientation="landscape"):
    df_all = pd.DataFrame()
    for s in scenario_list:
        for w in width_list:
            if rectangular==True:
                h = getRectangularImageHeight(w)
                image_shape = 'rect'
            else:
                h = w
                image_shape = 'square'
            opt_dict = getOptCNNHyperparams(w, h, s) 
            df = pd.DataFrame.from_dict(opt_dict, orient='index', columns=["Values"])#, columns= ["Hyperparameters", "Values"])
            print(df)
            df['Scenario'] = s
            df['Resolution'] = w
            try:
                df = df.drop(["tuner/bracket", "tuner/epochs", "tuner/initial_epoch", "tuner/round", "tuner/trial_id"])
            except:
                df = df.drop(["tuner/bracket", "tuner/epochs", "tuner/initial_epoch", "tuner/round"])
            df_all = df_all.append(df)
    df_all['Hyperparameters'] = df_all.index
    print(df_all['Hyperparameters'])
    df_all = df_all.reset_index() #['Scenario','Resolution'])
    pd.set_option('display.float_format', '{:.2E}'.format)
    if orientation=="landscape":
        df_all = df_all.pivot(index=['Hyperparameters'], columns=['Scenario','Resolution'], values='Values')
    elif orientation=="portrait":
        df_all = df_all.pivot(index=['Scenario','Hyperparameters'], columns=['Resolution'], values='Values')
    file = '../../results/opt-hyperparameter-summary-table-' + orientation + '-' + image_shape + '.csv' # TODO
    df_all.to_csv(file, index=True, float_format = ':.2E')
    return(df_all)


if __name__ == "__main__":
    getOptHyperparamsSummary(SCENARIO_LIST, IMAGE_WIDTH_LIST, rectangular=False, orientation='portrait')
    getOptHyperparamsSummary(SCENARIO_LIST, IMAGE_WIDTH_LIST, rectangular=False, orientation='landscape')

