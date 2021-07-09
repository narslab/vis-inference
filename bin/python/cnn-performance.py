#!/usr/bin/env python
# coding: utf-8


import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd

from timeit import default_timer as timer
import random

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tkr

from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

import json
import pickle
import os
import sys
sys.path.append("../python/")
from helpers import *
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
#from sklearn.preprocessing import OrdinalEncoder
#enc = OrdinalEncoder()

#from contextlib import redirect_stdout

# Globals
NUM_CHANNELS = 1
RESOLUTION_LIST = [128] # 64, 128, 224, 384]
SCENARIO_LIST = ["PrPo_Im"] #, "Pr_Im", "Pr_PoIm", "Pr_Po_Im"]
NUM_EPOCHS = 20
SAVED_MODEL_DIR = '../../results/models/'
MODEL_PERFORMANCE_METRICS_DIR = '../../results/model-performance/'

#
image_sets_square_train = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST, train=True, rectangular = False)
image_sets_square_test = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST, train=False, rectangular = False)

# image_sets = dict.fromkeys(RESOLUTION_LIST)
# for p in RESOLUTION_LIST:
#     image_sets[p] = dict.fromkeys(SCENARIO_LIST)
#     for s in SCENARIO_LIST:
#         image_sets[p][s] = np.load('../../data/tidy/preprocessed_images/size' + str(p) + '_exp5_' + s + '.npy', allow_pickle = True)

                                   
# https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb
# Metrics2 modified from https://stackoverflow.com/a/61856587/3023033
class Metrics(Callback):
    def __init__(self, val_data):#, batch_size = 64):
        super().__init__()
        self.validation_data = val_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        xVal, yVal = self.validation_data
        val_pred = np.argmax(np.asarray(self.model.predict(xVal)), axis=1)
        val_true = np.argmax(yVal, axis=1)        
        _val_f1 = f1_score(val_true, val_pred, average='macro', zero_division = 0)
        _val_precision = precision_score(val_true, val_pred, average='macro', zero_division = 0)
        _val_recall = recall_score(val_true, val_pred, average='macro', zero_division = 0)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        logs["val_f1"] = _val_f1
        logs["val_recall"] = _val_recall
        logs["val_precision"] = _val_precision
        print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        return


def trainModelWithDetailedMetrics(image_width, scenario, num_epochs = 10, trial_seed = 1, testing = True): 
    # IMAGES (former approach)
    # training_images_and_labels, test_images_and_labels = splitData(image_sets[image_size][scenario], prop = 0.8, seed_num = trial_seed)
    # training_images, training_labels = getImageAndLabelArrays(training_images_and_labels)
    # validation_images, validation_labels = getImageAndLabelArrays(test_images_and_labels)
    class_labels = getClassLabels(scenario)
    print("Class labels:", class_labels)
    training_images = np.array([np.expand_dims(x[0],axis=2) for x in image_sets_square_train[image_width][scenario]]) ## TOD)
    training_labels = np.array([x[1] for x in image_sets_square_train[image_width][scenario]]) 
    validation_images = np.array([np.expand_dims(x[0],axis=2) for x in image_sets_square_test[image_width][scenario]]) ## TOD)
    validation_labels = np.array([x[1] for x in image_sets_square_test[image_width][scenario]]) 

    print("Number of class training images:", training_labels.sum(axis=0), "total: ", training_labels.sum())
    print("Number of class validation images:", validation_labels.sum(axis=0), "total: ", validation_labels.sum())
    
    # CALLBACKS
    model_metrics = Metrics(val_data=(validation_images, validation_labels))
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    
    # INIT MODEL AND PARAMS, FIT
    K.clear_session()
    #input_shape = (image_size, image_size, NUM_CHANNELS) ## shape of images
    model = constructBaseCNN(image_size, scenario, num_channels = NUM_CHANNELS)    ## get model
    opt_learning_rate = getOptCNNHyperparams(image_size, scenario)['learning_rate']    ## learning rate
    opt = tf.keras.optimizers.Adam(learning_rate = opt_learning_rate)    
    reset_weights(model) # re-initialize model weights
    model.compile(loss='categorical_crossentropy',  optimizer = opt, metrics =  ['accuracy'])     ## compile and fit
    hist = model.fit(training_images, training_labels, batch_size = 32, epochs = num_epochs, verbose=1, 
                     validation_data=(validation_images, validation_labels),
                     callbacks = [model_metrics]) #, early_stopping])     
    
    # SAVE MODEL, SUMMARY AND PERFORMANCE
    if testing == True:
        model_name = "test-opt-cnn-" + scenario + "-" +str(image_size) + "-px"
    else:
        model_name = "opt-cnn-" + scenario + "-" +str(image_size) + "-px"
    model_folder = "model"
    if not os.path.exists(SAVED_MODEL_DIR):  
        os.makedirs(SAVED_MODEL_DIR)
    model.save(os.path.join(SAVED_MODEL_DIR, model_name, model_folder))     ## Save model summary
    #print(os.path.join(SAVED_MODEL_DIR, model_name, "summary.txt"))
    with open(os.path.join(SAVED_MODEL_DIR, model_name, "summary.txt"), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    with open(os.path.join(SAVED_MODEL_DIR, model_name, "history.txt"), 'w') as f:
        f.write(json.dumps(hist.history))    
   
    # ANALYZE PERFORMANCE AND SAVE OUTPUTS
    y_pred = np.argmax(model.predict(validation_images), axis=-1)     ## Params
    ## Classification report
    report = classification_report(np.argmax(validation_labels, axis=-1), y_pred, zero_division=0,
                                   labels = np.arange(len(class_labels)), target_names=class_labels, output_dict=True)
    print("Classification report for scenario " + scenario + ", resolution: " + str(image_size) + ":")
    report = pd.DataFrame(report).transpose().round(2)
    if not os.path.exists('../../results/classification-reports/'):  
        os.makedirs('../../results/classification-reports/')
    if testing == True:
        report.to_csv("../../results/classification-reports/test-opt-classification-report-" + scenario + "-" + str(image_size) + "-px.csv")
    else:
        report.to_csv("../../results/classification-reports/opt-classification-report-" + scenario + "-" + str(image_size) + "-px.csv")        
    print(report)
    
    ## Confusion matrix
    con_mat = tf.math.confusion_matrix(labels=np.argmax(validation_labels, axis=-1), predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index = class_labels, columns = class_labels)
    #print("Confusion matrix for scenario " + scenario + ", resolution: " + str(image_size) + ":")
    #print(con_mat_df)
    figure = plt.figure()#figsize=(4, 4))    ## Confusion matrix heatmap
    ax = sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues, fmt='g', cbar = False, annot_kws={"size": 16})
    #figure.tight_layout()
    plt.ylabel('True',fontsize=16)
    ax.set_yticklabels(class_labels,va='center',fontsize=14)
    ax.set_xticklabels(class_labels, ha='center',fontsize=14)
    plt.xlabel('Predicted',fontsize=16)
    #plt.show()
    if testing == True:
        con_mat_heatmap_file = "../../figures/test-opt-confusion-matrix-" + scenario + "-" + str(image_size) + "-px.png"
    else:
        con_mat_heatmap_file = "../../figures/opt-confusion-matrix-" + scenario + "-" + str(image_size) + "-px.png"
    figure.savefig(con_mat_heatmap_file, dpi=180)#, bbox_inches='tight')
    return(model, hist) 


def getScenarioModelPerformance(res = 64, num_epochs = 15, seed_val = 1, test_boolean = True):
    df = pd.DataFrame()
    for s in SCENARIO_LIST:
        m, h = trainModelWithDetailedMetrics(res, s, num_epochs, trial_seed = seed_val, testing = test_boolean)
        #visualizeCNN(m, s, res, images_per_class = 4, trial_seed = seed_val, testing = test_boolean)       
        perf = pd.DataFrame.from_dict(h.history)
        perf[['Scenario']] = s
        perf['epoch'] = perf.index + 1
        df = df.append(perf, ignore_index=True)
        #del m
    if test_boolean == True:
        df_filename = "../../results/test-opt-cnn-performance-metrics-summary-" + str(res) + "px.csv"
    else:
        df_filename = "../../results/opt-cnn-performance-metrics-summary-" + str(res) + "px.csv"
    df.to_csv(df_filename)
    return df

if __name__ == "__main__":
    getScenarioModelPerformance(res=128, num_epochs=13, seed_val = 2, test_boolean=False)
