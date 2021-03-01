#!/usr/bin/env python
# coding: utf-8

# This script explores CNN model performance based on training image resolution.
# We test the following training image sizes:
# 64 x 64, 128 x 128, 224 x 224, 384 x 384

import numpy as np
from timeit import default_timer as timer
import random

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

import json
import pickle
import os
import sys
sys.path.append("../python/")
from helpers import *

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Globals
NUM_CHANNELS = 1
RESOLUTION_LIST = [64, 128, 224] #, 224, 384]
SCENARIO_LIST = ["Pr_Im", "PrPo_Im", "Pr_PoIm", "Pr_Po_Im"]
NUM_MODEL_RUNS = 5
NUM_EPOCHS = 15
RESOLUTION_PERFORMANCE_METRICS_DIR = '../../results/sensitivity-tests-02272021' #'../../results/sensitivity-tests'



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

# def testResolutionScenarioPerformance(image_size, scenario, num_epochs = 10, trial_seed = 1): 
#     training_images_and_labels, test_images_and_labels = splitData(image_sets[image_size][scenario], prop = 0.80, seed_num = trial_seed)
#     training_images, training_labels = getImageAndLabelArrays(training_images_and_labels)
#     validation_images, validation_labels = getImageAndLabelArrays(test_images_and_labels)
#     # early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
#     # batch_training_histories = Histories()
#     # metrics_multiclass = Metrics(validation_images,validation_labels)  TODO
#     K.clear_session()
#     input_shape = (image_size, image_size, NUM_CHANNELS)
#     model = constructBaseCNN(image_size, scenario, num_channels = NUM_CHANNELS)
#     opt_learning_rate = getOptCNNHyperparams(image_size, scenario)['learning_rate']
#     reset_weights(model) # re-initialize model weights
#     opt = tf.keras.optimizers.Adam(learning_rate = opt_learning_rate)
#     model.compile(loss='categorical_crossentropy',  optimizer = opt, metrics = ['accuracy'])
#     hist = model.fit(training_images, training_labels, batch_size = 32, epochs = num_epochs, verbose=0, validation_data=(validation_images, validation_labels)) #, callbacks=[batch_training_histories])
#     performance_dict = {}    
#     performance_dict['scenario'] = scenario
#     performance_dict['image_size'] = image_size
#     performance_dict['metrics'] = hist.history
#     performance_dict['best_val_accuracy'] = np.max(hist.history['val_accuracy'])
#     return(performance_dict)


# def main(num_trials = NUM_MODEL_RUNS):
#     if not os.path.exists(RESOLUTION_PERFORMANCE_METRICS_DIR): # check if 'tidy/preprocessed_images' subdirectory does not exist
#         os.makedirs(RESOLUTION_PERFORMANCE_METRICS_DIR) # if not, create it    
#     for s in SCENARIO_LIST:
#         for p in RESOLUTION_LIST:
#             for i in range(num_trials):
#                 print("Conducting performance test: Scenario - " + s + "; Resolution - " + str(p) + "px; Trial - " + str(i+1))
#                 scenario_performance_dict = testResolutionScenarioPerformance(p, s, num_epochs = NUM_EPOCHS, trial_seed = 1 + i) #ultimately should be averaged across trials       
#                 scenario_filename = "scenario_resolution_performance_" + s + str(p) + "px_trial_" + str(i+1) + ".txt"
#                 with open(os.path.join(RESOLUTION_PERFORMANCE_METRICS_DIR, scenario_filename), 'w') as f:
#                    f.write(json.dumps(scenario_performance_dict )) # use `json.loads` to do the reverse)
#     return


def testResolutionScenarioPerformance(training_images, training_labels, validation_images, validation_labels, image_size, scenario, num_epochs = 10): #, trial_seed = 1): 
    K.clear_session()
    input_shape = (image_size, image_size, NUM_CHANNELS)
    model = constructBaseCNN(image_size, scenario, num_channels = NUM_CHANNELS)
    model_metrics = Metrics(val_data=(validation_images, validation_labels))
    opt_learning_rate = getOptCNNHyperparams(image_size, scenario)['learning_rate']
    reset_weights(model) # re-initialize model weights
    opt = tf.keras.optimizers.Adam(learning_rate = opt_learning_rate)
    model.compile(loss='categorical_crossentropy',  optimizer = opt, metrics = ['accuracy'])
    hist = model.fit(training_images, training_labels, batch_size = 32, epochs = num_epochs, verbose=1, 
                        validation_data=(validation_images, validation_labels), callbacks=[model_metrics])
    performance_dict = {}    
    performance_dict['scenario'] = scenario
    performance_dict['image_size'] = image_size
    performance_dict['metrics'] = hist.history
    performance_dict['total_params'] = model.count_params()
    del model
    return(performance_dict)

def main(num_trials = NUM_MODEL_RUNS):
    if not os.path.exists(RESOLUTION_PERFORMANCE_METRICS_DIR):  
        os.makedirs(RESOLUTION_PERFORMANCE_METRICS_DIR) 
    image_sets = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST)
    for s in SCENARIO_LIST:
        for p in RESOLUTION_LIST:
            skf = StratifiedKFold(n_splits = NUM_MODEL_RUNS, random_state = 1, shuffle=True)
            X = np.array([np.expand_dims(x[0],axis=2) for x in image_sets[p][s]])
            y = np.array([x[1] for x in image_sets[p][s]])
            for i, (train_index, test_index) in enumerate(skf.split(X, y.argmax(1))):
                print("Conducting performance test: Scenario - " + s + "; Resolution - " + str(p) + "px; Trial - " + str(i+1))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                scenario_performance_dict = testResolutionScenarioPerformance(X_train, y_train, X_test, y_test, p, s, 
                    num_epochs = NUM_EPOCHS)
                scenario_filename = "scenario_resolution_performance_" + s + str(p) + "px_trial_" + str(i+1) + ".txt"
                with open(os.path.join(RESOLUTION_PERFORMANCE_METRICS_DIR, scenario_filename), 'w') as f:
                   f.write(json.dumps(scenario_performance_dict )) # use `json.loads` to do the reverse)
    return


if __name__ == "__main__":
    main()