#!/usr/bin/env python
# coding: utf-8

from PIL import Image # used for loading images
import numpy as np
import os 
import imageio # used for writing images
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from timeit import default_timer as timer
from tensorflow.keras import backend as K

import sys
sys.path.append("../python/")
from helpers import *

"""HP Tuning"""
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

import IPython

import kerastuner as kt
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection

NUM_CHANNELS = 1
RESOLUTION_LIST = [64, 128, 224] #, 384]
SCENARIO_LIST = ["Pr_Im", "PrPo_Im", "Pr_PoIm", "Pr_Po_Im"]
OPTIMAL_HYPERPARAMETERS_PATH = '../../results/optimal-hyperparameters/'
HYPERBAND_MAX_EPOCHS = 10 #10
EXECUTIONS_PER_TRIAL = 2 #5
HYPERBAND_ITER = 3 #80

# TODO: make image_dict a function
image_dict = dict.fromkeys(RESOLUTION_LIST)
for p in RESOLUTION_LIST:
    image_dict[p] = dict.fromkeys(SCENARIO_LIST)
    for s in SCENARIO_LIST:
        image_dict[p][s] = np.load('../../data/tidy/preprocessed_images/size' + str(p) + '_exp5_' + s + '.npy', allow_pickle = True)


class CNNHyperModel(HyperModel):
    def __init__(self, input_image_shape, num_classes):
        self.input_image_shape = input_image_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = models.Sequential()

        ## Vary kernel size in first Conv Layer between 5 and 7
        hp_k_size = hp.Choice('kernel_size', values = [5, 7])

        # Tune the number of units in the first and second Dense layers
        # Choose an optimal value between 32-512
        dense_units_l = hp.Int('units_1', min_value = 32, max_value = 512, step = 32, default = 128)
        dense_units_2 = hp.Int('units_2', min_value = 32, max_value = 512, step = 32,  default = 64)  

        dropout_rate_1 = hp.Float('dropout_1', min_value = 0.0, max_value = 0.5, step = 0.05)
        dropout_rate_2 = hp.Float('dropout_2', min_value = 0.0, max_value = 0.5, step = 0.05)

        # Experiment with "relu" and "tanh" activation f-ns
        dense_activation_1 = hp.Choice('activation_1', values = ['relu', 'tanh'], default = 'relu')
        dense_activation_2 = hp.Choice('activation_2', values = ['relu', 'tanh'], default = 'relu')

        # Tune the learning rate for the optimizer 
        hp_learning_rate = hp.Float('learning_rate', min_value = 1e-4, max_value = 1e-2, sampling = 'LOG', default = 1e-3) 

        model.add(layers.Conv2D(filters = 64, kernel_size = hp_k_size, strides = 2, activation="relu", padding="same", input_shape = self.input_image_shape))

        model.add(layers.MaxPooling2D(2))
        model.add(layers.Conv2D(128, 3, activation="relu", padding="same"))
        model.add(layers.Conv2D(128, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D(2))
        model.add(layers.Conv2D(256, 3, activation="relu", padding="same"))
        model.add(layers.Conv2D(256, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D(2))
        model.add(layers.Flatten())

        model.add(layers.Dense(units = dense_units_l, activation = dense_activation_1))
        model.add(layers.BatchNormalization()) # Networks train faster & converge much more quickly
        model.add(layers.Dropout(dropout_rate_1))

        model.add(layers.Dense(units = dense_units_2, activation = dense_activation_2))
        model.add(layers.Dropout(dropout_rate_2))

        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Choose an optimal value from 0.01, 0.001, or 0.0001
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])
        return model


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

def optimizeCNNHyperparameters(scenario, image_size, seed_val = 1, save_results=True):
    if scenario=="Pr_Po_Im":
        NUM_CLASSES = 3
    else:
        NUM_CLASSES = 2
    
    hypermodel = CNNHyperModel(input_image_shape = (image_size, image_size, NUM_CHANNELS), num_classes=NUM_CLASSES)
    tuner = kt.Hyperband(hypermodel, seed = seed_val, hyperband_iterations = HYPERBAND_ITER, executions_per_trial=EXECUTIONS_PER_TRIAL, max_epochs = HYPERBAND_MAX_EPOCHS,
                         objective = 'val_accuracy', overwrite=True, #factor = 3,
                         directory = '../../results/opt', project_name = 'tuner_' + str(image_size) + 'px_' + scenario)
    training_images_and_labels, test_images_and_labels = splitData(image_dict[image_size][scenario], prop = 0.80, seed_num = 100 + seed_val)
    training_images, training_labels = getImageAndLabelArrays(training_images_and_labels)
    validation_images, validation_labels = getImageAndLabelArrays(test_images_and_labels)
    
    tuner.search(training_images, training_labels, validation_data = (validation_images, validation_labels), callbacks = [ClearTrainingOutput()])
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    best_hps_dict = best_hps.values
    if save_results:
        scenario_path = os.path.join(OPTIMAL_HYPERPARAMETERS_PATH, str(image_size), scenario)
        if not os.path.exists(scenario_path):
            os.makedirs(scenario_path)
        # with open(os.path.join(scenario_path, 'results-summary.txt'), 'w') as f:
        #     f.write(json.dumps(tuner.results_summary()))               
        with open(os.path.join(scenario_path, 'hyperparameters.txt'), 'w') as f:
            f.write(json.dumps(best_hps_dict))         
    return #(console_printout, summary, best_hps_dict)


def main():
    for i in RESOLUTION_LIST:
        for s in SCENARIO_LIST:     
            print("Beginning search for scenario: " + s + ", resolution: " + i)
            optimizeCNNHyperparameters(s, i, seed_val = 1, save_results = True)
            print("Search for scenario: " + s + ", resolution: " + i + " is complete.")
    return

if __name__ == "__main__":
    main()

