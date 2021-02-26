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


# Globals
NUM_CHANNELS = 1
RESOLUTION_LIST = [224] # 64, 128] #, 224, 384]
SCENARIO_LIST = ["Pr_Im", "PrPo_Im", "Pr_PoIm", "Pr_Po_Im"]
NUM_MODEL_RUNS = 10
NUM_EPOCHS = 10
RESOLUTION_PERFORMANCE_METRICS_DIR = '../../results/sensitivity-tests'


# def constructBaseCNN(image_size, scenario):
#     image_shape = (image_size, image_size, NUM_CHANNELS)
#     p_dict = getOptCNNHyperparams(image_size, scenario)
#     if scenario=="Pr_Po_Im":
#         num_classes = 3
#     else:
#         num_classes = 2
#     base_model = models.Sequential([
#         layers.Conv2D(filters = 64, kernel_size = p_dict['kernel_size'], strides = 2, activation="relu", padding="same", input_shape = image_shape),
#         layers.MaxPooling2D(2),
#         layers.Conv2D(128, 3, activation="relu", padding="same"),
#         layers.Conv2D(128, 3, activation="relu", padding="same"),
#         layers.MaxPooling2D(2),
#         layers.Conv2D(256, 3, activation="relu", padding="same"),
#         layers.Conv2D(256, 3, activation="relu", padding="same"),
#         layers.MaxPooling2D(2),
#         layers.Flatten(),
        
#         layers.Dense(p_dict['units_1'], activation = p_dict['activation_1']),
#         layers.BatchNormalization(),
#         layers.Dropout(p_dict['dropout_1']), 
        
#         layers.Dense(p_dict['units_2'], activation = p_dict['activation_2']), 
#         layers.Dropout(p_dict['dropout_2']),
        
#         layers.Dense(num_classes, activation="softmax")
#     ])
#     return(base_model)


image_sets = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST)


def testResolutionScenarioPerformance(image_size, scenario, num_epochs = 10, trial_seed = 1): 
    training_images_and_labels, test_images_and_labels = splitData(image_sets[image_size][scenario], prop = 0.80, seed_num = trial_seed)
    training_images, training_labels = getImageAndLabelArrays(training_images_and_labels)
    validation_images, validation_labels = getImageAndLabelArrays(test_images_and_labels)
    # early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    # batch_training_histories = Histories()
    # metrics_multiclass = Metrics(validation_images,validation_labels)  TODO
    K.clear_session()
    input_shape = (image_size, image_size, NUM_CHANNELS)
    model = constructBaseCNN(image_size, scenario, num_channels = NUM_CHANNELS)
    opt_learning_rate = getOptCNNHyperparams(image_size, scenario)['learning_rate']
    reset_weights(model) # re-initialize model weights
    opt = tf.keras.optimizers.Adam(learning_rate = opt_learning_rate)
    model.compile(loss='categorical_crossentropy',  optimizer = opt, metrics = ['accuracy'])
    hist = model.fit(training_images, training_labels, batch_size = 32, epochs = num_epochs, verbose=0, validation_data=(validation_images, validation_labels)) #, callbacks=[batch_training_histories])
    performance_dict = {}    
    performance_dict['scenario'] = scenario
    performance_dict['image_size'] = image_size
    performance_dict['metrics'] = hist.history
    performance_dict['best_val_accuracy'] = np.max(hist.history['val_accuracy'])
    return(performance_dict)


def main(num_trials = NUM_MODEL_RUNS):
    if not os.path.exists(RESOLUTION_PERFORMANCE_METRICS_DIR): # check if 'tidy/preprocessed_images' subdirectory does not exist
        os.makedirs(RESOLUTION_PERFORMANCE_METRICS_DIR) # if not, create it    
    for s in SCENARIO_LIST:
        for p in RESOLUTION_LIST:
            for i in range(num_trials):
                print("Conducting performance test: Scenario - " + s + "; Resolution - " + str(p) + "px; Trial - " + str(i+1))
                scenario_performance_dict = testResolutionScenarioPerformance(p, s, num_epochs = NUM_EPOCHS, trial_seed = 1 + i) #ultimately should be averaged across trials       
                scenario_filename = "scenario_resolution_performance_" + s + str(p) + "px_trial_" + str(i+1) + ".txt"
                with open(os.path.join(RESOLUTION_PERFORMANCE_METRICS_DIR, scenario_filename), 'w') as f:
                   f.write(json.dumps(scenario_performance_dict )) # use `json.loads` to do the reverse)
    return


if __name__ == "__main__":
    main()