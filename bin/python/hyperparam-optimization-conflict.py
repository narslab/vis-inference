from PIL import Image # used for loading images
import numpy as np
import os 
import imageio # used for writing images
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from timeit import default_timer as timer
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

import sys
sys.path.append("../python/")
from helpers import *

"""HP Tuning"""
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch

import IPython

import keras_tuner as kt
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

NUM_CHANNELS = 3
OPTIMAL_HYPERPARAMETERS_PATH = '../../results/conflict-detection/optimal-hyperparameters/'
HYPERBAND_MAX_EPOCHS = 12 #10
EXECUTIONS_PER_TRIAL = 2 #5
HYPERBAND_ITER = 3 #80
PATIENCE = 6

IMAGE_SET_TRAIN = np.load('../../data/tidy/preprocessed-images/conflict-tiles-train.npy', allow_pickle = True)
IMAGE_SET_VAL = np.load('../../data/tidy/preprocessed-images/conflict-tiles-validation.npy', allow_pickle = True)

class Metrics(Callback):
    def __init__(self, val_data):#, batch_size = 64):
        super().__init__()
        self.validation_data = val_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}): #changed average to binary
        xVal, yVal = self.validation_data
        val_pred = np.argmax(np.asarray(self.model.predict(xVal)), axis=1)
        val_true = np.argmax(yVal, axis=1)        
        _val_f1 = f1_score(val_true, val_pred, average='binary', zero_division = 0)
        _val_precision = precision_score(val_true, val_pred, average='binary', zero_division = 0)
        _val_recall = recall_score(val_true, val_pred, average='binary', zero_division = 0)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        logs["val_f1"] = _val_f1
        logs["val_recall"] = _val_recall
        logs["val_precision"] = _val_precision
        print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        return
        
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
        dense_units_l = hp.Int('units_1', min_value = 128, max_value = 512, step = 32, default = 128)
        dense_units_2 = hp.Int('units_2', min_value = 128, max_value = 512, step = 32,  default = 128)  

        dropout_rate_1 = hp.Float('dropout_1', min_value = 0.0, max_value = 0.5, step = 0.05)
        dropout_rate_2 = hp.Float('dropout_2', min_value = 0.0, max_value = 0.5, step = 0.05)

        # Experiment with "relu" and "tanh" activation f-ns
        dense_activation_1 = hp.Choice('activation_1', values = ['relu', 'tanh'], default = 'relu')
        dense_activation_2 = hp.Choice('activation_2', values = ['relu', 'tanh'], default = 'relu')

        # Tune the learning rate for the optimizer 
        hp_learning_rate = hp.Float('learning_rate', min_value = 1e-5, max_value = 1e-3, sampling = 'LOG', default = 1e-4) 

        model.add(layers.Conv2D(filters = 64, kernel_size = hp_k_size, strides = 2, activation="relu", padding="same", input_shape = self.input_image_shape))
       # model.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
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
                    metrics = ['accuracy']#, tf.keras.metrics.Precision(name='precision'), 
                    #tf.keras.metrics.Recall(name='recall')]
                    )
        #print(model.summary())
        return model

def optimizeConflictHyperparameters(seed_val = 1, save_results=True):
    hypermodel = CNNHyperModel(input_image_shape = (448, 336, 3), num_classes=2)
    tuner = kt.Hyperband(hypermodel, seed = seed_val, hyperband_iterations = HYPERBAND_ITER, executions_per_trial=EXECUTIONS_PER_TRIAL, max_epochs = HYPERBAND_MAX_EPOCHS,
                         objective = kt.Objective("val_f1", direction="max"), overwrite=True, #factor = 3,
                         directory = '../../results/conflict-detection/opt', project_name = 'tuner-w-' + str(448) + 'px-h-' + str(336) + 'px')
    training_images = np.squeeze(np.array([x[0] for x in IMAGE_SET_TRAIN]))
    training_labels = np.array([x[1] for x in IMAGE_SET_TRAIN])
    print("Training image shape: ", training_images[0].shape)
    validation_images = np.squeeze(np.array([x[0] for x in IMAGE_SET_VAL]))
    validation_labels = np.array([x[1] for x in IMAGE_SET_VAL])
    print("Validation image shape: ", validation_images[0].shape)
    
    model_metrics = Metrics(val_data=(validation_images, validation_labels))
    # early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta = 0.001, restore_best_weights=True, 
    #                             mode = "min") # not needed for Hyberband; already built-in.

    tuner.search(training_images, training_labels, validation_data = (validation_images, validation_labels), callbacks = [model_metrics, ClearTrainingOutput()])
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    best_hps_dict = best_hps.values
    if save_results:
        scenario_path = os.path.join(OPTIMAL_HYPERPARAMETERS_PATH, 'w-'+str(448) + 'px-h-' + str(336) + 'px')
        if not os.path.exists(scenario_path):
            os.makedirs(scenario_path)
        with open(os.path.join(scenario_path, 'hyperparameters.txt'), 'w') as f:
            f.write(json.dumps(best_hps_dict))         
    return #(console_printout, summary, best_hps_dict)
    
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)
    
def main():
    print("Beginning optimal hyperparameter search - conflict detection.")
    optimizeConflictHyperparameters(seed_val = 1, save_results = True)
    print("Search for optimal hyperparameter - conflict detection - is complete.")
    return

if __name__ == "__main__":
    main()        