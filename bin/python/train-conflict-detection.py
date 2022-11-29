#!/usr/bin/env python
# coding: utf-8


import numpy as np
from numpy.random import seed
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import time
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
#from tf_keras_vis.gradcam import Gradcam
#from tf_keras_vis.saliency import Saliency
#from tf_keras_vis.utils import normalize

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

"ResNet 50 dependencies"
from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.applications import resnet50

"GoogLeNet dependencies"
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"Produce the same results every time the experiment is run"
#seed(1)
#tf.random.set_seed(2)
#from contextlib import redirect_stdout
# geom_line()
# facet_wrap( vars(transportation_type), ncol = 1 )
# + geom_line(aes(col = region), color = "Gray")

IMAGE_SET_TRAIN = np.load('../../data/tidy/preprocessed-images/conflict-tiles-train.npy', allow_pickle = True)
IMAGE_SET_VAL = np.load('../../data/tidy/preprocessed-images/conflict-tiles-validation.npy', allow_pickle = True)
IMAGE_SET_TEST = np.load('../../data/tidy/preprocessed-images/conflict-tiles-test.npy', allow_pickle = True)

# Globals
NUM_CHANNELS = 3
PATIENCE = 7
# TESTING = False
# AUGMENTATION = 'fliplr'
# IMAGE_WIDTH_LIST = [336]#,252 189, 336
SCENARIO_LIST = ["none"] #"Pr_Im", "PrPo_Im", "Pr_PoIm", "Pr_Po_Im"
ARCHITECTURE_LIST = ["resnet50", "inception_v3"] #, "base", "resnet50", "inception_v3", "base-a", "base-b", "base-c", "all_conv"
NUM_EPOCHS = 30
SAVED_MODEL_DIR = '../../results/conflict-detection/models/'
MODEL_PERFORMANCE_METRICS_DIR = '../../results/conflict-detection/model-performance/'
TM = time.strftime('%d-%b-%Y-%H-%M-%S')

                                   
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

def constructIV3(image_size, scenario, num_channels = 3):
    image_shape = (image_size, image_size, num_channels)
    iv3 = tf.keras.applications.InceptionV3(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=image_shape,
        pooling='max',
        classes=2,
        classifier_activation="softmax"
    )
    return(iv3)

def constructRN50(image_size, scenario, num_channels = 3):
    image_shape = (image_size, image_size, num_channels)
    rn50 = resnet50.ResNet50(include_top=True, 
                           weights=None, 
                           input_tensor=None, 
                           input_shape=image_shape, 
                           pooling= 'max', 
                           classes=2)
    return(rn50)


def testAllConv(image_width, image_height,  scenario, num_channels=3, num_classes=2):
    image_shape = (image_width, image_height, num_channels)
    model = models.Sequential() 
        
    model.add(Convolution2D(128, 3, border_mode = 'same', input_shape=image_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, border_mode='same'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(256, 3, border_mode = 'same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, border_mode='same'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(512, 3, border_mode = 'same'))
    model.add(Activation('relu'))    

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    # Choose an optimal value from 0.01, 0.001, or 0.0001
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = .001),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
    return(model)

def trainModelWithDetailedMetrics(image_width, image_height, scenario, architecture, num_epochs = 30, trial_seed = 1, testing = True): 
    # image_width = 448
    # image_height = 336
    # if rectangular:
    #     image_height = getRectangularImageHeight(image_width)
    # else:
    #     image_height = image_width

    class_labels = ["conflict", "no_conflict"]
    print("Class labels:", class_labels)
    print("Image width: " + str(image_width))
    print("Image height: " + str(image_height))
#     if rectangular==True:
#         image_dictionary_train = IMAGE_SETS_RECT_TRAIN
#         image_dictionary_test = IMAGE_SETS_RECT_TEST
        
#     else:
#         image_dictionary_train = IMAGE_SETS_SQUARE_TRAIN
#         image_dictionary_test = IMAGE_SETS_SQUARE_TEST
#         image_dictionary_validation = IMAGE_SETS_SQUARE_VALIDATION
    
    # print("Optimal hyperparameters:\n",getOptConfHyperparams(image_width,image_height))
    
    train_images = [x[0] for x in IMAGE_SET_TRAIN]
    train_labels = [x[1] for x in IMAGE_SET_TRAIN]
    test_images = np.array([x[0] for x in IMAGE_SET_TEST])
    test_labels = np.array([x[1] for x in IMAGE_SET_TEST])
    validation_images = [x[0] for x in IMAGE_SET_VAL]
    validation_labels = [x[1] for x in IMAGE_SET_VAL]

    #Combining taining and validation images and lables
    train_images = np.concatenate((train_images,validation_images))
    train_labels = np.concatenate((train_labels,validation_labels))
    
    print(train_images[0].shape)
    print("Number of class training images:", train_labels.sum(axis=0), "total: ", train_labels.sum())
    print("Number of class test images:", test_labels.sum(axis=0), "total: ", test_labels.sum())
    
    # CALLBACKS
    model_metrics = Metrics(val_data=(test_images, test_labels))
    early_stopping = EarlyStopping(monitor='val_f1', patience=PATIENCE, min_delta = 0.001, restore_best_weights=True, 
                                    mode = "max")
    
    # INIT MODEL AND PARAMS, FIT
    K.clear_session()
    #input_shape = (image_size, image_size, NUM_CHANNELS) ## shape of images
    opt = tf.keras.optimizers.Adam(learning_rate=0.001) # default value will be used for all testing cases (including ResNet/Inception)

    if architecture == 'resnet50':
        model = constructRN50(image_width, scenario, NUM_CHANNELS)
    elif architecture == 'inception_v3':
        model = constructIV3(image_width, scenario, NUM_CHANNELS)
    elif architecture == 'base':
        if testing:
            model = testAllConv(image_width, image_height, scenario, num_channels=NUM_CHANNELS)
        else:
            model = constructOptBaseCNN(image_width, image_height, conflict = True, num_channels = NUM_CHANNELS)    ## get model
            opt_learning_rate = getOptConfHyperparams(image_width, image_height)['learning_rate']    ## learning rate
            opt = tf.keras.optimizers.Adam(learning_rate = opt_learning_rate)    
    
    reset_weights(model) # re-initialize model weights
    print(model.summary)
    start = timer()
    if testing:
        model.compile(loss='categorical_crossentropy', metrics =  ['accuracy'])     ## compile and fit
    else:
        model.compile(loss='categorical_crossentropy', optimizer = opt, metrics =  ['accuracy'])     ## compile and fit
    #with tf.device('/cpu:0'): 
    hist = model.fit(train_images, train_labels, batch_size = 32, epochs = num_epochs, verbose=1, 
                     validation_data=(test_images, test_labels),
                     callbacks = [model_metrics, early_stopping])
    print(opt)
    end = timer()
    
    # SAVE MODEL, SUMMARY AND PERFORMANCE
    if testing == True:
        model_name = "conf-det-test-" + architecture + "-" + "-w-" + str(image_width) + "-px-h-" + str(image_height) + "-px"
    else:
        model_name = "conf-det-opt-" + architecture + "-" + "-w-" +str(image_width) + "-px-h-" + str(image_height) + "-px"
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
    y_pred = np.argmax(model.predict(test_images), axis=-1)     ## Params
    ## Classification report
    report = classification_report(np.argmax(test_labels, axis=-1), y_pred, zero_division=0,
                                   labels = np.arange(len(class_labels)), target_names=class_labels, output_dict=True)
    print("Classification report for " + ", width: " + str(image_width) + ", height: " + str(image_height) + ", architecture: " + architecture + ":")
    class_report_path = os.path.join('../../results/conflict-detection/classification-reports/', TM + '/')
    report = pd.DataFrame(report).transpose().round(2)
    if not os.path.exists(class_report_path):  
        os.makedirs(class_report_path)
    classification_report_suffix = architecture + "-" + "-w-" + str(image_width) + "-h-" + str(image_height) + ".csv"
    if testing == True:
        report.to_csv(class_report_path + "test-classification-report-" + classification_report_suffix)
    else:
        report.to_csv(class_report_path + "opt-classification-report-" + classification_report_suffix)        
    print(report)
    print("Patience value: ", PATIENCE)
    print("Completion time in seconds: ", end - start)
    ## Confusion matrix
    con_mat_path = os.path.join('../../figures/conflict-detection/confusion-matrix/', TM + '/')
    print(con_mat_path)
    if not os.path.exists(con_mat_path):  
        os.makedirs(con_mat_path)
    con_mat = tf.math.confusion_matrix(labels=np.argmax(test_labels, axis=-1), predictions=y_pred).numpy()
    # con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat, index = class_labels, columns = class_labels)
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
    file_suffix =  architecture+"-"+scenario + "-w-" + str(image_width) + "-px-h-" + str(image_height) + '-px' + ".png"
    if testing == True:
        con_mat_heatmap_file = con_mat_path + "test-confusion-matrix-" + file_suffix
    else:
        con_mat_heatmap_file = con_mat_path + "opt-confusion-matrix-" + file_suffix
    figure.savefig(con_mat_heatmap_file, dpi=180)#, bbox_inches='tight')
    return(hist) #model


def getScenarioModelPerformance(architecture, s, width = 336, height = 336, num_epochs = 20, seed_val = 1, test_boolean = True):
    df = pd.DataFrame()
    print(TM)
    # if rect_boolean:
    #     height = getRectangularImageHeight(width)
    # else:
    #     height = width
    for s in SCENARIO_LIST:
        h = trainModelWithDetailedMetrics(width, height, s, architecture, num_epochs, trial_seed = seed_val, testing = test_boolean)
        #visualizeCNN(m, s, width, images_per_class = 4, trial_seed = seed_val, testing = test_boolean)       
        perf = pd.DataFrame.from_dict(h.history)
        perf['Scenario'] = s
        perf['epoch'] = perf.index + 1
        df = df.append(perf, ignore_index=True)
    if test_boolean == True:
        df_filename = "../../results/conflict-detection/test-performance-metrics-summary-" + architecture + "-w-" + str(width) + "-px-h-" + str(height) + "-px" + "-" + TM +".csv"
    else:
        df_filename = "../../results/conflict-detection/opt-performance-metrics-summary-" + architecture + "-w-" + str(width) + "-px-h-" + str(height) +  "-px-"+ TM + ".csv"
    df.to_csv(df_filename)
    return df

if __name__ == "__main__":
    for a in ARCHITECTURE_LIST:
        for s in SCENARIO_LIST:
            K.clear_session()
            getScenarioModelPerformance(a, s, 42, 42, num_epochs=NUM_EPOCHS, seed_val = 2, test_boolean=False)
            
