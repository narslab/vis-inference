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
RESOLUTION_LIST = [64, 128] # 64, 128] #, 224, 384]
SCENARIO_LIST = ["Pr_Im", "PrPo_Im", "Pr_PoIm", "Pr_Po_Im"]
NUM_EPOCHS = 20
SAVED_MODEL_DIR = '../../results/models/'
MODEL_PERFORMANCE_METRICS_DIR = '../../results/model-performance/'

#
image_sets = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST)

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


def trainModelWithDetailedMetrics(image_size, scenario, num_epochs = 10, trial_seed = 1, testing = True): 
    
    # IMAGES (former approach)
    training_images_and_labels, test_images_and_labels = splitData(image_sets[image_size][scenario], prop = 0.8, seed_num = trial_seed)
    training_images, training_labels = getImageAndLabelArrays(training_images_and_labels)
    validation_images, validation_labels = getImageAndLabelArrays(test_images_and_labels)
    class_labels = getClassLabels(scenario)
    print("Class labels:", class_labels)
    # training_images, validation_images, training_labels, validation_labels =  train_test_split(np.array([np.expand_dims(x[0],axis=2) for x in image_sets[image_size][scenario]]), 
    #                                                                                            np.array([x[1] for x in image_sets[image_size][scenario]]), 
    #                                                                                            stratify= np.array([x[1] for x in image_sets[image_size][scenario]]), 
    #                                                                                            test_size = .2, random_state = trial_seed)

    print("Number of class training images:", training_labels.sum(axis=0), "total: ", training_labels.sum())
    print("Number of class validation images:", validation_labels.sum(axis=0), "total: ", validation_labels.sum())
    
    # CALLBACKS
    model_metrics = Metrics(val_data=(validation_images, validation_labels))
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    
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
                     callbacks = [model_metrics,early_stopping])     
    
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
    figure = plt.figure(figsize=(4, 4))    ## Confusion matrix heatmap
    ax = sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues, fmt='g', cbar = False, annot_kws={"size": 16})
    figure.tight_layout()
    plt.ylabel('True',fontsize=16)
    ax.set_yticklabels(class_labels,va='center',fontsize=14)
    ax.set_xticklabels(class_labels, ha='center',fontsize=14)
    plt.xlabel('Predicted',fontsize=16)
    #plt.show()
    if testing == True:
        con_mat_heatmap_file = "../../figures/test-opt-confusion-matrix-" + scenario + "-" + str(image_size) + "-px.png"
    else:
        con_mat_heatmap_file = "../../figures/opt-confusion-matrix-" + scenario + "-" + str(image_size) + "-px.png"
    figure.savefig(con_mat_heatmap_file, dpi=180, bbox_inches='tight')
    return(model, hist) 


def visualizeCNN(model, scenario, image_size, images_per_class = 4, trial_seed = 1, saliency=False, testing=True):
    #ALT: use train_test_split
#    trimg,vaimg,trlab,valab =  train_test_split(np.array([np.expand_dims(x[0],axis=2) for x in image_sets[64]["Pr_Im"]]), 
#                 np.array([x[1] for x in image_sets[64]["Pr_Im"]]), stratify= np.array([x[1] for x in image_sets[64]["Pr_Im"]]), test_size=.2, random_state = 1  )

    training_images_and_labels, test_images_and_labels = splitData(image_sets[image_size][scenario], prop = 0.8, seed_num = trial_seed)
    training_images, training_labels = getImageAndLabelArrays(training_images_and_labels)
    validation_images, validation_labels = getImageAndLabelArrays(test_images_and_labels)
    class_labels = getClassLabels(scenario)
    print("Class labels:", class_labels)
    print(training_labels.sum(axis=0))
    print(validation_labels.sum(axis=0))    
    
    # GRAD CAM
    random.seed(trial_seed)
    # Randomly sample images from each class
    random_image_selection_class_0 = random.sample([i for i, j in enumerate(validation_labels) if np.argmax(j) == 0], k = images_per_class)    
    random.seed(trial_seed+1)
    random_image_selection_class_1 = random.sample([i for i, j in enumerate(validation_labels) if np.argmax(j) == 1], k = images_per_class)
    assert validation_labels[random_image_selection_class_0].mean(axis=0)[0] == 1 #assert that indices of class 0 labels are correct
    assert validation_labels[random_image_selection_class_1].mean(axis=0)[1] == 1 #assert that indices of class 1 labels are correct
    cam_list = random_image_selection_class_0 + random_image_selection_class_1 # join lists of indices in both classes
    if scenario=="Pr_Po_Im": # in 3-class case
        random.seed(trial_seed+2)
        random_image_selection_class_2 = random.sample([i for i, j in enumerate(validation_labels) if np.argmax(j) == 2], k = images_per_class)    
        cam_list = cam_list + random_image_selection_class_2 # join to prior list of class 0 and class 1
        assert validation_labels[random_image_selection_class_2].mean(axis=0)[2] == 1 #assert that indices of class 2 labels are correct
    # subset validation images to use for gradcam
    print("List of indices from validation images:", cam_list)
    gradcam_images = validation_images[cam_list] #tf.convert_to_tensor(validation_images[cam_list], dtype= tf.float32)
    print("Shape of gradcam image array:", gradcam_images.shape)
    print([np.argmax(j) for j in validation_labels[cam_list] ])
    
    # Loss function for gradcam
    def loss(output):
        """Returns score corresponding to class of given image index"""                                       
        #loss_tuple = (output[random_image_selection_class_0][0], output[random_image_selection_class_1][1])
        #first image loss (class 1); second image loss (class 2)
        #class_0_losses = [output[i][0] for i in np.arange(images_per_class)] 
        #class_1_losses = [output[i + images_per_class][1] for i in np.arange(images_per_class)]    
        loss_list = [output[i][j] for i, j in enumerate([np.argmax(j) for j in validation_labels[cam_list] ])]
        #print(loss_list)
        #(output[0][0], output[1][1]) #, output[2][2])                        
        #if scenario=="Pr_Po_Im":
            #loss_tuple = (output[random_image_selection_class_0][0], output[random_image_selection_class_1][1], output[random_image_selection_class_2][2])
            #loss_list = (output[0][0], output[1][1], output[2][2])           
        #    class_2_losses = [output[i + 2*images_per_class][2] for i in np.arange(images_per_class)]
        #    loss_list.extend(class_2_losses)
        return loss_list
    
    # Model_modifier function required for gradcam
    def model_modifier(m):
        """Remove softmax activation of last layer in model"""
        m.layers[-1].activation = tf.keras.activations.linear
        return m

    # Create Gradcam object
    gradcam = Gradcam(model, model_modifier = model_modifier)#, clone=False)

    # Generate heatmap with GradCAM
    subplot_args = { 'nrows': len(class_labels), 'ncols': images_per_class, 'figsize': (3*images_per_class,3*len(class_labels)), 
                    'subplot_kw': {'xticks': [], 'yticks': []} }    
    cam = gradcam(loss, gradcam_images, penultimate_layer = -1)
    cam = normalize(cam)
    print(len(cam))
    f, ax = plt.subplots(**subplot_args)
    f.set_facecolor("white")
    image_counter = 0
    for i, label in enumerate(class_labels):
        ax[i,0].set_ylabel(label, fontsize=14)
        for j in np.arange(images_per_class):
            print(i, j, image_counter)
            heatmap = np.uint8(cm.jet(cam[image_counter])[..., :3] * 255)            
            ax[i, j].imshow(gradcam_images[image_counter], cmap='gist_gray')
            ax[i, j].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
            image_counter += 1
    plt.tight_layout()
    #plt.show()
    if testing==True:
        f.savefig("../../figures/test-opt-gradcam-" + scenario + "-" + str(image_size) + "-px-" + str(images_per_class) + "-images.png")
    else:
        f.savefig("../../figures/opt-gradcam-" + scenario + "-" + str(image_size) + "-px-" + str(images_per_class) + "-images.png")
    if saliency==True:
        saliency = Saliency(model, model_modifier=model_modifier)#                    clone=False)

        # Generate saliency map with smoothing that reduce noise by adding noise
        saliency_map = saliency(loss, gradcam_images, 
                                smooth_samples=20, # The number of calculating gradients iterations.
                                smooth_noise=0.20) # noise spread level.
        saliency_map = normalize(saliency_map)
        #image_titles = class_labels
        f, ax = plt.subplots(**subplot_args)
        f.set_facecolor("white")
        image_counter = 0
        for i, label in enumerate(class_labels):
            ax[i,0].set_ylabel(label, fontsize=14)
            for j in np.arange(images_per_class):
                print(i, j, image_counter)
                ax[i, j].imshow(saliency_map[image_counter], cmap='jet', alpha=0.5) # overlay
                image_counter += 1
        plt.tight_layout()
        #plt.show()
        if testing == True:
            f.savefig("../../figures/test-opt-saliency-" + scenario + "-" + str(image_size) + "-px-" + str(images_per_class) + "-images.png")
        else:
            f.savefig("../../figures/opt-saliency-" + scenario + "-" + str(image_size) + "-px-" + str(images_per_class) + "-images.png")
    return


def getScenarioModelPerformance(res = 64, num_epochs = 15, seed_val = 1, test_boolean = True):
    df = pd.DataFrame()
    for s in SCENARIO_LIST:
        m, h = trainModelWithDetailedMetrics(res, s, num_epochs, trial_seed = seed_val, testing = test_boolean)
        visualizeCNN(m, s, res, images_per_class = 4, trial_seed = seed_val, testing = test_boolean)       
        perf = pd.DataFrame.from_dict(h.history)
        perf[['Scenario']] = s
        perf['epoch'] = perf.index + 1
        df = df.append(perf, ignore_index=True)
    if test_boolean == True:
        df_filename = "../../results/test-opt-cnn-performance-metrics-summary-" + str(res) + "px.csv"
    else:
        df_filename = "../../results/opt-cnn-performance-metrics-summary-" + str(res) + "px.csv"
    df.to_csv(df_filename)
    return df

if __name__ == "__main__":
    getScenarioModelPerformance(res=64, num_epochs=15, seed_val = 1, test_boolean=False)
