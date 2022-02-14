#!/usr/bin/env python
# coding: utf-8


import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from scipy.special import softmax

from timeit import default_timer as timer
import random
import cv2

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tkr

from tf_keras_vis.utils.callbacks import Print

from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from tf_keras_vis.activation_maximization import ActivationMaximization


import json
import pickle
import os
import sys
sys.path.append("../python/")
from helpers import *
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Globals
NUM_CHANNELS = 3
RESOLUTION_LIST = [336] 
SCENARIO_LIST = ["PrPo_Im"]
NUM_EPOCHS = 20
SAVED_MODEL_DIR = '../../results/models/'
MODEL_PERFORMANCE_METRICS_DIR = '../../results/model-performance/'
FULL_MODEL_PATH = '../../results/models/opt-cnn-base-a-PrPo_Im-w-336-px-h-336-px/model'
IMAGE_SETS_SQUARE_TRAIN = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST, train=True, rectangular = False)
IMAGE_SETS_SQUARE_TEST = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST, train=False, rectangular = False)
GLOBAL_MODEL = models.load_model(FULL_MODEL_PATH)

trial_seed = 1
class_labels = getClassLabels("PrPo_Im")

training_images, training_labels = getImageAndLabelArrays(IMAGE_SETS_SQUARE_TRAIN[336]["PrPo_Im"])
test_images, test_labels = getImageAndLabelArrays(IMAGE_SETS_SQUARE_TEST[336]["PrPo_Im"])

def grad_cam(index): 
    cam_img = index # class activation map corresponds to the specified image index 
    probPrediction = [] # store probability of prediction by the CNN
    predictedClass = [] # store CNN-predicted class 
    # class_labels = getClassLabels("PrPo_Im")
    print("Observed image class: ", class_labels[np.argmax(test_labels[cam_img])])
    m = GLOBAL_MODEL
    def loss(output):
        """Returns score corresponding to class of given image index"""                                       
        print("Loss output: ", output) ##
        loss_list = [output[i][j] for i, j in enumerate([np.argmax(j) for j in test_labels[cam_img] ])]
        print(loss_list)
        probPrediction.append(str([np.round(max(softmax(loss)), 2) for loss in output.numpy()])[1:-1])
        predictedClass.append(class_labels[np.argmax([softmax(loss) for loss in output.numpy()])])
        print('Probability of prediction: ', probPrediction)
        print('Predicted class: ', predictedClass)
        print([softmax(loss) for loss in output.numpy()])
        return loss_list

    print('Class labels: ', class_labels)
    
    # Model_modifier function required for gradcam
    def model_modifier(model):
        """Remove softmax activation of last layer in model"""
        model.layers[-1].activation = tf.keras.activations.linear # Assign linear activation function (pass-through) to the activation of layer at -1
        return model
    
    gradcam_image = np.squeeze(test_images[cam_img])
    gradcam = Gradcam(m, model_modifier = model_modifier)
    cam = gradcam(loss, gradcam_image, penultimate_layer = -1) # Penultimate layer is a fully-connected hidden layer. 
    cam = normalize(cam)
    cam = np.squeeze(cam)  
    print(gradcam_image.shape)
    print("Shape of heatmap matrix:", cam.shape )
    return cam, gradcam_image, probPrediction, predictedClass

def guided_backprop(index, activation_layer):
    # Reference: https://colab.research.google.com/drive/17tAC7xx2IJxjK700bdaLatTVeDA02GJn#scrollTo=jgTRCYgX4oz-&line=1&uniqifier=1  
    backprop_image = test_images[index].reshape(1, 336, 336, 3)
    @tf.custom_gradient
    def guidedRelu(x):
        def grad(dy):
            return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
        return tf.nn.relu(x), grad

    model = models.load_model(FULL_MODEL_PATH) # set model inside the function
    modified_model = models.Model(
        inputs = [model.inputs],
        outputs = [model.get_layer(activation_layer).output]
    )
    layer_dict = [layer for layer in modified_model.layers[1:] if hasattr(layer,'activation')]
    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guidedRelu

    with tf.GradientTape() as tape:
        inputs = tf.cast(backprop_image, tf.float32)
        tape.watch(inputs)
        outputs = modified_model(inputs)

    grads = tape.gradient(outputs,inputs)[0]
    return grads

def guided_gradcam(index, activation_layer='conv2d_4'):
    # element-wise multiplication in Python
    # https://stackoverflow.com/questions/40034993/how-to-get-element-wise-matrix-multiplication-hadamard-product-in-numpy
    gbp  = guided_backprop([index], activation_layer)
    gcam = grad_cam([index])[0]
#     if normalize:
#         gbp = deprocess_image(np.array(np.squeeze(gbp)))
#     else:
#         gbp = np.squeeze(gbp)
    
    ggcam = gbp * gcam[..., np.newaxis]
    print('Shape of GGCAM: ', ggcam.shape)
    return ggcam

def renderGradCam(index, save=False, grid=False, c_map='gist_gray'):
    '''Visualizing gradient-weighted class activation maps as an overlay (heatmap) to the original image.'''
    gradcam_dir = '../../figures/plottingGradCam/'
    gcam = grad_cam([index])
        
    if grid==False:
        plt.figure(figsize=(5, 5))
        # plt.suptitle('Grad-CAM Index '+str(index)[1:-1]+'\n')
        plt.title('Observed class: '+class_labels[np.argmax(test_labels[index])], y=-0.08)
        plt.text(-0.7,-0.05, "Probability of prediction: "+ str("{:.0%}".format(float(str(gcam[2])[2:-2])))+'\n', size=12)
        plt.text(-0.7,-0.35, 'Predicted class: '+ str(gcam[3])[2:-2], size=12)
        plt.axis('off')
        plt.tight_layout()
        heatmap = np.uint8(cm.jet(gcam[0])[..., :3] * 255)
        plt.imshow(np.squeeze(gcam[1]), cmap=c_map) # remove axes of length one from gradcam_images
        plt.imshow(heatmap, cmap='gist_gray', alpha=0.5) # overlay
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
    else:
        plt.axis('off')
        plt.tight_layout()
        heatmap = np.uint8(cm.jet(gcam[0])[..., :3] * 255)
        plt.imshow(np.squeeze(gcam[1]), cmap=c_map)
        plt.imshow(heatmap, cmap='gist_gray', alpha=0.5)
    if save:
        if not os.path.exists(gradcam_dir):
            os.makedirs(gradcam_dir)
        plt.savefig(gradcam_dir+"gradcam-"+str(index)+".png")
    return (heatmap)

def renderGBP(index, activation_layer='conv2d_4', save=False, grid=False, c_map='gist_gray'): #normalize=False
    '''Visualizing Guided Backpropagation with an option to use normalized gradients.'''
    gbp = deprocess_image(np.squeeze(guided_backprop([index], activation_layer)))
    gbp_dir = '../../figures/plottingBackProp/'
        
    if grid:
        plt.axis('off')
        plt.tight_layout()
#         plt.imshow(np.flip(gbp, -1), cmap=c_map) 
    else:
        standardizePlot(index, gbp_dir, 'Guided Backpropagation')    
        
    plt.imshow(np.flip(gbp, -1), cmap=c_map) # Reverse the order of elements, starting from the last axis, in order to compute saliency.
    
    if save:
        save_name = gbp_dir+"guided_backprop-"+str(index)+'-'+c_map+".png"
        plt.savefig(save_name)
    return (gbp)

def renderGGCAM(index, activation_layer='conv2d_4', save=False, grid = False, c_map='gist_gray'):
    '''Visualizing Guided Grad-CAM output with an option to use normalized guided backpropagation gradients.'''
    ggcam_dir = '../../figures/plottingGuided-GradCam/'
    plot_name = ''
    ggcam = deprocess_image(np.squeeze(guided_gradcam(index, activation_layer)))
    
    if grid:
        plt.axis('off')
        plt.tight_layout()
#         plt.imshow(np.flip(ggcam, -1), cmap=c_map) 
    else:
        standardizePlot(index, ggcam_dir, 'Guided Grad-CAM')
        
    plt.imshow(np.flip(ggcam, -1), cmap=c_map) # Reverse the order of elements, starting from the last axis, in order to compute saliency.   
    
    if save:
        plot_name = "guided-gradcam-"+str(index)+'-'+c_map+".png"
        plt.savefig(ggcam_dir+plot_name)
        print('Saving '+plot_name+' in '+ggcam_dir)
        
    return(ggcam)

def plotVisualizations(index, activation_layer='conv2d_4', c_map='gist_gray', save=False):
    subplot_args = { 'nrows': 1, 'ncols': 4, 'figsize': (15, 15), 
                    'subplot_kw': {'xticks': [], 'yticks': []} }
    grid_dir = '../../figures/'
    plot_name = 'plot_grid_'+str(index)
    f, axs = plt.subplots(**subplot_args)
    f.set_facecolor("white")
    axis_labels = ['Processed Input', 'GradCAM', 'Guided Backprop', 'Guided GradCAM']
    orig_image = test_images[index]
    gcam_heatmap = renderGradCam(index, False, True)    
    gbp = renderGBP(index, activation_layer, False, True)
    ggcam = renderGGCAM(index, activation_layer, False, True)
    f.suptitle('Observed: ' + class_labels[np.argmax(test_labels[index])]+ '. Predicted: ' + str(grad_cam([index])[3])[2:-2]+'. Probability: '+str("{:.0%}".format(float(str(grad_cam([index])[2])[2:-2])))+'.', y=0.595, fontsize=24, va='bottom') 
    axs[0].imshow(np.squeeze(orig_image))
    axs[1].imshow(np.squeeze(orig_image))
    axs[1].imshow(gcam_heatmap, cmap=c_map, alpha=0.5)
    axs[2].imshow(gbp, cmap=c_map)
    axs[3].imshow(ggcam, cmap=c_map)
    for axis in f.axes:
        axis.set_axis_on()
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_visible(False)        
    for i, axis in enumerate(f.axes):
        axis.set_xlabel(axis_labels[i], fontsize=18)
    f.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        f.savefig(grid_dir+plot_name, bbox_inches='tight')
        print('Saving '+plot_name+' in '+grid_dir)
    return 

if __name__ == "__main__":
    plotVisualizations(18, 'conv2d_4', 'gist_gray', False)