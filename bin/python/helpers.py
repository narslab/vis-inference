import os
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pydot
import random

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

        
from sklearn.metrics import recall_score, classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.datasets import make_classification

import json

# Handling files
def getListOfFiles(dirName):
    """Returns single list of the filepath of each of the training image files"""
    # source: https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def splitData(image_array, prop = 0.80, seed_num = 111):
	"""Returns training and test arrays of images with specified proportion - prop:1-prop"""
	random.Random(seed_num).shuffle(image_array)
	train_size = int(prop*np.shape(image_array)[0])
	train = image_array[:train_size]
	test = image_array[train_size:]
	return(train, test)

def getImageShape(image_array, num_channels):
	"""Returns shape of image from array of images, e.g. WIDTH x HEIGHT x NUM_CHANNELS"""
	if num_channels==1:
		image_shape = np.array([np.expand_dims(x[0],axis=2) for x in image_array]).shape[1:4]
	elif num_channels==3:
		image_shape = np.array([x[0] for x in image_array]).shape[1:4][::-1]
	print(image_shape)
	return image_shape

def getImageAndLabelArrays(image_label_tuple_array, num_channels = 1):
	"""Separates array of tuples (image matrix, vector label) into array of image matrices and array of image labels"""
	if num_channels == 1:
		image_array = np.array([np.expand_dims(x[0],axis=2) for x in image_label_tuple_array])
	elif num_channels == 3:
		image_array = np.array([x[0] for x in image_label_tuple_array]) 
		image_array = np.moveaxis(image_array, 1, -1)
	label_array = np.array([x[1] for x in image_label_tuple_array])
	return(image_array, label_array)

def getClassLabels(scenario):
    if scenario=="Pr_Po_Im":
        labels = ["Probable", "Possible", "Improbable"]
    elif scenario=="Pr_Im":
        labels = ["Probable", "Improbable"]
    elif scenario=="PrPo_Im":
        labels = ["Probable/Possible", "Improbable"]
    elif scenario=="Pr_PoIm":
        labels = ["Probable", "Possible/Improbable"]
    return(labels)	

def createResolutionScenarioImageDict(resolution_list, scenario_list):
    image_dict = dict.fromkeys(resolution_list)
    for p in resolution_list:
        image_dict[p] = dict.fromkeys(scenario_list)
        for s in scenario_list:
            image_dict[p][s] = np.load('../../data/tidy/preprocessed_images/size' + str(p) + '_exp5_' + s + '.npy', allow_pickle = True)
    return(image_dict)

def getOptCNNHyperparams(image_size, scenario):
    with open('../../results/optimal-hyperparameters/' + str(image_size) + '/' + scenario + '/hyperparameters.txt') as f: 
        data = f.read() 
    opt_params_dict = json.loads(data)   
    return(opt_params_dict)

def constructBaseCNN(image_size, scenario, num_channels = 1):
    image_shape = (image_size, image_size, num_channels)
    p_dict = getOptCNNHyperparams(image_size, scenario)
    if scenario=="Pr_Po_Im":
        num_classes = 3
    else:
        num_classes = 2
    base_model = models.Sequential([
        layers.Conv2D(filters = 64, kernel_size = p_dict['kernel_size'], strides = 2, activation="relu", padding="same", input_shape = image_shape),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        
        layers.Dense(p_dict['units_1'], activation = p_dict['activation_1']),
        layers.BatchNormalization(),
        layers.Dropout(p_dict['dropout_1']), 
        
        layers.Dense(p_dict['units_2'], activation = p_dict['activation_2']), 
        layers.Dropout(p_dict['dropout_2']),

        layers.Dense(num_classes, activation="softmax")   
        ])
    return(base_model)


#https://github.com/keras-team/keras/issues/5400#issuecomment-408743570
def check_units(y_true, y_pred):
	if y_pred.shape[1] != 1:
		y_pred = y_pred[:,1:2]
		y_true = y_true[:,1:2]
	return y_true, y_pred

def precision(y_true, y_pred):
	y_true, y_pred = check_units(y_true, y_pred)
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def recall(y_true, y_pred):
	y_true, y_pred = check_units(y_true, y_pred)
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def f1(y_true, y_pred):
	y_true, y_pred = check_units(y_true, y_pred)
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
def reset_weights(model):
	"""This function re-initializes model weights at each compile"""
	for layer in model.layers: 
		if isinstance(layer, tf.keras.Model):
			reset_weights(layer)
			continue
	for k, initializer in layer.__dict__.items():
		if "initializer" not in k:
			continue
		# find the corresponding variable
		var = getattr(layer, k.replace("_initializer", ""))
		var.assign(initializer(var.shape, var.dtype))
        
def getLayerWeights(model_path, layer_name):
    """This function loads a model and retrieves the weight of a specific layer"""
    m = models.load_model(model_path)
    layer_variables = m.get_layer(layer_name).get_weights()
    weights = layer_variables[0]
    w = np.array(weights)
    w = np.moveaxis(w, 2, 0)
    w = np.moveaxis(w, 3, 0)
    print(w.shape)
    return w
    

# define callback to save batch-wise loss/accuracy; Source: https://stackoverflow.com/a/52206330/3023033
class Histories(Callback):
	def on_train_begin(self,logs={}):
		self.losses = []
		self.accuracies = []
	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.accuracies.append(logs.get('accuracy'))


def plot_model_accuracy(hist):
	plt.plot(hist.history["accuracy"])
	plt.plot(hist.history["val_accuracy"])
	plt.title("Model Accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Epoch")
	plt.legend(["Train", "Validation"], loc="lower right")
	plt.show()

def plot_model_batch_accuracy(hist):
	#plt.plot(hist.accuracies)
	plt.plot(hist.val_accuracies)
	plt.title("Model Accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Iteration")
	#plt.legend(["Train", "Validation"], loc="lower right")
	plt.show()
	
def plot_model_loss(hist):
	plt.plot(hist.history["loss"])
	plt.plot(hist.history["val_loss"])
	plt.title("Model Loss")
	plt.ylabel("Loss")
	plt.xlabel("Epoch")
	plt.legend(["Train", "Validation"], loc="upper right")
	plt.show()