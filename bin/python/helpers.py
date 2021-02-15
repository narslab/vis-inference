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

        
from sklearn.metrics import recall_score, classification_report
from sklearn.datasets import make_classification

import json

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

# define callback to save batch-wise loss/accuracy; Source: https://stackoverflow.com/a/52206330/3023033
class Histories(Callback):
	def on_train_begin(self,logs={}):
		self.losses = []
		self.accuracies = []
	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.accuracies.append(logs.get('accuracy'))

# Callback to find metrics at epoch end (not perfectly implemented; TODO) # Source: https://stackoverflow.com/a/56485026/3023033
class Metrics(Callback):
	def __init__(self, x, y):
		self.x = x
		self.y = y if (y.ndim == 1 or y.shape[1] == 1) else np.argmax(y, axis=1)
		self.reports = []

	def on_epoch_end(self, epoch, logs={}):
		y_hat = np.asarray(self.model.predict(self.x))
		y_hat = np.where(y_hat > 0.5, 1, 0) if (y_hat.ndim == 1 or y_hat.shape[1] == 1)  else np.argmax(y_hat, axis=1)
		report = classification_report(self.y,y_hat,output_dict=True)
		self.reports.append(report)
		return
   
	# Utility method
	def get(self, metrics, of_class):
		return [report[str(of_class)][metrics] for report in self.reports]


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