import os
import numpy as np
from timeit import default_timer as timer
import pydot
import random

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

import matplotlib.pyplot as plt
        
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

def get_nth_key(dictionary, n=0):
    """Get dictionary keys by index"""
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")

def convertToCoords(d):
    """Transforms a sequential list l=d*d+1 into an xy coordinate space
    Example: convertToCoords(2) -> [(1, 1), (1, 2), (2, 1), (2, 2)]"""
    l = []
    for k in range(1, d*d+1,1):
        dm = (k//d,k%d) #divmod(k,d)
        if dm[1] != 0:
            t = (dm[0]+1, dm[1])
        else:
            t = (dm[0], d)
        l.append(t)
    return l    
    
def splitData(image_array, prop = 0.80, seed_num = 111):
    """Returns training and test arrays of images with specified proportion - prop:1-prop"""
    random.Random(seed_num).shuffle(image_array)
    train_size = int(prop*np.shape(image_array)[0])
    train = image_array[:train_size]
    test = image_array[train_size:]
    return(train, test)

# Reference: https://colab.research.google.com/drive/17tAC7xx2IJxjK700bdaLatTVeDA02GJn#scrollTo=4-OduFD-wH14&line=15&uniqifier=1
def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    Returns normalized guided backpropagation output
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
#     if K.image_data_format() == 'channels_first':
#         x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def eraser(input_img, p=1.0, s_l=0.01, s_h=0.05, r_1=0.3, r_2=1/0.3, v_l=0, v_h=1, pixel_level=True):
    """Regularizes the model by randomly masking parts of the training image with random values
    p : the probability that random erasing is performed
    s_l, s_h : minimum / maximum proportion of erased area against input image
    r_1, r_2 : minimum / maximum aspect ratio of erased area
    v_l, v_h : minimum / maximum pixel value for erased area
    pixel_level : pixel-level randomization for erased area
    """
    if input_img.ndim == 3:
        img_h, img_w, img_c = input_img.shape
    elif input_img.ndim == 2:
        img_h, img_w = input_img.shape

    p_1 = np.random.rand()

    if p_1 > p:
        return input_img

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        if input_img.ndim == 3:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        if input_img.ndim == 2:
            c = np.random.uniform(v_l, v_h, (h, w))
    else:
        c = np.random.uniform(v_l, v_h)

    output_img = np.copy(input_img)
    output_img[top:top + h, left:left + w] = c

    return output_img

def standardizePlot(index, plot_dir, title):
    '''Standardizing the plotting functionality when rendering the visual output of each technique'''
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.figure(figsize=(5, 5))
    plt.suptitle(title)
    plt.title('Index ' + str(index), y=-0.06)
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

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
		#image_array = np.moveaxis(image_array, 1, -1)
	label_array = np.array([x[1] for x in image_label_tuple_array])
	return(image_array, label_array)

def getClassLabels(scenario):
    if scenario=="Pr_Po_Im":
        labels = ["Improbable", "Possible", "Probable"]
    elif scenario=="Pr_Im":
        labels = ["Improbable", "Probable"]
    elif scenario=="PrPo_Im":
        labels = ["Improbable","Probable/Possible"]
    elif scenario=="Pr_PoIm":
        labels = ["Possible/Improbable","Probable"]
    return(labels)	

def getRectangularImageHeight(width):
    """Gets corresponding height for rectangular image width"""
    height = int(width * 4032/3024)
    return height

def createResolutionScenarioImageDict(image_width_list, scenario_list, train=True, rectangular=False, testing=False):
    image_dict = dict.fromkeys(image_width_list)
    if train==True:
        train_test = 'train'
    else:
        train_test = 'test'
    for w in image_width_list:
        image_dict[w] = dict.fromkeys(scenario_list)
        for s in scenario_list:
            if rectangular==True:
                h = getRectangularImageHeight(w)
            else:
                h = w
            if testing:
                image_dict[w][s] = np.load('../../data/tidy/preprocessed-images/testing-w-' + str(w) + 'px-h-' + str(h) + 'px-scenario-' + s + '-' + train_test + '.npy', allow_pickle = True)
            else:
                image_dict[w][s] = np.load('../../data/tidy/preprocessed-images/w-' + str(w) + 'px-h-' + str(h) + 'px-scenario-' + s + '-' + train_test + '.npy', allow_pickle = True)
    print(image_dict)
    return(image_dict)

def getOptCNNHyperparams(image_width, image_height, scenario):
    with open('../../results/optimal-hyperparameters/' + 'w-' + str(image_width) + 'px-h-' + str(image_height) + 'px/' + scenario + '/hyperparameters.txt') as f: 
        data = f.read() 
    opt_params_dict = json.loads(data)   
    return(opt_params_dict)

def constructOptBaseCNN(image_width, image_height, scenario, num_channels = 1):
    image_shape = (image_height, image_width, num_channels)
    p_dict = getOptCNNHyperparams(image_width, image_height, scenario)
    if scenario=="Pr_Po_Im":
        num_classes = 3
    else:
        num_classes = 2
    base_model = models.Sequential([
        layers.Conv2D(filters = 64, kernel_size = p_dict['kernel_size'], strides = 2, activation="relu", padding="same", 
            input_shape = image_shape),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
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
#     return w
    
    
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