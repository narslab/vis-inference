#!/usr/bin/env python
# coding: utf-8

'''Process images necessary for the calculation of VIS interpretability metric.'''

import os
import sys
import shutil
import time
from datetime import timedelta

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib import cm

from helpers import *
from analyze_predictions_new_gradcam import generateGradcam, renderGradcam

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

RESOLUTION_LIST = [336]
TILE_SIZE = 38
SCENARIO_LIST = ["PrPo_Im"]
AUGMENTATION = 'fliplr'
RP_MODEL_PATH = '../../results/rp/models/opt-cnn-base-PrPo_Im-w-336-px-h-336-px/model'
CD_MODEL_PATH = '../../results/cd/models/conf-det-test-inception_v3--w-336-px-h-336-px/model'
VIS_INDEX_PATH = '../../results/vis/index/tp_index.csv'

source_directory = '../../data/vis/input/tiles/'
destination_directory = '../../data/vis/output/non_processed'
dest_processed = '../../data/vis/output/processed/'

cd_model = models.load_model(CD_MODEL_PATH)
rp_model = models.load_model(RP_MODEL_PATH)
vis_index = pd.read_csv(VIS_INDEX_PATH)
# Edit the preprocessed_index column to exclude 'test-'
vis_index['preprocessed_index'] = vis_index['preprocessed_index'].str.replace('test-', '')
vis_index['labeled_image'] = vis_index['labeled_image'].str.rstrip('.jpg')

IMAGE_SETS_SQUARE_TRAIN = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST, augmentation=AUGMENTATION, type='train', rectangular = False, testing=False)
IMAGE_SETS_SQUARE_TEST = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST, augmentation=AUGMENTATION, type='test', rectangular = False, testing=False)

class_labels = getClassLabels(SCENARIO_LIST[0])
training_images, training_labels = getImageAndLabelArrays(IMAGE_SETS_SQUARE_TRAIN[RESOLUTION_LIST[0]][SCENARIO_LIST[0]])
test_images, test_labels = getImageAndLabelArrays(IMAGE_SETS_SQUARE_TEST[RESOLUTION_LIST[0]][SCENARIO_LIST[0]])

# Function to convert the image array into a grid of tiles
def convert_to_grid(image_array, tile_size=38, is_gradcam=True):
    if is_gradcam:
        ax = 0
    else:
        ax = -1
    # Convert RGB image to grayscale
    grayscale_image = np.mean(image_array, axis=ax)
    height, width = grayscale_image.shape[:2]
    grid_size_h = height // tile_size
    grid_size_w = width // tile_size
    tiles = np.zeros((grid_size_h, grid_size_w, tile_size, tile_size))

    for i in range(grid_size_h):
        for j in range(grid_size_w):
            tile = grayscale_image[i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size]
            tiles[i, j] = np.squeeze(tile)

    return tiles

def plot_img_to_grid(image_array, tile_size=38, remove_space=True):
    space_arg = {}
    if remove_space:
        space_arg = {'hspace': 0, 'wspace': 0}
    # Create the grid plot
    grid_size_h, grid_size_w = image_array.shape[:2]
    fig, axs = plt.subplots(grid_size_h, grid_size_w, figsize=(10, 10), gridspec_kw=space_arg)

    # Plot each tile in the grid
    for i in range(grid_size_h):
        for j in range(grid_size_w):
            axs[i, j].imshow(image_array[i, j], 'jet')
            axs[i, j].axis('off')

    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

def process_vis_tiles(labeled_image):
    '''Copy tiles to the destination directory and process them into .npy array'''
    # List all files in the source directory
    files = os.listdir(source_directory)

    shutil.rmtree(destination_directory, ignore_errors=True)
    os.makedirs(destination_directory)

    # Iterate through the files in the source directory
    for file in files:
        if file.split('_', 1)[0] == labeled_image:
            # If the file starts with 'possible_121', construct the source and destination paths
            source_file = os.path.join(source_directory, file)
            destination_file = os.path.join(destination_directory, file)

            # Copy the file from the source directory to the destination directory
            shutil.copyfile(source_file, destination_file)

    image_list = os.listdir(destination_directory)
    data_filename = labeled_image + '_draw_predictions.npy'
    processed_data_filepath = os.path.join(dest_processed, data_filename)

    shutil.rmtree(dest_processed, ignore_errors=True)
    os.makedirs(dest_processed)

    data = []
    for image in image_list:
        path = os.path.join(destination_directory, image)
        image = Image.open(path) # read in image
        image = image.resize((336, 336), Image.NEAREST) 
        scaled_image_array = np.array(image)/255.
        data.append([scaled_image_array])

    np.save(processed_data_filepath, data)
    return(processed_data_filepath)  

def calculate_single_vis(rp_data, cd_tiles):
    class_avg = {0: 0,
                 1: 0}
    avg_tile_intensity = []
    results = {}
    # Make predictions on the images using the loaded model
    predictions = cd_model.predict(np.squeeze(rp_data))
    # Interpret the predictions of the two-class classification problem with softmax activation
    class_labels = np.argmax(predictions, axis=1)
    unique_elements, counts = np.unique(class_labels, return_counts=True)
    frequency_dict = dict(zip(unique_elements, counts))
    # If a key doesn't exist, add it with the default value of 0
    if 0 not in frequency_dict:
        frequency_dict.setdefault(0, 0)
    elif 1 not in frequency_dict:
        frequency_dict.setdefault(1, 0)
    print(frequency_dict)
    
    for i in range(0,9):
        for j in range(0,9):
            avg_tile_intensity.append(np.mean(cd_tiles[i][j]))
            
    for c in range(len(class_labels)):
        if class_labels[c] == 0:
            class_avg[0] += avg_tile_intensity[c]
        else:
            class_avg[1] += avg_tile_intensity[c] 
    
    for key in class_avg.keys():
        if frequency_dict[key] != 0:
            results[key] = class_avg[key] / frequency_dict[key]
        else:
            results[key] = 0
            
    print(results)
    vis_score = results[1] - results[0]
    return vis_score

def calculate_vis():
    rp_predictions = rp_model.predict(np.squeeze(test_images))
    rp_model_vis = 0
    for loc in range(len(vis_index)):
        labeled_image = vis_index.iloc[loc].values[0]
        preprocessed_index = int(vis_index.iloc[loc].values[1])
        gcam = generateGradcam(rp_model, rp_predictions, preprocessed_index)
        gcam_resized = resize_image_array(gcam, (1, 342, 342))
        # Convert the Gradcam array into a grid of tiles
        conflict_tiles = convert_to_grid(gcam_resized)
        preprocessed_tiles_path = process_vis_tiles(labeled_image)
        image_data = np.load(preprocessed_tiles_path)
        img_vis = calculate_single_vis(image_data, conflict_tiles)
        rp_model_vis += img_vis
    network_vis = rp_model_vis / len(vis_index)
    print("VIS of risk prediction model is: ", str(network_vis))
    
if __name__ == "__main__":
    start = time.time()
    calculate_vis()
    elapsed = time.time() - start
    print("Calculated network VIS in (h/m/s/ms):", str(timedelta(seconds=elapsed)))
    