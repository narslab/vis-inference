#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shutil
import time

import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import models

from helpers import createResolutionScenarioImageDict, getClassLabels, getImageAndLabelArrays, generateTiles, plotTilesGrid, rotate_image_to_vertical

RESOLUTION_LIST = [336]
SCENARIO_LIST = ["PrPo_Im"]
AUGMENTATION = 'fliplr'
RESULTS_DIR = '../../results/'
LABELED_RP_IMAGES = '../../data/rp/tidy/labeled-images/'
VIS_DATA_DIR = '../../data/vis/input/'
RP_MODEL_PATH = RESULTS_DIR + 'rp/models/opt-cnn-base-PrPo_Im-w-336-px-h-336-px/model'
INDEX_PATH = RESULTS_DIR + 'rp/index/preprocessed_index.csv'
VIS_INDEX_PATH = RESULTS_DIR + 'vis/index/'

RP_MODEL = models.load_model(RP_MODEL_PATH)
IMAGE_SETS_SQUARE_TRAIN = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST, augmentation=AUGMENTATION, type='train', rectangular = False, testing=False)
IMAGE_SETS_SQUARE_TEST = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST, augmentation=AUGMENTATION, type='test', rectangular = False, testing=False)

class_labels = getClassLabels(SCENARIO_LIST[0])
training_images, training_labels = getImageAndLabelArrays(IMAGE_SETS_SQUARE_TRAIN[RESOLUTION_LIST[0]][SCENARIO_LIST[0]])
test_images, test_labels = getImageAndLabelArrays(IMAGE_SETS_SQUARE_TEST[RESOLUTION_LIST[0]][SCENARIO_LIST[0]])

def analyzePredictions(labels):
    """Returns the indices of the test images for both correct and incorrect predictions. 
    The total number for each class should equal the corresponding value in the confusion matrix.
    Currently only scenario PrPo_Im is supported."""
    im = labels[0]
    prPo = labels[1]
    pred = {im+'_correct': [],
                   im+'_incorrect': [],
                   prPo+'_correct':[],
                   prPo+'_incorrect':[]}
    model_predictions = RP_MODEL.predict(np.squeeze(test_images))
    for i in range(len(test_labels)):
        observed = class_labels[np.argmax(test_labels[i])]
        predicted = class_labels[np.argmax(model_predictions[i])]
        if observed == im:
            if predicted == observed:
                pred[im+'_correct'].append(i)
            else:
                pred[im+'_incorrect'].append(i)
        else:
            if predicted == observed:
                pred[prPo+'_correct'].append(i)
            else:
                pred[prPo+'_incorrect'].append(i)
    return pred

def retrieveTP(pred):
    ''''''
    preprocessed_index = pd.read_csv(INDEX_PATH)
    tp_ind = pred["Probable/Possible_correct"]
    tp_ind_filename = VIS_INDEX_PATH + 'tp_index.csv'
    shutil.rmtree(VIS_INDEX_PATH, ignore_errors=True)
    os.makedirs(VIS_INDEX_PATH)
    # Create a set of preprocessed_index values from tp_ind by concatenating 'test-' to each element
    tp_preprocessed_indices = set(f'test-{i}' for i in tp_ind)
    # Filter out the rows in index_all that are present in tp_preprocessed_indices
    index_tp = preprocessed_index[preprocessed_index['preprocessed_index'].isin(tp_preprocessed_indices)]
    # Reset the index and drop the original index column
    index_tp = index_tp.reset_index(drop=True)
    # Extract corresponding labeled image paths
    labeled_image_paths = []
    for index in tp_ind:
        labeled_image = preprocessed_index[preprocessed_index["preprocessed_index"]=="test-"+str(index)]["labeled_image"].values[0]
        labeled_image_path = os.path.join(LABELED_RP_IMAGES, labeled_image)
        labeled_image_paths.append(labeled_image_path)
    # Save the resulting DataFrame index_tp
    index_tp.to_csv(tp_ind_filename, index=False)
    return labeled_image_paths

def moveTPtoVIS(labeled_paths):
    # Clear any existing labeled TP images
    shutil.rmtree(VIS_DATA_DIR, ignore_errors=True)
    tp_copy_dir = os.path.join(VIS_DATA_DIR, 'tp_copy/')
    
    # Create VIS directory
    os.makedirs(VIS_DATA_DIR)
    os.makedirs(tp_copy_dir)

    # Iterate through the labeled image paths
    for path in labeled_paths:
        # Extract the filename from the path
        filename = os.path.basename(path)

        # Construct the destination path
        destination_path = os.path.join(tp_copy_dir, filename)

        # Copy the image file to the destination directory
        shutil.copyfile(path, destination_path)
        
    for filename in os.listdir(tp_copy_dir):
        image_path = os.path.join(tp_copy_dir, filename)
        rotate_image_to_vertical(image_path) 
        image = cv2.imread(image_path)
        if(image.shape[1] != 4032 | image.shape[2] != 3024):
            resized_image = cv2.resize(image, (4032, 3024))
            cv2.imwrite(image_path, resized_image)
            
def generateTilesGrids():
    tp_copy_dir = os.path.join(VIS_DATA_DIR, 'tp_copy/')
    collective_dir = os.path.join(VIS_DATA_DIR, 'tiles/')
    conf_split = os.path.join(VIS_DATA_DIR, 'grids/') 
    
    shutil.rmtree(collective_dir, ignore_errors=True)
    shutil.rmtree(conf_split, ignore_errors=True)
    
    # Create directory to hold all tiles
    os.makedirs(collective_dir, exist_ok=True)
    # # Create directory to hold all grids
    os.makedirs(conf_split, exist_ok=True)
    
    for filename in os.listdir(tp_copy_dir):
        if os.path.isfile(os.path.join(tp_copy_dir, filename)):
            t = generateTiles(filename, tp_copy_dir, collective_dir)
            plotTilesGrid(t, conf_split)
            
def main():
    start = time.time()
    predictions = analyzePredictions(class_labels)
    labeled_paths = retrieveTP(predictions)
    moveTPtoVIS(labeled_paths)
    generateTilesGrids()
    end = time.time()
    elapsed = end - start
    print('Time elapsed: {m}min {s}sec'.format(m=int(elapsed//60),s=int(np.round(elapsed%60,0))))
    
if __name__ == '__main__':
	main()