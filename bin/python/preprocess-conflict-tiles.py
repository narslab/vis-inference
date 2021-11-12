#!/usr/bin/env python
# coding: utf-8

import time
from datetime import timedelta
import sys
import os # used for navigating to image path
import shutil
import numpy as np
import pandas as pd
import csv
from PIL import Image
import imageio # used for writing images
from sklearn.model_selection import train_test_split
from helpers import *

SEED = 100
IMAGE_WIDTH_LIST = [189, 252]#, 336]
CONF_DIR = '../../data/tidy/conflict-detection/'
ORIGINAL_TILES = os.path.join(CONF_DIR,'split-images/all_tiles/')
CLASSIFIED_TILES_DIR = os.path.join(CONF_DIR, 'classified-tiles/')
PREPROCESSED_TILES_DIR = os.path.join(CONF_DIR, 'preprocessed-tiles/')
TILES_ASSIGNMENT = os.path.join(CONF_DIR, 'split-images/tiles_assignment.csv')

def assign_sample_tiles(unassigned_tiles_path):
    """Randomly classifies tiles as 'c'(conflict) or 'n'(no conflict)"""
    np.random.seed(SEED) # ensure repeatability
    data = pd.read_csv(unassigned_tiles_path)
    data['Classification'] = np.random.choice(['c','n'], size=len(data))
    
    return data

def generate_index(sample_data): #path_to_tiles_assignment
    """Serially labels all tiles by class and generates an index to data/tidy/conflict_split_images"""
    
#     data = pd.read_csv(path_to_tiles_assignment)
#     data = data.drop([0])
    
    df = pd.DataFrame(columns=['original', 'tile', 'classification'])
    
    counts = {'conflict':1,
             'no_conflict':1,
             'unknown':1}
        
    for row in sample_data.itertuples():
        tile = row.Filename
        original_image = str(os.path.splitext(tile)).split('_')[0][2:]
        classification = row.Classification

        if classification.lower() == 'c':        
            save_label = 'conflict' + '-' + str(counts['conflict']) + '.jpg'
            df.loc[row.Index] = [original_image] + [tile] + [save_label]
            counts['conflict'] += 1
        elif classification.lower() == 'n':
            save_label = 'no_conflict' + '-' + str(counts['no_conflict']) + '.jpg'
            df.loc[row.Index] = [original_image] + [tile] + [save_label]
            counts['no_conflict'] += 1
        else:
            save_label = 'unknown' + '-' + str(counts['unknown']) + '.jpg'
            df.loc[row.Index] = [original_image] + [tile] + [save_label]
            counts['unknown'] += 1
    
    df.to_csv(CONF_DIR+'tile_index_mapping.csv', encoding='utf-8', index=False)
    
    counts['conflict'] -= 1
    counts['no_conflict'] -= 1
    counts['unknown'] -= 1 
    
    print('Number of conflict tiles recorded:', counts['conflict'])    
    print('Number of no-conflict tiles recorded:', counts['no_conflict'])
    print('Number of unknown tiles:', counts['unknown'])
        
    return df, counts
    
def rename_tiles_conflict(data):
    """Rename each tile with its respective classification category"""
    shutil.rmtree(CLASSIFIED_TILES_DIR, ignore_errors=True)
    if not os.path.exists(CLASSIFIED_TILES_DIR):
        os.makedirs(CLASSIFIED_TILES_DIR)
    for filename in os.listdir(ORIGINAL_TILES):
        for row in data.itertuples():
            tile = row.tile + '.jpg'
            classification = row.classification
            if tile == filename:
                imageio.imwrite(CLASSIFIED_TILES_DIR+classification, imageio.imread(ORIGINAL_TILES+filename))
                continue
                
def getImageOneHotVector(image_file_name):
    """Returns one-hot vector encoding for each sub-image based on the image file name:
    Conflict: 1
    No conflict: 0
    Unknown: -1
    """
    word_label = image_file_name.split('-')[0]
    if word_label == 'conflict' : 
        return 1
    elif word_label == 'no_conflict': 
        return 0
    else:
        return -1 # if label is not present for current image

def processConflictTiles(image_width, seed_value, save_image_binary_files=True, rectangular = True, test = False): # original size 4032 × 3024 px
    data_train = []
    data_test = []
    if test==True: # test just a few images to see what is going on
        image_list = os.listdir(CLASSIFIED_TILES_DIR) #[0:10]
    else:
        image_list = os.listdir(CLASSIFIED_TILES_DIR)
    random.seed(seed_value) #seed for repeatability
    print("Preprocessing images for image width " + str(image_width) + "px")
    image_list_train, image_list_test =  train_test_split(image_list, test_size = .2, random_state = seed_value)

    for image_index in image_list:
        label = getImageOneHotVector(image_index)
        if label == -1: # if image unlabeled, move to next one
            continue
        path = os.path.join(CLASSIFIED_TILES_DIR, image_index)
        image = Image.open(path) # read in image
        print(np.array(image).shape)
        image_width = int(image_width) 
        if rectangular==True:
            image_height = getRectangularImageHeight(image_width)
        else:
            image_height = image_width
        resized_image = image.resize((image_width, image_height), Image.BICUBIC)  
        if test == True:
            pass
        resized_image_array = np.array(resized_image)/255. # convert to array and scale to 0-1
        print("Resized Image shape: " + str(resized_image_array.shape))  
        if image_index in image_list_train:
            data_train.append([resized_image_array, label])                        
        else:
            data_test.append([resized_image_array, label])            
    print(len(data_train))
    print(len(data_test))  
    print("Training Images:", (np.array([x[1] for x in data_train])).sum(axis=0))
    print("Test Images:", (np.array([x[1] for x in data_test])).sum(axis=0))
        
    filename_prefix = 'w-' + str(image_width) + 'px-h-' + str(image_height) + "px"
    data_filename_train = filename_prefix+ "-train.npy"
    data_filename_test = filename_prefix + "-test.npy"
    if not os.path.exists(PREPROCESSED_TILES_DIR): 
        os.makedirs(PREPROCESSED_TILES_DIR)
    if save_image_binary_files == True:
        if test == True:
            data_filename_train = 'testing-' + data_filename_train
            data_filename_test = 'testing-' + data_filename_test
        print(os.path.join(PREPROCESSED_TILES_DIR, data_filename_train))
        np.save(os.path.join(PREPROCESSED_TILES_DIR, data_filename_train), data_train) #save as .npy (binary) file
        np.save(os.path.join(PREPROCESSED_TILES_DIR, data_filename_test), data_test) #save as .npy (binary) file        
        print("Saved " + data_filename_train + " to " + PREPROCESSED_TILES_DIR.split("../../",1)[1])
        print("Saved " + data_filename_test + " to " + PREPROCESSED_TILES_DIR.split("../../",1)[1])        
    return  #(image_selection_array, class_list)
    
def main():
    #Use randomly classified tiles until all are labeled
    sample_df = assign_sample_tiles(TILES_ASSIGNMENT) 
    df, c = generate_index(sample_df)
    start = time.time()
    rename_tiles_conflict(df)
    for width in IMAGE_WIDTH_LIST:
        processConflictTiles(width, seed_value=SEED, rectangular = False, save_image_binary_files=True, test=False)
    elapsed = (time.time() - start)
    print("Tile renaming and preprocessing completed in (h/m/s/ms):", str(timedelta(seconds=elapsed)))
    
if __name__ == "__main__":
    main()