#!/usr/bin/env python
# coding: utf-8

import sys
import os # used for navigating to image path
import numpy as np
import pandas as pd
import csv
from PIL import Image
import imageio # used for writing images
from sklearn.model_selection import train_test_split
from helpers import *

SEED = 100
ALL_TILES = '../../data/tidy/conflict-split-images/all_tiles/'
PROCESSED_TILES_DIR = '../../data/tidy/conflict-split-images/preprocessed-tiles'
TILES_ASSIGNMENT = '../../data/tidy/conflict-split-images/tiles_assignment.csv'

def assign_sample_tiles(unassigned_tiles_path):
    """Randomly classifies tiles as 'c'(conflict) or 'n'(no conflict)"""
    
    data = pd.read_csv(unassigned_tiles_path)
    data['Classification'] = np.random.choice(['c','n'], size=len(data))
    
    return(data)
    
def generate_index(sample_data): #path_to_tiles_assignment
    """Serially labels all tiles by class and generates an index to data/tidy/conflict_split_images"""
    
#     data = pd.read_csv(path_to_tiles_assignment)
#     data = data.drop([0])
    
    df = pd.DataFrame(columns=['original', 'tile', 'classification'])
    
    conflict_counter    = 1
    no_conflict_counter = 1
    unknown_counter = 1
    counters = {}
        
    for row in sample_data.itertuples():
        tile = row.Filename
        original_image = str(os.path.splitext(tile)).split('_')[0][2:]
        classification = row.Classification

        if classification.lower() == 'c':        
            save_label = 'conflict' + '-' + str(conflict_counter) + '.jpg'
            df.loc[row.Index] = [original_image] + [tile] + [save_label]
            conflict_counter += 1
        elif classification.lower() == 'n':
            save_label = 'no_conflict' + '-' + str(no_conflict_counter) + '.jpg'
            df.loc[row.Index] = [original_image] + [tile] + [save_label]
            no_conflict_counter += 1
        else:
            save_label = 'unknown' + '-' + str(unknown_counter) + '.jpg'
            df.loc[row.Index] = [original_image] + [tile] + [save_label]
            unknown_counter += 1
    
    df.to_csv('../../data/tidy/conflict-split-images/tile_index_mapping.csv', encoding='utf-8', index=False)
    
    counters['conflict'] = conflict_counter - 1
    counters['no_conflict'] = no_conflict_counter - 1
    counters['unknown'] = unknown_counter -1 
    
    print('Number of conflict tiles recorded:', counters['conflict'])    
    print('Number of no-conflict tiles recorded:', counters['no_conflict'])
    print('Number of unknown tiles:', counters['unknown'])
        
    return df, counters
    
def rename_tiles_conflict(data):
    for filename in os.listdir(ALL_TILES):
        for row in data.itertuples():
            tile = row.tile
            classification = row.classification
            if tile == filename:
                os.rename(ALL_TILES + filename, ALL_TILES + classification)
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

def processConflictTiles(seed_value, save_image_binary_files=True, test = False): # original size 4032 × 3024 px
    data_train = []
    data_test = []
    image_list = os.listdir(ALL_TILES) 
    random.seed(seed_value) #seed for repeatability
    image_list_train, image_list_test =  train_test_split(image_list, test_size = .2, random_state = seed_value)

    for image_index in image_list:
        label = getImageOneHotVector(image_index)
        if label == -1: # if image labeled as unknown, move to next one
            continue
        path = os.path.join(ALL_TILES, image_index)
        image = Image.open(path) # read in image
        image_width, image_height = image.size
        if test == True:
            pass
        if image_index in image_list_train:
            data_train.append([image_list_train, label])
        else:
            data_test.append([image_list_train, label])            
    print("Training Images:", (np.array([x[1] for x in data_train])).sum(axis=0) )
    print("Test Images:", (np.array([x[1] for x in data_test])).sum(axis=0) )
    filename_prefix = 'w-' + str(image_width) + 'px-h-' + str(image_height)+ 'px'
    data_filename_train = filename_prefix+ "-train.npy"
    data_filename_test = filename_prefix + "-test.npy"
    if not os.path.exists(PROCESSED_TILES_DIR): # check if 'tidy/preprocessed_tiles' subdirectory does not exist
        os.makedirs(PROCESSED_TILES_DIR) # if not, create it    
    if save_image_binary_files == True:
        if test == True:
            data_filename_train = 'testing-' + data_filename_train
            data_filename_test = 'testing-' + data_filename_test
        np.save(os.path.join(PROCESSED_TILES_DIR, data_filename_train), data_train) #save as .npy (binary) file
        np.save(os.path.join(PROCESSED_TILES_DIR, data_filename_test), data_test) #save as .npy (binary) file        
        print("Saved " + data_filename_train + " to data/tidy/" + PROCESSED_TILES_DIR)
        print("Saved " + data_filename_test + " to data/tidy/" + PROCESSED_TILES_DIR)        
    return  #(image_selection_array, class_list)
    
def main():
    d = pd.read_csv(TILES_ASSIGNMENT)
    #Use randomly classified tiles until all are properly labeled
    sample_df = assign_sample_tiles(TILES_ASSIGNMENT) 
    df, c = generate_index(sample_df)
    rename_tiles_conflict(df)
    processConflictTiles(seed_value=SEED, save_image_binary_files=True, test=False)

if __name__ == "__main__":
    main()