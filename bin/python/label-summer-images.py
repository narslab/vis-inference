#!/usr/bin/env python
# coding: utf-8

"""
The purpose of this script is to label the images taken by arborists in Summer 2021.
The script obtains a list of all the files downloaded and extracted from SharePoint
and returns a list of unique images by splitting their file paths.
"""

import numpy as np
import pandas as pd

from PIL import Image # used for loading images
import os # used for navigating to image path
import imageio # used for writing images
import re # for matching image file name classes
import matplotlib.pyplot as plt
import random
import ntpath
import csv
from timeit import default_timer as timer

# Download and extract new images from sharepoint
SUMMER_21_IMAGES = '../../data/raw/Summer 2021 AI Photos'
TIDY_IMAGE_DIR   = '../../data/tidy/summer_21_labeled_images/'

def getListOfFiles(dirName):
    """Returns single list of the filepath of each of the training image files taken in the Summer of 2021"""
    # source: https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
    # modified
    level_one = [f.path for f in os.scandir(dirName) if f.is_dir()] # get all of the immediate subdirectories in the parent folder downloaded from SharePoint
    print(level_one)
    allFiles = list()
    for subdir in level_one:
        level_two = [f.path for f in os.scandir(subdir) if f.is_dir()] # get all of the immediate subdirectories for each arborist
        print(level_two)
        for subdir2 in level_two:
            if ('Away' in subdir2): # do not include images taken away powerlines
                print('Ignoring images away from power lines')
                #break
            else:
                listOfFile = os.listdir(subdir2) # retain only images taken near power lines
                # Iterate over all the entries
                for entry in listOfFile:
                    # Create full path
                    fullPath = os.path.join(subdir2, entry)
                    # If entry is a directory then get the list of files in this directory 
                    if os.path.isdir(fullPath):
                        allFiles = allFiles + getListOfFiles(fullPath)
                    else:
                        allFiles.append(fullPath)
                
    return allFiles
    
def splitIndexDescrArb(images):
    """Split the filepaths of the images taken in Summer 2021 a list of indices, descriptions and arborists"""
    index_list = []
    description_list = []
    arborist_list = []
    for fn in images:
        if ('_' in fn):
            index = fn.split('_')[0].split('\\')[-1]
            description = fn.split('_')[1]
            arborist = fn.split('_')[0].split('\\')[1].split('/')[-1]
        else:
            index = fn.split('-')[0].split('\\')[-1].rstrip(' ')
            description = fn.split('-')[1]
            arborist = fn.split('_')[0].split('\\')[1].split('/')[-1]        
        index = int(index)
        index_list.append(index)
        description_list.append(description)
        arborist_list.append(arborist)
    return index_list, description_list, arborist_list

def getUniqueImages(images, arb_list, idx_label, descr):
    """Get a list of unique Summer 2021 images by ignoring close-ups. Only selects an image with the classification present in the label."""
    df = pd.DataFrame([images, arb_list, idx_label, descr])
    df = df.transpose()
    df.columns=['Filename','Arborist', 'Index_Label', 'Description']
    unique_image_list = []
    
    for arborist in df.Arborist.unique():
        print(arborist)
        dfsub = df.loc[df.Arborist == arborist]
        for index in dfsub.Index_Label.unique():
            dfsub2 = dfsub.loc[dfsub.Index_Label == index]
            if len(dfsub2) == 1:
                selected_filename = dfsub2.Filename.tolist()[0]
            else:
                good_index_list = []
                for r in dfsub2.index:
                    if 'closeup' not in dfsub2.loc[r, 'Description']:
                        l = ['probable', 'possible', 'improbable']
                        if [i for i in l if re.findall(i, dfsub2.loc[r, 'Description'], re.IGNORECASE)]: # check if a classification is included in the file name
                            good_index_list.append(r)
                print(good_index_list)
                if len(good_index_list) == 1:
                    selected_filename = dfsub2.loc[good_index_list, 'Filename'].tolist()[0]
                else:
                    testdf = dfsub2.loc[good_index_list]
                    testdf['DescriptionLength'] = testdf.Description.str.len()
                    for i in good_index_list:
                        selected_filename = testdf.loc[testdf.DescriptionLength.idxmin(), 'Filename']
            print(selected_filename)                
            unique_image_list.append(selected_filename)
    return unique_image_list
    
def saveImageFiles(image_file_list):
    """Serially labels all images by class:  and saves them to the designated tidy image directory."""
    improbable_counter = 1
    possible_counter = 1
    probable_counter = 1
    if not os.path.exists(TIDY_IMAGE_DIR):
        os.makedirs(TIDY_IMAGE_DIR)
    for filename in image_file_list:
        if '.JPG' in filename or '.jpg' in filename:        
            if any(re.findall(r'improbable', filename, re.IGNORECASE)):
                save_name = TIDY_IMAGE_DIR + 'improbable' + '-' + str(improbable_counter) + '.jpg'
                improbable_counter += 1
            elif any(re.findall(r'probable', filename, re.IGNORECASE)):
                save_name = TIDY_IMAGE_DIR + 'probable' + '-' + str(probable_counter) + '.jpg'
                probable_counter += 1 
            elif any(re.findall(r'possible', filename, re.IGNORECASE)):
                save_name = TIDY_IMAGE_DIR + 'possible' + '-' + str(possible_counter) + '.jpg'
                possible_counter += 1 
            imageio.imwrite(save_name, np.array(Image.open(filename)))
    print('Number of improbable images saved:', improbable_counter - 1)    
    print('Number of possible images saved:', possible_counter - 1)
    print('Number of probable images saved:', probable_counter - 1)
    
def main():    
    summer_photos = getListOfFiles(SUMMER_21_IMAGES)
    print(len(summer_photos))
    index_list, description_list, arborist_list = splitIndexDescrArb(summer_photos)
    unique_image_list = getUniqueImages(summer_photos, arborist_list, index_list, description_list)
    saveImageFiles(unique_image_list)    

if __name__ == "__main__":
    main()