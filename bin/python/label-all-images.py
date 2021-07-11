#!/usr/bin/env python
# coding: utf-8

"""
The purpose of this script is to consolidate all images taken by arborists and label them according to their classification.
An index containing the newly labeled image name and its original file path is also genereated.
"""

import pandas as pd
from PIL import Image # used for loading images
import numpy as np
import os # used for navigating to image path
import imageio # used for writing images
import natsort
from natsort import natsorted
import re # for matching image file name classes
import matplotlib.pyplot as plt
import random
import ntpath
import shutil
import csv
from timeit import default_timer as timer

RAW_IMAGE_DIR = '../../data/raw/Pictures for AI'
RAW_IMAGE_DIR_SUMMER = '../../data/raw/Summer 2021 AI Photos'
TIDY_IMAGE_DIR = '../../data/tidy/labeled_images/'
INDEX_DIR = '../../results/index_raw/'

def getListOfFiles(dirName):
    """Returns single list of the filepath of each of the image files"""
    # source: https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
    allFiles = list()
    if 'Pictures for AI' in dirName: # 
        listOfFile = os.listdir(dirName)
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                allFiles = allFiles + getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)
    else:
        level_one = [f.path for f in os.scandir(dirName) if f.is_dir()] # get all of the immediate subdirectories in the parent folder downloaded from SharePoint
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
    unknown_counter = 1
    csv_col_index = ['Labeled Image', 'Original File Path']
    index = {}
    shutil.rmtree(TIDY_IMAGE_DIR, ignore_errors=True) # Deletes the directory containing any existing labeled images
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
            else:
                print(filename)
                unknown_counter += 1
            index[save_name] = filename
            imageio.imwrite(save_name, np.array(Image.open(filename)))
        else:
            print(filename)
            unknown_counter += 1
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    with open(INDEX_DIR + 'labels_index.csv', 'w', newline='') as f: # TODO: separate by tab not comma
        writer = csv.DictWriter(f, fieldnames=csv_col_index)
        writer.writeheader()
        for key in natsort.natsorted(index.keys()): # iterate through the alphanumeric keys in a natural order
            key_name = key.replace(TIDY_IMAGE_DIR,'')
            val_name = index[key]
            f.write("%s,%s\n"%(key_name,val_name))
    
    print('Number of improbable images saved:', improbable_counter-1)    
    print('Number of possible images saved:', possible_counter-1)
    print('Number of probable images saved:', probable_counter-1)
    print('Number of unknown images (not saved):', unknown_counter-1)
    print('Total number of images saved:', improbable_counter+possible_counter+probable_counter-3)

def main():    
    original_photos = getListOfFiles(RAW_IMAGE_DIR)
    summer_photos = getListOfFiles(RAW_IMAGE_DIR_SUMMER)
    index_list, description_list, arborist_list = splitIndexDescrArb(summer_photos)
    unique_image_list_summer = getUniqueImages(summer_photos, arborist_list, index_list, description_list)
    saveImageFiles(original_photos+unique_image_list_summer)

if __name__ == "__main__":
    main()    