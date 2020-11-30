#!/usr/bin/env python
# coding: utf-8

from PIL import Image # used for loading images
import numpy as np
import os # used for navigating to image path
import imageio # used for writing images
import re # for matching image file name classes

raw_image_dir = '../../data/raw/Pictures for AI'

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

def saveImageFilesBinary(image_file_list):
    """Serially labels all images by class (probable or improbable) and saves them to data/tidy/labeled_images"""
    probable_counter = 1
    improbable_counter = 1
    tidy_image_dir = '../../data/tidy/labeled_images_2_classes/'
    if not os.path.exists(tidy_image_dir):
        os.makedirs(tidy_image_dir)
    for filename in image_file_list:
        if '.JPG' in filename or '.jpg' in filename:        
            if any(re.findall(r'improbable', filename, re.IGNORECASE)):
                save_name = tidy_image_dir + 'improbable' + '-' + str(improbable_counter) + '.jpg'
                improbable_counter += 1
            elif any(re.findall(r'probable|possible', filename, re.IGNORECASE)):
                save_name = tidy_image_dir + 'probable' + '-' + str(probable_counter) + '.jpg'
                probable_counter += 1   
            imageio.imwrite(save_name, np.array(Image.open(filename)))
    print('Number of probable images saved:', probable_counter-1)
    print('Number of improbable images saved:', improbable_counter-1) 

def saveImageFilesTernary(image_file_list):
    """Serially labels all images by class (probable, possible or improbable) and saves them to data/tidy/labeled_images"""
    probable_counter = 1
    improbable_counter = 1
    possible_counter = 1
    tidy_image_dir = '../../data/tidy/labeled_images_3_classes/'
    if not os.path.exists(tidy_image_dir):
        os.makedirs(tidy_image_dir)
    for filename in image_file_list:
        if '.JPG' in filename or '.jpg' in filename:        
            if any(re.findall(r'improbable', filename, re.IGNORECASE)):
                save_name = tidy_image_dir + 'improbable' + '-' + str(improbable_counter) + '.jpg'
                improbable_counter += 1
            elif any(re.findall(r'probable', filename, re.IGNORECASE)):
                save_name = tidy_image_dir + 'probable' + '-' + str(probable_counter) + '.jpg'
                probable_counter += 1 
            elif any(re.findall(r'possible', filename, re.IGNORECASE)):
                save_name = tidy_image_dir + 'possible' + '-' + str(possible_counter) + '.jpg'
                possible_counter += 1 
            imageio.imwrite(save_name, np.array(Image.open(filename)))
    print('Number of probable images saved:', probable_counter-1)
    print('Number of possible images saved:', possible_counter-1)
    print('Number of improbable images saved:', improbable_counter-1) 

def main():    
    tree_image_list = getListOfFiles(raw_image_dir)
    saveImageFilesBinary(tree_image_list)
    saveImageFilesTernary(tree_image_list)

if __name__ == "__main__":
    main()

