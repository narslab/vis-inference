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

def saveImageFiles(image_file_list, classification_scenario):
    """Serially labels all images by class:
    Classification Scenario A: {probable, possible, improbable}
    Classification Scenario B: {probable, improbable}
    Classification Scenario C: {{probable, possible}, improbable}
    Classification Scenario D: {probable, {possible, improbable}}
    and saves them to data/tidy/labeled_images"""
    
    if classification_scenario=="A":
        improbable_counter = 1
        possible_counter = 1
        probable_counter = 1
        tidy_image_dir = '../../data/tidy/labeled_images_scenario_A/'
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
        print('Number of improbable images saved:', improbable_counter - 1)    
        print('Number of possible images saved:', possible_counter - 1)
        print('Number of probable images saved:', probable_counter - 1)
        
    elif classification_scenario=="B":
        improbable_counter = 1
        probable_counter = 1        
        tidy_image_dir = '../../data/tidy/labeled_images_scenario_B/'
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
                    imageio.imwrite(save_name, np.array(Image.open(filename)))
            print('Number of improbable images saved:', improbable_counter - 1) 
            print('Number of probable images saved:', probable_counter - 1)

    elif classification_scenario=="C":
        improbable_counter = 1
        possible_or_probable_counter = 1
        tidy_image_dir = '../../data/tidy/labeled_images_scenario_C/'
        if not os.path.exists(tidy_image_dir):
            os.makedirs(tidy_image_dir)
            for filename in image_file_list:
                if '.JPG' in filename or '.jpg' in filename:        
                    if any(re.findall(r'improbable', filename, re.IGNORECASE)):
                        save_name = tidy_image_dir + 'improbable' + '-' + str(improbable_counter) + '.jpg'
                        improbable_counter += 1
                    elif any(re.findall(r'possible|probable', filename, re.IGNORECASE)):
                        save_name = tidy_image_dir + 'possible_or_probable' + '-' + str(probable_counter) + '.jpg'
                        possible_or_probable_counter += 1   
                    imageio.imwrite(save_name, np.array(Image.open(filename)))
            print('Number of improbable images saved:', improbable_counter - 1) 
            print('Number of possible+probable images saved:', possible_or_probable_counter - 1)

    elif classification_scenario=="D":
        improbable_or_possible_counter = 1
        probable_counter = 1
        tidy_image_dir = '../../data/tidy/labeled_images_scenario_D/'
        if not os.path.exists(tidy_image_dir):
            os.makedirs(tidy_image_dir)
            for filename in image_file_list:
                if '.JPG' in filename or '.jpg' in filename:        
                    if any(re.findall(r'improbable|possible', filename, re.IGNORECASE)):
                        save_name = tidy_image_dir + 'improbable_or_possible' + '-' + str(improbable_counter) + '.jpg'
                        improbable_or_possible_counter += 1
                    elif any(re.findall(r'probable', filename, re.IGNORECASE)):
                        save_name = tidy_image_dir + 'probable' + '-' + str(probable_counter) + '.jpg'
                        probable_counter += 1   
                    imageio.imwrite(save_name, np.array(Image.open(filename)))
            print('Number of improbable+possible images saved:', improbable_or_possible_counter - 1) 
            print('Number of probable images saved:', probable_counter - 1)
        
def main():    
    tree_image_list = getListOfFiles(raw_image_dir)
    saveImageFiles(tree_image_list, "A")
    saveImageFiles(tree_image_list, "B")
    saveImageFiles(tree_image_list, "C")
    saveImageFiles(tree_image_list, "D")
    
if __name__ == "__main__":
    main()

