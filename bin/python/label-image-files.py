#!/usr/bin/env python
# coding: utf-8

from PIL import Image # used for loading images
import numpy as np
import os # used for navigating to image path
import imageio # used for writing images
import re # for matching image file name classes
import matplotlib.pyplot as plt
import random
import ntpath

raw_image_dir = '../../data/raw/Pictures for AI'
tidy_image_dir = '../../data/tidy/labeled_images/'

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

def saveImageFiles(image_file_list):
    """Serially labels all images by class:  and saves them to data/tidy/labeled_images"""
    
    improbable_counter = 1
    possible_counter = 1
    probable_counter = 1
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

def plotRawImages(image_file_list, images_per_class = 2):
    fig, axarr = plt.subplots(images_per_class, 3, sharex=True, sharey=True)
    class_index = 0
    class_list = ['probable', 'possible', 'improbable']
    #print(image_file_list)
    for image_class in class_list:
        class_images = [i for i in image_file_list if ntpath.basename(i).startswith(image_class)]
        print(len(class_images))
        random.seed(111)
        random_class_selection = random.choices(class_images, k = images_per_class)
        for i in range(images_per_class):
            axarr[i,class_index].imshow((Image.open(random_class_selection[i])).transpose(Image.TRANSPOSE), aspect='auto', 
                interpolation='antialiased', extent = [0, 3024, 0, 4032])
        class_index += 1
    for ax, col in zip(axarr[0,:], [i.title() for i in class_list]):
        ax.set_title(col, size=15)
    image_filename = '../../figures/raw_input_images_' + str(images_per_class) + '.png'
    plt.xticks([0,3024])
    plt.yticks([0,4032])
    fig.savefig(image_filename, dpi=120)

def main():    
    tree_image_list = getListOfFiles(raw_image_dir)
    saveImageFiles(tree_image_list)
    tree_image_list_labeled = getListOfFiles(tidy_image_dir)
    plotRawImages(tree_image_list_labeled, images_per_class = 2)

if __name__ == "__main__":
    main()

