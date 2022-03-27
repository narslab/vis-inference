#!/usr/bin/env python
# coding: utf-8

"""
The purpose of this script is to pre-process the tree images using the following approaches:
    - one-hot vector encoding
    - resizing
    - grayscaling 
    - normalization (scaling pixels from 0 to 1)
    - random cropping (based on specified expansion factor to multiply images)
    - horizontal flipping
"""

from PIL import Image, ImageOps, ExifTags # used for loading images
import numpy as np
import os # used for navigating to image path
import imageio # used for writing images
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helpers import *
import time
from datetime import timedelta


## GLOBAL VARIABLES
IMAGE_WIDTH_LIST = [336] #[189, 252, 336]
# Original image size: 3024 x 4032
# Reduction factor of 9: 336 x 448
# Reduction factor of 12: 252 x 336
# Reduction factor of 16: 189 x 252
NUM_CHANNELS = 3
# Train/test/validation: 60/20/20
TEST_SET_SIZE = 0.34
VALIDATION_SET_SIZE = 0.25
AUGMENTATION = 'occlusion_all'
CLASSIFICATION_SCENARIO = "Pr_Im"
CLASSIFICATION_SCENARIO_LIST = ["Pr_Im", "PrPo_Im", "Pr_PoIm", "Pr_Po_Im"]  
LABELED_IMAGES_DIR = '../../data/tidy/labeled-images'
PROCESSED_IMAGES_DIR = '../../data/tidy/preprocessed-images'

SEED = 100  
#NUM_PLOT_IMAGES_PER_CLASS = 1 #4 ## NOT USED IN CURRENT IMPLEMENTATION
#EXPANSION_FACTOR = 5 #5 of augmented images ## NOT USED IN CURRENT IMPLEMENTATION

def getImageOneHotVector(image_file_name, classification_scenario = "Pr_Im"):
    """Returns one-hot vector encoding for each image based on specified classification scenario:
    Classification Scenario Pr_Po_Im (3 classes): {probable, possible, improbable}
    Classification Scenario Pr_Im (2 classes): {probable, improbable}
    Classification Scenario PrPo_Im (2 classes): {{probable, possible}, improbable} 
    Classification Scenario Pr_PoIm (2 classes): {probable, {possible, improbable}}
    """
    word_label = image_file_name.split('-')[0]
    if classification_scenario == "Pr_Po_Im":
        if word_label == 'probable' : 
            return np.array([0, 0, 1])
        elif word_label == 'possible' : 
            return np.array([0, 1, 0])    
        elif word_label == 'improbable':
            return np.array([1, 0, 0])
        else :
            return np.array([0, 0, 0]) # if label is not present for current image
    elif classification_scenario == "Pr_Im":
        if word_label == 'probable' : 
            return np.array([0, 1])
        elif word_label == 'improbable' : 
            return np.array([1, 0])
        else :
            return np.array([0, 0]) # if label is not present for current image
    elif classification_scenario == "PrPo_Im":
        if word_label in ['probable', 'possible'] : 
            return np.array([0, 1])
        elif word_label == 'improbable' : 
            return np.array([1, 0])
        else :
            return np.array([0, 0]) # if label is not present for current image        
    elif classification_scenario == "Pr_PoIm":
        if word_label == 'probable' : 
            return np.array([0, 1])
        elif word_label in ['possible', 'improbable'] : 
            return np.array([1, 0])
        else :
            return np.array([0, 0]) # if label is not present for current image        

def processImageData(image_width, class_scenario, seed_value, channels=1, augmentation='fliplr', save_image_binary_files=True, rectangular = True, test = False): # original size 4032 × 3024 px
    """Processes labeled images into train/test numpy arrays based on a specified augmentation technique: fliplr (horizontal flipping) or occlusion"""
    data_train = []
    data_test = []
    data_validation = []
    if test==True: # test just a few images to see what is going on
        image_list = os.listdir(LABELED_IMAGES_DIR) #[0:10]
    else:
        image_list = os.listdir(LABELED_IMAGES_DIR)
    random.seed(seed_value) #seed for repeatability
    print("Preprocessing images for scenario " + class_scenario + "; image width " + str(image_width) + "px")
    image_list_train, image_list_test =  train_test_split(image_list, test_size = TEST_SET_SIZE, random_state = seed_value)
    image_list_train, image_list_validation = train_test_split(image_list_train, test_size = VALIDATION_SET_SIZE, random_state=seed_value)
    print("Total images (before augmentation):", len(image_list))
    print("Training images (initial): ", len(image_list_train)) 
    print("Test images (initial): ", len(image_list_test))
    print("Validation images (initial): ", len(image_list_validation))
    for image_index in image_list:
        label = getImageOneHotVector(image_index, class_scenario)
        if label.sum() == 0: # if image unlabeled, move to next one
            continue
        path = os.path.join(LABELED_IMAGES_DIR, image_index)
        image = Image.open(path) # read in image
        print(np.array(image).shape)
        image_width = int(image_width) 
        if rectangular==True:
            image_height = getRectangularImageHeight(image_width) #int(image.size[0] * image_width/image.size[1]) ##because of input orientation, this is flipped.
        else:
            image_height = image_width
        if channels == 1:
            image = image.convert('L') # convert image to monochrome 
            ## RANDOM CROPPING PRIOR IMPLEMENTATION
            # for i in range(expansion_factor):
            #     value = random.random()
            #     crop_value = int(value*(4032 - 3024))
            #     # crop image to 3024 x 3024 (original size: 4032 x 3024 (portrait) or 3024 x 4032 (lscape))
            #     if np.array(image).shape[0] == 3024: # if landscape mode (tree oriented sideways)
            #         cropped_image = image.crop((crop_value, 0, crop_value + 3024, 3024))                     
            #     else: # if portrait mode
            #         cropped_image = image.crop((0, crop_value, 3024, crop_value + 3024))
            #     cropped_image = cropped_image.resize((image_width, image_width), Image.BICUBIC) # resize image
            #     cropped_image_array = np.array(cropped_image)/255. # convert to array and scale to 0-1                
            #     if np.array(image).shape[0] == 3024: # if original image is landscape  
            #         cropped_image_array = cropped_image_array.T # transpose cropped/resized version
            #     if value <= 0.5: # flip horizontally with 50% probability
            #         cropped_image_array = np.fliplr(cropped_image_array)  
            #     data.append([cropped_image_array, label])
        if np.array(image).shape[1] == 4032: #3024 earlier; # if original image is landscape (rotated)
            print("Image is landscape")
            image = image.transpose(Image.ROTATE_270) # transpose cropped/resized version 
        print("Image shape: " + str(image.size))            
        resized_image = image.resize((image_width, image_height), Image.BICUBIC)  
        if test == True:
            pass
            # resized_image.rotate(270).show() # DISPLAY IMAGES if function is run in test mode
        resized_image_array = np.array(resized_image)/255. # convert to array and scale to 0-1
        flipped_resized_img_array = np.fliplr(resized_image_array)
        if augmentation == 'fliplr': 
            if image_index in image_list_train:
                data_train.append([resized_image_array, label])            
                data_train.append([flipped_resized_img_array, label])
                print("Flipped and Resized Image shape: " + str(flipped_resized_img_array.shape))              
            elif image_index in image_list_validation:
                data_validation.append([eraser(resized_image_array), label])
                data_validation.append([eraser(flipped_resized_img_array), label])
            else:
                data_test.append([resized_image_array, label]) 
        ## Occlusion implementation
        if augmentation == 'occlusion_all': # occludes a flipped and resized image with 100% probability
            if image_index in image_list_train:
                data_train.append([eraser(resized_image_array), label])
                data_train.append([eraser(flipped_resized_img_array), label]) 
                print("Occluded, Flipped and Resized Image shape: " + str(flipped_resized_img_array.shape))              
            elif image_index in image_list_validation:
                data_validation.append([eraser(resized_image_array), label])
                data_validation.append([eraser(flipped_resized_img_array), label])
            else:
                data_test.append([resized_image_array, label])
        # occludes a flipped and resized image with 50% probability 
        if augmentation == 'occlusion_half':
            if image_index in image_list_train:
                data_train.append([eraser(resized_image_array, p=0.5), label])
                data_train.append([eraser(flipped_resized_img_array, p=0.5), label]) 
                print("Occluded, Flipped and Resized Image shape: " + str(flipped_resized_img_array.shape))              
            elif image_index in image_list_validation:
                data_validation.append([eraser(resized_image_array), label])
                data_validation.append([eraser(flipped_resized_img_array), label])
            else:
                data_test.append([resized_image_array, label])
        if augmentation == 'occlusion_double':
            if image_index in image_list_train:
                for i in range(2):
                    data_train.append([eraser(resized_image_array), label])
                print("Occluded and Resized Image shape: " + str(resized_image_array.shape))              
            elif image_index in image_list_validation:
                data_validation.append([eraser(resized_image_array), label])
                data_validation.append([eraser(flipped_resized_img_array), label])
            else:
                data_test.append([resized_image_array, label])
    print("Training Images (without validation):", class_scenario, (np.array([x[1] for x in data_train])).sum(axis=0) )
    data_train = data_train + data_validation
    print("Training Images (with validation):", class_scenario, (np.array([x[1] for x in data_train])).sum(axis=0) )
    print("Test Images:", class_scenario, (np.array([x[1] for x in data_test])).sum(axis=0) )
    print("Validation Images:", class_scenario, (np.array([x[1] for x in data_validation])).sum(axis=0) )
    #data_filename = 'size' + str(image_size) + "_exp" + str(expansion_factor) + "_" + class_scenario + ".npy"
    filename_prefix = 'w-' + str(image_width) + 'px-h-' + str(image_height) + "px-scenario-" + class_scenario
    data_filename_train = filename_prefix+ "-train.npy"
    data_filename_test = filename_prefix + "-test.npy"
    data_filename_validation = filename_prefix + "-validation.npy"
    if not os.path.exists(PROCESSED_IMAGES_DIR): # check if 'tidy/preprocessed_images' subdirectory does not exist
        os.makedirs(PROCESSED_IMAGES_DIR) # if not, create it    
    if save_image_binary_files:
        if test == True:
            data_filename_train = 'testing-' + data_filename_train
            data_filename_test = 'testing-' + data_filename_test
        
        augmentation_directory = PROCESSED_IMAGES_DIR + '/' + augmentation + '/' #augmentation subdirectory: fliplr (default) OR occlusion
        if not os.path.exists(augmentation_directory):
            os.makedirs(augmentation_directory)
            
        np.save(os.path.join(augmentation_directory, data_filename_train), data_train) #save as .npy (binary) file
        np.save(os.path.join(augmentation_directory, data_filename_test), data_test) #save as .npy (binary) file    
        np.save(os.path.join(augmentation_directory, data_filename_validation), data_validation) #save as .npy (binary) file  
        
        print("Saved " + data_filename_train + " to " + augmentation_directory)
        print("Saved " + data_filename_test + " to " + augmentation_directory)     
        print("Saved " + data_filename_validation + " to " + augmentation_directory) 
    return  #(image_selection_array, class_list)

## The plotting routine can be here, but perhaps better in a separate fiel.
# def plotProcessedImages(class_scenario, image_array, class_list, images_per_class, resolution):
#     num_rows = images_per_class
#     num_cols = len(class_list)
#     fig, axarr = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
#     #print(image_file_list)
#     for i in range(num_rows):
#         for j in range(num_cols):
#             if num_rows==1:
#                 axarr[j].imshow(image_array[j][i], cmap = 'gist_gray', extent = [0, resolution, 0, resolution])
#             else:
#                 axarr[i, j].imshow(image_array[j][i], cmap = 'gist_gray', extent = [0, resolution, 0, resolution])
#     if num_rows==1:
#         for ax, row in zip(axarr[:], [i for i in class_list]):
#             ax.set_title(row, size=15)
#     else:
#         for ax, row in zip(axarr[0, :], [i for i in class_list]):
#             ax.set_title(row, size=15)
#     image_filename = '../../figures/processed_input_images_' + str(class_scenario) + '_' + str(resolution) + '_px.png'
#     #plt.xticks([0,3024])
#     #plt.yticks([0,4032])
#     #plt.tight_layout()
#     fig.savefig(image_filename, dpi=180)
#     return

def main(testing_boolean=False):
    for scenario in CLASSIFICATION_SCENARIO_LIST:
        for width in IMAGE_WIDTH_LIST:
            processImageData(width, scenario, seed_value=SEED, channels=NUM_CHANNELS, augmentation=AUGMENTATION, rectangular = False, save_image_binary_files=True, test=testing_boolean)
            #processImageData(width, scenario, seed_value=SEED, channels=NUM_CHANNELS, rectangular = True, save_image_binary_files=True, test=False)
            #plotProcessedImages(scenario, array_random_images, classes, images_per_class=NUM_PLOT_IMAGES_PER_CLASS, resolution=image_size)
    return 

if __name__ == "__main__":
    start = time.time()
    main(False)
    elapsed = (time.time() - start)
    print("Image preprocessing completed in (h/m/s/ms):", str(timedelta(seconds=elapsed)))
