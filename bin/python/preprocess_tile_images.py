from PIL import Image # used for loading images
import numpy as np
import csv
import natsort
from natsort import natsorted
import os # used for navigating to image path
import random
from sklearn.model_selection import train_test_split
import shutil
import sys
sys.path.append("../python/")
from helpers import *
import time
from datetime import timedelta

LABELED_IMAGES_DIR = '../../data/tidy/labeled-images/'
PROCESSED_IMAGES_DIR = '../../data/tidy/preprocessed-images/'
INDEX_DIR = '../../results/conflict-detection/index-tidy/'
INDEX_LABELS = INDEX_DIR + 'preprocessed_index.csv'

def getEncoding(image_file_name):
    """Returns binary encoding for each image.
    """
    word_label = image_file_name.split('_')[0]
    if word_label == 'c': 
        return np.array([0, 1])
    elif word_label == 'n': 
        return np.array([1, 0])
    else:
        return np.array([0, 0])
		
def processImageData(seed_value=10, test_set_size=0.2, save_image_binary_files=True):
    """Processes labeled images into train/test numpy arrays based on a specified augmentation technique: fliplr (horizontal flipping) or occlusion"""
    csv_col_index = ['labeled_image', 'preprocessed_index']
    preprocessed_image_dict = {}
    data_train = []
    data_test = []
    data_validation = []
    image_list = os.listdir(LABELED_IMAGES_DIR)
    train_test_ratio = 1-test_set_size
    val_ratio = (test_set_size/train_test_ratio)
    val_set_size = val_ratio*train_test_ratio
    print("Train set size: {tr}%\n test set size: {te}%\n validation set size: {val}%".format(
        tr=int((1-(test_set_size+val_set_size))*100),
        te=int(test_set_size*100),
        val=int(val_set_size*100))
         )
    random.seed(seed_value) #seed for repeatability
    image_list_train, image_list_test =  train_test_split(image_list, test_size = test_set_size, random_state = seed_value)
    image_list_train, image_list_validation = train_test_split(image_list_train, test_size = val_ratio, random_state=seed_value)
    train_cnt = val_cnt = test_cnt = 0
    shutil.rmtree(PROCESSED_IMAGES_DIR, ignore_errors=True) # Deletes the directory containing any existing preprocessed images
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    for image_name in image_list:
        label = getEncoding(image_name)
        path = os.path.join(LABELED_IMAGES_DIR, image_name)
        image = Image.open(path) # read in image
        scaled_image_array = np.array(image)/255.
        if label.sum() == 0: # if image unlabeled, move to next one
            continue
        if image_name in image_list_train:
            data_train.append([scaled_image_array, label])
            preprocessed_image_dict[image_name] = 'train-' + str(train_cnt)
            train_cnt += 1
        elif image_name in image_list_validation:
            data_validation.append([scaled_image_array, label])
            preprocessed_image_dict[image_name] = 'validation-' + str(val_cnt)
            val_cnt += 1
        else:
            data_test.append([scaled_image_array, label])
            preprocessed_image_dict[image_name] = 'test-' + str(test_cnt)
            test_cnt += 1
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    with open(INDEX_LABELS, 'w', newline='') as f: # TODO: separate by tab not comma
        writer = csv.DictWriter(f, fieldnames=csv_col_index)
        writer.writeheader()
        for key in preprocessed_image_dict.keys():
            f.write("%s,%s\n"%(key,preprocessed_image_dict[key]))
    filename_prefix = 'conflict-tiles'
    data_filename_train = filename_prefix+ "-train.npy"
    data_filename_test = filename_prefix + "-test.npy"
    data_filename_validation = filename_prefix + "-validation.npy"
    if not os.path.exists(PROCESSED_IMAGES_DIR): # check if 'tidy/preprocessed_images' subdirectory does not exist
        os.makedirs(PROCESSED_IMAGES_DIR) # if not, create
    if save_image_binary_files:
        np.save(os.path.join(PROCESSED_IMAGES_DIR, data_filename_train), data_train) #save as .npy (binary) file
        np.save(os.path.join(PROCESSED_IMAGES_DIR, data_filename_test), data_test) #save as .npy (binary) file    
        np.save(os.path.join(PROCESSED_IMAGES_DIR, data_filename_validation), data_validation) #save as .npy (binary) file  
        
        print("Saved " + data_filename_train + " to " + PROCESSED_IMAGES_DIR)
        print("Saved " + data_filename_test + " to " + PROCESSED_IMAGES_DIR)     
        print("Saved " + data_filename_validation + " to " + PROCESSED_IMAGES_DIR) 
    return
	
def main():
	start = time.time()
	processImageData()
	end = time.time()
	elapsed = end - start
	print('Time elapsed: {m}min {s}sec'.format(m=int(elapsed//60),s=int(np.round(elapsed%60,0))))
	
if __name__ == '__main__':
	main()