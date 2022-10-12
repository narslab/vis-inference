import pandas as pd
import time
from PIL import Image # used for loading images 
import numpy as np
import os # used for navigating to image path
import imageio # used for writing images
import natsort
from natsort import natsorted
import re # for matching image file name classes
import matplotlib.pyplot as plt
import ntpath
import shutil
import csv
from timeit import default_timer as timer
from datetime import timedelta
import platform

RAW_IMAGE_DIR = '../../data/raw/conflict-detection/'
TIDY_IMAGE_DIR = '../../data/tidy/labeled-images/'
RAW_TILES_DIR = RAW_IMAGE_DIR+'split-images/all_tiles/'

INDEX_DIR = '../../results/conflict-detection/index-raw/'

ORIGINAL_CLASSIFICATION = RAW_IMAGE_DIR + 'tiles_assignment.xlsx'
INDEX_LABELS = INDEX_DIR + 'labels_index.csv'
TILES_ASSIGNMENT_CSV = RAW_IMAGE_DIR+'tile_assignment.csv'

def getListOfFiles(dirName):
    """Returns single list of the filepath of each of the image files"""
    allFiles = list()
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
    return allFiles
    
def label_conflict_tiles(image_file_list):
    c_cnt = n_cnt = 0
    csv_col_index = ['labeled_tile', 'original_tile_path']
    index = {}
    read_file = pd.read_excel(ORIGINAL_CLASSIFICATION,engine='openpyxl')
    read_file.to_csv(TILES_ASSIGNMENT_CSV, index = None, header=True)
    df = pd.read_csv(TILES_ASSIGNMENT_CSV)
    df = df.iloc[1: , :]
    shutil.rmtree(TIDY_IMAGE_DIR, ignore_errors=True) # Deletes the directory containing any existing labeled images
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    if not os.path.exists(TIDY_IMAGE_DIR):
        os.makedirs(TIDY_IMAGE_DIR)
    for file in image_file_list:
        tile_name = file.split("all_tiles/",1)[1].split(".jpg",1)[0]
        for row in df.itertuples():
            image_name = row.Filename
            classification = row.Classification
            if image_name == tile_name:
                if classification == 'N':
                    save_name = TIDY_IMAGE_DIR+classification.lower()+'_'+str(n_cnt) + '.jpg'
                    n_cnt += 1
                else:
                    save_name = TIDY_IMAGE_DIR+classification.lower()+'_'+str(c_cnt) + '.jpg'
                    c_cnt += 1
                imageio.imwrite(save_name, np.array(Image.open(file)))
                index[save_name] = file
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    with open(INDEX_LABELS, 'w', newline='') as f: # TODO: separate by tab not comma
        writer = csv.DictWriter(f, fieldnames=csv_col_index)
        writer.writeheader()
        for key in natsort.natsorted(index.keys()): # iterate through the alphanumeric keys in a natural order
            key_name = key.replace(TIDY_IMAGE_DIR,'')
            val_name = index[key]
            f.write("%s,%s\n"%(key_name,val_name))
    print('Number of conflict tiles:', c_cnt)    
    print('Number of non-conflict tiles:', n_cnt)

def main():
    start = time.time()
    all_tiles = getListOfFiles(RAW_TILES_DIR)
    label_conflict_tiles(all_tiles)
    end = time.time()
    elapsed = end - start
    print('Time elapsed: {m}min {s}sec'.format(m=int(elapsed//60),s=int(np.round(elapsed%60,0))))

if __name__ == '__main__':
    main()