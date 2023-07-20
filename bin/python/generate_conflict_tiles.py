#!/usr/bin/env python
# coding: utf-8

"""
Splits a random subset of randomly selected images into dxd grids, where d is a factor reduction of the original image resolution.
The default factor is 9, yielding 81 subimages (tiles). Each tile is classified for conflict by a volunteer where conflict is defined
by the overlapping presence of a tree and a power line.
"""

import time
from datetime import timedelta
import pandas as pd
import numpy as np
import os # used for navigating to image path
import sys
from PIL import Image
import shutil
import re # for matching image file name classes
import natsort
from natsort import natsorted
import random
import matplotlib.pyplot as plt
# sys.path.append("../python/")
from helpers import *
import csv

SEED = 100
INDEX_FILE = '../../results/cd/index/labels_index.csv'
LABELED_IMAGES_DIR = '../../data/cd/tidy/labeled-images/'
CONF_DETECT_DIR = '../../data/cd/raw/'
CONF_SAMPLE = CONF_DETECT_DIR + 'sample/'
CONF_SPLIT = CONF_DETECT_DIR + 'split-images/'
COLLECTIVE_DIR = CONF_SPLIT+'all_tiles/'
RANDOM_SAMPLE_LOG = CONF_DETECT_DIR + 'fall21_random_selection.csv'


def selectConflictDetectionPhotos(index_file): 
    """Randomly picks 25 images from each category to be used for conflict detection"""
    shutil.rmtree(CONF_DETECT_DIR, ignore_errors=True)
    log_columns = ['conflict_detection_fall21']
    fall_subset = []
    random_photos = {'improbable':[],
                    'possible':[],
                    'probable':[]}
    df = pd.read_csv(index_file, error_bad_lines=False, warn_bad_lines=False)
    if not os.path.exists(CONF_DETECT_DIR):
        os.makedirs(CONF_DETECT_DIR)
    if not os.path.exists(CONF_SAMPLE):
        os.makedirs(CONF_SAMPLE)
    if not os.path.exists(CONF_SPLIT):
        os.makedirs(CONF_SPLIT)
    if not os.path.exists(COLLECTIVE_DIR):
        os.makedirs(COLLECTIVE_DIR)
    np.random.seed(SEED) # ensure repeatability
    for row in df.itertuples():
        if "Likelihood of Failure Images" in row.original_file_path:
            fall_subset.append(row.labeled_image)
    for k, v in random_photos.items():
        while len(v) < 25:
            rand = np.random.choice(fall_subset)
            if rand not in v:
                if re.match(rand.split(".",-1)[0].split("-",-1)[0], k, re.IGNORECASE):
                    v.append(rand)
    with open(RANDOM_SAMPLE_LOG, 'w', newline='') as f: # TODO: separate by tab not comma
        writer = csv.DictWriter(f, fieldnames=log_columns)
        writer.writeheader()
        for k,v in random_photos.items():
            for j in natsort.natsorted(v): # iterate through the alphanumeric keys in a natural order
                f.write("%s\n"%(j))        
    df1 = pd.read_csv(RANDOM_SAMPLE_LOG, error_bad_lines=False, warn_bad_lines=False)
    for row in df1.itertuples():
        src = os.path.join(LABELED_IMAGES_DIR+row.conflict_detection_fall21)
        dst = os.path.join(CONF_SAMPLE+row.conflict_detection_fall21)
        shutil.move(src, dst)
        
    return random_photos
    
    
def preprocessConflictImages():
    csv_col_classifier = ['Filename', 'Classification']
    classifier = {}
    selectConflictDetectionPhotos(INDEX_FILE)
    for filename in os.listdir(CONF_SAMPLE):
        t = generateTiles(filename, CONF_SAMPLE, COLLECTIVE_DIR)
        for k in t[0]:
            classifier[k] = ''
        plotTilesGrid(t, CONF_SPLIT)
    with open(CONF_SPLIT + 'tiles_assignment.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_col_classifier)
        writer.writeheader()
        for key in classifier.keys():
            f.write("%s,%s\n"%(key.split(".", 1)[0],classifier[key]))

if __name__ == "__main__":
    start = time.time()
    preprocessConflictImages()
    elapsed = (time.time() - start)
    print("Generated all tiles in (h/m/s/ms):", str(timedelta(seconds=elapsed)))   