#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import numpy as np
import os # used for navigating to image path
import imageio # used for writing images
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from helpers import *
from itertools import product
import csv

LABELED_IMAGES_DIR = '../../data/tidy/conflict-sample'
CONFLICT_SPLIT_DIR = '../../data/tidy/conflict-split-images/'

if not os.path.exists(CONFLICT_SPLIT_DIR): 
	os.makedirs(CONFLICT_SPLIT_DIR)
	
def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")
	
def generateTiles(filename, dir_in, dir_out, d=9):
    tiles = {}
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    rotate_img= img.transpose(Image.ROTATE_270)
    w, h = rotate_img.size
    grid = product(range(0, h, int(h/d)), range(0, w, int(w/d)))
    for i, j in grid:
        box = (j, i, j+int(w/d), i+int(h/d))
        tile_name = f'{name}_{i}_{j}{ext}'
        out = os.path.join(dir_out, tile_name)
        t = rotate_img.crop(box)
        t.save(out)
        tiles[tile_name] = t
    return tiles, filename
	
def plotTilesGrid(tiles, directory):
    plt.figure(figsize=(36,36)) # specifying the overall grid size
            
    for i in range(len(tiles[0])):
        plt.subplot(9,9,i+1)    # the number of images in the grid is 9*9 (81)
        plt.imshow(tiles[0][get_nth_key(tiles[0], i)])
        plt.title(get_nth_key(tiles[0], i))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(directory+"/split_grid_"+str(tiles[1]))
	
def preprocessConflictImages():
    csv_col_classifier = ['Filename', 'Classification']
    classifier = {}
    collective_dir = CONFLICT_SPLIT_DIR+'all_tiles/'
    print(collective_dir)
    if not os.path.exists(collective_dir):
            os.makedirs(collective_dir)
    for filename in os.listdir(LABELED_IMAGES_DIR):
        f_name = os.path.splitext(filename)
#         f_name_dir = CONFLICT_SPLIT_DIR+f_name[0]
        t = generateTiles(filename, LABELED_IMAGES_DIR, collective_dir)
        for k in t[0]:
            classifier[k] = ''
        plotTilesGrid(t, CONFLICT_SPLIT_DIR)
    with open(CONFLICT_SPLIT_DIR + 'tiles_assignment.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_col_classifier)
        writer.writeheader()
        for key in classifier.keys():
            f.write("%s,%s\n"%(key,classifier[key]))

if __name__ == "__main__":
    preprocessConflictImages()   