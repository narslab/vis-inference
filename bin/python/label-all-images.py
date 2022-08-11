#!/usr/bin/env python
# coding: utf-8

"""
The purpose of this script is to consolidate all images taken by arborists and label them according to their classification.
An index containing the newly labeled image name and its original file path is also genereated under '../../results/index-raw/'.
"""

import pandas as pd
import openpyxl
import time
from PIL import Image # used for loading images 
import numpy as np
import os # used for navigating to image path
import imageio # used for writing images
import natsort
from natsort import natsorted
import re # for matching image file name classes
import matplotlib.pyplot as plt
#import PythonMagick # used for .HEIC to .JPG conversion
import random
import ntpath
import shutil
import csv
from timeit import default_timer as timer
from datetime import timedelta
import platform
from pillow_heif import register_heif_opener

register_heif_opener()

SEED = 100

RAW_IMAGE_DIR = '../../data/raw/Pictures for AI'
RAW_IMAGE_DIR_SUMMER = '../../data/raw/Summer 2021 AI Photos'
RAW_IMAGE_DIR_FALL = '../../data/raw/Likelihood of Failure Images/'
RAW_IMAGE_DIR_APRIL22 = '../../data/raw/More AI Tree Pics 04_22/'
TIDY_IMAGE_DIR = '../../data/tidy/labeled-images/'
INDEX_DIR = '../../results/index-raw/'

INDEX_LABELS = INDEX_DIR + 'labels_index.csv'
TEMP = RAW_IMAGE_DIR_FALL+'likelihood.csv'
TRIMMED = RAW_IMAGE_DIR_FALL+'likelihood_of_failure_trimmed.csv'
RATINGS_APRIL22 = RAW_IMAGE_DIR_APRIL22+'Likelihood of Failure Ratings.xlsx'

def getListOfFiles(dirName):
    """Returns single list of the filepath of each of the image files"""
    # source: https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
    allFiles = list()
    if 'Pictures for AI' or 'Likelihood of Failure Images' in dirName: # 
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
            # print(level_two)
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
        # print('fn: ' + fn)
        if (platform.system() == 'Windows'):
            if ('_' in fn):
                index = fn.split('_')[0].split('\\')[-1]
                description = fn.split('_')[1]
                arborist = fn.split('_')[0].split('\\')[1].split('/')[-1]
            else:
                index = fn.split('-')[0].split('\\')[-1].rstrip(' ')
                description = fn.split('-')[1]
                arborist = fn.split('_')[0].split('\\')[1].split('/')[-1]        
        else: #Unix/Linux systems
            if ('_' in fn):
                index = fn.split('_')[0].split('/')[-1]
                # print("Index" + index)
                description = fn.split('_')[1]
                arborist = fn.split('_')[0].split('/')[-3] #.split('/')[-2]
                # print(arborist)
            else:
                index = fn.split('-')[0].split('/')[-1].rstrip(' ')
                description = fn.split('-')[1]
                arborist = fn.split('-')[0].split('/')[-3]                    
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
        # print(arborist)
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
                # print(good_index_list)
                if len(good_index_list) == 1:
                    selected_filename = dfsub2.loc[good_index_list, 'Filename'].tolist()[0]
                else:
                    testdf = dfsub2.loc[good_index_list]
                    testdf['DescriptionLength'] = testdf.Description.str.len()
                    for i in good_index_list:
                        selected_filename = testdf.loc[testdf.DescriptionLength.idxmin(), 'Filename']
            # print(selected_filename)                
            unique_image_list.append(selected_filename)
    return unique_image_list

def trimTrailingChars(data):
    """Converts original excel file containing classification categories for Fall 2021 images to csv"""
    read_file = pd.read_excel(data,engine='openpyxl')
    read_file.to_csv(TEMP, index = None, header=True)
    read_file = pd.read_csv(TEMP)
    read_file['Likelihood of Failure Rating'] = read_file['Likelihood of Failure Rating'].str.replace('\xa0','')
    df = read_file.rename({"Image":"image", "File Name":"file_name", "Likelihood of Failure Rating": "likelihood_of_failure_rating"}, axis='columns')
    df.to_csv(TRIMMED, index = None, header=True)
    # print(df.head())
    # print(df['likelihood_of_failure_rating'].unique())
    if os.path.isfile(TEMP):
        os.remove(TEMP)
        
# def encrypt(file_name):
#     """Cryptographically encrypts each failure likelihood category using Python's built-in hash function"""
#     if any(re.findall(r'improbable', file_name, re.IGNORECASE)):
#         h = hash('improbable')
#     elif any(re.findall(r'probable', file_name, re.IGNORECASE)):
#         h = hash('probable')
#     elif any(re.findall(r'possible', file_name, re.IGNORECASE)):
#         h = hash('possible')
#     else:
#         h = hash('unknown')
#     return h

# def updateNameCount(word, d):
#     """Creates a uniform label for all saved images and updates the global count"""
#     h  = encrypt(word)
#     save_name = ''
#     for key in d.keys():
#         if h == hash(key):
#             save_name = TIDY_IMAGE_DIR + key + '-' + str(d[key]) + '.jpg'
#             d[key] += 1
#     return save_name

def getFileName(fname):
    if any(re.findall(r'improbable', fname, re.IGNORECASE)):
        sname = 'improbable'
    elif any(re.findall(r'probable', fname, re.IGNORECASE)):
        sname = 'probable'
    elif any(re.findall(r'possible', fname, re.IGNORECASE)):
        sname = 'possible'
    else:
        sname = 'unknown'
    return sname

def saveImageFiles(image_file_list):
    """Serially labels all images by class:  and saves them to the designated tidy image directory."""
    # counts = {'improbable':0,
    #             'possible':0,
    #             'probable':0,
    #             'unknown':0}
    im_cnt = pr_cnt = po_cnt = unkn_cnt = 0
    csv_col_index = ['labeled_image', 'original_file_path']
    index = {}
    # jpg_images = []
    heic_image_list = []
    shutil.rmtree(TIDY_IMAGE_DIR, ignore_errors=True) # Deletes the directory containing any existing labeled images
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    # Convert April 2022 ratings from excel to csv
    read_file = pd.read_excel("../../data/raw/More AI Tree Pics 04_22/Likelihood of Failure Ratings.xlsx")
    read_file.to_csv ("april22_ratings.csv", 
                      index = None,
                      header=True)     
    df = pd.read_csv(TRIMMED)
    df1 = pd.read_csv("april22_ratings.csv")
    # Rename April 2022 dataframe cols to match dataframe format from Fall 2021
    df1_rename = df1.rename(columns={'Image number (file name is "IMG_18##.HEIC"':'image',
                                'Likelihood of Failure Rating':'likelihood_of_failure_rating'})
    df1_rename.insert(1, 'file_name', ['IMG_'+x+'.HEIC' for x in df1_rename.image])
    frames = [df, df1_rename]
    combined_df = pd.concat(frames)
    combined_df['likelihood_of_failure_rating'] = combined_df['likelihood_of_failure_rating'].fillna('').apply(str)
    combined_df['file_name'] = combined_df['file_name'].fillna('').apply(str)
    if not os.path.exists(TIDY_IMAGE_DIR):
        os.makedirs(TIDY_IMAGE_DIR)
    for filename in image_file_list:
        if '.JPG'.casefold() in filename.casefold():
            print(filename)
            label = getFileName(filename)
            if label == 'improbable':
                save_name = TIDY_IMAGE_DIR + label + '-' + str(im_cnt) + '.jpg'
                im_cnt += 1
            elif label == 'probable':
                save_name = TIDY_IMAGE_DIR + label + '-' + str(pr_cnt) + '.jpg'
                pr_cnt += 1
            elif label == 'possible':
                save_name = TIDY_IMAGE_DIR + label + '-' + str(po_cnt) + '.jpg'
                po_cnt += 1
            else:
                save_name = TIDY_IMAGE_DIR + label + '-' + str(unkn_cnt) + '.jpg'
                unkn_cnt += 1
            # save_name = updateNameCount(filename, counts)
            imageio.imwrite(save_name, np.array(Image.open(filename)))
            index[save_name] = filename
        elif '.HEIC'.casefold() in filename.casefold():
            heic_image_list.append(filename)
    print("\n",heic_image_list)
    for heic_image in heic_image_list:
        print(heic_image)
        for row in combined_df.itertuples():
            image_name = row.file_name
            rating = row.likelihood_of_failure_rating
            label = getFileName(rating)
            if 'Likelihood of Failure Images' in heic_image:
                if image_name == heic_image.split("Failure Images/",1)[1]:
                    if label == 'improbable':
                        save_name = TIDY_IMAGE_DIR + label + '-' + str(im_cnt) + '.jpg'
                        im_cnt += 1
                    elif label == 'probable':
                        save_name = TIDY_IMAGE_DIR + label + '-' + str(pr_cnt) + '.jpg'
                        pr_cnt += 1
                    elif label == 'possible':
                        save_name = TIDY_IMAGE_DIR + label + '-' + str(po_cnt) + '.jpg'
                        po_cnt += 1
                    else:
                        save_name = TIDY_IMAGE_DIR + label + '-' + str(unkn_cnt) + '.jpg'
                        unkn_cnt += 1
                    # heif_file = pillow_heif.read_heif(heic_image)
                    # image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
                    # image.save(save_name, format='JPEG')
                    heif_file = Image.open(heic_image)
                    heif_file.save(save_name, "JPEG")
                    index[save_name] = heic_image
            else:
                if image_name == heic_image.split("04_22/",1)[1]:
                    if label == 'improbable':
                        save_name = TIDY_IMAGE_DIR + label + '-' + str(im_cnt) + '.jpg'
                        im_cnt += 1
                    elif label == 'probable':
                        save_name = TIDY_IMAGE_DIR + label + '-' + str(pr_cnt) + '.jpg'
                        pr_cnt += 1
                    elif label == 'possible':
                        save_name = TIDY_IMAGE_DIR + label + '-' + str(po_cnt) + '.jpg'
                        po_cnt += 1
                    else:
                        save_name = TIDY_IMAGE_DIR + label + '-' + str(unkn_cnt) + '.jpg'
                        unkn_cnt += 1
                    # heif_file = pillow_heif.read_heif(heic_image)
                    # image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
                    # image.save(save_name, format='JPEG')
                    heif_file = Image.open(heic_image)
                    heif_file.save(save_name, "JPEG")
                    index[save_name] = heic_image
                # if 'Likelihood of Failure Images' in filename:
                #     if image_name == filename.split("Failure Images/",1)[1]:
                #         save_name = updateNameCount(rating, counts)
                #         # PythonMagick.Image(filename).write(save_name) # convert .HEIC to .JPG
                #         heif_file = pillow_heif.read_heif(filename)
                #         image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
                #         image.save(save_name, format='JPEG')
                #         # continue                    
                #     # index[save_name] = filename
                # else:
                #     if image_name == filename.split("04_22/",1)[1]:
                #         save_name = updateNameCount(rating, counts)
                #         # PythonMagick.Image(filename).write(save_name) # convert .HEIC to .JPG
                #         heif_file = pillow_heif.read_heif(filename)
                #         image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
                #         image.save(save_name, format='JPEG')                        
                #         # continue                    
                #     index[save_name] = filename
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    with open(INDEX_LABELS, 'w', newline='') as f: # TODO: separate by tab not comma
        writer = csv.DictWriter(f, fieldnames=csv_col_index)
        writer.writeheader()
        for key in natsort.natsorted(index.keys()): # iterate through the alphanumeric keys in a natural order
            key_name = key.replace(TIDY_IMAGE_DIR,'')
            val_name = index[key]
            f.write("%s,%s\n"%(key_name,val_name))
    os.remove("april22_ratings.csv")
    print('Number of improbable images:', im_cnt)    
    print('Number of possible images:', po_cnt)
    print('Number of probable images:', pr_cnt)
    print('Number of unknown images:', unkn_cnt)
    print('Total number of classified images:', im_cnt + pr_cnt + po_cnt)

def main():    
    trimTrailingChars(RAW_IMAGE_DIR_FALL+'Likelihood of Failure Images.xlsx')
    original_photos = getListOfFiles(RAW_IMAGE_DIR)
    summer_photos = getListOfFiles(RAW_IMAGE_DIR_SUMMER)
    fall_photos = getListOfFiles(RAW_IMAGE_DIR_FALL)
    april22_photos = getListOfFiles(RAW_IMAGE_DIR_APRIL22)
    index_list, description_list, arborist_list = splitIndexDescrArb(summer_photos)
    unique_image_list_summer = getUniqueImages(summer_photos, arborist_list, index_list, description_list)
    start = time.time()
    saveImageFiles(original_photos+unique_image_list_summer+fall_photos+april22_photos)
    elapsed = (time.time() - start)
    print("Saved images in (h/m/s/ms):", str(timedelta(seconds=elapsed)))

if __name__ == "__main__":
    main()    