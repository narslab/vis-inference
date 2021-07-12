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
## GLOBAL VARIABLES
IMG_SIZE = 108
IMAGE_WIDTH_LIST = [189, 252, 336]
# Original image size: 3024 x 4032
# Reduction factor of 9: 336 x 448
# Reduction factor of 12: 252 x 336
# Reduction factor of 16: 189 x 252
NUM_CHANNELS = 3
CLASSIFICATION_SCENARIO = "Pr_Im"
CLASSIFICATION_SCENARIO_LIST = ["Pr_Po_Im", "Pr_Im", "PrPo_Im", "Pr_PoIm"]  
LABELED_IMAGES_DIR = '../../data/tidy/labeled-images'
PROCESSED_IMAGES_DIR = '../../data/tidy/preprocessed-images'

SEED = 100  # 10 seed for repeatability ## NOT USED IN CURRENT IMPLEMENTATION
NUM_PLOT_IMAGES_PER_CLASS = 1 #4 ## NOT USED IN CURRENT IMPLEMENTATION
EXPANSION_FACTOR = 5 #5 of augmented images ## NOT USED IN CURRENT IMPLEMENTATION

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


def processImageData(image_width, class_scenario, seed_value, channels=1, save_image_binary_files=True, rectangular = True, test = False): # original size 4032 × 3024 px
    data_train = []
    data_test = []
    if test==True: # test just a few images to see what is going on
        image_list = os.listdir(LABELED_IMAGES_DIR)[0:10]
    else:
        image_list = os.listdir(LABELED_IMAGES_DIR)
    random.seed(seed_value) #seed for repeatability
    print("Preprocessing images for scenario " + class_scenario + "; image width" + str(image_width))
    image_list_train, image_list_test =  train_test_split(image_list, test_size = .2, random_state = seed_value)

    for img in image_list:
        label = getImageOneHotVector(img, class_scenario)
        if label.sum() == 0: # if image unlabeled, move to next one
            continue
        path = os.path.join(LABELED_IMAGES_DIR, img)
        img = Image.open(path) # read in image
        print(np.array(img).shape)
        img_width = int(image_width) 
        if rectangular==True:
            img_height = int(img.size[0] * image_width/img.size[1]) ##because of input orientation, this is flipped.
        else:
            img_height = img_width
        if channels == 1:
            img = img.convert('L') # convert image to monochrome 
            ## RANDOM CROPPING PRIOR IMPLEMENTATION
            # for i in range(expansion_factor):
            #     value = random.random()
            #     crop_value = int(value*(4032 - 3024))
            #     # crop image to 3024 x 3024 (original size: 4032 x 3024 (portrait) or 3024 x 4032 (lscape))
            #     if np.array(img).shape[0] == 3024: # if landscape mode (tree oriented sideways)
            #         cropped_img = img.crop((crop_value, 0, crop_value + 3024, 3024))                     
            #     else: # if portrait mode
            #         cropped_img = img.crop((0, crop_value, 3024, crop_value + 3024))
            #     cropped_img = cropped_img.resize((img_width, img_width), Image.BICUBIC) # resize image
            #     cropped_img_array = np.array(cropped_img)/255. # convert to array and scale to 0-1                
            #     if np.array(img).shape[0] == 3024: # if original image is landscape  
            #         cropped_img_array = cropped_img_array.T # transpose cropped/resized version
            #     if value <= 0.5: # flip horizontally with 50% probability
            #         cropped_img_array = np.fliplr(cropped_img_array)  
            #     data.append([cropped_img_array, label])
        if np.array(img).shape[1] == 3024: # if original image is landscape  
            print("Image is landscape")
            img = img.transpose(Image.rotate_270) # transpose cropped/resized version 
        print("Image shape: " + str(img.size))            
        resized_img = img.resize((img_width, img_height), Image.BICUBIC)  
        if test == True:
            pass
            # resized_img.rotate(270).show() # DISPLAY IMAGES if function is run in test mode
        resized_img_array = np.array(resized_img)/255. # convert to array and scale to 0-1
        print("Resized Image shape: " + str(resized_img_array.shape))  
        if img in image_list_train:
            flipped_resized_img_array = np.fliplr(resized_img_array)
            data_train.append([resized_img_array, label])            
            data_train.append([flipped_resized_img_array, label])
            print("Flipped and Resized Image shape: " + str(flipped_resized_img_array.shape))              
        else:
            data_test.append([resized_img_array, label])            
        
    print("Training Images:", class_scenario, (np.array([x[1] for x in data_train])).sum(axis=0) )
    print("Test Images:", class_scenario, (np.array([x[1] for x in data_test])).sum(axis=0) )
    # random_image_selection_class_0 = random.sample([i[0] for i in data if i[1][0] == 1], k = images_per_class)
    # random_image_selection_class_1 = random.sample([i[0] for i in data if i[1][1] == 1], k = images_per_class)
    # image_selection_array = [random_image_selection_class_0, random_image_selection_class_1]
    # if class_scenario == "Pr_Im":
    #     class_list = ["Improbable", "Probable"]
    # elif class_scenario == "PrPo_Im":
    #     class_list = ["Improbable", "Probable/Possible"]
    # elif class_scenario == "Pr_PoIm":
    #     class_list = ["Possible/Improbable", "Probable"]
    # elif class_scenario == "Pr_Po_Im":
    #     random_image_selection_class_2 = random.sample([i[0] for i in data if i[1][2] == 1], k = images_per_class)
    #     class_list = ["Improbable", "Possible", "Probable"]
    #     image_selection_array = [random_image_selection_class_0, random_image_selection_class_1, random_image_selection_class_2]
    
    #data_filename = 'size' + str(img_size) + "_exp" + str(expansion_factor) + "_" + class_scenario + ".npy"
    filename_prefix = 'w-' + str(img_width) + 'px-h-' + str(img_height) + "px-scenario-" + class_scenario
    data_filename_train = filename_prefix+ "-train.npy"
    data_filename_test = filename_prefix + "-test.npy"
    if not os.path.exists(PROCESSED_IMAGES_DIR): # check if 'tidy/preprocessed_images' subdirectory does not exist
        os.makedirs(PROCESSED_IMAGES_DIR) # if not, create it    
    if save_image_binary_files == True:
        np.save(os.path.join(PROCESSED_IMAGES_DIR, data_filename_train), data_train) #save as .npy (binary) file
        np.save(os.path.join(PROCESSED_IMAGES_DIR, data_filename_test), data_test) #save as .npy (binary) file        
        print("Saved " + data_filename_train + " to data/tidy/" + PROCESSED_IMAGES_DIR)
        print("Saved " + data_filename_test + " to data/tidy/" + PROCESSED_IMAGES_DIR)        
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

def main():
    for scenario in CLASSIFICATION_SCENARIO_LIST:
        for width in IMAGE_WIDTH_LIST:
            processImageData(width, scenario, seed_value=SEED, channels=NUM_CHANNELS, rectangular = True, save_image_binary_files=True, test=False)
            processImageData(width, scenario, seed_value=SEED, channels=NUM_CHANNELS, rectangular = False, save_image_binary_files=True, test=False)
            #plotProcessedImages(scenario, array_random_images, classes, images_per_class=NUM_PLOT_IMAGES_PER_CLASS, resolution=image_size)
    return 

if __name__ == "__main__":
    main()
