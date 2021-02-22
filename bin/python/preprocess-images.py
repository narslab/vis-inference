"""
The purpose of this script is to pre-process the tree images using the following approaches:
    - one-hot vector encoding
    - resizing
    - grayscaling 
    - normalization (scaling pixels from 0 to 1)
    - random cropping (based on specified expansion factor to multiply images)
    - horizontal flipping
"""

from PIL import Image # used for loading images
import numpy as np
import os # used for navigating to image path
import imageio # used for writing images
import random
import matplotlib.pyplot as plt

## GLOBAL VARIABLES
IMG_SIZE = 108
IMG_SIZE_LIST = [64]#, 128, 224]#, 384]
NUM_CHANNELS = 1
CLASSIFICATION_SCENARIO = "Pr_Im"
CLASSIFICATION_SCENARIO_LIST = ["Pr_Po_Im"]#, "Pr_Im", "PrPo_Im", "Pr_PoIm"]  
EXPANSION_FACTOR = 5 #5 of augmented images
LABELED_IMAGES_DIR = '../../data/tidy/labeled_images'
PROCESSED_IMAGES_DIR = '../../data/tidy/preprocessed_images'
SEED = 10 # seed for repeatability
NUM_PLOT_IMAGES_PER_CLASS = 4

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
            return np.array([1, 0, 0])
        elif word_label == 'possible' : 
            return np.array([0, 1, 0])    
        elif word_label == 'improbable':
            return np.array([0, 0, 1])
        else :
            return np.array([0, 0, 0]) # if label is not present for current image
    elif classification_scenario == "Pr_Im":
        if word_label == 'probable' : 
            return np.array([1, 0])
        elif word_label == 'improbable' : 
            return np.array([0, 1])
        else :
            return np.array([0, 0]) # if label is not present for current image
    elif classification_scenario == "PrPo_Im":
        if word_label in ['probable', 'possible'] : 
            return np.array([1, 0])
        elif word_label == 'improbable' : 
            return np.array([0, 1])
        else :
            return np.array([0, 0]) # if label is not present for current image        
    elif classification_scenario == "Pr_PoIm":
        if word_label == 'probable' : 
            return np.array([1, 0])
        elif word_label in ['possible', 'improbable'] : 
            return np.array([0, 1])
        else :
            return np.array([0, 0]) # if label is not present for current image        

def processImageData(img_size, expansion_factor, class_scenario, seed_value, channels=1, images_per_class=4): # original size 4032 × 3024 px
    data = []
    image_list = os.listdir(LABELED_IMAGES_DIR)
    random.seed(seed_value) #seed for repeatability
    print("Preprocessing images for scenario " + class_scenario + "and resolution " + str(img_size))
    for img in image_list:
        label = getImageOneHotVector(img, class_scenario)
        if label.sum() == 0: # if image unlabeled, move to next one
            continue
        path = os.path.join(LABELED_IMAGES_DIR, img)
        img = Image.open(path) # read in image
        if channels == 1:
            img = img.convert('L') # convert image to monochrome 
            for i in range(expansion_factor):
                value = random.random()
                crop_value = int(value*(4032 - 3024))
                # crop image to 3024 x 3024 (original size: 4032 x 3024 (portrait) or 3024 x 4032 (lscape))
                if np.array(img).shape[0] == 3024: # if landscape mode (tree oriented sideways)
                    cropped_img = img.crop((crop_value, 0, crop_value + 3024, 3024))                     
                else: # if portrait mode
                    cropped_img = img.crop((0, crop_value, 3024, crop_value + 3024))
                cropped_img = cropped_img.resize((img_size, img_size), Image.BICUBIC) # resize image
                cropped_img_array = np.array(cropped_img)/255. # convert to array and scale to 0-1                
                if np.array(img).shape[0] == 3024: # if original image is landscape  
                    cropped_img_array = cropped_img_array.T # transpose cropped/resized version
                if value <= 0.5: # flip horizontally with 50% probability
                    cropped_img_array = np.fliplr(cropped_img_array)  
                data.append([cropped_img_array, label])
        elif channels == 3:
            pass
            #TODO: cropping here needs to be updated
            #img = img.crop((left=400, top=0, r=3424, b=3024)) 
            #img = img.resize((img_size, img_size), Image.BICUBIC)
            #TODO: random cropping, flipping and resizing
    print("Images:", class_scenario, (np.array([x[1] for x in data])).sum(axis=0) )
    random_image_selection_class_0 = random.choices([i[0] for i in data if i[1][0] == 1], k = images_per_class)
    random_image_selection_class_1 = random.choices([i[0] for i in data if i[1][1] == 1], k = images_per_class)
    image_selection_array = [random_image_selection_class_0, random_image_selection_class_1]
    if class_scenario == "Pr_Im":
        class_list = ["Probable", "Improbable"]
    elif class_scenario == "PrPo_Im":
        class_list = ["Probable/Possible", "Improbable"]
    elif class_scenario == "Pr_PoIm":
        class_list = ["Probable", "Possible/Improbable"]
    elif class_scenario == "Pr_Po_Im":
        random_image_selection_class_2 = random.choices([i[0] for i in data if i[1][2] == 1], k = images_per_class)
        class_list = ["Probable", "Possible", "Improbable"]
        image_selection_array = [random_image_selection_class_0, random_image_selection_class_1, random_image_selection_class_2]
    
    data_filename = 'size' + str(img_size) + "_exp" + str(expansion_factor) + "_" + class_scenario + ".npy"
    if not os.path.exists(PROCESSED_IMAGES_DIR): # check if 'tidy/preprocessed_images' subdirectory does not exist
        os.makedirs(PROCESSED_IMAGES_DIR) # if not, create it    
    #np.save(os.path.join(PROCESSED_IMAGES_DIR, data_filename), data) #save as .npy (binary) file
    print("Saved " + data_filename + " to data/tidy/" + PROCESSED_IMAGES_DIR)
    return (image_selection_array, class_list)

def plotProcessedImages(class_scenario, image_array, class_list, images_per_class, resolution):
    num_rows = len(class_list)
    num_cols = images_per_class
    fig, axarr = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
    #print(image_file_list)
    for i in range(num_rows):
        for j in range(num_cols):
            axarr[i, j].imshow(image_array[i][j], cmap = 'gist_gray', extent = [0, resolution, 0, resolution])
    for ax, row in zip(axarr[:, 0], [i for i in class_list]):
        ax.set_ylabel(row, size=15)
    image_filename = '../../figures/processed_input_images_' + str(class_scenario) + '_' + str(resolution) + '_px.png'
    #plt.xticks([0,3024])
    #plt.yticks([0,4032])
    fig.savefig(image_filename, dpi=120)
    return

def main():
    for scenario in CLASSIFICATION_SCENARIO_LIST:
        for image_size in IMG_SIZE_LIST:
            array_random_images, classes = processImageData(image_size, EXPANSION_FACTOR, scenario, 
                seed_value=SEED, channels=NUM_CHANNELS, images_per_class=NUM_PLOT_IMAGES_PER_CLASS)
            plotProcessedImages(scenario, array_random_images, classes, images_per_class=NUM_PLOT_IMAGES_PER_CLASS, resolution=image_size)
    return

if __name__ == "__main__":
    main()
