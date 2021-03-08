import numpy as np
import matplotlib.pyplot as plt


IMAGE_ARRAY = np.load('../../data/tidy/preprocessed_images/size128_exp5_Pr_Po_Im.npy',allow_pickle = True)

def plotSelectedProcessedImages(image_array):
    ## Generates the subfigures in Figure 2 in the paper.
    example_image_list = [1790, 1792, 
                          2235, 2237, 
                          950, 951, 
                          1180, 1181, 
                          2110, 2111,
                          1301, 1302
                         ]
    example_image_filenames = ['probable-example-1.png', 'probable-example-1-flipped.png',
                              'probable-example-2.png', 'probable-example-2-flipped.png',
                              'possible-example-1.png', 'possible-example-1-flipped.png',
                              'possible-example-2.png', 'possible-example-2-flipped.png',   
                              'improbable-example-1.png', 'improbable-example-1-flipped.png',
                              'improbable-example-2.png', 'improbable-example-2-flipped.png',                            
                              ]
    for i,n in enumerate(example_image_list):
        plt.imshow(image_array[n][0], cmap = 'gist_gray')
        plt.tight_layout()
        plt.yticks([0,40,80,120],fontsize=16)
        plt.xticks([0,40,80,120],fontsize=16)
        plt.savefig('../../figures/processed-' + example_image_filenames[i], dpi=180,bbox_inches='tight')

def exploreProcessedImages(image_array):
    rows= 15
    cols = 15
    f, ax = plt.subplots(rows,cols, figsize=(50,30))
    k = 1120 #1120 #2240
    for i in np.arange(rows):
        for j in np.arange(cols):
            ax[i][j].set_title(str(k))
            ax[i][j].imshow(image_array[k][0], cmap = 'gist_gray')
            ax[i][j].set_xticklabels([])
            ax[i][j].set_yticklabels([])
            k += 5
    return

def main():
    plotSelectedProcessedImages(IMAGE_ARRAY)
    return

if __name__ == "__main__":
    main()