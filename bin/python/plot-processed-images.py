#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


IMAGE_ARRAY = np.load('../../data/tidy/preprocessed-images/w-336px-h-336px-scenario-Pr_Im-train.npy',allow_pickle = True)
IMAGE_ARRAY_TEST = np.load('../../data/tidy/preprocessed-images/w-336px-h-336px-scenario-PrPo_Im-test.npy',allow_pickle = True)

def getFigure2sub(images):
    pr = []
    po = []
    im = []
    for i in range(len(images)):
        if (images[i][1] == np.array([0, 0, 1])).all():
            pr.append(i)
        elif (images[i][1] == np.array([0, 1, 0])).all(): 
            po.append(i)
        elif (images[i][1] == np.array([1, 0, 0])).all():
            im.append(i)
    return pr, po, im

def plotSelectedProcessedImages(image_array, save=True):
    ## Generates the subfigures in Figure 2 in the paper.
    # pr, po, im = getFigure2sub(image_array)
    # example_image_list = [pr[3], pr[25], 
    #                       pr[91], po[17], 
    #                       po[39], po[71], 
    #                       im[23], im[45], im[87]]

    example_image_list = [764, 765, 252, 253,
                          480, 481, 482, 483,
                          898, 899, 548, 549]
    #print(example_image_list)
    example_image_filenames = ['probable-example-1.png', 'probable-example-1-flipped.png',
                               'probable-example-2.png', 'probable-example-2-flipped.png',
                               'possible-example-1.png', 'possible-example-1-flipped.png',
                               'possible-example-2.png', 'possible-example-2-flipped.png',
                               'improbable-example-1.png', 'improbable-example-1-flipped.png',
                               'improbable-example-2.png', 'improbable-example-2-flipped.png',
                              ]
    for i, n in enumerate(example_image_list):
        plt.imshow(image_array[n][0], cmap = 'gist_gray')
        #plt.imshow(np.rot90(image_array[n][0]), cmap = 'gist_gray')
        plt.tight_layout()
        plt.yticks(fontsize=16) #[0,40,80,120],
        plt.xticks(fontsize=16) #[0,40,80,120],
        #plt.show()
        if save==True:
          plt.savefig('../../figures/processed-336px-' + example_image_filenames[i], dpi=180,bbox_inches='tight')

def exploreProcessedImages(image_array):
    rows= 15
    cols = 15
    
    k = 0 #1120 #1120 #2240
    for b in np.arange(0, len(image_array), rows*cols):
      f, ax = plt.subplots(rows, cols, figsize=(50,30))
      while k < b + rows*cols:
        is_looping = True
        for i in np.arange(rows):
            for j in np.arange(cols):
                ax[i][j].set_title(str(k))
                ax[i][j].imshow(image_array[k][0])#, cmap = 'gist_gray')
                ax[i][j].set_xticklabels([])
                ax[i][j].set_yticklabels([])
                k  += 1
                if k > len(image_array) - 1:
                  is_looping = False
                  print("Max k reached; breaking out of loop")
                  break
            if not is_looping:
              print("Max k reached; breaking out of loop")
              break
        if not is_looping:
          print("Max k reached; breaking out of loop")
          break              
      plt.show()
      continue
    return

def main():
    #plotSelectedProcessedImages(IMAGE_ARRAY, save=True)
    #exploreProcessedImages(IMAGE_ARRAY)
    return

if __name__ == "__main__":
    main()