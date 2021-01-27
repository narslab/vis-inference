# tree-risk-ai
Artificial Intelligence for Tree Failure Identification and Risk Quantification

## Step 1: Label input data
Inputs are images (currently 3024 x 4032 pixels). These are currently saved locally and not accessible on the remote. Email the collaborators for data access. To perform labeling, run `process-image-files`. The current framework assumes theraw images are housed in `data/raw/Pictures for AI`.


## Step 2: Preprocess images
In this step, we perform image resizing and data augmentation (random cropping, horizontal flipping - probability of 50%). The user can specify the expansion factor for the original set of images. For instances, if there are 500 images in the original set and an expansion factor of 5 is specified for the preprocessing function, then the final augmented set will contain 2500 images. Finally, image training sets are generated for 4 classification scenarios and for user-specified resolutions, e.g. 64 x 64 px, 128 x 128 px, etc. One-hot-vector encoding is also performed. Each set of images and labels are saved as an array of tuples in a binary `.npy` file.

## Step 3: CNN hyperparameter optimization
We use the `HyperModel` module from `keras.tuner` to optimize the following parameters in our convolutional neural network:


