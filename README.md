# inference-tree-risk
Inference Methods for Tree Failure Identification and Risk Quantification

## Summary
The code and models in this repository implement convolutional neural network (CNN) models to predict tree likelihood of failure categories from a given input image. The categories are:
- Improbable: failure unlikely either during normal or extreme weather conditions
- Possible: failure expected under extreme weather conditions; but unlikely during normal weather conditions
- Probable: failure expected under normal weather conditions within a given time frame
Original input images are 3024 x 4032 pixels. We assess the performance of an optimized CNN using 64-pixel, 128-pixel and 224-pixel inputs (after data augmentation expands samples from 525 images to 2525 images).
We also evaluate performance under four classification scenarios (investigating how various category groupings impact classifier performance):
Pr_Im: {Probable, Improbable}; 2 classes
PrPo_Im: {Probable + Possible, Improbable}; 2 classes
Pr_PoIm: {Probable, Possible + Improbable}; 2 classes
Pr_Po_Im: {Probable, Possible, Improbable}; 3 classes

## Step 1: Label input data
Inputs are images (currently 3024 x 4032 pixels). These are currently saved locally and not accessible on the remote. Email the collaborators for data access. To perform labeling, run `label-image-files.py`. The user must specify the path to the raw images (`RAW_IMAGE_DIR`). The current framework assumes the raw images are housed in `data/raw/Pictures for AI`.


## Step 2: Preprocess images
In this step (`preprocess-images.py`), we perform image resizing and data augmentation (random cropping, horizontal flipping - probability of 50%). The user can specify the expansion factor for the original set of images. For instance, there are 525 images in the original dataset. if an expansion factor of 5 is specified for the preprocessing function, then the final augmented set will contain 2525 images. Finally, image training sets are generated for 4 classification scenarios and for user-specified resolutions, e.g. 64 x 64 px, 128 x 128 px, etc. One-hot-vector encoding is also performed. Each set of images and labels are saved as an array of tuples in a binary `.npy` file. The `preprocess-images.py` script also includes a `plotProcessedImages()` function that generates a specified number of randomly chosen input images for each scenario.

The user can also plot selected processed images using the functions in the `plot-processed-images.py` script. To explore all the processed images in a matrix plot, use `exploreProcessedImages()`. Figure 2 in the manuscript was generated using the `plotSelectedProcessedImages()` function.

## Step 3: CNN hyperparameter optimization
We use the `Hyperband` function from `keras.tuner` to optimize the following parameters in our convolutional neural network: kernel size of first convolutional layer, units in the 2 dense layers, their respective dropout rates and activation functions. The routine is carried out in `cnn-hyperparameter-optimization.py`. The search is performed for 12 cases (3 resolutions and 4 classification scenarios).
- The results are tabulated via `tabulate-optimal-hyperparameters.py` (which generates the CSV files used to create Table 4 in the manuscript).

## Step 4: Sensitivity tests
In `resolution-scenario-sensitivity.py`, the function `testResolutionScenarioPerformance()` conducts CNN model fitting for each combination of resolution and scenario as specified by the user in `RESOLUTION_LIST` and `SCENARIO_LIST` respectively. This is done via k-fold cross-validation. Validation metrics of macro-average precision, recall and $F_1$ are also implemented. Model histories are saved for each trial.

Tabulation and visualization summaries of the results are implemented in `senstivity-analysis.ipynb`.
- Figure 4 in the manuscript is generated using `plotMeanAccuracyLoss()`.
- Figure 5 is generated using `plotSummaryValidationMetrics()`

Furthermore, we aggregate performance statistics in `senstivity-analysis.ipynb` and performance Welch's tests to determine  if there are significant differences in outcomes.
- The function `getScenarioResolutionMeanPerformance()` generates Table 6.
- The function `resolutionPerformanceComparisonStats()` generates Table 7.
- The function `scenarioPerformanceComparisonStats()` generates Table 8.

## Step 5: Detailed CNN performance analysis
In `cnn-performance.py`, we define the function `trainModelWithDetailedMetrics()` which implements CNN model-fitting, along with sklearn classification metrics, including a confusion matrix, for a given resolution/scenario instance. The loss and performance results are visualized in the `plot-cnn-performance.ipynb` notebook, using the function `plotCNNPerformanceMetrics()`.
- Figure 6 in the manuscript is generated via `plotCNNPerformanceMetrics()`.
- Figure 7 is based on the confusion matrices saved from running `getScenarioModelPerformance()`, which in turns runs `trainModelWithDetailedMetrics()`.
The trained model is saved to `results/models/`.

## Step 6: CNN Visualization and Inference (in progress)
We implement GradCAM and saliency maps to understand how the CNN classifies an image. This is done using `plotGradCAM()` and `plotSaliency()` in `cnn-visualization.ipynb`. A prior trained model is loaded (e.g. `m = models.load_model('../../results/models/opt-cnn-Pr_Im-128-px/model')`) and used as an input to either of the functions mentioned.


Please note: Function and classes that are used in two or more scripts are housed in `helpers.py`
