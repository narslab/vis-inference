#import os
import sys
sys.path.append("../python/")
from helpers import *
#import pickle
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

from scipy.special import softmax

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as tkr
from itertools import combinations
from scipy.stats import ttest_ind, f_oneway # for independent t-test

import math

from collections import Counter
import statistics as stat
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# Globals
NUM_CHANNELS = 3
RESOLUTION_LIST = [336]
SCENARIO_LIST = ["PrPo_Im"] #, "Pr_Im", "Pr_PoIm", "Pr_Po_Im"]
NUM_EPOCHS = 20
AUGMENTATION = 'fliplr'
SAVED_MODEL_DIR = '../../results/models/'
MODEL_PERFORMANCE_METRICS_DIR = '../../results/model-performance/'
FULL_MODEL_PATH = '../../results/models/opt-cnn-base-PrPo_Im-w-336-px-h-336-px/model'
GROUPS = ["tp","fp","tn","fn"]

trial_seed = 1
class_labels = getClassLabels("PrPo_Im")

GLOBAL_MODEL = models.load_model(FULL_MODEL_PATH)

IMAGE_SETS_SQUARE_TRAIN = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST, augmentation=AUGMENTATION, type='train', rectangular = False)
IMAGE_SETS_SQUARE_TEST = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST, augmentation=AUGMENTATION, type='test', rectangular = False)
IMAGE_SETS_SQUARE_VALIDATION = createResolutionScenarioImageDict(RESOLUTION_LIST, SCENARIO_LIST, augmentation=AUGMENTATION, type='validation', rectangular = False)

training_images, training_labels = getImageAndLabelArrays(IMAGE_SETS_SQUARE_TRAIN[336]["PrPo_Im"])
test_images, test_labels = getImageAndLabelArrays(IMAGE_SETS_SQUARE_TEST[336]["PrPo_Im"])
validation_images, validation_labels = getImageAndLabelArrays(IMAGE_SETS_SQUARE_VALIDATION[336]["PrPo_Im"])

training_images = np.concatenate((training_images,validation_images))
training_labels = np.concatenate((training_labels,validation_labels))


def getPredictionGroupProbability(yPred, img_idx):
    """Returns the prediction group (TP/FP/TN/FN) and 
    the prediction probability of a test image index as a tuple."""
    prediction_probability = np.round(max(yPred[img_idx]),2)
    observed_class  = np.argmax(test_labels[img_idx])
    predicted_class = np.argmax(yPred[img_idx])
    if observed_class == 1:
        if predicted_class == observed_class: # true positive
            group = 'tp'
        else: # false negative
            group = 'fn' 
    else:
        if predicted_class == observed_class: # true negative
            group = 'tn'
        else: # false positive
            group = 'fp'
    return (group, prediction_probability)
    
def generateGradcam(model, yPred, img_idx, plus=False):
    """Returns a heatmap with a Gradcam or GradcamPlusPlus object.
    https://keisen.github.io/tf-keras-vis-docs/examples/attentions.html#GradCAM
    """
    img_array = np.squeeze(test_images[img_idx]) # Remove an axis from the image array
    score = CategoricalScore(np.argmax(yPred[img_idx])) # Returns the target score: 0 for negative, 1 for positive
    replace2linear = ReplaceToLinear() # Replaces the softmax activation f-n applied to the last layer
    # Creates Gradcam object
    if plus: # Improved Visual Explanations 
        gcam_obj = GradcamPlusPlus(model,
                   model_modifier=replace2linear,
                   clone=True)
    else: 
        gcam_obj = Gradcam(model,
                   model_modifier=replace2linear,
                   clone=True)
    # Generates heatmap with GradCAM
    cam = gcam_obj(score,
              img_array,
              penultimate_layer=-1)
    return cam
    
def renderGradcam(model,yPred,img_idx,plus=False,save_fig=False):
    """Visualizes the attention over input using the penultimate layer output"""
    gradcam_dir = '../../figures/plottingGradCam/7_15_2022/'
    prediction_probability = getPredictionGroupProbability(yPred, img_idx)
    gcam = generateGradcam(model,yPred,img_idx,plus)
    plt.figure(figsize=(5, 5))
    plt.tight_layout()
    heatmap = np.uint8(cm.jet(gcam[0])[..., :3] * 255)
    plt.imshow(np.squeeze(np.squeeze(test_images[img_idx]))) # remove axes of length one from the test image index
    plt.imshow(heatmap, alpha=0.5) # overlay
    # plt.colorbar(label="Gradcam score", orientation="vertical")
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if save_fig:
        if not os.path.exists(gradcam_dir):
            os.makedirs(gradcam_dir)
        save_name = gradcam_dir+"index_"+str(img_idx)+"_"+prediction_probability[0]+"_"+str(prediction_probability[1])
        if plus:
            save_name += '_plus'
        plt.savefig(save_name+".png")
    plt.show()

def analyzePredictions(labels, yPred):
    """Returns the indices and prediction probabilities of the test images for both correct and incorrect predictions. 
    The total number for each class should equal the corresponding value in the confusion matrix."""
    prediction_group = {
        "tp": [],
        "fp": [],
        "tn": [],
        "fn": []
    }
    for i in range(len(labels)):
        i_group, i_prediction_probability = getPredictionGroupProbability(yPred, i)
        prediction_group[i_group].append((i,i_prediction_probability))
    for v in prediction_group.values():
        v.sort(key=lambda x: x[1])
    return prediction_group

def expandPredictionProbabilityName(key):
    if key == 'tp':
        key = 'True positive'
    elif key == 'fp':
        key = 'False positive'
    elif key == 'tn':
        key = 'True negative'
    else:
        key = 'False negative'
    return key

def get_summary_statistics(prediction_groups):
    for key,val in prediction_groups.items():
        data = []
        print(expandPredictionProbabilityName(key))
        for v in val:
            data.append(v[1])
        mean = np.round(stat.mean(data), 2)
        median = np.round(stat.median(data), 2)
        std = np.round(stat.stdev(data), 2)
        min_value = np.round(min(data), 2)
        max_value = np.round(max(data), 2)
        # quartile_1 = np.round(np.quantile(data,0.25), 2)
        # quartile_3 = np.round(np.quantile(data,0.75), 2)
        # # Interquartile range
        # iqr = np.round(quartile_3 - quartile_1, 2)
        print('Min: %s' % min_value)
        print('Max: %s' % max_value)
        print('Mean: %s' % mean)        
        print('Median: %s' % median)
        # print('25th percentile: %s' % quartile_1)
        # print('75th percentile: %s' % quartile_3)
        # print('Interquartile range (IQR): %s' % iqr)
        print('Standard deviation: %s' % std,"\n")

def returnIdxByProb(group,prediction_group,prob,n):
    indices = []
    for v in prediction_group[group]:
        if len(indices) < n:
            if prob == 0.5:
                if(v[1] >= prob) and (v[1] <= prob+0.01):
                    indices.append(v)
            elif prob == 1:
                if(v[1] >= prob-0.01) and (v[1] <= prob):
                    indices.append(v)
            else:
                if(v[1] >= prob-0.01) and (v[1] <= prob + 0.01):
                    indices.append(v)
    return indices
    
def returnAllGradcamIndices(p_group,n):
    prob = [0.5, 0.6667, 0.833, 1]
    grad_cam_idx = {
        "tp": [],
        "fp": [],
        "tn": [],
        "fn": []
    }
    for p in prob:
        for k in p_group.keys():
            grad_cam_idx[k].append((returnIdxByProb(k,p_group,p,n),p))
    return grad_cam_idx   

def generateGradcamForGroup(model, gradcam_idx, group, save_fig=False):
    print(group, "Grad-CAMs\n")
    for val in gradcam_idx[group]:
        if len(val[0]) == 0:
            print("No prediction probabilities in the", val[1], "range (+\- 0.01).\n ")
        else:
            print(len(val[0]),"Gradcams for prediction probabilities in the", val[1], "range (+\- 0.01):", [x[0] for x in val[0]],"\n")
            for i in val[0]:
                print(i[0])
                renderGradcam(model,y_pred,i[0])
            print("\n")

def plotPredictionProbabilities(prediction_group,save_fig=False):
    """Generate summary statistics for True Positive, True Negative, False Positive
    and False Negative Rates"""
    rates = {
        "tp": [],
        "fp": [],
        "tn": [],
        "fn": []
    }
    for key,val in prediction_group.items():
        for v in val:
            rates[key].append(v[1])
    fig, ax = plt.subplots(figsize=(12, 7))
    # Remove top and right border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Remove y-axis tick marks
    ax.yaxis.set_ticks_position('none')
    # Add major gridlines in the y-axis
    ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
    # Set plot title
    # ax.set_title('Distribution of prediction probabilitiees', fontsize=24)
    data = [rates["tp"], rates["fp"], rates["tn"], rates["fn"]]
    lbls = [['TP'],['FP'],['TN'],['FN']]
    bp1 = ax.boxplot(data[0], positions=[0], notch=False, widths=0.35, 
                    patch_artist=True, boxprops=dict(facecolor="#E31A1C"), labels=lbls[0]) #
    # plt.xlabel(lbls[0],fontsize=18)
    bp2 = ax.boxplot(data[1], positions=[1], notch=False, widths=0.35, # median to black
                    patch_artist=True, boxprops=dict(facecolor="#FB9A99"), labels=lbls[1])
    bp3 = ax.boxplot(data[2], positions=[2], notch=False, widths=0.35, 
                    patch_artist=True, boxprops=dict(facecolor="#33A02C"), labels=lbls[2])
    bp4 = ax.boxplot(data[3], positions=[3], notch=False, widths=0.35, 
                    patch_artist=True, boxprops=dict(facecolor="#B2DF8A"), labels=lbls[3])
    # ax.set_ylabel('Prediction Probabilities', fontsize=18)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0]], 
              ['True Positive (TP)','False Positive (FP)','True Negative (TN)','False Negative (FN)'], 
              bbox_to_anchor=(1,0.5), loc='center left', prop={'size': 18})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(which='major', axis='y', linestyle='-', linewidth=0.25, alpha=0.5, zorder=-1.0)
    fig.tight_layout()
    if save_fig:
        plt.savefig('../../figures/boxplot_tp_tn_fp_fn.jpg', dpi=180)
    plt.show()

def generateViolinPlot(prediction_group,save_fig=False):
    """Generate a violin plot for True Positive, True Negative, False Positive
    and False Negative Rates"""
    rates = {
        "tp": [],
        "fp": [],
        "tn": [],
        "fn": []
    }
    for key,val in prediction_group.items():
        for v in val:
            rates[key].append(v[1])
    df = pd.DataFrame(columns = ['Group', 'Prediction Probability'])
    for key,value in rates.items():
        rates_df = pd.DataFrame({'Group': expandPredictionProbabilityName(key),
                                 'Prediction Probability':value})
        df = df.append(rates_df, ignore_index=True)
    fig, ax = plt.subplots()
    sns.violinplot(ax = ax,
               data = df,
               palette=['#E31A1C','#FB9A99','#33A02C','#B2DF8A'],
               x = 'Group',
               y = 'Prediction Probability')
    # plt.ylim(0.5,1)
    fig.tight_layout()
    if save_fig:
        plt.savefig('../../figures/violin_plot_tp_tn_fp_fn.png', dpi=180)
    plt.show()

def plotProbabilities(outcome, prediction_group, n_bins=10, font_size=22, save_fig=False):
    rates = {
        "tp": [],
        "fp": [],
        "tn": [],
        "fn": []
    }
    for key,val in prediction_group.items():
        for v in val:
            rates[key].append(v[1])
    plt.figure(figsize=(15,7)).gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    t = ''
    # plt.grid(which='major', axis='y', zorder=-1.0)  
    if outcome == 'tp':
        t = outcome #'true_positive'
        plt.hist(rates[t], color="#E31A1C", bins=n_bins)
    elif outcome == 'tn':
        t = outcome#'true_negative'
        plt.hist(rates[t], color="#33A02C", bins=n_bins)
    elif outcome == 'fp':
        t = outcome#'false_positive'
        plt.hist(rates[t], color="#FB9A99", bins=n_bins)
    elif outcome == 'fn':
        t = outcome#'false_negative'
        plt.hist(rates[t], color="#B2DF8A", bins=n_bins)
    plt.xlabel('Probabilities', fontsize=font_size)
    plt.ylabel('Counts', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # plt.title(t, fontsize=24)
    plt.grid(which='major', axis='y', zorder=-1.0)
    plt.tight_layout()
    # plt.title('Prediction probabilities for '+t+' Probable/Possible images')
    if save_fig:
        plt.savefig('../../figures/prpo_'+t+'_predictions_dist.jpg', dpi=180)
    
def main():
    m = GLOBAL_MODEL
    y_pred = m.predict(np.squeeze(test_images)) # Obtain model predictions
    # Overview of negative and positive class predictions
    df = pd.DataFrame(y_pred, columns = ['class 0', 'class 1'])
    print(df)
    sample_img = 29
    renderGradcam(m,y_pred,sample_img,False,True) # Regular GradCAM
    renderGradcam(m,y_pred,sample_img,True,True) # GradCAM Plus Plus
    pred_group = analyzePredictions(test_labels, y_pred)
    get_summary_statistics(pred_group)
    gcam_idx = returnAllGradcamIndices(pred_group,1)
#    for g in GROUPS: # Returns gradcams for each group (TP/FP/TN/FN) with the following probabilities: 0.5,0.6667,0.833.100 with 1% margin
#        generateGradcamForGroup(model, gcam_idx, g)
#        plotProbabilities(g, pred_group, font_size=28, save_fig=True)
    plotPredictionProbabilities(pred_group)
    generateViolinPlot(pred_group)
    
if __name__ == "__main__":
    main()