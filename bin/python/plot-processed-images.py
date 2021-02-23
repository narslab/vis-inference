#TODO (in order to not have to regenerate images each time)
def plotProcessedImages(class_scenario, image_array, class_list, images_per_class, resolution):
    num_cols = images_per_class
    num_rows = len(class_list)
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
    fig.savefig(image_filename, dpi=180)
    return

def main():
    for scenario in CLASSIFICATION_SCENARIO_LIST:
        for image_size in IMG_SIZE_LIST:
            imd = np.load('../../data/tidy/preprocessed_images/size128_exp5_Pr_Po_Im.npy',allow_pickle = True)
            array_random_images = 
            plotProcessedImages(scenario, array_random_images, classes, images_per_class=NUM_PLOT_IMAGES_PER_CLASS, resolution=image_size)
    return

if __name__ == "__main__":
    main()