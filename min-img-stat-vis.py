import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from make_small_dataset import text_to_small_label


PATH_TO_DATA = '../min-img-data/'
PATH_TO_STATS = PATH_TO_DATA + 'stats/'
PATH_TO_CLASSIFICATION_RESULTS = PATH_TO_DATA + 'classification_results/'

crop_metrics = [0.2, 0.4, 0.6, 0.8]
models = ['vgg16', 'resnet', 'inception']
image_scale = 1.0


# Percent minimal images inside/outside bbx vs. model; crop metric 0.2

# Shift

def percent_min_imgs_inside_bbx_vs_model():

    loose_pcts = {model: np.load(PATH_TO_STATS + '0.2/' + model + '/1.0/loose/shift/percent-min-img-in-bbx.npy') for model in models}
    strict_pcts = {model: np.load(PATH_TO_STATS + '0.2/' + model + '/1.0/strict/shift/percent-min-img-in-bbx.npy') for model in models}

    # filter out 0 rows
    # loose_pcts = {model: loose_pcts[model][]}

    general_min_img_loose_pcts = {model: loose_pcts[model][:, 0] for model in loose_pcts}      # first column of each pct matrix
    general_min_img_strict_pcts = {model: strict_pcts[model][:, 0] for model in strict_pcts}

    all_loose_data = [pd.DataFrame(data={'model': model,
                                         'percent of minimal images inside bbx': general_min_img_loose_pcts[model],
                                         'minimal image axis': 'loose'}) for model in general_min_img_loose_pcts]
    all_loose_df = pd.concat(all_loose_data)
    all_strict_data = [pd.DataFrame(data={'model': model,
                                         'percent of minimal images inside bbx': general_min_img_strict_pcts[model],
                                         'minimal image axis': 'strict'}) for model in general_min_img_strict_pcts]
    all_strict_df = pd.concat(all_strict_data)
    all_df = pd.concat([all_loose_df, all_strict_df])

    ax = sns.boxplot(x='minimal image axis', y='percent of minimal images inside bbx', hue='model', data=all_df)
    plt.show()


# Number of minimal images vs. proportion of image that is bbx; separated by model; crop metric 0.2

def discrete_sizes(proportion_of_im_in_bbx):

    if proportion_of_im_in_bbx < 0.2:
        return 'XS'
    elif proportion_of_im_in_bbx < 0.4:
        return 'S'
    elif proportion_of_im_in_bbx < 0.6:
        return 'M'
    elif proportion_of_im_in_bbx < 0.8:
        return 'L'
    else:
        return 'XL'


def vis_num_min_imgs_vs_prop_in_bbx_models(crop_metric, image_scale, strictness, axis):

    sns.set_style('whitegrid')

    nums = {model: json.load(open(PATH_TO_STATS + str(crop_metric) + '/' + model + '/1.0/' + strictness + '/' + axis + '/id-to-measurements.json')) for model in models}
    nums_discretesize = {}
    size_test_errors = {}
    for model in models:        # maintain order

        size_test_errors[model] = test_error(model)[0]

        for smalldataset_id in nums[model]:
            entry = nums[model][smalldataset_id]
            entry[0] = discrete_sizes(entry[0])
            nums_discretesize[model] = entry

    nums_df = pd.concat([pd.DataFrame(data={'model': model,
                                     'object size': [nums[model][smalldataset_id][0] for smalldataset_id in nums[model]],
                                     'percent of images that are minimal': [nums[model][smalldataset_id][1][0] / float(nums[model][smalldataset_id][1][3]) for smalldataset_id in nums[model]]})
                     for model in models])
    ax = sns.pointplot(x='object size', y='percent of images that are minimal', hue='model', data=nums_df, order=['XS', 'S', 'M', 'L', 'XL'])
    ax.set_title('Number of ' + (strictness + ' ' + axis).title() + ' Minimal Images vs. Relative Size of Bound-in Box '
                                                                    '(Crop Size = ' + str(crop_metric) + ')')

    plotdata = []
    datalines = ax.get_lines()
    for i in range(len(datalines)):
        if not i % (len(SIZES) + 1):
            plotdata.append(datalines[i].get_data())
    for i in range(len(models)):
        model = models[i]
        test_errors = size_test_errors[model]
        xdata, ydata = plotdata[i]
        for i in range(len(SIZES)):
            acc_label = test_errors[SIZES[i]]
            acc_text = '%.5f' % acc_label
            ax.text(xdata[i], ydata[i], acc_text)

    plt.show()


SIZES = ['XS', 'S', 'M', 'L', 'XL']

def test_error(model_name, crop='img'):

    # if crop == 'bbx', check bbx test performance instead of full image test performance

    # By object size
    with open(PATH_TO_STATS + '0.2/resnet/1.0/loose/shift/id-to-measurements.json', 'r') as measurementsfile:
        id_to_measurements = json.load(measurementsfile)        # just take any - all we want is the proportion in bbx
    object_proportions = {smalldataset_id: id_to_measurements[smalldataset_id][0] for smalldataset_id in id_to_measurements}      # mapping of smalldataset_id to object proportion
    object_sizes = {smalldataset_id: discrete_sizes(object_proportions[smalldataset_id]) for smalldataset_id in object_proportions}

    with open(PATH_TO_CLASSIFICATION_RESULTS + model_name + '-small-dataset-classification-correctness.json') as resultsfile:
        classification_results = json.load(resultsfile)

    size_results = {size: [] for size in SIZES}
    for smalldataset_id in object_sizes:
        # gather the classification results for each size (dict mapping size to list of results)
        size_results[object_sizes[smalldataset_id]].append(classification_results[smalldataset_id][0] if crop == 'img' else classification_results[smalldataset_id][1])
    size_test_errors = {size: float(sum(size_results[size])) / len(size_results[size]) for size in size_results}

    # By small dataset class
    small_label_to_text = {text_to_small_label[text]: text for text in text_to_small_label}     # map integer small dataset category to text version
    with open('small-dataset-to-imagenet.txt', 'r') as smalldatafile:
        true_labels = [int(line.split()[-1]) for line in list(smalldatafile.readlines())]        # get the last element (the true class) for each image in the small dataset, ordered by smalldataset_id

    category_results = {category: [] for category in small_label_to_text}
    for smalldataset_id in classification_results:
        if smalldataset_id not in classification_results:       # in case it was intractable
            continue
        ind_result = classification_results[smalldataset_id][0] if crop == 'img' else classification_results[smalldataset_id][1]
        smalldataset_id = int(smalldataset_id)
        category_results[true_labels[smalldataset_id]].append(ind_result)

    category_test_errors = {small_label_to_text[category]: float(sum(category_results[category])) / len(category_results[category]) for category in category_results}

    return size_test_errors, category_test_errors


def pct_minimal_images_vs_correctness(crop_metric, image_scale, strictness, axis):

    # bar (violin) chart with one zone being correct, one zone being incorrect, and the average (distribution of) minimal images for each of those?

    sns.set_style('whitegrid'
                  '')
    nums = {model: json.load(open(PATH_TO_STATS + '0.2/' + model + '/1.0/' + strictness + '/' + axis + '/id-to-measurements.json')) for model in models}

    # make df: x -> correct or incorrect. y -> pct of image that is minimal. hue -> model. df with those three columns.
    all_dfs = []
    for model in models:

        with open(PATH_TO_CLASSIFICATION_RESULTS + model + '-small-dataset-classification-correctness.json', 'r') as resultsfile:
            classification_results = json.load(resultsfile)

        with open(PATH_TO_STATS + str(crop_metric) + '/' + model + '/1.0/' + strictness + '/' + axis + '/id-to-measurements.json', 'r') as pctfile:
            pct_min_imgs = json.load(pctfile)
            tractable_ids = list(pct_min_imgs.keys())              # have to get a specific order. Can't just do range(settings.SMALL_DATASET_SIZE) because of intractable images

        model_df = pd.DataFrame(data={'model': model,
                                      'correctly classified': [classification_results[smalldataset_id][0] for smalldataset_id in tractable_ids],
                                      'percent of image that is minimal': [pct_min_imgs[smalldataset_id][1][0] / float(pct_min_imgs[smalldataset_id][1][3]) for smalldataset_id in tractable_ids]})
        all_dfs.append(model_df)

    data = pd.concat(all_dfs)
    ax = sns.barplot(x='correctly classified', y='percent of image that is minimal', hue='model', data=data)
    ax.set_title('Percent of Image Consisting of ' + (strictness + ' ' + axis).title() + ' Minimal Images vs. Correctness of Image Classification (Crop Size = ' + str(crop_metric) + ')')
    plt.figure()
    ax2 = sns.violinplot(x='correctly classified', y='percent of image that is minimal', hue='model', data=data)
    ax2.set_title('Percent of Image Consisting of ' + (strictness + ' ' + axis).title() + ' Minimal Images vs. Correctness of Image Classification (Crop Size = ' + str(crop_metric) + ')')
    plt.show()

def pct_correct_in_bbx():

    '''
    Shows a plot of percent correctness within bbx as a function of crop size, hued by model
    '''

    # need DF with x = crop size; y = percent correct within bbx; hue = model. I want to do a scatter plot or box plot, not a point or bar plot.
    # DF also needs id field for images just to keep things straight.
    all_dfs = []
    for crop_metric in crop_metrics:
        for model_name in models:
            with open(PATH_TO_STATS + os.path.join(str(crop_metric), model_name, str(image_scale), 'all-img-pct-correct-in-bbx.json'), 'r') as f:
                pct_correct = json.load(f)
                num_imgs = len(pct_correct)
            pct_correct_df = pd.DataFrame.from_dict(pct_correct, orient='index')
            pct_correct_df['model'] = np.array([model_name] * num_imgs)
            pct_correct_df['crop size'] = np.array([crop_metric] * num_imgs)
            all_dfs.append(pct_correct_df)

    all_pct_correct_df = pd.concat(all_dfs)
    all_pct_correct_df = all_pct_correct_df.rename(index=str, columns={0: 'percent correct minimal images within bound-in box'})

    ax = sns.violinplot(x='crop size', y='percent correct minimal images within bound-in box', hue='model', data=all_pct_correct_df)
    # ax.set_ylim((0, 1))
    plt.figure()
    ax2 = sns.boxplot(x='crop size', y='percent correct minimal images within bound-in box', hue='model', data=all_pct_correct_df)
    plt.figure()
    ax3 = sns.pointplot(x='crop size', y='percent correct minimal images within bound-in box', hue='model', data=all_pct_correct_df)

    plt.show()


if __name__ == '__main__':

    # vis_num_min_imgs_vs_prop_in_bbx_models(0.2, 1.0, 'loose', 'scale')
    # vis_num_min_imgs_vs_prop_in_bbx_models(0.2, 1.0, 'strict', 'scale')
    pct_minimal_images_vs_correctness(0.2, 1.0, 'loose', 'scale')
    # pct_correct_in_bbx()
