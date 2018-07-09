import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
import seaborn as sns
import settings

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


def arrange_test_errors():

    '''
    Returns:
        { <vgg16, inception, resnet>:
            { <img, bbx>:
                { <bbx-size, category>:
                    { accuracies }
                }
            }
        }

    '''

    errors = {}
    crop_types = ['img', 'bbx']
    for model_name in models:
        img_size_test_errors, img_category_test_errors = test_error(model_name)
        bbx_size_test_errors, bbx_category_test_errors = test_error(model_name, crop='bbx')
        errors[model_name] = {
                                'img': {
                                            'bbx-size': img_size_test_errors,
                                            'category': img_category_test_errors
                                       },
                                'bbx': {
                                            'bbx-size': bbx_size_test_errors,
                                            'category': bbx_category_test_errors
                                }
                             }

    with open(PATH_TO_CLASSIFICATION_RESULTS + 'arranged-errors.json', 'w') as errfile:
        json.dump(errors, errfile)


def pct_minimal_images_vs_correctness(crop_metric, image_scale, strictness, axis):

    # bar (violin) chart with one zone being correct, one zone being incorrect, and the average (distribution of) minimal images for each of those?

    sns.set_style('whitegrid')
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


def pct_minimal_images_vs_crop_size():

    # chart showing four line plots: pct minimal images as a function of crop size for each strictness/axis combo

    sns.set(font_scale=1.3, style='whitegrid')
    image_scale = 1.0

    strictnesses = ['loose', 'strict']
    axes = ['shift', 'scale']
    for axis in axes:
        sns.set(font_scale=1.3, style='whitegrid')
        i = 0
        fig = plt.figure()
        for strictness in strictnesses:

            all_dfs = []
            for crop_metric in crop_metrics:
                for model in models:

                    with open(PATH_TO_STATS + os.path.join(str(crop_metric), model, str(image_scale), strictness, axis, 'id-to-measurements.json')) as mfile:
                        id_to_measurements = json.load(mfile)

                    pct_min_imgs = []
                    for im_id in sorted(id_to_measurements.keys()):         # get pct of min map that is minimal for all image ids
                        pct_min_imgs.append((id_to_measurements[im_id][1][1] / float(id_to_measurements[im_id][1][3])) * 100)

                    crop_size_column = [crop_metric for __ in pct_min_imgs]
                    model_column = [model for __ in pct_min_imgs]

                    all_dfs.append(pd.DataFrame(data={'crop size': crop_size_column,
                                                      'percent minimal images': pct_min_imgs,
                                                      'model': model_column}))

            data = pd.concat(all_dfs)

            fig.add_subplot(121 + i)
            ax = sns.pointplot(x='crop size', y='percent minimal images', hue='model', data=data)
            ax.set_title('Percent of Images that are ' + (strictness + ' ' + axis).title() + ' Minimal vs. Crop Size')
            i += 1

    plt.show()


def accuracy_vs_crop_size():

    '''
    Basically, a graph of top5 map white percentage. x axis: crop size. y axis: top5 white percentage. hue: model.
    '''

    all_dfs = []
    for crop_metric in crop_metrics:
        for model in models:
            result = np.load(PATH_TO_STATS + os.path.join(str(crop_metric), model, str(image_scale), 'crop-classification-correctness.npy'))
            mean_result = np.nanmean(result)
            all_dfs.append(pd.DataFrame(data={'DNN': [model],
                                              '% crops correctly classified': [mean_result],
                                              'proportionality constant': [crop_metric]}))
    data = pd.concat(all_dfs)
    print(data)
    sns.set(style='whitegrid', font_scale=2.0)
    ax = sns.pointplot(y='% crops correctly classified', x='proportionality constant', hue='DNN', data=data)
    # plt.show()


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


    # ax = sns.violinplot(x='crop size', y='percent correct minimal images within bound-in box', hue='model', data=all_pct_correct_df)
    # ax.set_ylim((0, 1))
    # plt.figure()
    # ax2 = sns.boxplot(x='crop size', y='percent correct minimal images within bound-in box', hue='model', data=all_pct_correct_df)
    plt.figure()
    ax3 = sns.pointplot(x='crop size', y='percent correct minimal images within bound-in box', hue='model', data=all_pct_correct_df)
    ax3.set_title('Percent Correctly Classified Minimal Images Within Object Bound-in Box')

    plt.show()


def pct_min_img_vs_bbx_size():

    '''
    line plot of % min imgs vs. bbx size (object size) - strict scale, all three models
    TODO should I hold a constant crop metric? If so, I choose 0.2
    '''

    crop_metric = 0.2
    image_scale = 1.0
    strictness = 'strict'
    axis = 'scale'

    all_dfs = []
    for model in models:
        with open(PATH_TO_STATS + os.path.join(str(crop_metric), model, str(image_scale), strictness, axis, 'id-to-measurements.json')) as mfile:
            id_to_measurements = json.load(mfile)

        im_ids = sorted(id_to_measurements.keys())  # df col 0: smalldataset_id (not actually put into df)
        bbx_sizes = []                              # df col 1: bbx size
        pct_min_imgs = []                           # df col 2: pct min imgs
        model_column = [model for __ in im_ids]     # df col 3: model
        for im_id in im_ids:
            bbx_sizes.append(discrete_sizes(id_to_measurements[im_id][0]))
            pct_min_imgs.append((id_to_measurements[im_id][1][0] / float(id_to_measurements[im_id][1][3])) * 100)   # (num min imgs / num pixels) * 100

        all_dfs.append(pd.DataFrame(data={'Object size': bbx_sizes,
                                          'Percent minimal images': pct_min_imgs,
                                          'Model': model_column}))
    data = pd.concat(all_dfs)
    # data = data.sort_values(['Object size'], ascending=SIZES)

    sns.set(font_scale=1.5, style='whitegrid')
    ax = sns.pointplot(x='Object size', y='Percent minimal images', hue='Model', data=data, order=SIZES)
    ax.set_title('% Strict shift minimal images vs. object size')
    ax.set_ylabel('% Strict shift minimal images')
    plt.show()


def accuracy_vs_bbx_size():

    '''
    line plot of accuracy vs. bbx size - strict scale 0.2, all three models
    '''

    crop_metric = 0.2
    image_scale = 1.0
    strictness = 'strict'
    axis = 'scale'

    all_dfs = []

    for model in models:
        with open(PATH_TO_CLASSIFICATION_RESULTS + model + '-small-dataset-classification-correctness.json', 'r') as cfile:
            classification_results = json.load(cfile)
        with open(PATH_TO_STATS + os.path.join(str(crop_metric), model, str(image_scale), strictness, axis, 'id-to-measurements.json'), 'r') as mfile:
            id_to_measurements = json.load(mfile)

        im_ids = id_to_measurements.keys()
        size_classification_results = {size: [] for size in SIZES}
        for im_id in im_ids:        # map size string to a list of correctness values for its images
            bbx_size = discrete_sizes(id_to_measurements[im_id][0])     # get the size of the bbx
            size_classification_results[bbx_size].append(classification_results[str(im_id)][0])         # append the full image correctness to that size's list
        avg_size_classification_results = {size: np.mean(size_classification_results[size]) * 100 for size in size_classification_results}        # get averages as percentages

        bbx_sizes = SIZES
        accuracies = [avg_size_classification_results[size] for size in SIZES]      # get accuracies in order of SIZES
        model_column = [model for __ in SIZES]

        all_dfs.append(pd.DataFrame(data={'Object size': bbx_sizes,
                                          '% Accuracy of DNN on full image': accuracies,
                                          'DNN': model_column}))

    data = pd.concat(all_dfs)
    sns.set(font_scale=1.5, style='whitegrid')
    ax = sns.pointplot(x='Object size', y='% Accuracy of DNN on full image', hue='DNN', data=data)
    ax.set_title('Accuracy of DNN on full image vs. object size ')
    ax.set_ylabel('% Accuracy of DNN on full image')
    plt.show()


SMALL_LABEL_TO_TEXT = {text_to_small_label[text]: text for text in text_to_small_label}
CATEGORIES = [SMALL_LABEL_TO_TEXT[i] for i in range(len(SMALL_LABEL_TO_TEXT))]              # text categories in order of ordinal value


def pct_min_img_vs_category():

    '''
    line plot of % min imgs vs. category - strict scale 0.2, all three models
    '''

    crop_metric = 0.2
    image_scale = 1.0
    strictness = 'strict'
    axis = 'scale'

    with open('small-dataset-to-imagenet.txt', 'r') as labelfile:
        vals = list(labelfile.readlines())
    category_inds = [int(val.split()[-1]) for val in vals]

    all_dfs = []
    for model in models:
        with open(PATH_TO_STATS + os.path.join(str(crop_metric), model, str(image_scale), strictness, axis, 'id-to-measurements.json')) as mfile:
            id_to_measurements = json.load(mfile)

        im_ids = sorted(id_to_measurements.keys())  # df col 0: smalldataset_id (not actually put into df)
        categories = []                             # df col 1: categories
        pct_min_imgs = []                           # df col 2: pct min imgs
        model_column = [model for __ in im_ids]     # df col 3: model
        for im_id in im_ids:
            categories.append(SMALL_LABEL_TO_TEXT[category_inds[int(im_id)]])       # get the im_idth image's text category and append it to the list of categories
            pct_min_imgs.append((id_to_measurements[im_id][1][0] / float(id_to_measurements[im_id][1][3])) * 100)   # (num min imgs / num pixels) * 100

        all_dfs.append(pd.DataFrame(data={'Object category': categories,
                                          'Percent minimal images': pct_min_imgs,
                                          'DNN': model_column}))
    data = pd.concat(all_dfs)

    sns.set(font_scale=1.5, style='whitegrid')
    ax = sns.pointplot(x='Object category', y='Percent minimal images', hue='DNN', data=data, order=CATEGORIES)
    ax.set_title('% Strict shift minimal images vs. object category')
    ax.set_ylabel('% Strict shift minimal images')
    plt.show()


def accuracy_vs_category():

    '''
    line plot of accuracy vs. category - strict scale 0.2, all three models
    '''

    crop_metric = 0.2
    image_scale = 1.0
    strictness = 'strict'
    axis = 'scale'

    all_dfs = []

    for model in models:
        with open(PATH_TO_CLASSIFICATION_RESULTS + model + '-small-dataset-classification-correctness.json', 'r') as cfile:
            classification_results = json.load(cfile)
        with open('small-dataset-to-imagenet.txt', 'r') as labelfile:
            vals = list(labelfile.readlines())
            category_inds = [int(val.split()[-1]) for val in vals]

        im_ids = sorted([int(key) for key in classification_results.keys()])
        category_classification_results = {category: [] for category in category_inds}
        for im_id in im_ids:        # map size string to a list of correctness values for its images
            category = category_inds[im_id]    # get the category of the image
            category_classification_results[category].append(classification_results[str(im_id)][0])         # append the full image correctness to that size's list
        avg_category_classification_results = {category: np.mean(category_classification_results[category]) * 100 for category in category_classification_results}        # get averages as percentages


        accuracies = [avg_category_classification_results[category] for category in range(len(CATEGORIES))]      # get accuracies in order of SIZES
        model_column = [model for __ in CATEGORIES]


        all_dfs.append(pd.DataFrame(data={'Object category': CATEGORIES,
                                          '% Accuracy of DNN on full image': accuracies,
                                          'DNN': model_column}))

    data = pd.concat(all_dfs)
    print(data)
    sns.set(font_scale=1.5, style='whitegrid')
    ax = sns.pointplot(x='Object category', y='% Accuracy of DNN on full image', hue='DNN', data=data, order=CATEGORIES)
    ax.set_title('Accuracy of DNN on full image vs. object category ')
    ax.set_ylabel('% Accuracy of DNN on full image')

    plt.show()


def pct_min_imgs_vs_non_min_imgs_in_bbx():

    '''
    Plots percent of bbx that is minimal for loose shift minimal images vs. crop size; hued by model
    '''

    image_scale = 1.0
    # strictness = 'loose'
    # axis = 'shift'
    strictnesses = ['loose', 'strict']
    axes = ['shift', 'scale']

    for strictness in strictnesses:
        for axis in axes:

            all_dfs = []

            for crop_metric in crop_metrics:
                for model in models:
                    # df: model, proportionality constant, pct bbx is min img
                    # get matrix: row i is smalldataset_id=i's
                    pct_of_bbx_min_imgs = np.load(PATH_TO_STATS + os.path.join(str(crop_metric), model, str(image_scale), strictness, axis, '') + 'percent-of-bbx-minimal.npy')
                    pct_of_bbx_min_imgs = np.nanmean(pct_of_bbx_min_imgs)

                    print(pct_of_bbx_min_imgs)
                    all_dfs.append(pd.DataFrame(data={'Proportionality constant': [crop_metric],
                                                      '% of bound-in box images that are minimal': [pct_of_bbx_min_imgs],
                                                      'DNN': [model]}))

            plt.figure()
            data = pd.concat(all_dfs)
            sns.set(font_scale=2, style='whitegrid')
            ax = sns.pointplot(x='Proportionality constant', y='% of bound-in box images that are minimal', hue='DNN', data=data)
            plt.show()


def pct_min_imgs_in_bbx_vs_outside():

    '''
    Plots percent of min imgs that are in bbx (as opposed to outside bbx) vs. crop size; hued by model
    '''

    image_scale = 1.0
    strictnesses = ['loose', 'strict']
    axes = ['shift', 'scale']

    for strictness in strictnesses:
        for axis in axes:
            all_dfs = []

            for crop_metric in crop_metrics:
                for model in models:
                    pct_min_imgs_in_bbx = np.load(PATH_TO_STATS + settings.make_stats_foldername(crop_metric, model, image_scale, strictness, axis) + 'percent-min-img-in-bbx.npy')
                    pct_min_imgs_in_bbx = pct_min_imgs_in_bbx[:, 0]

                    all_dfs.append(pd.DataFrame(data={'Proportionality constant': [crop_metric for __ in pct_min_imgs_in_bbx],
                                                      '% of all minimal images within bound-in box': pct_min_imgs_in_bbx,
                                                      'DNN': [model for __ in pct_min_imgs_in_bbx]}))

            plt.figure()
            data = pd.concat(all_dfs)
            sns.set(font_scale=2, style='whitegrid')
            ax = sns.pointplot(x='Proportionality constant', y='% of all minimal images within bound-in box', hue='DNN', data=data)
            plt.show()




if __name__ == '__main__':

    # vis_num_min_imgs_vs_prop_in_bbx_models(0.2, 1.0, 'loose', 'scale')
    # vis_num_min_imgs_vs_prop_in_bbx_models(0.2, 1.0, 'strict', 'scale')
    # pct_minimal_images_vs_correctness(0.2, 1.0, 'loose', 'scale')
    # pct_correct_in_bbx()
    # pct_minimal_images_vs_crop_size()
    # pct_min_img_vs_bbx_size()
    # accuracy_vs_bbx_size()
    # pct_min_img_vs_category()
    # accuracy_vs_category()
    # pct_min_imgs_vs_non_min_imgs_in_bbx()
    # pct_min_imgs_in_bbx_vs_outside
    accuracy_vs_crop_size()
    # pass