import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PATH_TO_DATA = '../min-img-data/'
PATH_TO_STATS = PATH_TO_DATA + 'stats/'

models = ['vgg16', 'resnet', 'inception']


# Percent minimal images inside/outside bbx vs. model; crop metric 0.2

# Shift

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
sns.set_style('darkgrid')

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
    nums = {model: json.load(open(PATH_TO_STATS + '0.2/' + model + '/1.0/' + strictness + '/' + axis + '/id-to-measurements.json')) for model in models}
    nums_discretesize = {}
    for model in nums:
        for smalldataset_id in nums[model]:
            entry = nums[model][smalldataset_id]
            entry[0] = discrete_sizes(entry[0])
            nums_discretesize[model] = entry

    loose_nums_df = pd.concat([pd.DataFrame(data={'model': model,
                                     'proportion of image in bbx': [nums[model][smalldataset_id][0] for smalldataset_id in nums[model]],
                                     'number of minimal images': [nums[model][smalldataset_id][1][0] for smalldataset_id in nums[model]]})
                     for model in nums])
    ax = sns.pointplot(x='proportion of image in bbx', y='number of minimal images', hue='model', data=loose_nums_df, order=['XS', 'S', 'M', 'L', 'XL'])
    ax.set_title('Number of ' + (strictness + ' ' + axis).title() + ' Minimal Images vs. Relative Size of Bound-in Box')
    plt.show()

# vis_num_min_imgs_vs_prop_in_bbx_models(0.2, 1.0, 'loose', 'scale')
# vis_num_min_imgs_vs_prop_in_bbx_models(0.2, 1.0, 'strict', 'scale')





