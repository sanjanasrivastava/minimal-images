import json
import os

from alexnet import alexnet 
from imagenet_classes import class_names
from inception import inception
from resnet import resnet 
from vgg16 import vgg16


IMAGENET_SET = 'ILSVRC2012'
DATASET = 'val'
MULTIPLE_CROPS_DATATYPES = ['crop', 'gbvs', 'itti', 'rand']
BBX_DATATYPES = ['crop', 'backblk', 'backnoise']
BIN_THRESHOLDS = [0.25, 0.5, 0.75]  # These are used in MATLAB, which doesn't reference this file, but still for names and stuff I need this data
DATA_FIELDNAMES = ['testID', 'imagenet_set', 'dataset', 'datatype', 'num_images', 'top1_accuracy', 'top5_accuracy', 'aggregation', 'num_crops']


NUM_CLASSES = 1000
MODELS = {'vgg16': vgg16, 'alexnet': alexnet, 'inception': inception, 'resnet': resnet}
# PULL_LAYERS = {'vgg16': [model.probs, model.conv1_1, model.conv1_2, model.pool1, model.conv2_1, model.conv2_2, model.pool2, model.conv3_1, model.conv3_2, model.conv3_3, model.pool3, model.conv4_1, model.conv4_2, model.conv4_3, model.pool4, model.conv5_1, model.conv5_2, model.conv5_3, model.pool5, model.fc1, model.fc2, model.fc3l],
#                'alexnet': [model.probs, model.conv1, model.conv2, model.conv3, model.conv4, model.conv5, model.fc6, model.fc7, model.fc8]}



CROP_MIN_WIDTH = 20
CROP_MIN_HEIGHT = 20
# NUM_RAND_BBXS = 10    # Replaced this with a command line arg so that we could decide how many crops we wanted
CROP_CONFIDENCE_THRESHOLD = 0.5


def folder_name(datatype):
    return IMAGENET_SET + '_' + datatype + '_' + DATASET + '/'


def get_ind_name(image_id):

    return IMAGENET_SET + '_' + DATASET + '_' + '%08d' % image_id


def get_all_names(num, datatype, startpoint=1):
    """
    Parameters
    ----------
    num (int): number of names desired (will get names of first num images)
    datatype (str): ya know
    startpoint (int): starting image you want to get

    Returns
    -------
    basic names for a dataset's images ([IMAGENET_SET]_[dataset]_[image ID])
    """

    bounds = range(startpoint, num + startpoint)
    names = []
    for i in bounds:
        ind_name = get_ind_name(i)
        names.append(ind_name)

    return names


def get_random_crops_results_name(image_id):

    return get_ind_name(image_id) + '_crop_results.json'


def get_class_labels(image_ids):

    labels_file = open('caffe_ilsvrc12/' + DATASET + '-labels.json')
    true_labels = json.load(labels_file)
    for image_id in image_ids:
        image_tag = get_ind_name(image_id)
        true_class = true_labels[image_tag]
        print('IMAGE ID NUMBER:', image_id, '; TRUE CLASS:', true_class, '; CLASSNAME:', class_names[true_class])


MIN_IMGS_PATH_TO_DATA = '../poggio_urop/poggio-urop-data/'
MIN_IMGS_PATH_TO_WEIGHTS = MIN_IMGS_PATH_TO_DATA + 'weights/'

# MAPTYPES
CONFIDENCE_MAPTYPE = 'confidence'
TOP5_MAPTYPE = 'top5'
TOP1_MAPTYPE = 'top1'
DISTANCE_MAPTYPE = 'distance'
BACKPROP_MAPTYPE = 'backprop'

BACKPROP_INDICES_FILENAME = 'backprop/backprop-indices.json'

def map_folder_name(maptype, crop_metric, model, image_scale):
    return os.path.join(maptype, str(crop_metric), model, str(image_scale), '')


def map_filename(maptype, crop_metric, model, image_scale, image_id):

    return map_folder_name(maptype, crop_metric, model, image_scale) + str(image_id) 


# MAXDIFF CROPS PREFIX
MAXDIFF_PREFIX = 'maxdiff'

# def maxdiff_folder_name(proportion):
#     return IMAGENET_SET + '_' + MAXDIFF_PREFIX + '_' + str(proportion) + '/'


def maxdiff_folder_name(axis, crop_metric, model_name, image_scale, corr, conf=None):
    if conf:    
        return os.path.join(MAXDIFF_PREFIX, axis, str(crop_metric), model_name, str(image_scale), corr, conf, '')
    else: 
        return os.path.join(MAXDIFF_PREFIX, axis, str(crop_metric), model_name, str(image_scale), corr, '')

