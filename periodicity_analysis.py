import ast 
import json 
import numpy as np
import os
from PIL import Image
from scipy.misc import imresize
import sys
import tensorflow as tf

from confidence_maps_parallel import get_crop_size
import settings


# DATA PARAMS
PATH_TO_DATA = '../../poggio-urop-data/'


def mask_confidence(image_id, crop_metric, model_name, image_scale, correctness='top5'):
    
    conf = np.load(PATH_TO_DATA + settings.map_filename(settings.CONFIDENCE_MAPTYPE, crop_metric, model_name, image_scale, image_id))
    corr = np.load(PATH_TO_DATA + settings.map_filename(correctness, crop_metric, model_name, image_scale, image_id))
    
    # TODO figure out the best masking method 


def pick_backprop_indices(image_id, crop_metric, model_name, image_scale, indices):
 
    '''
    indices: a list of (x1, y1) tuples for crops that should be evaluated
    '''

    backprop_ind_filename = PATH_TO_DATA + settings.BACKPROP_INDICES_FILENAME
    with open(backprop_ind_filename, 'r') as indfile:
        all_inds = json.load(indfile)
    key = str((crop_metric, model_name, image_scale, image_id)) 
    if key in all_inds:
        existing = set([tuple(index) for index in all_inds[key]])
        new = set(indices)
        to_add = new.difference(new.intersection(existing))
        print(to_add)
        all_inds[key].extend(list(to_add))	# to avoid repeats in separate calls
    else:
        all_inds[key] = indices
    print(all_inds)
    with open(backprop_ind_filename, 'w') as indfile:
        json.dump(all_inds, indfile)
       

def backprop_crops(image_id, crop_metric, model_name, image_scale):

    '''
    note: even with the graph inefficiency, it makes more sense for this function to be one image at a time because the indices are different for each one. This would only change if I figured out a programmatic way to find the right indices. 
    '''
   
    # Get indices for this map
    with open(PATH_TO_DATA + settings.BACKPROP_INDICES_FILENAME, 'r') as indfile:
        all_inds = json.load(indfile)
        indices = all_inds[str((crop_metric, model_name, image_scale, image_id))]
 
    with tf.Session() as sess:
        
        # Set up CNN
        model = settings.MODELS[model_name]
        batch_size = len(indices)
        imgs = tf.placeholder(tf.float32, [batch_size, model.im_size, model.im_size, 3])
        network = model(imgs, sess)

        # Create backprop objects 
        true_labels = json.load(open('caffe_ilsvrc12/' + settings.DATASET + '-labels.json'))   
        true_label = true_labels[settings.get_ind_name(image_id)]
        y = tf.constant([true_label for __ in range(batch_size)])
        err = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=network.logits)
        dy_dx = tf.gradients(err, network.imgs)

        # Scale image, get all crops and run backprop
        image = Image.open(PATH_TO_DATA + settings.folder_name('img') + settings.get_ind_name(image_id) + '.JPEG')
        width, height = image.size
        image = imresize(image, (int(width * image_scale), int(height * image_scale)))
        crop_type = 'proportional' if crop_metric <= 1. else 'constant'
        crop_size = get_crop_size(height, crop_metric, crop_type) if height <= width else get_crop_size(width, crop_metric, crop_type)
        all_crops = []
        for x1, y1 in indices:
            crop = image[y1:y1 + crop_size, x1:x1 + crop_size, :]
            # crop = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))    
            all_crops.append(imresize(crop, (model.im_size, model.im_size)))
            # all_crops.append(imresize(image, (model.im_size, model.im_size)))	# TODO remove after making sure this code is working

        backprops = sess.run(dy_dx, feed_dict={network.imgs: model.preprocess(np.array(all_crops))})[0]

    # Make backprop results visualizable
    backprops = backprops - np.min(backprops, axis=(1, 2), keepdims=True)
    backprops = backprops / np.max(backprops, axis=(1, 2), keepdims=True)
    backprops = backprops * 255.
    backprops = backprops.astype(np.uint8)
    
    # savez results
    folder = PATH_TO_DATA + settings.map_folder_name(settings.BACKPROP_MAPTYPE, crop_metric, model_name, image_scale)
    filename = PATH_TO_DATA + settings.map_filename(settings.BACKPROP_MAPTYPE, crop_metric, model_name, image_scale, image_id)
    if not os.path.exists(folder):
        os.makedirs(folder)    
    np.savez(filename, **{str(indices[i]): backprops[i] for i in range(len(indices))})


def get_backprops_by_index(crop_metric, model_name, image_scale, image_id, indices):
    
    backprops = np.load(PATH_TO_DATA + settings.map_filename(settings.BACKPROP_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '.npz')
    for index in indices: 
        yield backprops[str(list(index))]


if __name__ == '__main__':
    test_inds = [(37, 68), (45, 68), (51, 68), (32, 68), (37, 69), (45, 69), (51, 69), (32, 69)]	# these were for 0.2,resnet,1.0,1
    # test_inds = [(242, 54), (237, 54), (247, 54), (242, 49), (237, 49), (247, 49)]
    pick_backprop_indices(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], float(sys.argv[4]), test_inds)
    backprop_crops(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], float(sys.argv[4])) 
        
    outfile = np.load(PATH_TO_DATA + settings.map_filename(settings.BACKPROP_MAPTYPE, float(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[1] + '.npz'))
    print(outfile.keys())






