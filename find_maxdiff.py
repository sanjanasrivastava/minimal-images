import json
import numpy as np
import os
from PIL import Image
from scipy.misc import imread, imresize, imshow
import tensorflow as tf

import settings
from confidence_maps_parallel import get_crop_size


# DATA PARAMS
PATH_TO_DATA = settings.MIN_IMGS_PATH_TO_DATA


def get_maxdiff_size_crops(start_id, end_id, crop_metric, model_name, image_scale, compare_corr=True):

    # For each image id in range(start_id, end_id + 1), finds the crop of size crop_metric and the crop ~2 pixels smaller that are maximally different in confidence. If compare_corr is True, it necessarily finds crops where the smaller one is classified incorrectly and the larger one is classified correctly. This uses the top5 maps for the two scales.

    crop_type = 'proportional' if crop_metric <= 1. else 'constant'
    folders = [PATH_TO_DATA + settings.maxdiff_folder_name('size', crop_metric, model_name, image_scale, 'diff' if compare_corr else 'any', conf) for conf in ['high', 'low']]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    map_folder = PATH_TO_DATA + settings.maxdiff_folder_name('size', crop_metric, model_name, image_scale, 'map')
    if not os.path.exists(map_folder):
        os.makedirs(map_folder)
 
    # with tf.Session() as sess:	# TODO use for testing
    # model = settings.MODELS[model_name]
    # imgs = tf.placeholder(tf.float32, [None, model.im_size, model.im_size, 3])
    # network = model(imgs, sess)

    true_labels = json.load(open('caffe_ilsvrc12/' + settings.DATASET + '-labels.json'))

    for image_id in range(start_id, end_id + 1):
        image_tag = settings.get_ind_name(image_id)
        image_filename = PATH_TO_DATA + settings.folder_name('img') + image_tag + '.JPEG'
        true_class = true_labels[image_tag]
        im = Image.open(image_filename)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        width, height = im.size
        im = im.resize((int(width * image_scale), int(height * image_scale)))
        width, height = im.size
        im = np.asarray(im)

        # Get the small crop_metric, crop sizes for the large and small crop_metrics
        size_dim = height if height <= width else width
        large_size = get_crop_size(size_dim, crop_metric, crop_type) if height <= width else get_crop_size(width, crop_metric, crop_type)
        small_metric = 0.194 if crop_metric == 0.2 else 0.394	# TODO change to be a calculation and command
        small_size = get_crop_size(size_dim, small_metric, crop_type)
        metrics = [crop_metric, small_metric]        

        # Get the correctness maps (top5, may become a choice between top5 and top1 in the future), and if the call requires diff correctness, check that that's possible
        corr_fns = [PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, metric, model_name, image_scale, image_id) + '.npy' for metric in metrics]	# TODO change to allow choice of top1 or top5
        lcor, scor = cor_maps = [np.load(corr_fn) for corr_fn in corr_fns]
        if compare_corr:
            for cor_map in cor_maps:
                if not cor_map.any():
                    print('%s has no correctly classified crops.' % image_tag)
                    continue
                elif cor_map.all():
                    print('%s has only correctly classified crops.' % image_tag)
                    continue            

        # Get confidence maps 
        con_fns = [PATH_TO_DATA + settings.map_filename(settings.CONFIDENCE_MAPTYPE, metric, model_name, image_scale, image_id) + '.npy' for metric in metrics]
        lcon, scon = [np.load(con_fn) for con_fn in con_fns]
        
        # Calculate difference matrices
        lrows, lcols = lcon.shape
        offset = large_size - small_size	# get the metric that bottom and right are off by 

        tl_sub = scon[:lrows, :lcols]	# for top left, get the top left small crops that correspond to big crops - same shape. It's all of them because the large crops are all adjacent, even if the difference in pixels is >1. 
        tr_sub = scon[:lrows, offset:lcols + offset]	# for top right, everything that is 'offset' cols over 
        bl_sub = scon[offset:lrows + offset, :lcols]
        br_sub = scon[offset:lrows + offset, offset:lcols + offset]
        ctoffset = int(offset / 2)
        ct_sub = scon[ctoffset:lrows + ctoffset, ctoffset:lcols + ctoffset]

        diffs = {'tl': lcon - tl_sub,	# use subtraction because we are looking for increase in conf from increase in size
                 'tr': lcon - tr_sub,
                 'bl': lcon - bl_sub,
                 'br': lcon - br_sub,
                 'ct': lcon - ct_sub}

        # Make map of the largest size change in confidence across all directions of shrinking
        change_map = np.maximum.reduce(list(diffs.values()))
        np.save(map_folder + str(image_id), change_map)

        # Find maxdiff pair by searching for maximally different pairs until one with different correctness is found (if diffcor. Else, this will terminate after one loop as the first pair found will be the maximally different one and therefore the right one for anycor.)
        while True:

            maxes = {corner: np.unravel_index(np.argmax(diffs[corner]), diffs[corner].shape) for corner in diffs}	# map each corner diff to its argmax (index of maximum confidence diff)
            max_dir = max([corner for corner in maxes], key=lambda corner: diffs[corner][tuple(maxes[corner])])	# get the corner id of the diff whose max change is the highest out of the four max changes 
            # getting the indices of the maximal confidence increase. Indices are based on the size of the large crop size map. The first index is the index for the large crop, and the second is for the small crop. 
            corner_max = maxes[max_dir]
            if max_dir == 'tl':
                lcell, scell = tuple(corner_max), tuple(corner_max)
            elif max_dir == 'tr':
                lcell, scell = tuple(corner_max), (corner_max[0], corner_max[1] + offset)
            elif max_dir == 'bl':
                lcell, scell = tuple(corner_max), (corner_max[0] + offset, corner_max[1])
            elif max_dir == 'br':
                lcell, scell = tuple(corner_max), (corner_max[0] + offset, corner_max[1] + offset)
            else:
                lcell, scell = tuple(corner_max), (corner_max[0] + ctoffset, corner_max[1] + ctoffset)

            diff_corr = lcor[lcell] != scor[scell]
            if diff_corr or not compare_corr:
                sy, sx = scell
                ly, lx = lcell
                lcropped = im[lcell[0]:lcell[0] + large_size, lcell[1]:lcell[1] + large_size]
                scropped = im[scell[0]:scell[0] + small_size, scell[1]:scell[1] + small_size]
                # lcropped = imresize(lcropped, (network.im_size, network.im_size))
                # scropped = imresize(scropped, (network.im_size, network.im_size))
                # result = sess.run(network.probs, feed_dict={network.imgs: np.array([lcropped, scropped])})	# run and see if it works later. Without this, the sess isn't actually being used - this is for internal test. 
                break 
            else:	# if that location wasn't diffcorr, set the diff's entry to -2. 
                diffs[max_dir][lcell] = -2.
                
        lfolder, sfolder = folders
        np.save(lfolder + str(image_id), lcropped)
        np.save(sfolder + str(image_id), scropped)


def get_max_diff_adjacent_crops(start_id, end_id, crop_metric, model_name, image_scale, compare_corr=True, use_top5=True):	# eventually add step size, right now defaults to 1 in code

    # compare_correctness: bool indicating whether the final crops should necessarily have different classification correctness
    # use_top5: bool indicating whether using top5 correctness or top1 correctness
    # returns the two crops. If compare_correctness==True and there are no correctly classified crops, returns None.

    crop_type = 'proportional' if crop_metric <= 1. else 'constant'
    folders = [PATH_TO_DATA + settings.maxdiff_folder_name('size', crop_metric, model_name, image_scale, 'diff' if compare_corr else 'any', conf) for conf in ['high', 'low']]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    sess = tf.Session()
    model = settings.MODELS[model_name]
    imgs = tf.placeholder(tf.float32, [None, model.im_size, model.im_size, 3])
    network = model(imgs, sess)

    image_ids = range(start_id, end_id + 1)
    hc_correct = []
    hc_incorrect = []
    lc_correct = []
    lc_incorrect = []
    h_activations = {}
    l_activations = {}
    for image_id in image_ids:

        f = open('small-dataset-to-imagenet.txt')
        lines = f.readlines()
        image_tag = lines[image_id].split(" ", 1)[0]
        print(image_tag)

        #image_tag = settings.get_ind_name(image_id)
        image_filename = PATH_TO_DATA + settings.folder_name('img') + image_tag #+ ('.png' if image_id == 50001 else '.JPEG')
        image = Image.open(image_filename)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.asarray(image)

        image_label = image_tag[:-5] + '_' + str(crop_metric) + '_' + model_name + '_'
        correctness_type = 'top5' if use_top5 else 'top1'
        correctness_filename = PATH_TO_DATA + (settings.map_folder_name(settings.TOP5_MAPTYPE, crop_metric) if use_top5 else settings.map_folder_name(settings.TOP1_MAPTYPE, crop_metric)) + image_label + (settings.TOP5_MAPTYPE if use_top5 else settings.TOP1_MAPTYPE) + '.npy'
        cor_map = np.load(correctness_filename)
        corr_extension = '_diffcorrectness' if compare_corr else '_anycorrectness'

        if compare_corr:
            if not cor_map.any():
                print('%s has no correctly classified crops.' % image_tag)
                continue
            elif cor_map.all():
                print('%s has only correctly classified crops.' % image_tag)
                continue

        con_map_filename = PATH_TO_DATA + settings.map_folder_name(settings.CONFIDENCE_MAPTYPE, crop_metric) + image_label + settings.CONFIDENCE_MAPTYPE + '.npy'
        con_map = np.load(con_map_filename)

        down_diff = np.diff(con_map, axis=0)	# apparently assumes step_size=1 (adjacency)
        up_diff = -1. * down_diff
        right_diff = np.diff(con_map)
        left_diff = -1. * right_diff
        diffs = {'up': up_diff, 'down': down_diff, 'left': left_diff, 'right': right_diff}

        # TESTER: CLASSIFYING CROPS TO SEE IF THERE'S A DIFFERENCE BETWEEN WHAT THE TESTER REPORTS AND WHAT THIS REPORTS
        true_labels = json.load(open('caffe_ilsvrc12/' + settings.DATASET + '-labels.json'))

        while True:
            maxes = {direction: np.unravel_index(np.argmax(diffs[direction]), diffs[direction].shape) for direction in diffs}	# map each directional diff to its argmax (index of its maximum confidence diff)
            max_dir = max([direction for direction in maxes], key=lambda direction: diffs[direction][tuple(maxes[direction])])	# get the direction of the diff whose max confidence is the highest out of the four max confidences
            # depending on the max-confidence direction, get the argmax of that direction. The more confident crop will be offset by 1 in a way that depends on the direction.
            if max_dir == 'up':
                up_max = maxes['up']
                gcell, cell = (tuple(up_max), (up_max[0] + 1, up_max[1]))	# up (and left) are like this because when up and left diffs are made, the negation also changes the direction in which your step goes. it goes down -> up; right -> left.
            elif max_dir == 'down':
                down_max = maxes['down']
                cell, gcell = (tuple(down_max), (down_max[0] + 1, down_max[1]))
            elif max_dir == 'left':
                left_max = maxes['left']
                gcell, cell = (tuple(left_max), (left_max[0], left_max[1] + 1))
            else:
                right_max = maxes['right']
                cell, gcell = (tuple(right_max), (right_max[0], right_max[1] + 1))

            diff_correctness = cor_map[cell] != cor_map[gcell]
            if diff_correctness or not compare_correctness:
                y, x = cell
                gy, gx = gcell
                height, width, channels = image.shape
                crop_size = get_crop_size(height, proportion) if height <= width else get_crop_size(width, proportion)
                dim_used = 'height' if height <= width else 'width'

                cropped = image[cell[0]:cell[0] + crop_size, cell[1]:cell[1] + crop_size]
                gcropped = image[gcell[0]:gcell[0] + crop_size, gcell[1]:gcell[1] + crop_size]
                true_value = true_labels[image_tag[:-5]]
                hc  = imresize(gcropped, (network.im_size, network.im_size))
                lc = imresize(cropped, (network.im_size, network.im_size))

                hresult = sess.run(network.pull_layers, feed_dict={network.imgs: [hc]})
                hcprob = hresult[0][0]
                h_activations[image_id] = hresult[1:]

                result = sess.run(network.pull_layers, feed_dict={network.imgs: [lc]})
                lcprob = lresult[0][0]
                l_activations[image_id] = lresult[1:]

                hcpreds = (np.argsort(hcprob)[::-1])[0:5]
                lcpreds = (np.argsort(lcprob)[::-1])[0:5]

                if true_value in hcpreds:
                    hc_correct.append(image_id)
                else:
                    hc_incorrect.append(image_id)
                if true_value in lcpreds:
                    lc_correct.append(image_id)
                else:
                    lc_incorrect.append(image_id)

                maxdiff_folder = settings.maxdiff_folder_name(crop_metric)
                np.save(PATH_TO_DATA + maxdiff_folder + image_label + 'maxdiff_lowconf' + corr_extension, cropped)
                np.save(PATH_TO_DATA + maxdiff_folder + image_label + 'maxdiff_highconf' + corr_extension, gcropped)
                break

            else:
                if max_dir in ['up', 'left']:
                    diffs[max_dir][gcell] = -2.	# for the diff where that was the argmax, mark the cell containing it to something lower than any real entry (-1. <= real entry <= 1.) This is gcell for up, left and cell for down, right because the lower-indexed cell is always the one that contained the confidence originally
                else:
                    diffs[max_dir][cell] = -2.

    print('INTERNAL TEST')
    print("High confidence crop correctly classified:", hc_correct)
    print('High confidence crop incorrectly classified:', hc_incorrect)
    print('Low confidence crop correctly classified:', lc_correct)
    print('Low confidence crop incorrectly classified:', lc_incorrect)
    sess.close()
    return h_activations, l_activations

           
if __name__ == '__main__':
    get_maxdiff_size_crops(1, 5, 0.2, 'vgg16', 1.0)
     
                




