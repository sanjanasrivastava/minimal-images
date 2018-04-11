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
    folders = [settings.maxdiff_folder_name('size', crop_metric, model_name, image_scale, compare_corr, conf) for conf in ['high', 'low']]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    print('MADE FOLDER')   
 
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
        print('GOT IMAGE') 

        # Get the small crop_metric, crop sizes for the large and small crop_metrics
        size_dim = height if height <= width else width
        large_size = get_crop_size(size_dim, crop_metric, crop_type) if height <= width else get_crop_size(width, crop_metric, crop_type)
        small_metric = 0.194 if crop_metric == 0.2 else 0.394	# TODO change to be a calculation and command
        small_size = get_crop_size(size_dim, small_metric, crop_type)
        metrics = [crop_metric, small_metric]        
        print('GOT METRICS')

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
        print('CHECKED FOR VIABILITY')

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

        loop_counter = 0
        print('LCON SIZE:', lcon.size)
        print('SCON SIZE:', scon.size)
        for corner in diffs:
            print('CORNER:', corner, ', DIFF SIZE:', diffs[corner].size)
        return 
        while True:
            
            loop_counter += 1
            if not loop_counter % lcon.size:
                print('went through one lcon size') 
            if loop_counter > lcon.size * 4 + 100:
                print('INFINITE LOOP')
                return

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
            else:
                lcell, scell = tuple(corner_max), (corner_max[0] + offset, corner_max[1] + offset)

            diff_corr = lcor[lcell] != scor[scell]
            if diff_corr or not compare_corr:
                sy, sx = scell
                ly, lx = lcell
                lcropped = im[lcell[0]:lcell[0] + large_size, lcell[1]:lcell[1] + large_size]
                scropped = im[scell[0]:scell[0] + small_size, scell[1]:scell[1] + small_size]
                # lcropped = imresize(lcropped, (network.im_size, network.im_size))
                # scropped = imresize(scropped, (network.im_size, network.im_size))
                # result = sess.run(network.probs, feed_dict={network.imgs: np.array([lcropped, scropped])})	# run and see if it works later. Without this, the sess isn't actually being used - this is for internal test. 
                print('SCELL:', scell, 'LCELL:', lcell)
                return 
            else:	# if that location wasn't diffcorr, set the diff's entry to -2. 
                diffs[max_dir][lcell] = -2.
                
        np.save('ltest', lcropped)
        np.save('stest', scropped)

           
if __name__ == '__main__':
    get_maxdiff_size_crops(1, 1, 0.2, 'resnet', 1.0)
     
                




