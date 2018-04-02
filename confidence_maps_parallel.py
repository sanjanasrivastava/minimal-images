import itertools
import json
import numpy as np
import os
from PIL import Image, ImageFilter
from scipy.misc import imread, imresize, imshow
import sys
import tensorflow as tf
import time

import settings
from vgg16 import vgg16


# DATA PARAMS
# PATH_TO_DATA = '../../poggio-urop-data/'
PATH_TO_DATA = '../poggio_urop/poggio-urop-data/'


# TEST PARAMS
CROP_TYPE = 'proportional'          # choose between 'proportional' and 'constant'
CONSTANT = 224                       # pixel-length of square crop (must be odd)
BATCH_SIZE = 160
NUM_GPUS = 8


def get_crop_size(dimension, crop_metric, crop_type):
    crop_size = int(((crop_metric * dimension) // 2) * 2 + 1) if crop_type == 'proportional' else int(crop_metric)
    return crop_size


def create_confidence_map(start_id, end_id, crop_metric, model_name, image_scale, make_cmap=True, make_top5=True, make_top1=True, make_distance=True):

    # Get the crop type - if the crop_metric is a fraction, it's a proportion. If it's larger than 1, it's a constant crop size.
    crop_type = 'proportional' if crop_metric <= 1. else 'constant'
   
    # Prepare folders 
    maptypes = [settings.CONFIDENCE_MAPTYPE, settings.TOP5_MAPTYPE, settings.TOP1_MAPTYPE, settings.DISTANCE_MAPTYPE]
    folders = [PATH_TO_DATA + settings.map_folder_name(maptype, crop_metric, model_name, image_scale) for maptype in maptypes]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
 
    # Get the right model
    model = settings.MODELS[model_name]

    # Before anything else happens, so we only set up once despite multiple images, make a network for each GPU
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
    networks = []
    for i in range(NUM_GPUS):
        gpu = 'device:GPU:%d' % i 
        with tf.device(gpu):
            imgs = tf.placeholder(tf.float32, [BATCH_SIZE, model.im_size, model.im_size, 3])
            network = model(imgs, sess, reuse=None if i == 0 else True)
            networks.append(network)

    # Get each map called for!
    labels_file = open('caffe_ilsvrc12/' + settings.DATASET + '-labels.json')
    true_labels = json.load(labels_file)
    image_ids = range(start_id, end_id + 1)    
    
    for image_id in image_ids:
   
        image_tag = settings.get_ind_name(image_id)
        image_filename = PATH_TO_DATA + settings.folder_name('img') + image_tag + ('.png' if image_id == 50001 else'.JPEG') 
        true_class = true_labels[image_tag] if image_id != 50001 else 266
        im = Image.open(image_filename)
        if im.mode != 'RGB':
            im = im.convert('RGB')	# make sure bw images are 3-channel, because they get opened in L mode
        width, height = im.size

        # Resize image based on image_scale
        im = im.resize((int(width * image_scale), int(height * image_scale)))


        width, height = im.size
       
        # TODO do resizing experiment    im = imresize(im, (width*image_scale, height*image_scale))	# resize image as needed 
    
        crop_size = get_crop_size(height, crop_metric, crop_type) if height <= width else get_crop_size(width, crop_metric, crop_type)
        crop_x1s = range(width - crop_size + 1)
        crop_y1s = range(height - crop_size + 1)
        crop_dims = itertools.product(crop_x1s, crop_y1s)
        
        C, O, F, D = (np.zeros((height - crop_size + 1, width - crop_size + 1)) for __ in range(4))    
        crop_dims = list(crop_dims)
        total_crops = len(crop_dims)
        crop_index = 0
        
        overall_start_time = time.clock()
        print('TOTAL CROPS:', total_crops)
    
        while crop_index < total_crops:	# While we haven't exhausted crops TODO see if the logic needs to be changed
            all_cropped_imgs = []
            stub_crops = 0		# Initializing to 0 in case there's an image with no stub crops (i.e. number of crops is a multiple of 64)
            map_indices = []	# The indices that these confidences will be mapped to 
    
            for i in range(BATCH_SIZE * NUM_GPUS):	# Get BATCH_SIZE crops for each GPU (total of NUM_GPUS)
    
                if crop_index == total_crops:	# If we're on the last round and it's not all 64, repeat the last crop for the rest 
                    stub_crops = (BATCH_SIZE * NUM_GPUS) - i 	# the number of crops still needed, because i in this round is the number of crops that have already been filled in 
                    cropped = imresize(cropped, (model.im_size, model.im_size))		# resize the last one (from previous round) permanently. There will be a previous one due to the while logic
                    for i in range(stub_crops):
                        all_cropped_imgs.append(cropped)    # fill in the rest of the slots with the resized last crop 
                    break 
               
                # If not on the last one, continue on to fill in the next crop
                x1, y1 = crop_dims[crop_index]
                map_indices.append((x1, y1))
                cropped = im.crop((x1, y1, x1 + crop_size, y1 + crop_size))
                all_cropped_imgs.append(imresize(cropped, (model.im_size, model.im_size)))
                crop_index += 1	# Increment to get the next crop dimensions
    
            start_time = time.clock()    
            network_probs = list(map(lambda x: x.probs, networks))
            num_crops = len(all_cropped_imgs)
            partition_cropped_imgs = [all_cropped_imgs[int(i*num_crops/NUM_GPUS):min(num_crops, int((i+1)*num_crops/NUM_GPUS))] for i in range(NUM_GPUS)]	# TODO yikes is this correct
            prob = np.array(sess.run(network_probs, feed_dict={networks[i].imgs: model.preprocess(np.array(partition_cropped_imgs[i])) for i in range(NUM_GPUS)}))       
           
            #  prob = sess.run(vgg.probs, feed_dict={vgg.imgs: all_cropped_imgs})
            end_time = time.clock()
            print ('Time for running one size-' + str(BATCH_SIZE), 'batch:', end_time - start_time, 'seconds')
            print('CROPS COMPLETED SO FAR:', crop_index)
    
            # plot the confidences in the map. For final iteration, which likely has <BATCH_SIZE meaningful crops, the index list being of shorter length will cause them to be thrown. 
            confidence = prob[:, :, true_class].reshape(BATCH_SIZE * NUM_GPUS)
            for i in range(len(map_indices)):
                c, r = map_indices[i]
                C[r, c] = confidence[i]           

            # plot the top-5 and top-1 binary correctness maps
            flat_probs = prob.reshape((NUM_GPUS * BATCH_SIZE, settings.NUM_CLASSES))
            sorted_classes = flat_probs.argsort(axis=1)
            top_5 = sorted_classes[:, -5:]
            top_1 = sorted_classes[:, -1].squeeze()
            for i in range(len(map_indices)):
                c, r = map_indices[i]
                O[r, c] = 255. if top_1[i] == true_class else 0.
                F[r, c] = 255. if true_class in top_5[i] else 0.
                
            # plot the distance map
            sixth = sorted_classes[:, 6]
            for i in range(len(map_indices)):
                c, r = map_indices[i]
                if true_class in top_5[i]:
                    D[r, c] = 255. * flat_probs[i][true_class] - flat_probs[i][sixth[i]]
                else:
                    D[r, c] = 0.
                
        overall_end_time = time.clock()
        print('Time for overall cropping and map-building process with size-' + str(BATCH_SIZE), 'batch:', overall_end_time - overall_start_time, 'seconds')
    
        # Make confidence map, save to confidence map folder
        if make_cmap:
            np.save(PATH_TO_DATA + settings.map_filename(settings.CONFIDENCE_MAPTYPE, crop_metric, model_name, image_scale, image_id), C)

        # Save binary correctness maps
        if make_top5:
            np.save(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id), F)
        if make_top1:
            np.save(PATH_TO_DATA + settings.map_filename(settings.TOP1_MAPTYPE, crop_metric, model_name, image_scale, image_id), O)
             
        # Save distance maps
        if make_distance:
            np.save(PATH_TO_DATA + settings.map_filename(settings.DISTANCE_MAPTYPE, crop_metric, model_name, image_scale, image_id), D)
 
        print(image_tag)


def get_max_diff_adjacent_crops(start_id, end_id, proportion, model_name='vgg16', compare_correctness=True, use_top5=True):	# eventually add step size, right now defaults to 1 in code
    # compare_correctness: bool indicating whether the final crops should necessarily have different classification correctness
    # use_top5: bool indicating whether using top5 correctness or top1 correctness
    # returns the two crops. If compare_correctness==True and there are no correctly classified crops, returns None.

    crop_metric = proportion if CROP_TYPE == 'proportional' else CONSTANT
    maxdiff_folder = PATH_TO_DATA + settings.maxdiff_folder_name(crop_metric)
    if not os.path.exists(maxdiff_folder):
        os.mkdir(maxdiff_folder)

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

        image_tag = settings.get_ind_name(image_id)
        image_filename = PATH_TO_DATA + settings.folder_name('img') + image_tag + ('.png' if image_id == 50001 else '.JPEG') 
        image = Image.open(image_filename)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.asarray(image)
    
        image_label = image_tag + '_' + str(crop_metric) + '_' + model_name + '_'
        correctness_type = 'top5' if use_top5 else 'top1'
        correctness_filename = PATH_TO_DATA + (settings.map_folder_name(settings.TOP5_MAPTYPE, crop_metric) if use_top5 else settings.map_folder_name(settings.TOP1_MAPTYPE, crop_metric)) + image_label + (settings.TOP5_MAPTYPE if use_top5 else settings.TOP1_MAPTYPE) + '.npy'
        cor_map = np.load(correctness_filename)
        corr_extension = '_diffcorrectness' if compare_correctness else '_anycorrectness'
    
        if compare_correctness:
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
                true_value = true_labels[image_tag]
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



def test_maxdiff_crops(start_id, end_id, crop_metric, model_name='vgg16'):
   
    # crop_metric = proportion if CROP_TYPE == 'proportional' else CONSTANT
 
    sess = tf.Session()
    model = settings.MODELS[model_name]
    imgs = tf.placeholder(tf.float32, [None, model.im_size, model.im_size, 3])
    network = model(imgs, sess)
    
    true_labels = json.load(open('caffe_ilsvrc12/' + settings.DATASET + '-labels.json'))
    image_ids = range(start_id, end_id + 1)
    
    hc_incorrect = []
    lc_incorrect = []
    hc_correct = []
    lc_correct = []
    h_activations = {}
    l_activations = {}
    for image_id in image_ids:
        image_tag = settings.get_ind_name(image_id)
        image_label = image_tag + '_' + str(crop_metric) + '_' # TODO I think I got ahead of myself and put in model name to this function when maxdiff crops haven't been adjusted, but eventually, fix this with the new filesystem
        true_value = true_labels[image_tag]
        maxdiff_folder = settings.maxdiff_folder_name(crop_metric)

        try:
            highconf_filename = PATH_TO_DATA + maxdiff_folder + image_label + 'maxdiff_highconf_diffcorrectness.npy'
            highconf_crop = np.load(highconf_filename)
        except FileNotFoundError:
            print(image_tag, 'does not have diff-correctness crops')
            continue
        lowconf_filename = PATH_TO_DATA + maxdiff_folder + image_label + 'maxdiff_lowconf_diffcorrectness.npy'
        lowconf_crop = np.load(lowconf_filename)

        y = tf.constant([true_value])
        err = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=network.fc3l) # TODO adapt to other models beyond VGG
        dy_dx = tf.gradients(err, network.imgs)

        hc = imresize(highconf_crop, (network.im_size, network.im_size))
        lc = imresize(lowconf_crop, (network.im_size, network.im_size))
        hcresult = sess.run([network.probs, dy_dx], feed_dict={network.imgs: [hc]})
        lcresult = sess.run([network.probs, dy_dx], feed_dict={network.imgs: [lc]})
        hcprob = hcresult[0][0]
        lcprob = lcresult[0][0]
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
        h_activations[image_id] = hcresult[1:]
        l_activations[image_id] = lcresult[1:]

        hgrad = np.squeeze(hcresult[1][0])
        lgrad = np.squeeze(lcresult[1][0])

        hgrad = hgrad - hgrad.min()
        lgrad = lgrad - lgrad.min()
        hgrad = hgrad / hgrad.max()
        lgrad = lgrad / lgrad.max()
        hgrad = hgrad * 255.
        lgrad = lgrad * 255.
        hgrad = hgrad.astype(np.uint8)
        lgrad = lgrad.astype(np.uint8)
        print('H Shape:', hgrad.shape)
        print('L Shape:', lgrad.shape)

        hgrad_vis = Image.fromarray(hgrad, mode='RGB')
        lgrad_vis = Image.fromarray(lgrad, mode='RGB')
        hgrad_vis.save(PATH_TO_DATA + str(image_id) + 'hgrad_vis.JPEG', 'JPEG')
        lgrad_vis.save(PATH_TO_DATA + str(image_id) + 'lgrad_vis.JPEG', 'JPEG')
        
        # SANITY CHECK for backprop code
        image_filename = PATH_TO_DATA + settings.folder_name('img') + image_tag + ('.png' if image_id == 50001 else'.JPEG') 
        im = imread(image_filename, mode='RGB')
        im = imresize(im, (network.im_size, network.im_size))
        fullresult = sess.run([network.probs, dy_dx], feed_dict={network.imgs: [im]})
        fullgrad = np.squeeze(fullresult[1][0])

        fullgrad = fullgrad - fullgrad.min()
        fullgrad = fullgrad / fullgrad.max()
        fullgrad = fullgrad * 255.
        fullgrad = fullgrad.astype(np.uint8) # np.uint8)


        print('OG SHAPE:', im.shape)
        print('FG SHAPE:', fullgrad.shape)

        print('EQUIVALENT:', np.array_equal(im, fullgrad))
        print('OG TYPE:', type(im[0][0][0]))
        print('FG TYPE:', type(fullgrad[0][0][0]))    
 
        debug_folder = 'debugging/'
        original = Image.fromarray(im, mode='RGB')
        original.save(debug_folder + 'og.JPEG', 'JPEG')

        print('FULLGRAD:', fullgrad)
        fullgrad_vis = Image.fromarray(fullgrad, mode='RGB')
        fullgrad_vis.save(debug_folder + 'fg.JPEG', 'JPEG')

    print('EXTERNAL TEST')
    print("High confidence crop correctly classified:", hc_correct)
    print('High confidence crop incorrectly classified:', hc_incorrect)
    print('Low confidence crop correctly classified:', lc_correct)
    print('Low confidence crop incorrectly classified:', lc_incorrect)
    return h_activations, l_activations


if __name__ == '__main__':
    create_confidence_map(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], float(sys.argv[5])) 

 
    # h_activations, l_activations = get_max_diff_adjacent_crops(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), compare_correctness=True) 
    
    # test_maxdiff_crops(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))



