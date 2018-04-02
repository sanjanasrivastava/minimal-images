########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import csv
import json
import numpy as np
import os
from scipy.misc import imread, imresize, imshow
import shutil
import sys
import tensorflow as tf

import settings



class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))


# Testing

CORRECT_THRESHOLD = 5
DATA_PATH = '../../poggio-urop-data/'   # path from this file to the data folder

if sys.argv[0] == 'vgg16_imagenet.py':   # if the python script is being run directly
    COMMAND_PATH = ''
else:
    COMMAND_PATH = 'poggio_urop_code/vgg_tensorflow/'   # path from main shell script to here, to be attached to anything this file references

if sys.argv[1] == 'test':
    DATATYPE = sys.argv[4]
elif sys.argv[1] == 'random' or 'best':
    DATATYPE = 'randind'
else:
    DATATYPE = None


def test():     # Run a test on a large set of (potentially transformed) images across all crops with some kind of aggregation

    num_images = int(sys.argv[3])
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, COMMAND_PATH + 'vgg16_weights.npz', sess)

    folder = COMMAND_PATH + DATA_PATH + settings.folder_name(DATATYPE)
    nametags = settings.get_all_names(num_images, DATATYPE)
    names = os.listdir(folder)
    total_names = len(names)

    labels_file = open(COMMAND_PATH + 'caffe_ilsvrc12/' + settings.DATASET + '-labels.json')
    true_labels = json.load(labels_file)

    top1_accuracy = 0.
    top5_accuracy = 0.

    results = {}

    name_index = 0
    aggregation = sys.argv[2]
    for image_tag in nametags:
        probs = []
        while image_tag in names[name_index]:   # if the current filename contains the image tag
            image_name = names[name_index]
            img1 = imread(folder + image_name, mode='RGB')
            img1 = imresize(img1, (224, 224))

            prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
            probs.append(prob)
            name_index += 1
            if name_index >= total_names:
                break

        all_probs = np.array(probs)
        if aggregation == 'avg':
            prob = np.average(all_probs, 0)
        else:
            prob = np.amax(all_probs, 0)

        preds = (np.argsort(prob)[::-1])[0:5]
        true_value = true_labels[image_tag]

        if true_value in preds:
            top5_accuracy += 1.
            results[image_tag] = True
            if true_value == preds[0]:
                top1_accuracy += 1.
        else:
            results[image_tag] = False

        # for p in preds:
        #     print p
        #     print class_names[p], prob[p]

    # Save results of individual images (whether classified correctly or not)

    with open(COMMAND_PATH + DATA_PATH + settings.IMAGENET_SET + '_results/' + settings.IMAGENET_SET + '_' + DATATYPE +
              '_' + settings.DATASET + '_results' + '.json', 'w') as results_file:
        json.dump(results, results_file)

    # Print accuracies and save in test statistics CSV

    top1_accuracy /= num_images
    top5_accuracy /= num_images

    fieldnames = settings.DATA_FIELDNAMES
    with open(COMMAND_PATH + DATA_PATH + 'test_statistics.csv', 'rU') as stats_file:
        stats_reader = csv.DictReader(stats_file, fieldnames=fieldnames)
        # TODO make this code less disgusting
        testID = 0
        for line in stats_reader:
            testID = line['testID']
        testID = int(testID) + 1

    with open(COMMAND_PATH + DATA_PATH + 'test_statistics.csv', 'a') as stats_file:
        entry = {'testID': testID,
                 'imagenet_set': settings.IMAGENET_SET,
                 'dataset': settings.DATASET,
                 'datatype': DATATYPE,
                 'num_images': num_images,
                 'top1_accuracy': top1_accuracy,
                 'top5_accuracy': top5_accuracy,
                 'aggregation': aggregation,
                 'num_crops': None}
        stats_writer = csv.DictWriter(stats_file, fieldnames=fieldnames)
        stats_writer.writerow(entry)
        print entry

    if not DATATYPE == 'img':   # We don't want to remove the images, but everything else yes
        shutil.rmtree(folder)

    print 'TOP 1 ACCURACY:', top1_accuracy
    print 'TOP 5 ACCURACY:', top5_accuracy


def random():

    # Start VGG
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, COMMAND_PATH + 'vgg16_weights.npz', sess)

    # Get random crop list
    folder = COMMAND_PATH + DATA_PATH + settings.folder_name(DATATYPE)
    rand_crops = os.listdir(folder)

    # Get true value
    image_id = int(sys.argv[2])
    image_tag = settings.get_ind_name(image_id)
    labels_file = open(COMMAND_PATH + 'caffe_ilsvrc12/' + settings.DATASET + '-labels.json')
    true_labels = json.load(labels_file)
    true_value = true_labels[image_tag]

    # Get old classifications if they exist
    results_folder = COMMAND_PATH + DATA_PATH + settings.IMAGENET_SET + '_randind_results/'
    old_results_name = settings.get_random_crops_results_name(image_id)
    if old_results_name in os.listdir(results_folder):
        with open(results_folder + old_results_name, 'r') as old_results_file:
            old_crop_classifications = json.load(old_results_file)
    else:
        old_crop_classifications = {}

    # Classify crops and store results in dictionary
    with open(COMMAND_PATH + DATA_PATH + settings.IMAGENET_SET + '_randind_results/' + image_tag + '_' + 'dim_ids.json',
              'r') as dim_ids_file:
        dim_ids = json.load(dim_ids_file)

    crop_classifications = {}
    beginning = 'ILSVRC2012_val_00000000_crop'
    end = '.JPEG'
    for rand_crop in rand_crops:
        crop_id = rand_crop[len(beginning):-1 * len(end)]  # from the end of 'crop' to beginning of '.JPEG'. TODO fix; super abusive
        img1 = imread(folder + rand_crop, mode='RGB')
        img1 = imresize(img1, (224, 224))
        prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
        max_prob = np.amax(prob)
        pred = (np.argsort(prob))[-1]
        print 'PRED:', pred
        print 'PROB:', max_prob
        dims = dim_ids[crop_id]
        if pred == true_value and max_prob > settings.CROP_CONFIDENCE_THRESHOLD:
            crop_classifications[str(dims)] = True
        else:
            crop_classifications[str(dims)] = False

    # Update old classifications with new ones and save everything
    old_crop_classifications.update(crop_classifications)
    with open(results_folder + image_tag + '_crop_results.json', 'w') as results_file:
        json.dump(old_crop_classifications, results_file)
        print old_crop_classifications

    shutil.rmtree(folder)


def best_crops():

    # Start VGG
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, COMMAND_PATH + 'vgg16_weights.npz', sess)

    # Get best crop records - CHANGED TO just start a new best crop record
    # results_folder = COMMAND_PATH + DATA_PATH + settings.IMAGENET_SET + '_randind_results/'
    # if 'best-crops.json' in os.listdir(results_folder):
    #     with open(COMMAND_PATH + DATA_PATH + settings.IMAGENET_SET + '_randind_results/best-crops.json', 'r') as best_crop_file:
    #         best_crop_record = json.load(best_crop_file)
    # else:
    #     best_crop_record = {}
    best_crop_record = {}

    start_id = int(sys.argv[2])
    end_id = int(sys.argv[3])
    num_crops = int(sys.argv[4])
    for image_id in range(start_id, end_id + 1):

        # Get random crop list
        folder = folder = COMMAND_PATH + DATA_PATH + settings.folder_name(DATATYPE + '%08d' % image_id)
        rand_crops = os.listdir(folder)

        # Get true value
        image_tag = settings.get_ind_name(image_id)
        labels_file = open(COMMAND_PATH + 'caffe_ilsvrc12/' + settings.DATASET + '-labels.json')
        true_labels = json.load(labels_file)
        true_value = true_labels[image_tag]

        # Classify crops and get the best crop
        dim_id_path = COMMAND_PATH + DATA_PATH + settings.IMAGENET_SET + '_randind_results/' + image_tag + '_dim_ids.json'
        with open(dim_id_path,
                  'r') as dim_ids_file:
            dim_ids = json.load(dim_ids_file)

        best_prob = 0
        best_dims = None
        beginning = 'ILSVRC2012_val_00000000_crop'
        end = '.JPEG'
        for rand_crop in rand_crops:
            crop_id = rand_crop[len(beginning):-1 * len(end)]  # from the end of 'crop' to beginning of '.JPEG'. TODO fix; super abusive
            img1 = imread(folder + rand_crop, mode='RGB')
            img1 = imresize(img1, (224, 224))
            prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]

            spec_prob = prob[true_value]    # get the probability of this image being the true value

            if spec_prob > best_prob:   # if it's better, save its dimensions and check whether it correctly classified the image
                best_prob = spec_prob
                best_dims = dim_ids[crop_id]

                pred = (np.argsort(prob))[-5:]   # get the actual top 5 predictions

                if true_value in pred:
                    correct = True
                else:
                    correct = False

        print 'IMAGE ID:', image_id
        print type(best_crop_record)
        print len(best_crop_record)

        # if str(image_id) not in best_crop_record or float(best_crop_record[str(image_id)][0]) < best_prob:    # if it doesn't exist yet or this new probability is better than the old one
        #     best_crop_record[image_id] = (str(best_prob), best_dims, correct)     # update the best crop to this
        best_crop_record[image_id] = (str(best_prob), best_dims, correct)   # just add it - no longer updating an existing one

        os.remove(dim_id_path)  # Get rid of the dimensions file
        shutil.rmtree(folder)   # Get rid of the crops

    # Update test records with an entry on number of images and performance on those images

    with open(COMMAND_PATH + DATA_PATH + settings.IMAGENET_SET + '_randind_results/best-crops.json', 'w') as best_crop_file:
        json.dump(best_crop_record, best_crop_file)

    success = sum(best_crop_record[image_id][2] for image_id in best_crop_record)
    total = len(best_crop_record)

    top5_accuracy = float(success)/float(total)

    fieldnames = settings.DATA_FIELDNAMES
    with open(COMMAND_PATH + DATA_PATH + 'test_statistics.csv', 'rU') as stats_file:
        stats_reader = csv.DictReader(stats_file, fieldnames=fieldnames)
        # TODO make this code less disgusting
        testID = 0
        for line in stats_reader:
            testID = line['testID']
        testID = int(testID) + 1

    with open(COMMAND_PATH + DATA_PATH + 'test_statistics.csv', 'a') as stats_file:
        entry = {'testID': testID,
                 'imagenet_set': settings.IMAGENET_SET,
                 'dataset': settings.DATASET,
                 'datatype': DATATYPE,
                 'num_images': total,
                 # 'top1_accuracy': None,
                 'top5_accuracy': top5_accuracy,
                 'aggregation': None,
                 'num_crops': num_crops}
        stats_writer = csv.DictWriter(stats_file, fieldnames=fieldnames)
        stats_writer.writerow(entry)


if __name__ == '__main__':

    # folder = COMMAND_PATH + DATA_PATH + settings.folder_name(DATATYPE)

    if sys.argv[1] == 'test':
        test()
    elif sys.argv[1] == 'random':
        random()
    elif sys.argv[1] == 'best':
        best_crops()