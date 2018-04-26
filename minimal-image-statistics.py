import json
import numpy as np
import os.path
from PIL import Image
import random
from scipy.misc import imresize
import sys

import confidence_maps_parallel as c_m_p
import settings


PATH_TO_DATA = "/om/user/xboix/share/minimal-images/"
# PATH_TO_DATA = '../min-img-data/'	# uncomment only when working on my laptop
#""./backup/"


def create_location_minimal_image_maps(image_id, crop_metric, model_name, image_scale, loose, k=1):

    '''
    image_id (int): the small dataset id of the image we are finding minimal images for 
    crop_metric (float): the crop metric we are referencing
    model_name (string): the model that we are referencing
    image_scale (float): the image scale we are referencing
    loose (bool): loose minimal images if True else strict minimal images
    k (int): the square size that we are looking for minimal image change within; should be even
    '''

    fname = PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '.npy'

    if not os.path.isfile(fname):
        return -1, -1

    top5map = np.load(fname)
    r, c = top5map.shape
    
    M = np.zeros((r, c))

    for i in range(r):
        for j in range(c):
            offset = int(k / 2)
            self = top5map[i, j]

            # make minimal image map 
            if loose:
                window = top5map[max(0, i - offset):min(r-1, i + offset)+1, max(0, j - offset):min(c-1, j + offset)+1]	# get the k-side-length window centered at current cell
                if self:	# if the current cell is nonzero...                
                     if not np.all(window):	# ...and if any part of the window is zero...
                         M[i, j] = 1.	# ...this is a positive minimal image. If no other part of the window is zero, i.e. everything is nonzero, this is not a minimal image.
                else:		# if the current cell is zero...
                     if np.any(window):		# ...and if any part of the window is nonzero...
                         M[i, j] = -1.	# ...this is a negative minimal image. If no other part of the window is nonzero, i.e. everything is zero, this is not a minimal image.

            else:	# we are looking for strict minimal images
                if self:	# if the current cell is nonzero...
                    top5map[i, j] = 0.	# temporarily set the current cell to zero          
                    window = top5map[max(0, i - offset):min(r-1, i + offset)+1, max(0, j - offset):min(c-1, j + offset)+1]	# get the k-side-length window centered at current cell
                    if not np.any(window):	# ...and if no part of the window is nonzero...
                        M[i, j] = 1.	# ...this is a positive minimal image. If some part of the window is nonzero, i.e. a surrounding pixel is nonzero, this is not a minimal image.
                    top5map[i, j] = self	# reset current cell
                else:	# if the current cell is zero...
                    top5map[i, j] = 255.	# temporarily set the current cell to nonzero
                    window = top5map[max(0, i - offset):min(r-1, i + offset)+1, max(0, j - offset):min(c-1, j + offset)+1]	# get the k-side-length window centered at current cell
                    if np.all(window):		# ...and if the entire window is nonzero...
                        M[i, j] = -1.	# ...this is a negative minimal image. If some part of the window is zero, i.e. a surrounding pixel is zero, this is not a minimal image.
                    top5map[i, j] = self	# reset current cell

    #  save map
    if loose:
        np.save(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '_lmap.npy', M)
    else:
        np.save(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '_map.npy', M)

    # calculate map statistics
    num_pos_min_imgs = (M > 0.).sum()
    num_neg_min_imgs = (M < 0.).sum()

    return num_pos_min_imgs/float(M.size), num_neg_min_imgs/float(M.size)


BBX_FILE = 'ILSVRC2012_val_bbx_dimensions.json'


def get_crop_size(smalldataset_id, crop_metric):
    imagenetval_id = settings.convert_id_small_to_imagenetval(smalldataset_id)
    image_tag = settings.get_ind_name(imagenetval_id)
    image_filename = PATH_TO_DATA + settings.folder_name('img') + image_tag + '.JPEG'
    im = Image.open(image_filename)
    width, height = im.size
    crop_type = 'proportional' if crop_metric <= 1. else 'constant'
    crop_size = c_m_p.get_crop_size(height, crop_metric, crop_type) if height <= width else c_m_p.get_crop_size(width, crop_metric, crop_type)
    return crop_size


def minimal_image_distribution(num_imgs, crop_metric, model_name, image_scale, loose):

    resize_dim = 150
    minimal_image_aggregation = np.zeros((resize_dim, resize_dim))

    img_ids = random.sample(range(100), num_imgs)     # for testing on my machine: only a subset of the maps. TODO remove for full job

    for smalldataset_id in img_ids:

        # get bbx dimensions
        imagenetval_id = settings.convert_id_small_to_imagenetval(smalldataset_id)
        image_tag = settings.get_ind_name(imagenetval_id)
        with open(BBX_FILE, 'r') as bbx_file:
            all_bbxs = json.load(bbx_file)
            crop_dims = [bbx[0] for bbx in all_bbxs[image_tag]]     # get all x1, y1, x2, y2 crops

        minimal_map_f = PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, smalldataset_id)
        minimal_map_f = minimal_map_f + '_' + ('l' if loose else '') + 'map'
        minimal_map = np.load(minimal_map_f + '.npy')

        image_filename = PATH_TO_DATA + settings.folder_name('img') + image_tag + '.JPEG'
        try:                                    # for testing on my machine: if the image is not on my machine, move on. TODO remove for full job
            im = Image.open(image_filename)
        except OSError:
            continue
        width, height = im.size
        crop_type = 'proportional' if crop_metric <= 1. else 'constant'
        crop_size = c_m_p.get_crop_size(height, crop_metric, crop_type) if height <= width else c_m_p.get_crop_size(width, crop_metric, crop_type)
        for x1, y1, x2, y2 in crop_dims:
            minmap_sub = minimal_map[y1:y2 - crop_size + 1, x1:x2 - crop_size + 1]
            minmap_sub = imresize(minmap_sub, (resize_dim, resize_dim))
            minimal_image_aggregation += minmap_sub

    vis = (minimal_image_aggregation - np.min(minimal_image_aggregation))
    vis /= np.max(vis)
    vis *= 255.
    Image.fromarray(vis).show()

    return minimal_image_aggregation


def minmap_bbx_mask(minmap_shape, bbx_dims, crop_size):

    '''
    returns mask of minmap shape with 1s in bbx pixels, 0s elsewhere
    '''

    mask = np.zeros(minmap_shape)
    for x1, y1, x2, y2 in bbx_dims:
        mask[y1:y2 - crop_size + 1, x1:x2 - crop_size + 1] = 1.
    return mask


def total_min_imgs(minmap):
  
    '''
    returns size (3,) array with num general min imgs, num pos min imgs, num neg min imgs
    '''
   
    return np.array([np.sum(minmap != 0), np.sum(minmap > 0.), np.sum(minmap < 0.)])


def percent_min_img_in_bbx(crop_metric, model_name, image_scale, loose, axis):
    '''
    defaults to calculating for all images 
    loose: boolean indicating loose if True else strict
    axis: 'shift' or 'scale'

    returns a matrix shape (500, 3), where row i is a three-array: the percentage of min imgs in bbx for image i, the percentage of positive min imgs in bbx for image i, the percentage of negative min imgs in bbx for image i 
    '''
   
    smalldataset_ids = range(settings.SMALL_DATASET_SIZE)
    with open(BBX_FILE, 'r') as bbx_file:
        all_bbxs = json.load(bbx_file)

    result = np.zeros((settings.SMALL_DATASET_SIZE, 3))

    for smalldataset_id in smalldataset_ids: 
        if smalldataset_id in settings.INTRACTABLE_IMAGES:
            continue 
       
        # get minimal image map 
        minimal_map_filename = PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, smalldataset_id)
        minimal_map_filename = minimal_map_filename + '_' + ('small_' if axis == 'scale' else '') + ('l' if loose else '') + 'map'
        minmap = np.load(minimal_map_filename + '.npy')

        # get total min img counts - total general, total positive, total negative
        totals = total_min_imgs(minmap)

        # construct mask of minmap size with 1s in bbx regions, 0s otherwise, mask minmap so that minimal images outside the bbx region look like non-minimal images
        bbx_dims = settings.get_bbx_dims(all_bbxs, smalldataset_id)
        crop_size = get_crop_size(smalldataset_id, crop_metric) 
        bbx_region_mask = minmap_bbx_mask(minmap.shape, bbx_dims, crop_size)
        bbx_minmap = minmap * bbx_region_mask
        
        # get min img counts within bbx regions - total general, total positive, total negative
        bbx_minimgs = np.array([np.sum(bbx_minmap != 0.), np.sum(bbx_minmap > 0.), np.sum(bbx_minmap < 0.)])
        
        # get percentages
        percentages = bbx_minimgs / totals 

        result[smalldataset_id] = percentages

    np.save('percent-min-img-in-bbx.npy', result)

    return result

    
def num_min_imgs_vs_bbx_coverage(crop_metric, model_name, image_scale, loose, axis):
    
    '''
    defaults to calculating for all images     

    returns a dict mapping smalldataset_id to (proportion of image that is bbx, [num pos min imgs, num neg min imgs]) 
    '''        
    
    smalldataset_ids = range(settings.SMALL_DATASET_SIZE)
    with open(BBX_FILE, 'r') as bbx_file:
        all_bbxs = json.load(bbx_file)
    
    id_to_measurements = {} 
    for smalldataset_id in smalldataset_ids:
        if smalldataset_id in settings.INTRACTABLE_IMAGES:
            continue 
        
        # get minimal image map 
        minimal_map_filename = PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, smalldataset_id)
        minimal_map_filename = minimal_map_filename + '_' + ('small_' if axis == 'scale' else '') + ('l' if loose else '') + 'map'
        minmap = np.load(minimal_map_filename + '.npy')
        
        # get total min img counts - total general, total positive, total negative
        totals = [int(total) for total in total_min_imgs(minmap)]
      
        # get bbx mask, apply, and get proportion of image that is bbx
        bbx_dims = settings.get_bbx_dims(all_bbxs, smalldataset_id)
        crop_size = get_crop_size(smalldataset_id, crop_metric)
        bbx_region_mask = minmap_bbx_mask(minmap.shape, bbx_dims, crop_size)
        proportion = np.sum(bbx_region_mask) / float(minmap.size)

        # map smalldataset_id to measurements
        id_to_measurements[smalldataset_id] = (proportion, totals)

    with open('id-to-measurements.json', 'w') as writefile:
        json.dump(id_to_measurements, writefile)

    return id_to_measurements


if __name__ == '__main__':
    percent_min_img_in_bbx(float(sys.argv[1]), sys.argv[2], float(sys.argv[3]), bool(sys.argv[4]), sys.argv[5])
    num_min_imgs_vs_bbx_coverage(float(sys.argv[1]), sys.argv[2], float(sys.argv[3]), bool(sys.argv[4]), sys.argv[5])
    
        
 



# results = - np.ones([ 4, 2, 5, 500, 2])
#
# image_scale = '1.0'
#
# model_name = sys.argv[1:][0]
#
# for idx_metric, crop_metric in enumerate([0.2, 0.4, 0.6, 0.8]):
#     print(idx_metric)
#     sys.stdout.flush()
#     for idx_loose, loose in enumerate([False, True]):
#         for idx_k, k in enumerate([3]):#enumerate([3, 5, 7, 11, 17]):
#             print(k)
#             sys.stdout.flush()
#             for image_id in range(500):
#                 print(image_id)
#                 sys.stdout.flush()
#
#                 a, b = \
#                     create_location_minimal_image_maps(image_id, crop_metric, model_name, image_scale, loose, k)
#                 results[idx_metric][idx_loose][idx_k][image_id][0] = a
#                 results[idx_metric][idx_loose][idx_k][image_id][1] = b
#
#         #np.save('tmp_results_' + model_name +'.npy', results)




