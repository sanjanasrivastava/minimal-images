import numpy as np
import settings
import sys

import os.path


PATH_TO_DATA = "/om/user/xboix/share/minimal-images/"
#""./backup/" #


def create_size_minimal_image_maps(image_id, crop_metric, model_name, image_scale, loose):

    fname = PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '.npy'
    if not os.path.isfile(fname):
        return -1, -1

    l_top5 = np.load(fname)

    fname = PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '_small.npy'
    if not os.path.isfile(fname):
        return -1, -1
    s_top5 = np.load(fname)

    r, c = l_top5.shape
    M = np.zeros((r, c))
    
    for i in range(r):
        for j in range(c):
            self = l_top5[i, j]

            window = s_top5[i:i + 3, j:j + 3]	# get all the possible shrinks for this crop
            if loose:
                if self:	# if the current crop is correctly classified...
                    if not np.all(window):	# if any cell in the window is incorrectly classified... 
                        M[i, j] = 1.	# ...the current crop is a positive minimal image. Otherwise, it's not minimal.
                else:		# if the current crop is incorrectly classified...
                    if np.any(window):		# if any cell in the window is correctly classified...
                        M[i, j] = -1.	# ...the current crop is a negative minimal image. Otherwise, it's not minimal.
            else:	# we are looking for strict minimal image maps 
                if self: 	# if the current crop is correctly classified...
                    if not np.any(window):	# if all crops in the window are incorrectly classified...
                        M[i, j] = 1.	# ...the current crop is a positive minimal image. Otherwise, it's not minimal. 
                else:		# if the current crop is incorrectly classified...
                    if np.all(window):	# if all the crops in the window are correctly classified...
                        M[i, j] = -1.	# ...the current crop is a negative minimal image. Otherwise, it's not minimal.
    
    #  save map
    if loose:
        np.save(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '_small_lmap.npy', M)
    else:
        np.save(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '_small_map.npy', M)

    # calculate map statistics
    num_pos_min_imgs = (M > 0.).sum()
    num_neg_min_imgs = (M < 0.).sum()
    return num_pos_min_imgs/float(M.size), num_neg_min_imgs/float(M.size) 

def create_human_size_minimal_image_maps(image_id, crop_metric, model_name, image_scale, loose):

    PATH_TO_DATA = '/om/user/sanjanas/min-img-data/'

    fname = PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '.npy'
    if not os.path.isfile(fname):
        return -1, -1

    l_top5 = np.load(fname)

    fname = PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '_small.npy'
    if not os.path.isfile(fname):
        return -1, -1
    s_top5 = np.load(fname)

    r, c = l_top5.shape
    M = np.zeros((r, c))

    for i in range(r):
        for j in range(c):
            self = l_top5[i, j]

            window = s_top5[i:i + 3, j:j + 3]	# get all the possible shrinks for this crop
            if loose:
                if self:	# if the current crop is correctly classified...
                    if not np.all(window):	# if any cell in the window is incorrectly classified...
                        M[i, j] = 1.	# ...the current crop is a positive minimal image. Otherwise, it's not minimal.
                else:		# if the current crop is incorrectly classified...
                    if np.any(window):		# if any cell in the window is correctly classified...
                        M[i, j] = -1.	# ...the current crop is a negative minimal image. Otherwise, it's not minimal.
            else:	# we are looking for strict minimal image maps
                if self: 	# if the current crop is correctly classified...
                    if not np.any(window):	# if all crops in the window are incorrectly classified...
                        M[i, j] = 1.	# ...the current crop is a positive minimal image. Otherwise, it's not minimal.
                else:		# if the current crop is incorrectly classified...
                    if np.all(window):	# if all the crops in the window are correctly classified...
                        M[i, j] = -1.	# ...the current crop is a negative minimal image. Otherwise, it's not minimal.

    #  save map
    if loose:
        np.save(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '_small_lmap.npy', M)
    else:
        np.save(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '_small_map.npy', M)

    # calculate map statistics
    num_pos_min_imgs = (M > 0.).sum()
    num_neg_min_imgs = (M < 0.).sum()
    return num_pos_min_imgs/float(M.size), num_neg_min_imgs/float(M.size)


results = - np.ones([ 4, 2, 7, 2])

image_scale = '1.0'

model_name = sys.argv[1:][0]

# for idx_metric, crop_metric in enumerate([0.2, 0.4, 0.6, 0.8]):
for idx_metric, crop_metric in enumerate([0.4]):
    print(idx_metric)
    sys.stdout.flush()
    for idx_loose, loose in enumerate([False, True]):
        for image_id in range(50002, 50008):
            print(image_id)
            sys.stdout.flush()

            a, b = \
                create_human_size_minimal_image_maps(image_id, crop_metric, model_name, image_scale, loose)
            results[idx_metric][idx_loose][image_id][0] = a
            results[idx_metric][idx_loose][image_id][1] = b

        np.save('tmp_results_' + model_name + '_small.npy', results)




