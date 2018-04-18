import numpy as np

import settings


PATH_TO_DATA = None # TODO 


def create_location_minimal_image_maps(image_id, crop_metric, model_name, image_scale, loose, k=1):

    '''
    image_id (int): the small dataset id of the image we are finding minimal images for 
    crop_metric (float): the crop metric we are referencing
    model_name (string): the model that we are referencing
    image_scale (float): the image scale we are referencing
    loose (bool): loose minimal images if True else strict minimal images
    k (int): the square size that we are looking for minimal image change within; should be even
    '''

    top5map = np.load(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id))
    r, c = top5map.shape
    
    M = np.zeros((r, c))

    for i in range(r):
        for j in range(c):
            offset = int(k / 2)
            self = top5map[i, j]

            # make minimal image map 
            if loose:
                window = top5map[max(0, i - offset):min(r, i + offset), max(0, j - offset):min(c, j + offset)]	# get the k-side-length window centered at current cell
                if self:	# if the current cell is nonzero...                
                     if not np.all(window):	# ...and if any part of the window is zero...
                         M[i, j] = 1.	# ...this is a positive minimal image. If no other part of the window is zero, i.e. everything is nonzero, this is not a minimal image.
                else:		# if the current cell is zero...
                     if np.any(window):		# ...and if any part of the window is nonzero...
                         M[i, j] = -1.	# ...this is a negative minimal image. If no other part of the window is nonzero, i.e. everything is zero, this is not a minimal image.

            else:	# we are looking for strict minimal images
                if self:	# if the current cell is nonzero...
                    top5map[i, j] = 0.	# temporarily set the current cell to zero          
                    window = top5map[max(0, i - offset):min(r, i + offset), max(0, j - offset):min(c, j + offset)]	# get the k-side-length window centered at current cell
                    if not np.any(window):	# ...and if no part of the window is nonzero...
                        M[i, j] = 1.	# ...this is a positive minimal image. If some part of the window is nonzero, i.e. a surrounding pixel is nonzero, this is not a minimal image.
                    top5map[i, j] = self	# reset current cell
                else:	# if the current cell is zero...
                    top5map[i, j] = 255.	# temporarily set the current cell to nonzero
                    window = top5map[max(0, i - offset):min(r, i + offset), max(0, j - offset):min(c, j + offset)]	# get the k-side-length window centered at current cell
                    if np.all(window):		# ...and if the entire window is nonzero...
                        M[i, j] = -1.	# ...this is a negative minimal image. If some part of the window is zero, i.e. a surrounding pixel is zero, this is not a minimal image.
                    top5map[i, j] = self	# reset current cell

    # TODO save map
    
    # calculate map statistics
    num_pos_min_imgs = (M > 0.).sum()
    num_neg_min_imgs = (M < 0.).sum()
 
    return num_pos_min_imgs/float(M.size), num_neg_min_imgs/float(M.size)


def create_size_minimal_image_maps(image_id, crop_metric, model_name, image_scale, loose):

    l_top5 = np.load(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id))
    s_top5 = np.load(PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '_small')
    
    r, c = l_top5.shape
    M = zeros((r, c))
    
    for i in range(r):
        for j in range(c):
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
    
    # TODO save map 

    # calculate map statistics
    num_pos_min_imgs = (M > 0.).sum()
    num_neg_min_imgs = (M < 0.).sum()
    return num_pos_min_imgs/float(M.size), num_neg_min_imgs/float(M.size) 





