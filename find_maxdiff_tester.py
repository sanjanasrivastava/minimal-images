import json
import numpy as np
import os
from PIL import Image
from scipy.misc import imread, imresize, imshow
import tensorflow as tf

import settings
from minimal_image_statistics import get_crop_size

PATH_TO_DATA = settings.MIN_IMGS_PATH_TO_DATA


def get_maxdiff_coordinates(start_id, end_id, crop_metric, model_name, image_scale, axis, compare_corr=True):

    # for all smalldataset images start_id through end_id, return coordinates of maximally different crops

    crop_type = 'proportional' if crop_metric <= 1. else 'constant'
    maxdiff_coordinates = {}

    for smalldataset_id in range(start_id, end_id + 1):

        if axis == 'scale':         # scale (size)-based minimal images
            sfxs = ['', '_small']
            large_size = get_crop_size(smalldataset_id, crop_metric)
            small_size = large_size - 2

            # Get correctness maps and check if diffcor possible
            corr_fns = (PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, smalldataset_id) + sfx + '.npy' for sfx in sfxs)
            lcor, scor = cor_maps = [np.load(corr_fn) for corr_fn in corr_fns]
            if compare_corr:
                for cor_map in cor_maps:
                    if not cor_map.any():
                        print('%s has no correctly classified crops.' % smalldataset_id)
                        continue
                    elif cor_map.all():
                        print('%s has only correctly classified crops.' % smalldataset_id)
                        continue

            # Get confidence maps
            con_fns = (PATH_TO_DATA + settings.map_filename(settings.CONFIDENCE_MAPTYPE, crop_metric, model_name, image_scale, smalldataset_id) + sfx + '.npy' for sfx in sfxs)
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
                    break
                else:	# if that location wasn't diffcorr, set the diff's entry to -2.
                    diffs[max_dir][lcell] = -2.

            maxdiff_coordinates[smalldataset_id] = (lcell, scell, lcon[lcell] - scon[scell])

    return maxdiff_coordinates


if __name__ == '__main__':
    coords = get_maxdiff_coordinates(0, 499, 0.2, 'resnet', 1.0, 'scale')
    top_results = sorted(coords, reverse=True, key=lambda x: coords[x][2])[:50]          # sort entries of coords by conf diff
    print(top_results)
