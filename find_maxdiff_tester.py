import numpy as np
import os
from PIL import Image
import pprint

import settings

from minimal_image_statistics import get_crop_size

PATH_TO_DATA = settings.MIN_IMGS_PATH_TO_DATA
PATH_TO_OUTPUT_DATA = '/om/user/sanjanas/min-img-data/'


def get_maxdiff_coordinates(start_id, end_id, crop_metric, model_name, image_scale, axis, compare_corr=True):

    # for all smalldataset images start_id through end_id, return coordinates of maximally different crops

    print('ENTERED COORDS FUNCTION')

    crop_type = 'proportional' if crop_metric <= 1. else 'constant'
    maxdiff_coordinates = {}

    print('ENTERING FOR LOOP')

    for smalldataset_id in range(start_id, end_id + 1):
        print('CURRENTLY:', smalldataset_id)

        if axis == 'scale':         # scale (size)-based minimal images
            sfxs = ['', '_small']
            large_size = get_crop_size(smalldataset_id, crop_metric)
            small_size = large_size - 2

            # Get correctness maps and check if diffcor possible
            corr_fns = (PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, smalldataset_id) + sfx + '.npy' for sfx in sfxs)
            try:
                lcor, scor = cor_maps = [np.load(corr_fn) for corr_fn in corr_fns]
            except FileNotFoundError:
                continue
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

                print('ENTERED WHILE LOOP')

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

        else:           # shift-based maxdiff

            correctness_filename = PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, smalldataset_id) + '.npy'
            try:
                cor_map = np.load(correctness_filename)
            except FileNotFoundError:
                continue
            if compare_corr:
                if not cor_map.any():
                    print('%s has no correctly classified crops.' % smalldataset_id)
                    continue
                elif cor_map.all():
                    print('%s has only correctly classified crops.' % smalldataset_id)
                    continue

            con_map_filename = PATH_TO_DATA + settings.map_filename(settings.CONFIDENCE_MAPTYPE, crop_metric, model_name, image_scale, smalldataset_id) + '.npy'
            con_map = np.load(con_map_filename)

            down_diff = np.diff(con_map, axis=0)	# apparently assumes step_size=1 (adjacency)
            up_diff = -1. * down_diff
            right_diff = np.diff(con_map)
            left_diff = -1. * right_diff
            diffs = {'up': up_diff, 'down': down_diff, 'left': left_diff, 'right': right_diff}

            while True:

                print('ENTERED WHILE LOOP')

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
                if diff_correctness or not compare_corr:
                    break

                else:
                    if max_dir in ['up', 'left']:
                        diffs[max_dir][gcell] = -2.	# for the diff where that was the argmax, mark the cell containing it to something lower than any real entry (-1. <= real entry <= 1.) This is gcell for up, left and cell for down, right because the lower-indexed cell is always the one that contained the confidence originally
                    else:
                        diffs[max_dir][cell] = -2.

            maxdiff_coordinates[smalldataset_id] = (gcell, cell, con_map[gcell] - con_map[cell])

    print('COORDS COMPLETE')

    return maxdiff_coordinates


def save_crops(coords, crop_metric, model_name, image_scale, axis, compare_corr, num_samples=50):

    folder = PATH_TO_OUTPUT_DATA + settings.maxdiff_folder_name(axis, crop_metric, model_name, image_scale, 'diff' if compare_corr else 'any')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # top_ids = sorted(coords, reverse=True, key=lambda x: coords[x][2])[num_samples:num_samples * 2]

    # print(top_ids)

    for smalldataset_id in coords:

        print('CURRENTLY SAVING:', smalldataset_id)

        im_filename = PATH_TO_DATA + settings.folder_name('img') + settings.get_ind_name(settings.convert_id_small_to_imagenetval(smalldataset_id)) + '.JPEG'
        im = Image.open(im_filename)
        hcell, lcell = coords[smalldataset_id][:2]
        hfn, lfn = (PATH_TO_OUTPUT_DATA + settings.maxdiff_file_name(smalldataset_id, axis, crop_metric, model_name, image_scale, 'diff' if compare_corr else 'any', conf) for conf in ['high', 'low'])

        if axis == 'scale':
            high_size = get_crop_size(smalldataset_id, crop_metric)
            low_size = high_size - 2
            hcrop = im.crop((hcell[0], hcell[1], hcell[0] + high_size, hcell[1] + high_size))
            lcrop = im.crop((lcell[0], lcell[1], lcell[0] + low_size, lcell[1] + low_size))

        elif axis == 'shift':
            size = get_crop_size(smalldataset_id, crop_metric)
            hcrop = im.crop((hcell[0], hcell[1], hcell[0] + size, hcell[1] + size))
            lcrop = im.crop((lcell[0], lcell[1], lcell[0] + size, lcell[1] + size))

        print(hfn)
        print(lfn)

        hcrop.save(hfn, 'JPEG')
        lcrop.save(lfn, 'JPEG')

    print('SAVES COMPLETE')


def get_human_maxdiff_coordinates(crop_metric, model_name, image_scale, compare_corr=True):

    # for all human images, return coordinates of maximally different crops (shift)

    PATH_TO_DATA = '/om/user/sanjanas/min-img-data/human-image-comparison/'

    crop_type = 'proportional' if crop_metric <= 1. else 'constant'
    maxdiff_coordinates = {}

    for image_id in range(50002, 50009):

        correctness_filename = PATH_TO_DATA + settings.map_filename(settings.TOP5_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '.npy'
        try:
            cor_map = np.load(correctness_filename)
        except FileNotFoundError:
            continue
        if compare_corr:
            if not cor_map.any():
                print('%s has no correctly classified crops.' % image_id)
                continue
            elif cor_map.all():
                print('%s has only correctly classified crops.' % image_id)
                continue

        con_map_filename = PATH_TO_DATA + settings.map_filename(settings.CONFIDENCE_MAPTYPE, crop_metric, model_name, image_scale, image_id) + '.npy'
        con_map = np.load(con_map_filename)

        down_diff = np.diff(con_map, axis=0)	# apparently assumes step_size=1 (adjacency)
        up_diff = -1. * down_diff
        right_diff = np.diff(con_map)
        left_diff = -1. * right_diff
        diffs = {'up': up_diff, 'down': down_diff, 'left': left_diff, 'right': right_diff}

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
            if diff_correctness or not compare_corr:
                break

            else:
                if max_dir in ['up', 'left']:
                    diffs[max_dir][gcell] = -2.	# for the diff where that was the argmax, mark the cell containing it to something lower than any real entry (-1. <= real entry <= 1.) This is gcell for up, left and cell for down, right because the lower-indexed cell is always the one that contained the confidence originally
                else:
                    diffs[max_dir][cell] = -2.

        maxdiff_coordinates[image_id] = (gcell, cell, con_map[gcell] - con_map[cell])

    return maxdiff_coordinates


def save_human_crops(coords, crop_metric, model_name, image_scale, compare_corr, num_samples=50):

    PATH_TO_DATA = '/om/user/sanjanas/min-img-data/human-image-comparison/'
    PATH_TO_OUTPUT_DATA = PATH_TO_DATA
    axis = 'shift'

    folder = PATH_TO_OUTPUT_DATA + settings.maxdiff_folder_name(axis, crop_metric, model_name, image_scale, 'diff' if compare_corr else 'any')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # top_ids = sorted(coords, reverse=True, key=lambda x: coords[x][2])[num_samples:num_samples * 2]

    for image_id in range(50002, 50009):

        if image_id in coords:

            im_filename = PATH_TO_DATA + 'human-images/' + settings.get_ind_name(image_id) + '.png'
            im = Image.open(im_filename)
            hcell, lcell = coords[image_id][:2]
            hfn, lfn = (PATH_TO_OUTPUT_DATA + settings.maxdiff_file_name(image_id, axis, crop_metric, model_name, image_scale, 'diff' if compare_corr else 'any', conf) for conf in ['high', 'low'])

            width, height = im.size
            dimension = height if height <= width else width
            size = int(((crop_metric * dimension) // 2) * 2 + 1)
            hcrop = im.crop((hcell[0], hcell[1], hcell[0] + size, hcell[1] + size))
            lcrop = im.crop((lcell[0], lcell[1], lcell[0] + size, lcell[1] + size))

            hcrop.save(hfn, 'PNG')
            lcrop.save(lfn, 'PNG')


if __name__ == '__main__':
    # coords = get_maxdiff_coordinates(0, 499, 0.2, 'resnet', 1.0, 'scale')
    # top_results = sorted(coords, reverse=True, key=lambda x: coords[x][2])[:10]          # sort entries of coords by conf diff
    # pp = pprint.PrettyPrinter()
    # pp.pprint([coords[top_result] for top_result in top_results])
    #
    # save_crops(coords, 0.2, 'resnet', 1.0, 'scale', True)
    #
    # coords = get_maxdiff_coordinates(0, 499, 0.2, 'resnet', 1.0, 'shift')
    # top_results = sorted(coords, reverse=True, key=lambda x: coords[x][2])[:10]          # sort entries of coords by conf diff
    # pp = pprint.PrettyPrinter()
    # pp.pprint([coords[top_result] for top_result in top_results])
    #
    # save_crops(coords, 0.2, 'resnet', 1.0, 'shift', True)

    # coords = get_maxdiff_coordinates(0, 30, 0.6, 'resnet', 1.0, 'shift')
    # save_crops(coords, 0.6, 'resnet', 1.0, 'shift', True)

    # print('IN SCRIPT')
    #
    # coords = get_maxdiff_coordinates(0, 249, 0.6, 'resnet', 1.0, 'scale')
    #
    # print('COMPLETED COORDS')
    #
    # save_crops(coords, 0.6, 'resnet', 1.0, 'scale', True)
    #
    # print('COMPLETED SCRIPT')
    sample_ids = [363, 220, 162, 151, 474, 203, 176, 469, 112, 401, 389, 452, 45, 342, 206, 134, 166, 200, 242, 58, 54, 460, 350, 319, 292, 237, 433, 352, 21, 344]
    for small_id in sample_ids:
        coords = get_maxdiff_coordinates(small_id, small_id, 0.6, 'resnet', 1.0, 'scale')
        save_crops(coords, 0.6, 'resnet', 1.0, 'scale', True)


