import pickle
import numpy as np
from scipy import spatial

import confidence_maps_parallel_tester as cmpt
import settings


PATH_TO_DATA = '../../poggio-urop-data/'
VGG_LAYERS = ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3', 'pool5']


def compare_stride(image_id, crop_metric, which_layers=VGG_LAYERS):
   
    with open('high-act-sample', 'rb') as readfile:
        h_act = pickle.load(readfile)
    with open('low-act-sample', 'rb') as readfile:
        l_act = pickle.load(readfile) 

    maxdiff_folder = settings.maxdiff_folder_name(crop_metric)
    hconfcrop = np.load(PATH_TO_DATA + maxdiff_folder + settings.get_ind_name(image_id) + '_' + str(crop_metric) + '_maxdiff_highconf_diffcorrectness.npy')
    lconfcrop = np.load(PATH_TO_DATA + maxdiff_folder + settings.get_ind_name(image_id) + '_' + str(crop_metric) + '_maxdiff_lowconf_diffcorrectness.npy')

    direction = None
    if np.array_equal(lconfcrop[:, 0, :], hconfcrop[:, 1, :]):
        direction = 'left'
    elif np.array_equal(lconfcrop[:, 1, :], hconfcrop[:, 0, :]):
        direction = 'right'
    elif np.array_equal(lconfcrop[0, :, :], hconfcrop[1, :, :]):
        direction = 'up'
    elif np.array_equal(lconfcrop[1, :, :], hconfcrop[0, :, :]):
        direction = 'down'
    if direction is None:
        print ('Finding direction failed.')
        return

    hlayers = dict(zip(VGG_LAYERS, [np.squeeze(mat) for mat in h_act]))
    llayers = dict(zip(VGG_LAYERS, [np.squeeze(mat) for mat in l_act]))
    hlayers = {k:v for k,v in hlayers.items() if k in which_layers}
    llayers = {k:v for k,v in llayers.items() if k in which_layers}

    differences = []

    for layer in which_layers:
        
        hlayer = hlayers[layer]
        llayer = llayers[layer]
        if direction == 'left':
            hcommon = hlayer[:, 1:, :]
            lcommon = llayer[:, :-1, :]
        elif direction == 'right':
            hcommon = hlayer[:, :-1, :]
            lcommon = llayer[:, 1:, :]
        elif direction == 'up':
            hcommon = hlayer[1:, :, :]
            lcommon = llayer[:-1, :, :]
        else:
            hcommon = hlayer[:-1, :, :]
            lcommon = llayer[1:, :, :]
     
        D = np.linalg.norm(hcommon - lcommon, axis=(0, 1))
        D2 = np.linalg.norm(hcommon - lcommon, axis=0)
        D3 = hcommon - lcommon

        np.save(str(image_id) + '_' + layer + '_high', hlayer)
        np.save(str(image_id) + '_' + layer + '_low', llayer)

if __name__ == '__main__':
    
    # Following five lines are only run if the tester activations need to be updated 
    '''
    h_act, l_act = cmpt.test_maxdiff_crops(1, 1, 0.2)
    with open('high-act-sample', 'wb') as writefile:
        pickle.dump(h_act[1], writefile)
    with open('low-act-sample', 'wb') as writefile:
        pickle.dump(l_act[1], writefile)
    '''
    compare_stride(1, 0.2, which_layers=['pool1', 'pool2', 'pool3', 'pool4', 'pool5'])
