import json
# import matlab.engine
import os
from PIL import Image, ImageDraw, ImageFilter
import random
import sys
import xml.etree.ElementTree as ET

import settings


DATA_PATH = '../poggio_urop/poggio-urop-data/'
print sys.argv
if sys.argv[0] == 'image_alterations.py':   # if the python script is being run directly
    COMMAND_PATH = ''
else:
    COMMAND_PATH = 'poggio_urop_code/vgg_tensorflow/'   # path from main shell script to here, to be attached to anything this file references
BOUNDINBOX_FILE = settings.IMAGENET_SET + '_' + settings.DATASET + '_bbx_dimensions.json'


## GENERAL FUNCTIONS

def folder_name(datatype):
    '''
    Parameters
    ----------
    datatype (str): the type of data, e.g. crop or backblk

    Returns
    -------
    folder name with command path and data path attached, because all functions here deal with data files
    '''

    return DATA_PATH + settings.folder_name(datatype)


def fill_datatype(num_images, datatype):
    # Check if you already have the datatype you want, and if you have enough

    folder = folder_name(datatype)
    print folder
    num_images = int(num_images)

    if not os.path.exists(folder):
        print 'IT DIDNT EXIST'
        os.mkdir(folder)
    else:
        print 'IT EXISTED'
    num_existing_images = len(os.listdir(folder))   # TODO this is wrong now because of multiple crops

    if num_existing_images < num_images:
        num_extra_images = num_images - num_existing_images
        startpoint = num_existing_images + 1
    else:
        return

    for bbxtype in settings.BBX_DATATYPES:
        if bbxtype in datatype: # i.e. equivalent or that one case where it's like, is backnoise in it
            collect_all_boundinbox_data(num_existing_images, startpoint=startpoint)
            break

    if datatype == 'img':   # just here for record - this wouldn't ever need to be used. img is always already there.
        pass
    elif datatype == 'crop':
        collect_all_crops(num_extra_images, startpoint=startpoint)
    elif datatype == 'backblk':
        collect_all_blackouts(num_extra_images, startpoint=startpoint)
    elif datatype[:9] == 'backnoise':   # this better follow the 'backnoise####' convention
        collect_all_blur(num_extra_images, int(datatype[9:]), startpoint=startpoint)
    elif datatype == 'gbvs':
        collect_all_gbvs(num_extra_images, startpoint=startpoint)
    elif datatype == 'itti':
        collect_all_itti(num_extra_images, startpoint=startpoint)
    else:
        print 'Please enter a valid datatype.'


## BOUND-IN BOXES

def get_boundinbox(img_name, dimensions_dict):
    '''
    Parameters
    ----------
    img_name (str): ImageNet image identifier
    dimensions_dict (dict): TODO

    Returns
    -------
    TODO
    '''

    tree = ET.parse(folder_name('bbx') + img_name + '.xml')
    root = tree.getroot()

    # Get image name
    filename = root.find('filename').text

    # Get dimensions
    objects = root.findall('object')

    dimensions_dict[filename] = []
    for object in objects:
        dims = tuple(map(lambda x: int(x.text), list(object.find('bndbox'))))
        dimensions_dict[filename].append((dims, object.find('name').text))

    return dimensions_dict[filename]


def get_class(img_name):

    tree = ET.parse(folder_name('bbx') + '/' + img_name + '.xml')
    root = tree.getroot()

    filename = root.find('filename').text
    objects = root.find('object')

    # TODO get classnames


# def get_all_names():
#     """
#     Returns
#     -------
#     names for a dataset's images
#     """
#
#     # TODO correct all these
#     if DATASET == 'train':
#         bounds = range(1, 2)
#     elif DATASET == 'train_t3':
#         bounds = range(1, 2)
#     elif DATASET == 'val':
#         bounds = range(1, 1001)
#     elif DATASET == 'test':
#         bounds = range(1, 2)
#     else:
#         return 'Invalid dataset'
#
#     names = []
#     for i in bounds:
#         names.append(IMAGENET_SET + '_' + DATASET + '_' + '%08d' % i)
#
#     return names


def collect_all_boundinbox_data(num_images, startpoint=1):
    """
    make a json dump of all boundinbox data
    """

    with open(BOUNDINBOX_FILE) as infile:
        salient_regions = json.load(infile)
    names = settings.get_all_names(num_images, 'bbx', startpoint=startpoint)
    for name in names:
        # TODO I haven't actually confirmed that this is exactly correct for anything other than the validation
        get_boundinbox(name, salient_regions)

    with open(BOUNDINBOX_FILE, 'w') as outfile:
        json.dump(salient_regions, outfile)


## CROPS

def crop_box(img_name, datatype, folder, all_dims):
    """
    Parameters
    ----------
    img_name (str): image you want to crop
    datatype (str): type of crops you want to make
    folder (str): folder you want to put it in (probably based on datasets, I did this a while ago)
    all_dims (list of lists): all crops being made

    Returns
    -------
    dict mapping number of crop to tuple of dims

    Save the crop to the folder and map dims to crop id
    """

    im = Image.open(folder + '/' + img_name + '.JPEG')
    dim_ids = {}
    i = 1
    for dims in all_dims:
        cropped = im.crop(dims)
        cropped.save(folder_name(datatype) + img_name + '_crop' + str(i) + '.JPEG', 'JPEG')
        dim_ids[i] = tuple(dims)
        i += 1

    return dim_ids


def collect_all_crops(num_images, startpoint=1):
    """
    get all crops for the image set
    """

    infile = open(BOUNDINBOX_FILE)
    salient_regions = json.load(infile)

    names = settings.get_all_names(num_images, 'crop', startpoint=startpoint)
    for name in names:
        all_dims = map(lambda x: x[0], salient_regions[name])
        crop_box(name, 'crop', folder_name('img'), all_dims)


def draw_boundinbox(img_name, folder, all_dims):
    '''
    Parameters
    ----------
    img_name (str): image you want to crop
    folder (str): folder you want to get it from (probably based on datasets, I did this a while ago)
    all_dims (list of lists): all crops being made

    display boundinbox on image
    '''

    im = Image.open(folder + img_name + '.JPEG')
    draw = ImageDraw.Draw(im)
    for dims in all_dims:
        print dims
        draw.rectangle([(dims[0], dims[1]), (dims[2], dims[3])], outline="rgb(255,0,0)")

    im.show()


## BACKGROUND ISOLATION (BLACKED OUT BOUNDINBOX)

def blackout_boundinbox(img_name):
    """
    Parameters
    ----------
    img_name (str): name of image you want to blackout

    saves image with blacked-out boundinbox to correct folder
    """

    infile = open(BOUNDINBOX_FILE)
    salient_regions = json.load(infile)

    image_folder = folder_name('img')

    im = Image.open(image_folder + '/' + img_name + '.JPEG')
    all_dims = map(lambda x: x[0], salient_regions[img_name])
    draw = ImageDraw.Draw(im)

    for dims in all_dims:
        draw.rectangle(dims, fill='black')

    im.save(folder_name('backblk') + '/' + img_name + '.JPEG', 'JPEG')


def collect_all_blackouts(num_images, startpoint=1):
    """
    save all blackout boundinbox images
    """

    names = settings.get_all_names(num_images, 'backblk', startpoint=startpoint)
    for name in names:
        blackout_boundinbox(name)


## BACKGROUND ISOLATION (GAUSSIAN NOISE)

def blur_boundinbox(img_name, blur_level):
    """
    Parameters
    ----------
    img_name (str): name of image you want to blur

    saves image with gaussian-blurred boundinbox to correct folder
    """

    infile = open(BOUNDINBOX_FILE)
    salient_regions = json.load(infile)

    image_folder = folder_name('img')

    im = Image.open(image_folder + '/' + img_name + '.JPEG')
    all_dims = map(lambda x: x[0], salient_regions[img_name])
    for dims in all_dims:
        box = im.crop(dims)
        box = box.filter(ImageFilter.GaussianBlur(radius=blur_level))
        im.paste(box, dims)

    im.save(folder_name('backnoise' + str(blur_level)) + '/' + img_name + '.JPEG', 'JPEG')


def collect_all_blur(num_images, blur_level, startpoint=1):
    """
    save all blur images
    """

    names = settings.get_all_names(num_images, 'backnoise' + str(blur_level), startpoint=startpoint)
    for name in names:
        blur_boundinbox(name, blur_level)


## GBVS and ITTI

# def get_sal_map(img_name, map):
#
#     img_name = folder_name('img') + img_name + '.JPEG'
#     save_name = folder_name(map) + img_name + '.JPEG'
#     eng = matlab.engine.start_matlab()
#     save = eng.process_images(img_name, map, save_name)


def collect_all_gbvs(num_images, startpoint=1):

    eng = matlab.engine.start_matlab()
    names = settings.get_all_names(num_images, 'gbvs', startpoint=1)
    for name in names:
        img_name = folder_name('img') + name + '.JPEG'
        save_name = folder_name('gbvs') + name + '.JPEG'
        save = eng.process_images(img_name, 'gbvs', save_name)


def collect_all_itti(num_images, startpoint=1):

    eng = matlab.engine.start_matlab()
    names = settings.get_all_names(num_images, 'itti', startpoint=1)
    for name in names:
        img_name = folder_name('img') + name + '.JPEG'
        save_name = folder_name('itti') + name + '.JPEG'
        save = eng.process_images(img_name, 'itti', save_name)


## RANDOM CROPS

def random_crops(img_name, num_crops):

    image_folder = folder_name('img')
    im = Image.open(image_folder + '/' + img_name + '.JPEG')
    width, height = im.size

    beginning = 'ILSVRC2012_val_'
    end = '.JPEG'

    image_id = int(img_name[len(beginning):-1 * len(end)])

    crop_max_width = width/2
    crop_max_height = height/2

    all_dims = []

    for __ in xrange(num_crops):

        xmin = 0
        xmax = 0
        ymin = 0
        ymax = 0
        while xmax - xmin < settings.CROP_MIN_WIDTH \
                or ymax - ymin < settings.CROP_MIN_HEIGHT:
                #or xmax - xmin > crop_max_width \
                #or ymax - ymin > crop_max_height:
            xmin = random.randint(0, width)
            xmax = random.randint(xmin, width)
            ymin = random.randint(0, height)
            ymax = random.randint(ymin, height)

        all_dims.append([xmin, ymin, xmax, ymax])

    dim_ids = crop_box(img_name, 'randind' + '%08d' % img_id, folder_name('img'), all_dims)

    with open(COMMAND_PATH + DATA_PATH + settings.IMAGENET_SET + '_randind_results/' + img_name + '_' + 'dim_ids.json',
              'w') as dim_ids_file:    # 'w' because this is replaced with every trial
        json.dump(dim_ids, dim_ids_file)


def highlight_random_crops(img_id, visualization='black'):

    img_name = settings.get_ind_name(img_id)
    img_folder = folder_name('img')
    results_folder = COMMAND_PATH + DATA_PATH + settings.IMAGENET_SET + '_randind_results/'

    results_name = settings.get_random_crops_results_name(img_id)
    if results_name in os.listdir(results_folder):
        with open(results_folder + results_name, 'r') as results_file:
            results = json.load(results_file)

        all_dims = []
        for dims in results:
            if results[dims]:   # if it was classified correctly
                all_dims.append(json.loads(dims))   # They were stringified, so load them

        draw_boundinbox(img_name, img_folder, all_dims)

    else:
        print "This image hasn't been randomly cropped yet."


def highlight_best_crop(img_id):

    img_name = settings.get_ind_name(img_id)
    img_folder = folder_name('img')
    results_folder = COMMAND_PATH + DATA_PATH + settings.IMAGENET_SET + '_randind_results/'

    with open(results_folder + 'best-crops.json', 'r') as crops_file:
        crops = json.load(crops_file)

    if str(img_id) in crops:
        draw_boundinbox(img_name, img_folder, [tuple(crops[str(img_id)][1])])
    else:
        print "This image hasn't been randomly cropped yet."


if __name__ == '__main__':

    # if COMMAND_PATH:    # If being called from the control script
    #
    #     if sys.argv[1] == 'test':
    #         num_images = sys.argv[2]
    #         datatype = sys.argv[3]
    #         fill_datatype(num_images, datatype)
    #
    #     elif sys.argv[1] == 'random':
    #
    #         os.mkdir(folder_name('randind'))
    #         img_id = int(sys.argv[2])
    #         img_name = settings.get_ind_name(img_id)
    #         random_crops(img_name, int(sys.argv[3]))
    #
    #     elif sys.argv[1] == 'randompt2':
    #
    #         highlight_random_crops(int(sys.argv[2]))
    #
    #     elif sys.argv[1] == 'best':
    #         for img_id in range(int(sys.argv[2]), int(sys.argv[3]) + 1):
    #             os.mkdir(folder_name('randind' + '%08d' % img_id))
    #             img_name = settings.get_ind_name(img_id)
    #             random_crops(img_name, int(sys.argv[4]))
    #
    # else:   # For testing
    #     pass

    # collect_all_boundinbox_data(50000)

    pass



