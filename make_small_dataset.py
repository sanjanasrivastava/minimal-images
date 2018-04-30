from caffe_classes import class_names

text_to_small_label = {'dog': 0, 'snake': 1, 'monkey': 2, 'fish': 3, 'vegetable': 4, 'instrument': 5, 'boat': 6, 'land vehicle': 7, 'drinks': 8, 'furniture': 9}
small_label_to_imagenet_label = {0: [153, 157, 158, 163, 187, 204, 207, 219, 232, 236],
                                 1: [52,  55,  58,  63,  65,  67,  66,  68,  60,  56],
                                 2: [370, 372, 373, 377, 380, 382, 381, 374, 378, 379],
                                 3: range(391, 398),
                                 4: [937, 938, 939, 942, 945, 944, 943, 940, 936, 947],
                                 5: [683, 684, 776, 401, 402, 589, 881, 546, 558, 875],
                                 6: [554, 625, 693, 814, 472, 576, 724, 628],
                                 7: [466, 829, 864, 555, 569, 575, 717, 867, 436, 565],
                                 8: [966, 967, 969, 440],
                                 9: [610, 846, 423, 559, 765, 532, 736]}

def make_data():
    
    with open('caffe_ilsvrc12/val.txt', 'r') as val_file:
        val_list = [line.split() for line in val_file.readlines()]
        val_imgs = {imgid: classid for imgid, classid in val_list}

    with open('small-dataset-to-imagenet.txt', 'w') as datafile:
        for classlabel in small_label_to_imagenet_label:		# for each of the small dataset's labels...
            options = small_label_to_imagenet_label[classlabel]		# get the acceptable imagenet class labels 
            counter = 0     
            for imgid, classid in val_list:				# for each image in val.txt...
                if counter >= 50:
                    break
                elif int(classid) in options:				# if its imagenet class label is in the acceptable list, write it t0 the data file
                    counter += 1
                    datafile.write(imgid + ' ' + classid + ' ' + str(classlabel) + '\n')


make_data()
            
