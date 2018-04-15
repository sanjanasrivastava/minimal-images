#!/bin/bash

cut -d' ' -f1 small-dataset-to-imagenet.txt | while read i; do cp /cbcl/cbcl01/sanjanas/poggio_urop/poggio-urop-data/ILSVRC2012_img_val/$i /om/user/xboix/share/minimal-images/ilsvrc-dataset/;  done
