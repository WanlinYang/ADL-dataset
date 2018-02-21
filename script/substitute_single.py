from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re

"""
Take images and masks, and substitute with backgrounds.
We put one object in one image and use the orignal mask
rather than COCO-polygon mask.
output: only full-rgb images
"""

masks = {}
# {'frame_name': }

verbose = True

class Object_mask:
    def __init__(self, datadir):
        self.locpath = datadir
        self.croploc = [None]*4
        self.split_locpath = re.split('_|/', self.locpath)
        self.frame_name = self.split_locpath[-5] + '_' \
                + self.split_locpath[-4] + '_' \
                + self.split_locpath[-3] + '_' \
                + self.split_locpath[-2]

    def load_location(self):
        with open(self.locpath) as f:
            loc = f.readlines()
            self.croploc[0:2] = re.split(',|\n', loc[0])[0:2]
            self.croploc[2:4] = re.split(',|\n', loc[1])[0:2]
            self.croploc = [int(i) for i in self.croploc]
        self.width = self.croploc[2] - self.croploc[0]
        self.height = self.croploc[3] - self.croploc[1]
        self.x_start = self.croploc[0]
        self.y_start = self.croploc[1]

    def load_mask(self, datadir):
        self.maskpath = datadir
        self.maskcrop = cv2.imread(self.maskpath, cv2.IMREAD_GRAYSCALE)
#        self.full_mask = np.zeros((480,640), dtype=np.uint8)
#        for x in xrange(self.width):
#            for y in xrange(self.height):
#                self.full_mask[self.y_start + y, self.x_start + x] = \
#                        self.maskcrop[y, x]


def get_mask(datadir):
    """
    go through crop location and mask files
    """
    if verbose:
        print('going through locations ...')
    for root, dirs, files in os.walk(datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('loc.txt'):
                obj_mask = Object_mask(filepath)
                obj_mask.load_location()
                masks[obj_mask.frame_name] = obj_mask

    if verbose:
        print('going through masks ...')
    for root, dirs, files in os.walk(datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('maskcrop.png'):
                split_filepath = re.split('_|/', filepath)
                frame_name = split_filepath[-5] + '_' \
                        + split_filepath[-4] + '_' \
                        + split_filepath[-3] + '_' \
                        + split_filepath[-2]
                masks[frame_name].load_mask(filepath)

def get_background(bg_datadir):
    """
    output a dictionary containing background images
    """
    backgrounds = {}
    # {'deg' : 'cv2.image'}

    if verbose:
        print('getting backgrounds ...')
    for root, dirs, files in os.walk(bg_datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('30.png'):
                backgrounds['30deg'] = cv2.imread(filepath)
            elif filepath.endswith('45.png'):
                backgrounds['45deg'] = cv2.imread(filepath)
            elif filepath.endswith('60.png'):
                backgrounds['60deg'] = cv2.imread(filepath)
    return backgrounds

def substitute_background(obj_mask, full_img, output_img):

    x_start = obj_mask.x_start
    y_start = obj_mask.y_start

    width = obj_mask.width
    height = obj_mask.height

    maskcrop = obj_mask.maskcrop

    for x in xrange(width):
        for y in xrange(height):
            if maskcrop[y][x] != 0:
                output_img[y_start+y][x_start+x] = \
                        full_img[y_start+y][x_start+x]

def output_data(obj_datadir, bg_datadir, output_dir):
    """
    go through rgb images and substitute background
    """
    mask_datadir = obj_datadir + '/ADL_mask'
    rgb_datadir = obj_datadir + '/ADL_rgb'
    get_mask(mask_datadir)

    backgrounds = get_background(bg_datadir)

    if verbose:
        print('going through rgb and outputing rgb ...')
    for root, dirs, files in os.walk(rgb_datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('rgboriginal.png'):
                split_filepath = re.split('_|/', filepath)
                cat = split_filepath[-5]
                instance = split_filepath[-5] + '_' \
                        + split_filepath[-4]
                frame_name = split_filepath[-5] + '_' \
                        + split_filepath[-4] + '_' \
                        + split_filepath[-3] + '_' \
                        + split_filepath[-2]
                full_img = cv2.imread(filepath)
                obj_mask = masks[frame_name]
                angle_type = int(split_filepath[-3])


                if angle_type == 1:
                    output_img = backgrounds['30deg'].copy()
                    angletxt = '30deg'
                elif angle_type == 2:
                    output_img = backgrounds['45deg'].copy()
                    angletxt = '45deg'
                elif angle_type == 3:
                    output_img = backgrounds['60deg'].copy()
                    angletxt = '60deg'

                substitute_background(obj_mask, full_img, output_img)

                output_filedir = output_dir + '/' + cat + '/' \
                        + instance + '/' + angletxt
                if not os.path.exists(output_filedir):
                    os.makedirs(output_filedir)
                output_filepath = output_filedir + '/' + frame_name \
                        + '_rgboriginal.png'
                cv2.imwrite(output_filepath, output_img)

if __name__ == '__main__':

    obj_datadir = '/home/wanlin/Downloads'
    bg_datadir = '/home/wanlin/Pictures/background'
    output_dir = '/home/wanlin/Downloads/sub_single'

    output_data(obj_datadir, bg_datadir, output_dir)
