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

    def load_mask(self, datadir):
        self.maskpath = datadir
        self.maskcrop = cv2.imread(self.maskpath, cv2.IMREAD_GRAYSCALE)
        self.full_mask = np.zeros((480,640), dtype=np.uint8)
        self.x_start = self.croploc[0]
        self.y_start = self.croploc[1]
        for x in xrange(self.width):
            for y in xrange(self.height):
                self.full_mask[self.y_start + y, self.x_start + x] = \
                        self.maskcrop[y, x]


def get_mask(datadir):
    """
    go through crop location and mask files
    """
    for root, dirs, files in os.walk(datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('loc.txt'):
                obj_mask = Object_mask(filepath)
                obj_mask.load_location()
                masks[obj_mask.frame_name] = obj_mask

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

def output_data(obj_datadir, bg_datadir, output_dir):
    """
    go through rgb images and substitute background
    """
    mask_datadir = obj_datadir + '/crop'
    rgb_datadir = obj_datadir + '/full_rgb'
    get_mask(mask_datadir)

    bg_30 = []; bg_45 = []; bg_60 = []
    for root, dirs, files in os.walk(bg_datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('30.png'):
                bg_30 = cv2.imread(filepath)
            elif filepath.endswith('45.png'):
                bg_45 = cv2.imread(filepath)
            elif filepath.endswith('60.png'):
                bg_60 = cv2.imread(filepath)

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

                x_start = obj_mask.x_start; y_start = obj_mask.y_start
                width = obj_mask.width; height = obj_mask.height
                maskcrop = obj_mask.maskcrop

                if angle_type == 1:
                    output_img = bg_30.copy()
                    angletxt = '30deg'
                elif angle_type == 2:
                    output_img = bg_45.copy()
                    angletxt = '45deg'
                elif angle_type == 3:
                    output_img = bg_60.copy()
                    angletxt = '60deg'

                for x in xrange(width):
                    for y in xrange(height):
                        if maskcrop[y][x] != 0:
                            output_img[y_start+y][x_start+x] = \
                                    full_img[y_start+y][x_start+x]
                output_filedir = output_dir + '/' + cat + '/' \
                        + instance + '/' + angletxt
                if not os.path.exists(output_filedir):
                    os.makedirs(output_filedir)
                output_filepath = output_filedir + '/' + frame_name \
                        + '_rgboriginal.png'
                print(output_filepath)

if __name__ == '__main__':

    obj_datadir = '/home/wanlin/Pictures/sub_single'
    bg_datadir = '/home/wanlin/Pictures/sub_single/background'
    output_dir = '/home/wanlin/Pictures/sub_single'

    output_data(obj_datadir, bg_datadir, output_dir)
