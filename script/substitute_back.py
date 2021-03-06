from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import random
import json
import pickle

categories = [
    {'id': 1, 'name': 'cracker-box', 'supercategory': 'box','superid': 1},
    {'id': 2, 'name': 'can', 'supercategory': 'container','superid': 2},
    {'id': 3, 'name': 'bowl', 'supercategory': 'container','superid': 2},
    {'id': 4, 'name': 'plate', 'supercategory': 'container','superid': 2},
    {'id': 5, 'name': 'cup', 'supercategory': 'container','superid': 2},
    {'id': 6, 'name': 'plastic-tumbler', 'supercategory': 'container','superid': 2},
    {'id': 7, 'name': 'knife', 'supercategory': 'food-utensil','superid': 3},
    {'id': 8, 'name': 'spoon', 'supercategory': 'food-utensil','superid': 3},
    {'id': 9, 'name': 'fork', 'supercategory': 'food-utensil','superid': 3},
    #{'id':10, 'name': 'medicine-bottle', 'supercategory': 'medicine','superid': 2},
    {'id':11, 'name': 'towel', 'supercategory': 'scrub','superid': 4},
    {'id':12, 'name': 'dish-scouring-pad', 'supercategory': 'scrub','superid': 4},
    {'id':13, 'name': 'flashlight', 'supercategory': 'device','superid': 5},
    {'id':14, 'name': 'screwdriver', 'supercategory': 'tool','superid': 6},
    {'id':15, 'name': 'hammer', 'supercategory': 'tool','superid': 6},
    {'id':16, 'name': 'wrench', 'supercategory': 'tool','superid': 6},
    {'id':17, 'name': 'toothbrush', 'supercategory': 'hygiene-teeth','superid': 7},
    {'id':18, 'name': 'toothpaste', 'supercategory': 'hygiene-teeth','superid': 7},
    {'id':19, 'name': 'soap', 'supercategory': 'hygiene-body','superid': 8},
    {'id':20, 'name': 'handsoap', 'supercategory': 'hygiene-body','superid': 8},
    {'id':21, 'name': 'wallet', 'supercategory': 'dressing','superid': 9},
    {'id':22, 'name': 'hairbrush', 'supercategory': 'grooming','superid': 10},
    {'id':23, 'name': 'book', 'supercategory': 'printed-matter','superid': 11},
    {'id':24, 'name': 'pen', 'supercategory': 'writing','superid': 12},
    {'id':25, 'name': 'tape', 'supercategory': 'office-tool','superid': 13},
    {'id':26, 'name': 'stapler', 'supercategory': 'office-tool','superid': 13},
    {'id':27, 'name': 'headphone', 'supercategory': 'digital-device','superid': 14},
    {'id':28, 'name': 'mouse', 'supercategory': 'digital-device','superid': 14}
]

supercategories = [
    {'id': 1, 'name': 'box', 'supercategory': 'food-preparation'},
    {'id': 2, 'name': 'container', 'supercategory': 'food-preparation'},
    {'id': 3, 'name': 'food-utensil', 'supercategory': 'food-preparation'},
    {'id': 4, 'name': 'scrub', 'supercategory': 'housekeeping'},
    {'id': 5, 'name': 'device', 'supercategory': 'housekeeping'},
    {'id': 6, 'name': 'tool', 'supercategory': 'housekeeping'},
    {'id': 7, 'name': 'hygiene-teeth', 'supercategory': 'hygiene'},
    {'id': 8, 'name': 'hygiene-body', 'supercategory': 'hygiene'},
    {'id': 9, 'name': 'dressing', 'supercategory': 'hygiene'},
    {'id': 10,'name': 'grooming', 'supercategory': 'hygiene'},
    {'id': 11,'name': 'printed-matter', 'supercategory': 'office-tasks'},
    {'id': 12,'name': 'writing', 'supercategory': 'office-tasks'},
    {'id': 13,'name': 'office-tool', 'supercategory': 'office-tasks'},
    {'id': 14,'name': 'digital-device', 'supercategory': 'office-tasks'}
]

"""
This is the attempt of putting several objects that masked
out from the original dataset over several pre-labeled
candidate background
output: 480x640 image and COCO-format json file
"""
object_dict = {}
# {'frame_name': 'Object'}

category_ids = {}
# {'category', 'id'}

supercategory_ids = {}
# {'category':{'supercategory','superid'}}

super_category = True
test_data = False
verbose = True

CAPACITY = 6
# object capacity of one background image

class Object:
    def __init__(self, datadir):
        self.locpath = datadir
        self.croploc = [None]*4
        self.split_locpath = re.split('_|/', self.locpath)
        self.frame_name = get_frame_name(self.locpath)
        self.category = self.split_locpath[-5]
        self.supercategory = supercategory_ids[self.category]['supercategory']
        self.category_id = category_ids[self.category]
        self.supercategory_id = supercategory_ids[self.category]['superid']
        self.bbox = self.croploc_to_bbox()

    def croploc_to_bbox(self):
        with open(self.locpath) as f:
            loc = f.readlines()
            self.croploc[0:2] = re.split(',|\n', loc[0])[0:2]
            self.croploc[2:4] = re.split(',|\n', loc[1])[0:2]
            self.croploc = [int(i) for i in self.croploc]

        [x1, y1, x2, y2] = self.croploc
        width = x2 - x1
        height = y2 - y1
        x = x1
        y = y1
        return [x, y, width, height]

    def get_maskpath(self, maskpath):
        self.maskpath = maskpath

    def get_rgbpath(self, rgbpath):
        self.rgbpath = rgbpath

    def load_data(self):
        """ load cropped rgb data and cropped mask data """
        full_img = cv2.imread(self.rgbpath)

        ul_x = self.bbox[0]; ul_y = self.bbox[1]
        lr_x = ul_x + self.bbox[2]
        lr_y = ul_y + self.bbox[3]

        self.crop_loc = [ul_x, ul_y, lr_x, lr_y]
        self.cropped_img = full_img[ul_y:lr_y,ul_x:lr_x]

        self.cropped_mask = cv2.imread(self.maskpath, cv2.IMREAD_GRAYSCALE)

class Background:
    def __init__(self, annotation, bgdatadir):
        self.bgdatadir = bgdatadir # bg folder path
        self.bgname = annotation['name']
        self.img_path = bgdatadir + '/' + self.bgname + '.jpg'
        self.mask_path = bgdatadir + '/' + self.bgname + '.pkl'
        self.img = cv2.imread(self.img_path)
        self.scale_range = annotation['scale_range']
        self.coord_range = annotation['coord_range']

    def load_data(self):
        with open(self.mask_path, 'rb') as f:
            d = pickle.load(f)
        self.mask = d['mask']

def get_frame_name(filepath):
    split_filepath = re.split('_|/', filepath)
    frame_name = split_filepath[-5] + '_' \
            + split_filepath[-4] + '_' \
            + split_filepath[-3] + '_' \
            + split_filepath[-2]
    return frame_name

def get_category_id():
    global category_ids
    for cat in categories:
        cat_name = cat['name']
        cat_id = cat['id']
        category_ids[cat_name] = cat_id

    global supercategory_ids
    for cat in categories:
        cat_name = cat['name']
        supercat_name = cat['supercategory']
        supercat_id = cat['superid']
        supercategory_ids[cat_name] = {}
        supercategory_ids[cat_name]['superid'] = supercat_id
        supercategory_ids[cat_name]['supercategory'] = supercat_name

def traverse_datapath(rgb_datadir, mask_datadir):
    """
    traverse loc and mask, return a list of dict
    """
    global object_dict

    if verbose:
        print('going through locations ...')
    for root, dirs, files in os.walk(mask_datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('loc.txt'):
                split_filepath = re.split('_|/', filepath)
                cat = split_filepath[-5]

                if cat in category_ids:
                    frame_name = get_frame_name(filepath)
                    obj = Object(filepath)
                    object_dict[frame_name] = obj

    if verbose:
        print('going through masks ...')
    for root, dirs, files in os.walk(mask_datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('maskcrop.png'):
                frame_name = get_frame_name(filepath)
                if frame_name in object_dict:
                    object_dict[frame_name].get_maskpath(filepath)

    if verbose:
        print('going through rgbs ...')
    for root, dirs, files in os.walk(rgb_datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('rgboriginal.png'):
                frame_name = get_frame_name(filepath)
                if frame_name in object_dict:
                    object_dict[frame_name].get_rgbpath(filepath)

def put_objects(obj_list, bg):
    """
    obj_list: a list of Object
    bg: a Background
    """
    bg_img = bg.img
    bg_mask = bg.mask
    bg_scale_range = bg.scale_range
    [y_min, y_max, x_min, x_max] = bg.coord_range

    def pick_obj_pos(obj_bbox, scale):
        """
        pick random position in background for an object
        """
        [x, y, width, height] = obj_bbox
        width = int(scale * width)
        height = int(scale * height)
        if y+height >= 480 or x+width >= 640:
            return [-1, -1]
        bbox_area = width*height
        overlap_area = 0
        for i in xrange(y, y+height):
            for j in xrange(x, x+width):
                if bg_mask[i][j] == 1:
                    overlap_area += 1
        overlap_rate = float(overlap_area) / float(bbox_area)
        if overlap_rate >= 0.9:
            for i in xrange(y, y+height):
                for j in xrange(x, x+width):
                    bg_mask[i][j] = 0
            return [x, y]
        else:
            return [-1, -1]

    annotations = []

    for obj in obj_list:
        obj_bbox = obj.bbox
        obj_cropped_img = obj.cropped_img
        obj_cropped_mask = obj.cropped_mask
        [x, y] = [-1, -1]
        scale = 0

        for i in xrange(10):
            scale_cand = random.uniform(bg_scale_range[0], bg_scale_range[1])
            x_cand = random.randint(x_min, x_max)
            y_cand = random.randint(y_min, y_max)
            obj_bbox_cand = [x_cand, y_cand, \
                            obj_bbox[2], obj_bbox[3]]
            [x, y] = pick_obj_pos(obj_bbox_cand, scale_cand)
            if [x, y] != [-1, -1]:
                scale = scale_cand
                break

        if scale != 0:
            obj_scaled_img = cv2.resize(obj_cropped_img, None, \
                                       fx = scale, fy = scale)
            obj_scaled_mask = cv2.resize(obj_cropped_mask, None, \
                                        fx = scale, fy = scale)

            rows, cols, channel = obj_scaled_img.shape
            for i in xrange(rows):
                for j in xrange(cols):
                    if obj_scaled_mask[i][j] != 0:
                        bg_img[y+i][x+j] = obj_scaled_img[i][j]

            annot_dict = {}
            annot_dict['bbox'] = [x, y, cols, rows]
            annot_dict['category_id'] = obj.category_id
            if super_category:
                annot_dict['category_id'] = obj.supercategory_id
            annot_dict['segmentation'] = None
            annot_dict['area'] = None
            annotations.append(annot_dict)

    return bg_img, annotations


def output_data(rgb_datadir, mask_datadir, bg_datadir, output_dir, total):
    """
    put random number of objects into random background,
    and output total merged images and json file
    """

    output_img_dir = output_dir + '/ADL_train'
    if test_data:
        output_img_dir = output_dir + '/ADL_test'
    output_json_dir = output_dir + '/annotations'
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_json_dir):
        os.makedirs(output_json_dir)

    if verbose:
        print('loading background pickle ...')

    bg_annotations = []
    for root, dirs, files in os.walk(bg_datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    d = pickle.load(f)
                bg_annotations.append(d)

    traverse_datapath(rgb_datadir, mask_datadir)
    # traverse loc, mask and rgb, and save path into object_dict

    def prone_test_id_cands(output_dir):
        """ assume already has training data """
        train_json_path = output_dir + \
                '/annotations/cluster_train2018.json'
        with open(train_json_path, 'r') as f:
            train_data = json.load(f)
        num_imgs = 0
        for img in train_data['images']:
            num_imgs += 1
            train_id = img['id']
            image_id_cands.remove(train_id)
        return num_imgs

    image_id_cands = [i for i in xrange(10000)]
    start_id = 0
    if test_data:
        start_id = prone_test_id_cands(output_dir)
    data = {}
    data['images'] = []
    data['annotations'] = []
    data['categories'] = categories
    if super_category:
        data['categories'] = supercategories

    if verbose:
        print('producing and saving cluster images ...')

    for total_i in xrange(start_id, start_id + total):
        image_id = random.choice(image_id_cands)
        image_id_cands.remove(image_id)
        image_name = 'ADL2018_cluster_' + str(total_i) + '.png'

        bg_annotation = random.choice(bg_annotations)
        bg = Background(bg_annotation, bg_datadir)
        bg.load_data()

        num_objs = random.randint(3, CAPACITY)
        obj_list = []
        for obj_i in xrange(num_objs):
            obj_frame = random.choice(object_dict.keys())
            obj = object_dict[obj_frame]
            obj.load_data()
            obj_list.append(obj)

        img, annotations = put_objects(obj_list, bg)
        if not annotations:
            continue
        img_file_path = output_img_dir + '/' + image_name
        cv2.imwrite(img_file_path, img)

        image = {}
        image['id'] = image_id
        image['height'] = 480
        image['width'] = 640
        image['file_name'] = image_name
        data['images'].append(image)

        for annot in annotations:
            annot['image_id'] = image_id
            annot['id'] = image_id
            annot['iscrowd'] = 0
            data['annotations'].append(annot)

    if verbose:
        print('saving cluster json file ...')

    output_json_path = output_json_dir + '/cluster_train2018.json'
    if test_data:
        output_json_path = output_json_dir + '/cluster_test2018.json'
    with open(output_json_path, 'w') as f:
        json.dump(data, f)

    return data


if __name__ == '__main__':
    get_category_id()

    rgb_datadir = '/home/wanlin/Downloads/ADL_rgb'
    mask_datadir = '/home/wanlin/Downloads/ADL_mask'
    bg_datadir = '/home/wanlin/Downloads/ADL_cluster/scene'

    output_dir = '/home/wanlin/Downloads/ADL_cluster'

    test_data = False
    super_category = True
    data = output_data(rgb_datadir, mask_datadir, bg_datadir, output_dir, 5000)


