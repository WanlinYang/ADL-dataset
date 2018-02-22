from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import cv2
import os
import random
import json
import pickle

"""
This is the attempt of putting several objects that masked
out from the original dataset over several pre-labeled
candidate background
output: 480x640 image and COCO-format json file
"""
image_names = {}
# {'image_id': 'image_file_name'}

test_data = False
verbose = True

CAPACITY = 12
# object capacity of one background image

class Object:
    def __init__(self, annotation, rgbdatadir, maskdatadir):
        self.image_id = annotation['image_id']
        self.area = annotation['area']
        self.bbox = annotation['bbox']
        self.category_id = annotation['category_id']
        self.polygon = None
        self.rgb_path = rgbdatadir # rgb image path
        self.mask_path = maskdatadir # mask image path

    def load_data(self):
        """ load cropped rgb data and cropped mask data """
        full_img = cv2.imread(self.rgb_path)

        ul_x = self.bbox[0]; ul_y = self.bbox[1]
        lr_x = ul_x + self.bbox[2]
        lr_y = ul_y + self.bbox[3]

        self.crop_loc = [ul_x, ul_y, lr_x, lr_y]
        self.cropped_img = self.full_img[ul_y:lr_y,ul_x:lr_x]

        self.cropped_mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)

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

def get_obj_annotations(mask_datadir):
    """
    traverse loc and mask, return a list of dict
    """
    obj_annotations = []

    return obj_annotations

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
                    if obj_scaled_mask[i][j] == 1:
                        bg_img[y+i][x+j] = obj_scaled_img[i][j]

            annot_dict = {}
            annot_dict['bbox'] = [x, y, cols, rows]
            annot_dict['category_id'] = obj.category_id
            annot_dict['segmentation'] = None
            annot_dict['area'] = None
            annotations.append(annot_dict)

    return bg_img, annotations


def load_image_names(data_image_list):
    """
    data_image is a dictionary, containing image_id and
    corresponding file name.
    It is included in json file.
    """
    global image_names

    for data_image in data_image_list:
        image_id = data_image['id']
        file_name = data_image['file_name']
        image_names[image_id] = file_name

def output_data(rgb_datadir, mask_datadir, bg_datadir, output_dir, total):
    """
    put random number of objects into random background,
    and output total merged images and json file
    """
    obj_img_folder = rgb_datadir
    mask_img_folder = mask_datadir
    bg_img_folder = bg_datadir

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
    for root, dirs, files in os.walk(bg_img_folder):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    d = pickle.load(f)
                bg_annotations.append(d)

    if verbose:
        print('getting annotations ...')

    obj_annotations = get_obj_annotations(mask_datadir)
    # traverse loc and mask, save as a list of dict

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
    data['categories'] = obj_json_data['categories']

    if verbose:
        print('producing and saving cluster images ...')

    for total_i in xrange(start_id, start_id + total):
        image_id = random.choice(image_id_cands)
        image_id_cands.remove(image_id)
        image_name = 'ADL2018_cluster_' + str(total_i) + '.png'

        bg_annotation = random.choice(bg_annotations)
        bg = Background(bg_annotation, bg_img_folder)
        bg.load_data()

        num_objs = random.randint(3, CAPACITY)
        obj_list = []
        for obj_i in xrange(num_objs):
            obj_annotation = random.choice(obj_annotations)
            obj = Object(obj_annotation, obj_img_folder)
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

    obj_datadir = '/home/wanlin/Downloads/ADL2018'
    bg_datadir = '/home/wanlin/Downloads/ADL_cluster/scene'
    output_dir = '/home/wanlin/Downloads/ADL_cluster'

    test_data = True
    data = output_data(obj_datadir, bg_datadir, output_dir, 1000)

#    from IPython import embed
#    embed()


