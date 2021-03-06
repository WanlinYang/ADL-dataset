
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import cv2
import os
import re
import random

from substitute_single import Object_mask, get_background, \
                                substitute_background
from shutil import copyfile
import json

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
    {'id':10, 'name': 'towel', 'supercategory': 'scrub','superid': 4},
    {'id':11, 'name': 'dish-scouring-pad', 'supercategory': 'scrub','superid': 4},
    {'id':12, 'name': 'flashlight', 'supercategory': 'device','superid': 5},
    {'id':13, 'name': 'screwdriver', 'supercategory': 'tool','superid': 6},
    {'id':14, 'name': 'hammer', 'supercategory': 'tool','superid': 6},
    {'id':15, 'name': 'wrench', 'supercategory': 'tool','superid': 6},
    {'id':16, 'name': 'toothbrush', 'supercategory': 'hygiene-teeth','superid': 7},
    {'id':17, 'name': 'toothpaste', 'supercategory': 'hygiene-teeth','superid': 7},
    {'id':18, 'name': 'soap', 'supercategory': 'hygiene-body','superid': 8},
    {'id':19, 'name': 'handsoap', 'supercategory': 'hygiene-body','superid': 8},
    {'id':20, 'name': 'wallet', 'supercategory': 'dressing','superid': 9},
    {'id':21, 'name': 'hairbrush', 'supercategory': 'grooming','superid': 10},
    {'id':22, 'name': 'book', 'supercategory': 'printed-matter','superid': 11},
    {'id':23, 'name': 'pen', 'supercategory': 'writing','superid': 12},
    {'id':24, 'name': 'tape', 'supercategory': 'office-tool','superid': 13},
    {'id':25, 'name': 'stapler', 'supercategory': 'office-tool','superid': 13},
    {'id':26, 'name': 'headphone', 'supercategory': 'digital-device','superid': 14},
    {'id':27, 'name': 'mouse', 'supercategory': 'digital-device','superid': 14}
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

# keys: id, name, supercategory
category_ids = {}
# {'category': id}

supercategory_ids = {}
# {'category': {'superid': superid, 'supercategory': supercategory}}

image_ids = {}
# {'ADL_file_name': 'image_id'}

image_annotations = {}
# {'ADL_file_name': Annotation}

masks = {}
# {'frame_name', 'Object_mask'}

id_list = [i for i in xrange(30000)]

verbose = True
test_data = False
substitution = False

class Annotation:
    def __init__(self, category_id, image_id):
        self.category_id = category_id
        self.image_id = image_id
        self.area = 0
        self.bbox = []
        self.segmentation = []

    def json_serializable(self):
        self.area = float(self.area)
        for i in xrange(len(self.segmentation[0])):
            self.segmentation[0][i] = \
                    float(self.segmentation[0][i])

    def __str__(self):
        display = 'categopy_id: ' + str(self.category_id) \
                + '\nimage_id: ' + str(self.image_id) \
                + '\narea: ' + str(self.area) \
                + '\nbbox: ' + str(self.bbox) \
                + '\nsegmentation: ' + str(self.segmentation)
        return display

def get_category_id():
    """
    go through categories and save cat_id into category_ids
    """
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

def croploc_to_bbox(croploc):
    """
    croploc [x1, y1, x2, y2]  bbox [x, y, width, height]
    """
    [x1, y1, x2, y2] = croploc

    width = x2 - x1
    height = y2 - y1
    x = x1
    y = y1
    return [x, y, width, height]

def find_polygon(maskcrop, bbox):
    """
    return coco-compatible covex polygon for a 480x640 image,
    given cropped mask and bbox
    """
    full_mask = np.zeros((480, 640), dtype=np.uint8)
    [x, y, width, height] = bbox

    for i in xrange(width):
        for j in xrange(height):
            full_mask[y+j][x+i] = maskcrop[j][i]

    im2, contours, hierarchy = cv2.findContours(full_mask, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
    cont = contours[0]
    for i in xrange(1, len(contours)):
        cont = np.concatenate((cont, contours[i]), axis=0)

    hull = cv2.convexHull(cont)
    polygon = []

    for p in hull:
        [x,y] = p[0]
        polygon.append(x)
        polygon.append(y)

    return [polygon]

def show_mask(polygon):
    """
    show convex mask given polygon
    """
    Rs = mask_util.frPyObjects(polygon, 480, 640)
    mask = mask_util.decode(Rs)
    mask = np.reshape(mask, (480, 640))

    ret, thresh = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    cv2.imshow('mask', thresh)
    cv2.waitKey(0)

def brightness(img, value):
    """
    change brightness of image
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    elif value < 0:
        lim = 0 - value
        v[v <= lim] = 0
        v[v > lim] += np.uint8(value)

    final_hsv = cv2.merge((h, s, v))
    final_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return final_img

def get_data(datadir):
    """
    go through crop location and mask files
    """
    global id_list

    if verbose:
        print('getting crop locations ...')

    for root, dirs, files in os.walk(datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('loc.txt'):
                split_filepath = re.split('_|/', filepath)
                num_frame = int(split_filepath[-2])
                cat = split_filepath[-5]

                if ((not test_data) and (num_frame % 15 == 0) and \
                    (cat in category_ids)) or \
                   ((test_data) and ((num_frame+10) % 97 == 0) and \
                    (cat in category_ids)):

                    frame_name = split_filepath[-5] + '_' \
                            + split_filepath[-4] + '_' \
                            + split_filepath[-3] + '_' \
                            + split_filepath[-2]

                    if substitution:
                        obj_mask = Object_mask(filepath)
                        obj_mask.load_location()
                        masks[obj_mask.frame_name] = obj_mask

                    croploc = [None]*4
                    with open(filepath) as f:
                        loc = f.readlines()
                        croploc[0:2] = re.split(',|\n', loc[0])[0:2]
                        croploc[2:4] = re.split(',|\n', loc[1])[0:2]
                        croploc = [int(i) for i in croploc]

                    img_id = random.choice(id_list)
                    id_list.remove(img_id)
                    image_ids[frame_name] = img_id
                    image_annotations[frame_name] = \
                            Annotation(category_ids[cat], img_id)
                    image_annotations[frame_name].bbox = \
                            croploc_to_bbox(croploc)

    if verbose:
        print('going through cropped masks ...')

    for root, dirs, files in os.walk(datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('maskcrop.png'):
                split_filepath = re.split('_|/', filepath)
                frame_name = split_filepath[-5] + '_' \
                        + split_filepath[-4] + '_' \
                        + split_filepath[-3] + '_' \
                        + split_filepath[-2]
                if frame_name not in image_annotations:
                    continue

                if substitution:
                    masks[frame_name].load_mask(filepath)

                bbox = image_annotations[frame_name].bbox
                maskcrop = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                polygon = find_polygon(maskcrop, bbox)
                image_annotations[frame_name].segmentation = polygon

                #print('id ', image_annotations[frame_name].image_id)
                #print('cat ', image_annotations[frame_name].category_id)
                #show_mask(polygon)

                Rs = mask_util.frPyObjects(polygon, 480, 640)
                area = mask_util.area(Rs)[0]
                image_annotations[frame_name].area = area

def prone_test_id_list(outputdir):
    """
    test image_id should not conflict with trained ids
    """
    global id_list
    train_json_path = outputdir + \
            '/annotations/segmentation_train2018.json'
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    for img in train_data['images']:
        train_id = img['id']
        id_list.remove(train_id)

def save_data(bg_datadir, datadir, outputdir):
    """
    copy corresponding RGB image and ouput json file for training
    """
    rgbdest_path = outputdir + '/ADL_train'
    if test_data:
        rgbdest_path = outputdir + '/ADL_test'

    if substitution:
        backgrounds = get_background(bg_datadir)

    if not os.path.exists(rgbdest_path):
        os.makedirs(rgbdest_path)

    if verbose:
        print('copying big RGB images ...')

    for root, dirs, files in os.walk(datadir):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('rgboriginal.png'):
                split_filepath = re.split('_|/', filepath)
                frame_name = split_filepath[-5] + '_' \
                        + split_filepath[-4] + '_' \
                        + split_filepath[-3] + '_' \
                        + split_filepath[-2]
                if frame_name not in image_annotations:
                    continue

                destfilename = filepath.split('/')[-1]
                destfilepath = rgbdest_path + '/' + destfilename
                num_frame = int(split_filepath[-2])

                if substitution:
                    if num_frame % 2 != 0:
                        # substitution
                        full_img = cv2.imread(filepath)
                        obj_mask = masks[frame_name]
                        angle_type = int(split_filepath[-3])
                        if angle_type == 1:
                            output_img = backgrounds['30deg'].copy()
                        elif angle_type == 2:
                            output_img = backgrounds['45deg'].copy()
                        elif angle_type == 3:
                            output_img = backgrounds['60deg'].copy()
                        substitute_background(obj_mask, full_img, output_img)
                        brt = random.randint(-50,50)
                        brt_img = brightness(output_img, brt)
                        cv2.imwrite(destfilepath, brt_img)
                        continue
                    else:
                        # no substitution
                        output_img = cv2.imread(filepath)
                        brt = random.randint(-50,50)
                        brt_img = brightness(output_img, brt)
                        cv2.imwrite(destfilepath, brt_img)
                else:
                    copyfile(filepath, destfilepath)

    if verbose:
        print('saving json file ...')

    data = {}
    data['categories'] = categories
    data['images'] = []
    data['annotations'] = []
    for key, value in image_annotations.iteritems():
        data_image = {}
        data_image['file_name'] = key + '_rgboriginal.png'
        data_image['height'] = 480
        data_image['width'] = 640
        data_image['id'] = value.image_id
        data['images'].append(data_image)

        data_annotation = {}
        value.json_serializable()
        data_annotation['area'] = value.area
        data_annotation['bbox'] = value.bbox
        data_annotation['category_id'] = value.category_id
        data_annotation['image_id'] = value.image_id
        data_annotation['id'] = value.image_id
        data_annotation['iscrowd'] = 0
        data_annotation['segmentation'] = value.segmentation
        data['annotations'].append(data_annotation)

    json_name = 'segmentation_train2018.json'
    if test_data:
        json_name = 'segmentation_test2018.json'
    jsondest_path = outputdir + '/annotations'
    if not os.path.exists(jsondest_path):
        os.makedirs(jsondest_path)
    json_path = jsondest_path + '/' + json_name
    with open(json_path, 'w') as fid:
        json.dump(data, fid)

    return data


if __name__ == '__main__':
    get_category_id()

    datadir = '/home/wanlin/Downloads/ADL_mask'
    bg_datadir = '/home/wanlin/Pictures/background'
    rgbdir = '/home/wanlin/Downloads/ADL_rgb'

    outputdir = '/home/wanlin/Downloads/ADL2018'

    substitution = True
    test_data = True
    if test_data:
        prone_test_id_list(outputdir)

    get_data(datadir)
    data = save_data(bg_datadir, rgbdir, outputdir)

#    for key, value in image_annotations.iteritems():
#        print(value)

#    from IPython import embed
#    embed()

