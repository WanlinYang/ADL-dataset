from __future__ import print_function

import numpy as np
import json

categories = []

evaluation = {}
# image_id : Evalueate_unit

class Evaluate_unit:
    """
    has both test data and ground truth data
    """
    def __init__(self, image_id):
        self.image_id = image_id

    def load_pred(self, pred):
        self.pred_bbox = pred['bbox']
        self.pred_catid = pred['category_id']
        self.pred_score = pred['score']

    def load_gt(self, gt):
        self.gt_bbox = gt['bbox']
        self.gt_catid = gt['category_id']

    def print_error(self):
        if(self.gt_catid != self.pred_catid):
            print('gt = ', self.gt_catid)
            print('pred = ', self.pred_catid)

    def calculate_IoU(self):
        # Intersection over Union
        gt_area = self.gt_bbox[2] * self.gt_bbox[3]
        pred_area = self.pred_bbox[2] * self.pred_bbox[3]
        [x1, y1, w1, h1] = self.gt_bbox
        [x2, y2, w2, h2] = self.pred_bbox
        ul_x = max(x1, x2); ul_y = max(y1, y2)
        lr_x = min(x1+w1, x2+w2)
        lr_y = min(y1+h1, y2+h2)
        Union = gt_area + pred_area - \
                abs(lr_x-ul_x) * abs(lr_y-ul_y)
        self.IoU = float(pred_area) / float(Union)


def load_data(datadir):
    """
    load data from json file
    """
    global categories

    pred_file_path = datadir + '/bbox_adl_2018_test_results.json'
    gt_file_path = datadir + '/segmentation_test2018.json'

    with open(pred_file_path, 'r') as f:
        pred_data = json.load(f)
    with open(gt_file_path, 'r') as f:
        gt_data = json.load(f)

    categories = gt_data['categories']
    return pred_data, gt_data

def traverse_data(datadir):
    """
    load all prediction and ground truth data
    """
    global evaluation
    pred_data, gt_data = load_data(datadir)

    for gt in gt_data['annotations']:
        image_id = gt['image_id']
        evalu_unit = Evaluate_unit(image_id)
        evalu_unit.load_gt(gt)
        evaluation[image_id] = evalu_unit

    for pred in pred_data:
        image_id = pred['image_id']
        evaluation[image_id].load_pred(pred)
        evaluation[image_id].calculate_IoU()

if __name__ == '__main__':
    datadir = '/home/wanlin/Downloads/ADL_CNN_2.20'
    traverse_data(datadir)

    for key, value in evaluation.iteritems():
        print(value.IoU)

    from IPython import embed
    embed()
