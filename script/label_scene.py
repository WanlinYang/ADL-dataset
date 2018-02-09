import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import pickle
import json

COLOR_LIB = [[250, 222, 200], [205, 0, 0],
            [34, 139, 34], [128, 192, 192],
            [42, 42, 165], [128, 64, 128],
            [0, 102, 204], [11, 134, 184],
            [153, 153, 0], [141, 134, 0],
            [141, 0, 184], [0, 134, 184],
            [223, 134, 184], [141, 134, 43],
            [141, 23, 11], [141, 34, 14],
            [41, 134, 14], [241, 14, 233],
            [241, 24, 182], [141, 13, 123],
            [141, 164, 13], [141, 174, 84],
            [41, 14, 184], [231, 34, 184]]

pen_size = 5
eraser_size = 4

mode_nums = 2
mode = 1
colors = COLOR_LIB[:mode_nums]
drawing = False
erasing = False
mask = np.zeros((480, 640), dtype = np.uint8)

def draw(event, x, y, flags, param):
    global drawing, erasing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (x-pen_size,y-pen_size),
                      (x+pen_size,y+pen_size), colors[mode], -1)
        label_mask(x, y)

    elif event == cv2.EVENT_MBUTTONDOWN:
        erasing = True

    elif event == cv2.EVENT_MBUTTONUP:
        erasing = False
        erase_img(x, y)
        erase_mask(x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(img, (x-pen_size,y-pen_size),
                          (x+pen_size,y+pen_size), colors[mode], -1)
            label_mask(x, y)
        if erasing:
            erase_img(x, y)
            erase_mask(x, y)

def erase_img(x, y):
    global img
    for i in xrange(x-eraser_size, x+eraser_size):
        for j in xrange(y-eraser_size, y+eraser_size):
            i_temp = max(0,i); i_temp = min(i_temp, 639)
            j_temp = max(0,j); j_temp = min(j_temp, 479)
            img[j_temp,i_temp] = origin_img[j_temp,i_temp]

def label_mask(x, y):
    global mask
    for i in xrange(x-pen_size, x+pen_size):
        for j in xrange(y-pen_size, y+pen_size):
            i_temp = max(0,i); i_temp = min(i_temp, 639)
            j_temp = max(0,j); j_temp = min(j_temp, 479)
            mask[j_temp,i_temp] = mode

def erase_mask(x, y):
    global mask
    for i in xrange(x-pen_size, x+pen_size):
        for j in xrange(y-pen_size, y+pen_size):
            i_temp = max(0,i); i_temp = min(i_temp, 639)
            j_temp = max(0,j); j_temp = min(j_temp, 479)
            mask[j_temp,i_temp] = 0

def label_img(datadir, scale_range):
    global mode, img, origin_img, pen_size, eraser_size

    split_filepath = datadir.split('/')
    mask_name = split_filepath[-1][:-4] + '.pkl'
    mask_path = datadir[:-(len(mask_name))] + mask_name

    img = cv2.imread(datadir)
    img = cv2.resize(img, (640, 480))
    origin_img = cv2.imread(datadir)
    origin_img = cv2.resize(origin_img, (640, 480))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)

    def nothing(x):
        pass
    cv2.createTrackbar('pen_size', 'image', 5, 50, nothing)
    cv2.createTrackbar('eraser_size', 'image', 5, 20, nothing)

    while True:
        pen_size = cv2.getTrackbarPos('pen_size', 'image')
        eraser_size = cv2.getTrackbarPos('eraser_size', 'image')
        cv2.imshow('image', img)
        k = cv2.waitKey(1)
        if k in [ord(str(i)) for i in xrange(10)]:
            mode = int(chr(k))
            print('mode = ', mode)
        elif k == 27:
            plt.imshow(mask)
            plt.show()
            y_min = 480; y_max = 0 # mask.shape[0]
            x_min = 640; x_max = 0 # mask.shape[1]
            for i in xrange(mask.shape[0]):
                for j in xrange(mask.shape[1]):
                    if mask[i][j] == 1:
                        y_min = min(y_min, i)
                        y_max = max(y_max, i)
                        x_min = min(x_min, j)
                        x_max = max(x_max, j)
            annotation = {}
            annotation['coord_range'] = [y_min, y_max, x_min, x_max]
            annotation['name'] = split_filepath[-1][:-4]
            annotation['scale_range'] = scale_range
            annotation['mask'] = mask
            with open(mask_path, 'wb') as f:
                pickle.dump(annotation, f, protocol=pickle.HIGHEST_PROTOCOL)
            cv2.imwrite(datadir, origin_img)
            break

if __name__ == '__main__':
    datadir = '/home/wanlin/Downloads/ADL_cluster/scene/Scene11.jpg'
    label_img(datadir, [0.7, 1.0])

#    from IPython import embed
#    embed()
