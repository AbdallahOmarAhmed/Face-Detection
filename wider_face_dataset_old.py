import os
import time

import torch
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
# from data_aug.data_aug import *
# from data_aug.bbox_util import *
import random
import torch.nn as nn
import albumentations as Aug
import cv2


grid_size = 14
img_size = 448

def getAug(train):
    if train :
        transforms = Aug.Compose([
            Aug.RandomResizedCrop(img_size, img_size, scale=(0.5, 0.9)),
            Aug.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30),
            Aug.HorizontalFlip(p=0.5),
            Aug.Normalize(),
            ToTensorV2()
        ], bbox_params=Aug.BboxParams(format='pascal_voc', min_visibility=0.33))
        return transforms
    else:
        transforms = Aug.Compose([
            Aug.Resize(img_size, img_size),
            Aug.Normalize(),
            ToTensorV2()
        ], bbox_params=Aug.BboxParams(format='pascal_voc', min_visibility=0.33))
        return transforms

class WiderDataset(Dataset):
    def __init__(self, train=True, max_faces=-1):
        self.train = train
        self.X=[]
        self.Y=[]
        self.path = 'WIDER_train/images/' if train else 'WIDER_val/images/'
        self.txtPath = 'wider_face_split/wider_face_train_bbx_gt.txt' if train\
            else 'wider_face_split/wider_face_val_bbx_gt.txt'
        with open(self.txtPath) as f:
            while True:
                imgName = f.readline().strip('\n')
                if not imgName:
                    break
                c = int(f.readline())
                if c == 0:
                    _ = f.readline()
                    continue
                if c > max_faces and max_faces != -1:
                    for i in range(c):
                        _ = f.readline()
                    continue
                anno=[]
                for i in range(c):
                    y = f.readline().split()
                    y = list(map(float, y))
                    anno.append(y)
                self.X.append(imgName)
                self.Y.append(anno)
        print('finished loading dataset : ', len(self.X))
        # transforms = Sequence([RandomTranslate(0.1), RandomScale(0.2), Resize(img_size), RandomHorizontalFlip(0.5), RandomHSV(30, 20, 20)])\

    def __len__(self):
        return len(self.X)
    def __getitem__(self,index):
        aug = getAug(self.train)
        v = 0.0001
        y0 = self.Y[index]
        labels = []
        image = cv2.imread(self.path+self.X[index])
        for k in y0:
            l = (k[0], k[1], k[0]+k[2]+v, k[1]+k[3]+v, 1)
            if l[2] <= image.shape[1]+v and l[3] <= image.shape[0]+v:
                labels.append(l)
        labels = np.array(labels)
        transformed = aug(image=image, bboxes=labels)
        x = transformed['image']
        y = transformed['bboxes']
        return x,y


def myCollate(batch):
    images = []
    boxes = []
    for x,y in batch:
        images.append(x)
        boxes.append(y)
    return torch.stack(images), boxes


def draw(frame, face_locations):
    frame = frame.numpy()
    c = 0
    size = img_size / grid_size
    colors = np.array([0, 0, 254])
    for out in face_locations:
        if out[4] == 1:
            colm = c % grid_size
            row = int(c / grid_size)
            print(colm, row)

            centreX = (out[0] + colm) * size
            centreY = (out[1] + row) * size
            w = out[2] * img_size
            h = out[3] * img_size

            # print(out[0],out[1])
            k = np.array([centreX-w/2, centreY-h/2, centreX+w/2, centreY+h/2])
            k = list(map(int, k))
            # print(k)
            color = (int(colors[0]), int(colors[1]), int(colors[2]))
            cv2.rectangle(frame, (k[0], k[1]), (k[2], k[3]), color, 1)
            colors[0] += 127
            colors[2] -= 127
        c += 1
    cv2.imshow('image', frame)
    cv2.waitKey(0)

# data = WiderDataset(False, 5)
# x,y = data[515]
# print(y.shape)
# draw(x,y)
# cv2.imshow('image', x.numpy())
# cv2.waitKey(0)