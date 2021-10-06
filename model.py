import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm

from wider_face_dataset import WiderDataset, img_size
from train import batch_size,device
import torch.nn.functional as F

class FaceModel(nn.Module):
    def __init__(self, name):
        super(FaceModel, self).__init__()
        # model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='')
        self.model = timm.create_model(name, pretrained=True, num_classes=0, global_pool='')
        self.pool = nn.AvgPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(512, 5, 1)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(5)
        self.drop = nn.Dropout2d(p=0.25)
        self.conv2 = nn.Conv2d(5, 5, 1)
        self.conv3 = nn.Conv2d(5, 5, 1)
        self.convEnd = nn.Conv2d(5, 5, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # input 3 * 256 * 256
        out = self.model(x)
        out = self.pool(out)
        out = self.drop(out)

        out = self.conv1(out)
        out = self.norm(out)
        out = self.activation(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.norm(out)
        out = self.activation(out)
        out = self.drop(out)

        out = self.conv3(out)
        out = self.norm(out)
        out = self.activation(out)

        out = self.convEnd(out)
        out = self.sig(out)
        return out


