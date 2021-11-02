
# load data
import random

import cv2
import numpy
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import albumentations as Aug

from model import FaceModel, FaceModelFC
from wider_face_dataset_old import WiderDataset

img_size = 448


model = FaceModelFC().cuda()
model.load_state_dict(torch.load("models/BestFixed3.pth"))

path = 'WIDER_train/images/33--Running/33_Running_Running_33_71.jpg'

transforms = Aug.Compose([
    Aug.Resize(img_size, img_size),
    Aug.Normalize(),
    ToTensorV2()])

image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transformed = transforms(image=image)
image = transformed['image']
x = image.unsqueeze(0).cuda()
out = model(x)
print(out.shape)
print(out)

# image = image.permute(1,2,0)
# cv2.imshow('image', image.numpy())
# cv2.waitKey(0)