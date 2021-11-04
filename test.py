
# load data
import random

import cv2
import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import FaceModel
from wider_face_dataset_old import WiderDataset

img_size = 448

model = FaceModel('resnet18')
model.load_state_dict(torch.load("models/BestIou.pth"))
model = model.cuda()
model.eval()

def draw(frame, face_locations):
    grid_size = 14
    frame = frame.numpy()
    c = 0
    size = img_size / grid_size
    face_locations = face_locations.view(-1,5)
    for out in face_locations:
        if out[4] >= 0.5:
            # k *= 256

            colm = c % grid_size
            row = int(c / grid_size)

            centreX = (out[0] + colm) * size
            centreY = (out[1] + row) * size
            w = out[2] * img_size
            h = out[3] * img_size

            # print(out[0],out[1])
            k = np.array([centreX-w/2, centreY-h/2, centreX+w/2, centreY+h/2])
            k = list(map(int, k))
            # print(k)
            cv2.rectangle(frame, (k[0], k[1]), (k[2], k[3]), (256,0,0), 2)
        c += 1
    cv2.imshow('image', frame)
    cv2.waitKey(0)

with torch.no_grad():
    data = WiderDataset(7,False)
    for i in range(100):
        x0,_ = data[1000 + i]
        x = torch.unsqueeze(x0,0)
        images = x.cuda()

        output = model(images)
        # output = torch.reshape(output, (49, 5))
        output = output.squeeze()
        x0 = x0.permute(1,2,0)
        draw(x0.cpu(),output.cpu())