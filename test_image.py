import argparse
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as Aug
from model import FaceModel
from wider_face_dataset import img_size

parser = argparse.ArgumentParser(description='add batch size')
parser.add_argument('model_path', type=str, help='the path of your model')
parser.add_argument('image_path', type=str, help='the path of the image that u want to test')
args = parser.parse_args()


def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def post_process(face_locations):
    grid_size = face_locations.shape[1]
    size = img_size / grid_size
    face_locations = face_locations.reshape(-1, 5)
    output0 = []
    for i, out in enumerate(face_locations):
        if out[4] >= 0.02:
            colm = i % grid_size
            row = int(i / grid_size)

            x = (out[0] + colm) * img_size / grid_size
            y = (out[1] + row) * img_size / grid_size
            w = out[2] * img_size
            h = out[3] * img_size

            k = [x-w/2, y-h/2, x+w/2, y+h/2]
            k.append(out[4])
            output0.append(k)
    return output0


def draw(frame, face_location):
    for k in face_location:
        if k[4] >= 0.3:
            k = list(map(int, k))
            cv2.rectangle(frame, (k[0], k[1]), (k[2], k[3]), (256, 0, 0), 2)
    cv2.imshow('image', frame)
    cv2.waitKey(0)

model = FaceModel('resnet18').cuda()
model.load_state_dict(torch.load(args.model_path))

transforms = Aug.Compose([
    Aug.Resize(img_size, img_size),
    Aug.Normalize(),
    ToTensorV2()])
image = cv2.imread(args.image_path)
orig_size = (image.shape[0], image.shape[1])
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transformed = transforms(image=image2)
x = transformed['image']
x = x.unsqueeze(0).cuda()
output = model(x)

with torch.no_grad():
    dets1 = post_process(torch.squeeze(output[0].cpu()))
    dets2 = post_process(torch.squeeze(output[1].cpu()))
    dets3 = post_process(torch.squeeze(output[2].cpu()))
    dets = np.array(dets1 + dets2 + dets3)
    keep = nms(dets, 0.25)
    dets = dets[keep]
    dets[..., 0] = dets[..., 0] * orig_size[1] / img_size
    dets[..., 1] = dets[..., 1] * orig_size[0] / img_size
    dets[..., 2] = dets[..., 2] * orig_size[1] / img_size
    dets[..., 3] = dets[..., 3] * orig_size[0] / img_size
draw(image, dets)
