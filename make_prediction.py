import argparse
import os.path
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as Aug
from glob import glob
from model import FaceModel
from tqdm import tqdm


parser = argparse.ArgumentParser(description='add batch size')
parser.add_argument('path', type=str, help='your model path')
args = parser.parse_args()


path = '/home/abdallah/projects/Face-Detection/WIDER_val/images'
output_path = "output"
images = glob(os.path.join(path, "*", "*.jpg"))
grid_size = 14
input_size = 448

model = FaceModel('resnet18')
model.load_state_dict(torch.load(args.path))
model = model.cuda()
model.eval()

aug = Aug.Compose([
    Aug.Resize(448, 448),
    Aug.Normalize(),
    ToTensorV2()])

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


def post_process(face_locations, input_size, img_size):
    grid_size = face_locations.shape[1]
    size = input_size / grid_size
    face_locations = face_locations.reshape(-1, 5)
    output0 = []
    for i, out in enumerate(face_locations):
        if out[4] >= 0.02:
            colm = i % grid_size
            row = int(i / grid_size)

            x = (out[0] + colm) * img_size[1] / grid_size
            y = (out[1] + row) * img_size[0] / grid_size
            w = out[2] * img_size[1]
            h = out[3] * img_size[0]

            k = [x-w/2, y-h/2, x+w/2, y+h/2]
            k.append(out[4])
            output0.append(k)
    return output0


for i, image_path in tqdm(enumerate(images)):

    dir = image_path.split('/')[-2]

    output_name = image_path.split('/')[-1][:-4] + ".txt"

    dir_name = os.path.join(output_path,dir)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    save_name = os.path.join(dir_name, output_name)

    # performing detection
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img_size = (img.shape[0], img.shape[1])
    transformed = aug(image=img)
    transformed_img = transformed['image']
    image = transformed_img.cuda()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
    dets1 = post_process(torch.squeeze(output[0].cpu()), input_size, img.shape)
    dets2 = post_process(torch.squeeze(output[1].cpu()), input_size, img.shape)
    dets3 = post_process(torch.squeeze(output[2].cpu()), input_size, img.shape)
    dets = np.array(dets1 + dets2 + dets3)

    # do NMS
    keep = nms(dets, 0.5)
    dets = dets[keep]

    with open(save_name, "w") as fd:
        bboxs = dets
        file_name = os.path.basename(save_name)[:-4] + "\n"
        if len(bboxs) == 0:
            bboxs_num = str(len(bboxs)+1) + "\'n"
            fd.write(file_name)
            fd.write('1\n')
            fd.write('5 5 10 10 0.001\n')
        else:
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)

        for box in bboxs:
            x = int(box[0])
            y = int(box[1])
            w = int(box[2]) - int(box[0])
            h = int(box[3]) - int(box[1])
            confidence = str(box[4])
            line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + "\n"
            fd.write(line)
