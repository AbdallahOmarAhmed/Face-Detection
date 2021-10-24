import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as Aug
import cv2


grid_size = 7
img_size = 448


def getAug(train):
    if train :
        transforms = Aug.Compose([
            Aug.RandomResizedCrop(img_size, img_size, scale=(0.5, 0.9)),
            Aug.RGBShift(r_shift_limit=26, g_shift_limit=26, b_shift_limit=26),
            Aug.RandomBrightnessContrast(p=0.4),
            Aug.HorizontalFlip(p=0.5)
        ], bbox_params=Aug.BboxParams(format='pascal_voc', min_visibility=0.75))
        return transforms
    else:
        transforms = Aug.Compose([
            Aug.Resize(img_size, img_size)
        ], bbox_params=Aug.BboxParams(format='pascal_voc', min_visibility=0.75))
        return transforms


class WiderDataset(Dataset):
    def __init__(self,max_faces = -1,train = True):
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
        if self.train:
            print('finished loading dataset :', len(self.X), 'image')

    def __len__(self):
        return len(self.X)
    def __getitem__(self,index):
        aug = getAug(self.train)
        v = 0.0001
        y0 = self.Y[index]
        labels = []
        for k in y0:
            #l = (k[0]-k[2]/2, k[1]-k[3]/2, k[0]+k[2]/2, k[1]+k[3]/2, 256)
            #l = torch.unsqueeze(torch.tensor(l),0)
            l = (k[0], k[1], k[0]+k[2]+v, k[1]+k[3]+v, img_size)
            labels.append(l)
        labels = np.array(labels)
        image = cv2.imread(self.path+self.X[index])
        transformed = aug(image=image, bboxes=labels)
        x = transformed['image']
        y = transformed['bboxes']
        y = np.array(y)/img_size
        # y = calcY(y)/img_size
        # y[:,-1] *= img_size
        # print(y.shape)
        x = torch.from_numpy(x)/255
        return x,y


def calcY(Ys, preds):
    output = []
    if len(Ys) == 0:
        return torch.zeros(preds.shape)
    for Y in Ys:
        # Y *= img_size
        out = torch.zeros(preds[0].shape)
        grid_size = preds[0].shape[0] ** 0.5
        size = img_size / grid_size
        for y in Y:
            w = y[2] - y[0]
            h = y[3] - y[1]

            centerX = (y[0] + y[2]) / 2
            centerY = (y[1] + y[3]) / 2

            colm = int(centerX * img_size / size)
            row = int(centerY * img_size / size)

            normX = colm / grid_size
            normY = row / grid_size

            centerX = (centerX - normX) * grid_size
            centerY = (centerY - normY) * grid_size
            out[colm][row] = torch.tensor([centerX, centerY, w, h, 1])
        output.append(out)
    output = torch.stack(output)
    return output


def my_collate(batch):
    batch_size = len(batch)
    x,y = map(list, zip(*batch))
    x = torch.stack(list(x), dim=0)
    dummy = torch.zeros(batch_size, 7, 7, 5)
    y = calcY(y,dummy)
    return x,y

