import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class FaceModel(nn.Module):
    def __init__(self, name):
        super(FaceModel, self).__init__()
        # features_only=True,
        self.model = timm.create_model(name, pretrained=True, features_only=True, num_classes=0, global_pool='')

        self.convStart1 = nn.Conv2d(128, 64, 1)
        self.activation1 = nn.LeakyReLU()
        self.convEnd1 = nn.Conv2d(64, 5, 1)
        self.sigmoid1 = nn.Sigmoid()

        self.convStart2 = nn.Conv2d(256, 128, 1)
        self.activation2 = nn.LeakyReLU()
        self.convEnd2 = nn.Conv2d(128, 5, 1)
        self.sigmoid2 = nn.Sigmoid()

        self.convStart3 = nn.Conv2d(512, 256, 1)
        self.activation3 = nn.LeakyReLU()
        self.convEnd3 = nn.Conv2d(256, 5, 1)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        # torch.Size([1, 64, 224, 224])
        # torch.Size([1, 64, 112, 112])
        # torch.Size([1, 128, 56, 56])
        # torch.Size([1, 256, 28, 28])
        # torch.Size([1, 512, 14, 14])
        out = self.model(x)  # out1 = 56, out2 = 28, out3 = 14
        output = []

        out1 = self.convStart1(out[-3])
        out1 = self.activation1(out1)
        out1 = self.convEnd1(out1)
        out1 = self.sigmoid1(out1)
        out1 = torch.permute(out1, (0, 2, 3, 1)).contiguous()
        output.append(out1)

        out2 = self.convStart2(out[-2])
        out2 = self.activation2(out2)
        out2 = self.convEnd2(out2)
        out2 = self.sigmoid2(out2)
        out2 = torch.permute(out2, (0, 2, 3, 1)).contiguous()
        output.append(out2)

        out3 = self.convStart3(out[-1])
        out3 = self.activation3(out3)
        out3 = self.convEnd3(out3)
        out3 = self.sigmoid3(out3)
        out3 = torch.permute(out3, (0, 2, 3, 1)).contiguous()
        output.append(out3)

        return output


class YOLOLoss(torch.nn.Module):
    def __init__(self, size, l_coord=5, l_noobj=.5, image_size=448):
        super(YOLOLoss, self).__init__()
        self.img_size = image_size
        self.grid_size = size
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def forward(self, preds, ans):
        losses = np.array([0.,0.,0.])
        loss = 0
        for i, pred in enumerate(preds):
            target = self.create_targets(ans, pred).cuda()
            loss0, iou = self.Loss(pred, target)
            loss += loss0
            losses[i] = loss0
        return loss, losses, preds, ans, iou, target

    def Loss(self, pred, target):
        batch_size = pred.shape[0]
        grid_size = pred.shape[-2]
        obj_mask = target[..., 4]
        no_obj_mask = torch.ones_like(obj_mask) - obj_mask
        with torch.no_grad():
            iou = self.compute_iou(pred[..., 0:4], target[..., 0:4])

        # obj_loss = torch.sum(obj_mask * F.mse_loss(pred[..., 4], target[..., 4], reduction='none'))
        obj_loss = torch.sum(obj_mask * F.mse_loss(pred[:, :, :, 4], iou, reduction='none'))
        no_obj_loss = self.l_noobj * torch.sum(no_obj_mask * F.mse_loss(pred[..., 4], target[..., 4], reduction='none'))
        coord_loss = self.l_coord * torch.sum(obj_mask * F.mse_loss(pred[..., 0:4], target[..., 0:4], reduction='none').sum(-1))

        norm = grid_size / 14
        return (obj_loss + no_obj_loss + coord_loss) / (batch_size * norm**2), iou

    def create_targets(self, Y, pred):
        out = torch.zeros(pred.shape)
        grid_size = pred.shape[-2]
        size = self.img_size / grid_size
        for i, y0 in enumerate(Y):
            for y in y0:
                w = y[2] - y[0]
                h = y[3] - y[1]
                centerX = (y[0] + y[2]) / 2
                centerY = (y[1] + y[3]) / 2

                colm = int(centerX / size)
                row = int(centerY / size)

                normX = colm / grid_size * self.img_size
                normY = row / grid_size * self.img_size

                centerX = (centerX - normX) * grid_size
                centerY = (centerY - normY) * grid_size

                out[i, row, colm, 0] = centerX / self.img_size
                out[i, row, colm, 1] = centerY / self.img_size
                out[i, row, colm, 2] = w / self.img_size
                out[i, row, colm, 3] = h / self.img_size
                out[i, row, colm, 4] = 1
                # import ipdb;ipdb.set_trace()
        return out

    def compute_iou(self, boxes1, boxes2):
        boxes1_xy = torch.zeros_like(boxes1)
        boxes1_xy[..., :2] = boxes1[..., :2] / self.grid_size - .5 * boxes1[..., 2:4]
        boxes1_xy[..., 2:4] = boxes1[..., :2] / self.grid_size + .5 * boxes1[..., 2:4]

        boxes2_xy = torch.zeros_like(boxes2)
        boxes2_xy[..., :2] = boxes2[..., :2] / self.grid_size - .5 * boxes2[..., 2:4]
        boxes2_xy[..., 2:4] = boxes2[..., :2] / self.grid_size + .5 * boxes2[..., 2:4]

        tl = torch.max(boxes1_xy[..., :2], boxes2_xy[..., :2])
        br = torch.min(boxes1_xy[..., 2:], boxes2_xy[..., 2:])

        intersection = torch.clamp(br - tl, 0)

        intersection_area = intersection[..., 0] * intersection[..., 1]

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        union_area = boxes1_area + boxes2_area - intersection_area

        return intersection_area / torch.clamp(union_area, min=1e-9)


# model = FaceModel('resnet18')
# dummy = torch.randn(1,3,448,448)
# x = model(dummy)
# for i in x:
#     print(i.shape)