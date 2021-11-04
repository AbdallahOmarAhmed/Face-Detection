import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class FaceModel(nn.Module):
    def __init__(self, name):
        super(FaceModel, self).__init__()
        # model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='')
        self.model = timm.create_model(name, pretrained=True, num_classes=0, global_pool='')
        #self.pool = nn.AvgPool2d(2, stride=2) # 14 * 14 output size
        self.convStart = nn.Conv2d(512, 256, 1)
        self.activation = nn.LeakyReLU()
        self.convEnd = nn.Conv2d(256, 5, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # input 3 * 256 * 256
        out = self.model(x)
        #out = self.pool(out)
        out = self.convStart(out)
        out = self.activation(out)
        out = self.convEnd(out)
        out = self.sig(out)
        out = torch.permute(out, (0, 2, 3, 1)).contiguous()
        return out


class YOLOLoss(torch.nn.Module):
    def __init__(self, s, l_coord = 5, l_noobj = .5, image_size=448):
        super(YOLOLoss, self).__init__()
        self.image_size = image_size
        self.grid_size = s
        self.l_coord = l_coord
        self.l_noobj  = l_noobj

    def forward(self, pred, true):

        #import ipdb;ipdb.set_trace()
        batch_size = pred.shape[0]
        # target = creat_targets(true, self.s, self.image_size).cuda()
        target = true
        obj_mask = target[..., 4]
        no_obj_mask = torch.ones_like(obj_mask) - obj_mask
        with torch.no_grad():
            iou = self.compute_iou(pred[..., 0:4], target[..., 0:4])

        # obj_loss = torch.sum(obj_mask * F.mse_loss(pred[..., 4], target[..., 4], reduction='none'))
        obj_loss = torch.sum(obj_mask * F.mse_loss(pred[:, :, :, 4], iou, reduction='none'))
        no_obj_loss = self.l_noobj * torch.sum(no_obj_mask * F.mse_loss(pred[..., 4], target[..., 4], reduction='none'))
        coord_loss = self.l_coord * torch.sum(obj_mask * F.mse_loss(pred[..., 0:4], target[..., 0:4], reduction='none').sum(-1))

        return (obj_loss + no_obj_loss + coord_loss) / batch_size

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

        return intersection_area / union_area


