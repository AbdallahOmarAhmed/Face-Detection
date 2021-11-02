import torch
import numpy as np
import cv2
from time import time
import torch.nn.functional as F



class YOLOLoss(torch.nn.Module):

    def __init__(self, s, l_coord = 5, l_noobj = .5, image_size=448):
        super(YOLOLoss, self).__init__()
        self.image_size = image_size
        self.s = s
        self.l_coord = l_coord
        self.l_noobj  = l_noobj


    def forward(self, pred, true):

        batch_size = pred.shape[0]
        # target = creat_targets(true, self.s, self.image_size).cuda()
        target = true
        obj_mask = target[:,:,4]
        no_obj_mask = torch.ones_like(obj_mask) - obj_mask

        obj_loss = torch.sum(obj_mask * F.mse_loss(pred[..., 4], target[:, :, 4], reduction='none'))
        # obj_loss = torch.sum(obj_mask * F.mse_loss(pred[:, :, :, 4], iou, reduction='none'))
        no_obj_loss = self.l_noobj * torch.sum(no_obj_mask * F.mse_loss(pred[..., 4], target[:, :, 4], reduction='none'))
        coord_loss = self.l_coord * torch.sum(obj_mask * F.mse_loss(pred[..., 0:4], target[:, :, 0:4], reduction='none').sum(-1))

        # coord_loss = self.l_coord * torch.sum(obj_mask * (F.mse_loss(pred[...,0:2], target[...,0:2], reduction='none') +
        #                                                   F.mse_loss(torch.sqrt(pred[...,2:4]), torch.sqrt(target[...,2:4]), reduction='none')).sum(-1))

        return (obj_loss + no_obj_loss + coord_loss) / batch_size


# if __name__ == "__main__":
#
#     boxes = [  [[200,3,2,2],[400,3,2,2]], [[200,200,2,2]]  ]
#
#     pred = torch.rand([2,7,7,5])
#     loss_func = YOLOLoss(7)
#
#     l = loss_func(pred,boxes)
#     print(l)
#     s = time()
#     out = creat_targets(boxes,7,448)
#     e = time()
#     print(e-s)
#
#     print(out[:,:,:,4])

