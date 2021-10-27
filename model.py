import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

def Loss(pred, ans):
    pred = torch.permute(pred,(0,2,3,1))
    mask_obj = ans[:,:,:,4]
    mask_no_obj = (ans[:,:,:,4] == 0)
    lamda = 5

    loss_obj = F.mse_loss(mask_obj * pred[:, :, :, 4], ans[:, :, :, 4], reduction='sum')
    loss_x = F.mse_loss(mask_obj * pred[:, :, :, 0], ans[:, :, :, 0], reduction='sum') * lamda
    loss_y = F.mse_loss(mask_obj * pred[:, :, :, 1], ans[:, :, :, 1], reduction='sum') * lamda
    loss_w = F.mse_loss(mask_obj * torch.sqrt(pred[:, :, :, 2]), torch.sqrt(ans[:, :, :, 2]), reduction='sum') * lamda
    loss_h = F.mse_loss(mask_obj * torch.sqrt(pred[:, :, :, 3]), torch.sqrt(ans[:, :, :, 3]), reduction='sum') * lamda
    loss_no_obj = F.mse_loss(mask_no_obj * pred[:, :, :, 4], torch.zeros_like(ans[:, :, :, 4]), reduction='sum') * 0.5

    loss = (loss_obj + loss_x + loss_y + loss_w + loss_h + loss_no_obj) / ans.shape[0]
    return loss

class FaceModelFC(nn.Module):
    def __init__(self, name, grid_size):
        super(FaceModel, self).__init__()
        # model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='')
        self.grid_size = grid_size
        self.model = timm.create_model(name, pretrained=True, num_classes=grid_size*grid_size*5)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # input 3 * 256 * 256
        out = self.model(x)
        out = self.sig(out)
        return out


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

# model = FaceModel('resnet18').to('cuda')
# test = torch.rand(1, 3, 448, 448).to('cuda')
# ans = torch.rand(1, 7, 7, 5)
# out = model(test).to('cpu')
# loss = Loss(out, ans)
# print(loss)


