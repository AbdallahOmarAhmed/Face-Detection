import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F



def LossVec(pred, ans):
    grid_size = 7
    loss = 0
    pred = torch.reshape(pred, (-1, grid_size*grid_size, 5))
    noObj = (ans[:,:,4] == 0) * 0.5
    lamda = 5
    sqrt_ans = torch.sqrt(ans[:, :, 2:4])
    sqrt_pred = torch.sqrt(pred[:, :, 2:4])
    loss += sum(sum(ans[:,:,4] * (pred[:,:,4] - ans[:,:,4])**2))
    loss += sum(sum(noObj * (pred[:,:,4] - ans[:,:,4])**2))
    loss += sum(sum(sum((lamda * (pred[:, :, 0:2] - ans[:, :, 0:2]) ** 2) * torch.unsqueeze(ans[:, :, 4], -1))))
    loss += sum(sum(sum((lamda * (sqrt_pred - sqrt_ans) ** 2) * torch.unsqueeze(ans[:, :, 4], -1))))
    return loss

def Loss(pred, ans):
    mask_obj = ans[:,:,4]
    mask_no_obj = (ans[:,:,4] == 0)
    lamda = 5

    loss_obj = F.mse_loss(mask_obj * pred[:, :, 4], ans[:, :, 4], reduction='sum')
    loss_x = F.mse_loss(mask_obj * pred[:, :, 0], ans[:, :, 0], reduction='sum') * lamda
    loss_y = F.mse_loss(mask_obj * pred[:, :, 1], ans[:, :, 1], reduction='sum') * lamda
    loss_w = F.mse_loss(mask_obj * torch.sqrt(pred[:, :, 2]), torch.sqrt(ans[:, :, 2]), reduction='sum') * lamda
    loss_h = F.mse_loss(mask_obj * torch.sqrt(pred[:, :, 3]), torch.sqrt(ans[:, :, 3]), reduction='sum') * lamda
    loss_no_obj = F.mse_loss(mask_no_obj * pred[:, :, 4], torch.zeros_like(ans[:, :, 4]), reduction='sum') * 0.5

    loss = (loss_obj + loss_x + loss_y + loss_w + loss_h + loss_no_obj) / ans.shape[0]
    return loss


class FaceModelFC(nn.Module):
    def __init__(self):
        super(FaceModelFC, self).__init__()
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=7*7*5)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # input 3 * 256 * 256
        out = self.model(x)
        out = self.sig(out)
        out = torch.reshape(out, (-1, 49, 5))
        return out

class FaceModel(nn.Module):
    def __init__(self, name):
        super(FaceModel, self).__init__()
        # model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='')
        self.model = timm.create_model(name, pretrained=True, num_classes=0, global_pool='')
        self.pool = nn.AvgPool2d(2, stride=2)

        self.convStart = nn.Conv2d(512, 256, 1)
        self.activation = nn.LeakyReLU()
        self.convEnd = nn.Conv2d(256, 5, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # input 3 * 256 * 256
        out = self.model(x)
        out = self.pool(out)

        out = self.convStart(out)
        out = self.activation(out)
        out = self.convEnd(out)
        out = self.sig(out)

        out = torch.permute(out, (0, 3, 2, 1))
        out = torch.reshape(out, (-1, 49, 5))
        return out

# model = FaceModel('resnet18').to('cuda')
# test = torch.rand(1, 3, 448, 448).to('cuda')
# ans = torch.rand(1, 7, 7, 5)
# out = model(test).to('cpu')
# loss = Loss(out, ans)
# print(loss)


