'''
Author: HW
Date: 2025-04-28 16:40:10
LastEditors: [huowei]
LastEditTime: 2025-04-28 18:23:02
Description: 
'''
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        resnet = models.resnet50(pretrained=False)
        pretrained_dict = torch.load('/gpfs/home/huowei/SMG/codes/models/archs/pretrained_model/resnet50-11ad3fa6.pth')
        resnet.load_state_dict(pretrained_dict)
        resnet = nn.Sequential(*list(resnet.children())[:-2])
        resnet.lastConv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.lastConv.parameters():
            param.requires_grad = True

        self.resnet = nn.Sequential(*list(resnet.children()))

    def forward(self, x):
        return self.resnet(x)

if __name__ == '__main__':
    model = ResNetModel()
    print(model.parameters())
