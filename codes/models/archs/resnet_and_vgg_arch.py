'''
Author: HW
Date: 2025-04-28 16:22:23
LastEditors: [huowei]
LastEditTime: 2025-04-28 16:39:35
Description: 
'''
import torch
import torch.nn as nn
import torchvision
from models.archs.pretrained_model.resnet import ResNetModel


class VGG_128(nn.Module):
    def __init__(self, in_nc, nf):
        super(VGG_128, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)

        self.linear1 = nn.Linear(115200, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.conv0_1(fea))

        fea = self.lrelu(self.conv1_0(fea))
        fea = self.lrelu(self.conv1_1(fea))

        fea = self.lrelu(self.conv2_0(fea))
        fea = self.lrelu(self.conv2_1(fea))

        fea = self.lrelu(self.conv3_0(fea))
        fea = self.lrelu(self.conv3_1(fea))

        fea = self.lrelu(self.conv4_0(fea))
        fea = self.lrelu(self.conv4_1(fea))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True):
        super(ResNetFeatureExtractor, self).__init__()
        device = torch.device('cuda')
        self.use_input_norm = use_input_norm
        model = ResNetModel()
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = model

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output
