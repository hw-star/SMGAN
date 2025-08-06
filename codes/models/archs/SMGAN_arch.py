import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.net_utils import *
from models.archs.mamba import MambaBlock


class SMGANNet(nn.Module):
    """
    PyTorch Module for RRSNet.
    Now x4 is only supported.

    Parameters
    ---
    ngf : int, optional
        the number of filterd of generator.
    n_blocks : int, optional
        the number of residual blocks for each module.
    """
    def __init__(self, ngf=64, n_blocks=16):
        super(SMGANNet, self).__init__()
        self.get_g_nopadding = Get_gradient_nopadding()

        self.Encoder_First = ContentExtractor(ngf, n_blocks)
        self.Encoder_First_grad = ContentExtractor(ngf, n_blocks)

        self.Encoder_Second = Encoder(ngf)
        self.Encoder_Second_grad = Encoder(ngf)

        self.Encoder_Third = MambaBlock(in_channels = 3, depth=4)
        self.Encoder_Third_grad = MambaBlock(in_channels = 3, depth=4)

        self.merge = Merge()
        self.ram = RAM(nf=64)
        self.Reconstruction = EnhancedReconstruction()

        self.cross_ram = RAM(nf=64)
        self.cross_fuser = CrossFusion(64)


    def forward(self, LR, weights=None):
        LR_grad = self.get_g_nopadding(LR)

        cnn_LR = self.Encoder_First(LR) # content_feat-LR:4 64 120 120
        cnn_LR_grad = self.Encoder_First_grad(LR_grad) # content_feat-LR:4 64 120 120
        cnn_feat = self.ram(cnn_LR, cnn_LR_grad) #############

        st_LR = self.Encoder_Second(LR)
        st_LR_grad = self.Encoder_Second_grad(LR_grad) 
        st_feat = self.ram(st_LR, st_LR_grad)

        mb_LR = self.Encoder_Third(LR)
        mb_LR_grad = self.Encoder_Third_grad(LR_grad)
        mb_feat = self.ram(mb_LR, mb_LR_grad)

        global_guide = self.cross_fuser(cnn_feat, st_feat, mb_feat)
        cnn_feat = self.cross_ram(cnn_feat, global_guide)
        st_feat  = self.cross_ram(st_feat, global_guide)
        mb_feat  = self.cross_ram(mb_feat, global_guide)

        maps = [cnn_feat, st_feat, mb_feat]
        merge = self.merge(maps)

        return self.Reconstruction(merge)


class CrossFusion(nn.Module):
    def __init__(self, channels):
        super(CrossFusion, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x1, x2, x3):
        fused = torch.cat([x1, x2, x3], dim=1)
        return self.fuse_conv(fused)
    
    
class Merge(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, fused_channels=80):
        super(Merge, self).__init__()
        self.conv_cnn = nn.Conv2d(in_channels, fused_channels, 3, 1, 1, bias=True)
        self.conv_st = nn.Conv2d(in_channels, fused_channels, 3, 1, 1, bias=True)
        self.conv_mb = nn.Conv2d(in_channels, fused_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fusion_conv = nn.Conv2d(fused_channels * 3, out_channels, kernel_size=3, padding=1)



    def forward(self, maps):
        LR_cnn_feat = self.lrelu(self.conv_cnn(maps[0]))
        LR_st_feat = self.lrelu(self.conv_st(maps[1]))
        LR_map_feat = self.lrelu(self.conv_mb(maps[2]))
        fused = torch.cat([LR_cnn_feat, LR_st_feat, LR_map_feat], dim=1)
        fused_feat = self.lrelu(self.fusion_conv(fused))

        return fused_feat
        

class ContentExtractor(nn.Module):
    def __init__(self, ngf=64, n_blocks=15):
        super(ContentExtractor, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )
        
    def forward(self, x):
        h = self.head(x)
        h = self.body(h) + h
        return h
    
class Reconstruction(nn.Module):
    def __init__(self):
        super(Reconstruction, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(8, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, merge):
        x = self.lrelu(self.conv1(merge))
        x = self.pixel_shuffle(x)
        x = self.lrelu(self.conv2(x))

        return x
             

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        # Fusion layer
        self.fusion_conv = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)  # Adjust in_channels as needed
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        
        fused_features = torch.cat((x1, x2), dim=1)  # Output: (batch_size, 128, 30, 30)
        
        fused_features = self.fusion_conv(fused_features)  # Output: (batch_size, 64, 30, 30)
        fused_features = self.bn(fused_features)
        fused_features = self.relu(fused_features)
        return fused_features

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match the input and output dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        # Sobel kernels
        sobel_v = [[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]]
        sobel_h = [[-1,  0,  1],
                   [-2,  0,  2],
                   [-1,  0,  1]]
        kernel_h = torch.FloatTensor(sobel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(sobel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)

        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)


    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)

        return x

class RAM(nn.Module):
    def __init__(self, nf=64):
        super(RAM, self).__init__()
        self.mul_conv1 = nn.Conv2d(nf * 2, 128, kernel_size=3, stride=1, padding=1)
        self.mul_conv2 = nn.Conv2d(128, nf, kernel_size=3, stride=1, padding=1)
        self.add_conv1 = nn.Conv2d(nf * 2, 128, kernel_size=3, stride=1, padding=1)
        self.add_conv2 = nn.Conv2d(128, nf, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, features, conditions):
        cat_input = torch.cat((features, conditions), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.lrelu(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.lrelu(self.add_conv1(cat_input)))
        return features * mul + add


class EnhancedReconstruction(nn.Module):
    def __init__(self):
        super(EnhancedReconstruction, self).__init__()
        self.rrdb = RRDB(128)
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(8, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, merge):
        x = self.rrdb(merge)
        x = self.lrelu(self.conv1(x))
        x = self.pixel_shuffle(x)
        x = self.lrelu(self.conv2(x))
        return x


class RRDB(nn.Module):
    def __init__(self, nf):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(nf)
        self.rdb2 = ResidualDenseBlock(nf)
        self.rdb3 = ResidualDenseBlock(nf)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.3 + x


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + x