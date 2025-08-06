'''
Author: HW
Date: 2025-05-19 17:04:24
LastEditors: [huowei]
LastEditTime: 2025-05-23 15:47:54
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.net_utils import *
from mamba_ssm import Mamba
from timm.layers import DropPath


class MambaBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, depth=6, drop_path_rate=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.project_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.project_out = nn.Conv2d(hidden_dim, 64, kernel_size=1)  

        self.pos_embed = nn.Parameter(torch.zeros(1, hidden_dim, 1))

        self.blocks = nn.ModuleList()
        self.gammas = nn.ParameterList()  # LayerScale

        for i in range(depth):
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                Mamba(d_model=hidden_dim, d_state=16, expand=2 * hidden_dim),
                nn.GELU(),
                DropPath(drop_path_rate)
            ]))
            self.gammas.append(nn.Parameter(1e-2 * torch.ones(hidden_dim)))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.project_in(x).flatten(2).transpose(1, 2)  # [B, L, C]
        x = x + self.pos_embed.transpose(1, 2) 

        for (norm, mamba, act, drop), gamma in zip(self.blocks, self.gammas):
            x_ = norm(x)
            x_ = mamba(x_)
            x_ = act(x_)
            x = x + drop(gamma * x_)

        x = x.transpose(1, 2).view(B, -1, H, W)
        return self.project_out(x)