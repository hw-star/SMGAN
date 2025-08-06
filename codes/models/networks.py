'''
Author: HW
Date: 2025-04-28 16:12:15
LastEditors: [huowei]
LastEditTime: 2025-04-28 18:19:45
Description: 
'''
import torch
from torch.nn import init
import models.archs.resnet_and_vgg_arch
from models.archs.SMGAN_arch import SMGANNet
from models.archs.resnet_and_vgg_arch import VGG_128, ResNetFeatureExtractor
import functools
import logging
logger = logging.getLogger('base')

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        if m.affine != False:

            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    return SMGANNet(ngf=opt_net['nf'], n_blocks=opt_net['n_blocks'])


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    return VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])


def define_D_grad(opt):
    opt_net = opt['network_D']
    return VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])


# Define network used for perceptual loss
def define_F(opt):
    gpu_ids = opt['gpu_ids']
    netF = ResNetFeatureExtractor(use_input_norm=True)
    netF.eval()  # No need to train
    return netF
