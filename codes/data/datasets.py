'''
Author: HW
Date: 2025-04-28 15:54:22
LastEditors: [huowei]
LastEditTime: 2025-04-28 18:09:35
Description: 
'''
import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
# from data.util import imresize_np

class SMGANDataset(data.Dataset):

    def __init__(self, opt):
        super(SMGANDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environments for lmdb

        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])


    def _init_lmdb(self):
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.GT_env is None):
            self._init_lmdb()
        LQ_path, GT_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        GT_path = self.paths_GT[index]
        resolution = [int(s) for s in self.sizes_GT[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        if self.opt['phase'] != 'train':  # modcrop in the validation / test phase
            img_GT = util.modcrop(img_GT, scale)
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            resolution = [int(s) for s in self.sizes_LQ[index].split('_')
                          ] if self.data_type == 'lmdb' else None
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)

        if self.opt['phase'] == 'train':
            H, W, C = img_GT.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
    
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()

        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
