'''
Author: HW
Date: 2025-04-28 16:07:04
LastEditors: [huowei]
LastEditTime: 2025-04-28 16:09:00
Description: 
'''
import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .SMGAN_model import SMGANModel
    model = SMGANModel(opt)
    logger.info('Model [{:s}] is created.'.format(model.__class__.__name__))
    return model
