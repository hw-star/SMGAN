#!/bin/sh
###
 # @Author: HW
 # @Date: 2025-04-28 17:29:17
 # @LastEditors: [huowei]
 # @LastEditTime: 2025-06-20 22:45:10
 # @Description: 
### 
#JSUB -m gpu05
#JSUB -q qgpu
#JSUB -n 1
#JSUB -e /gpfs/home/huowei/SMG/log/error.%J
#JSUB -o /gpfs/home/huowei/SMG/log/output.%J
#JSUB -J SMG-Baseline

CUDA_VISIBLE_DEVICES=2 /gpfs/home/huowei/anaconda3/envs/DGT/bin/python train.py -opt /gpfs/home/huowei/SMG/codes/configs/SMGAN.yml
