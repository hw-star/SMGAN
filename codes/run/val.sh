#!/bin/sh
###
 # @Author: HW
 # @Date: 2025-06-09 15:53:53
 # @LastEditors: [huowei]
 # @LastEditTime: 2025-06-23 15:16:56
 # @Description: 
### 
#JSUB -m gpu05
#JSUB -q qgpu
#JSUB -n 1
#JSUB -e /gpfs/home/huowei/SMG/log/val/error.%J
#JSUB -o /gpfs/home/huowei/SMG/log/val/output.%J
#JSUB -J SMGAN-val

your_model=
dataset=val_7hw   # val_1st, val_2nd, val_3rd, val_4th, val_5th

CUDA_VISIBLE_DEVICES=0 /gpfs/home/huowei/anaconda3/envs/DGT/bin/python val.py \
  --model_path=${your_model} \
  --exp_name=SMGAN_val \
  --dataset=${dataset} \
  --save_path=./results
