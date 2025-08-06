# SMGAN
PyTorch implementation of SMGAN: A Multi-Path Generative Adversarial Network for Remote Sensing Image Super-Resolution Reconstruction

## 1.Dependencies
- Python 3.9
- Python packages: `pip install numpy opencv-python lmdb pyyaml mamba`
- Hardware: NVIDIA A100-80G

## 2.Datasets
- Download link for the RS-SR19 dataset: [Google Drive](https://drive.google.com/file/d/1-XKL9lfDZ7URWZWVBFatmd2FScAyECJC/view?usp=sharing)| [Baidu Netdisk code: b2ik](https://pan.baidu.com/s/1mDbax-JI_ypH0OOXVERGwQ)
- Download link for the AID dataset: [Google Drive](https://drive.google.com/file/d/1FU2JtrTQ6lDvPur8bc5oWwmC1KnLAkDc/view?usp=sharing)| [Baidu Netdisk code: wvzc](https://pan.baidu.com/s/1SYTuVfxeWPBqbprzm8sV8w) Note: This is a public dataset. You can also download it by yourself.
- Download link for the Sanjiangyuan dataset: [Google Drive](https://drive.google.com/file/d/1HDu9hMrPYufyFKNbVSyl47TcQyYXl8VB/view?usp=sharing)| [Baidu Netdisk code: gd7d](https://pan.baidu.com/s/1uoajCBpFtYibsaITIi0kcA)
- ResNet50.pth：[Google Drive](https://drive.google.com/file/d/1UbubPmfFGaXypr31_YwrkVyjysmR7q3x/view?usp=sharing)| [Baidu Netdisk code: 6i7c](https://pan.baidu.com/s/1TMLfCAHIxbD3HnDgySBVeg)

## 3.Train
jsub/bash >./train.sh

## 4.Test
jsub/bash >./val.sh

## 5.important
The final presentation order of the entire file is as follows:
- codes
- dataset
-- train
-- val
-- data_script

- project
  - data
    - raw
    - processed
  - src
    - main.py
    - utils.py
ResNet50.pth is placed in the pretrained_model folder.

## The SMGAM structure diagram is as follows:[Demo](https://hw-star.github.io/SMGAN/)
![image text](https://github.com/hw-star/SMGAN/blob/main/readmeIamges/SMGAM.png)

## Matters needing attention

If you have any question, please feel free to contact the author by 3111314916@qq.com.
