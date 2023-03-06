
# import
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision

from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image


data_dir = './flower_data/'
train_dir = data_dir + '/train'
test_dir = data_dir + '/valid'

data_transforms = {
    'train' :
        transforms.Compose([
            transforms.Resize([96, 96]), # resize
            transforms.RandomRotation(45), # rotation
            transforms.CenterCrop(64), # 中心裁剪
            transforms.RandomHorizontalFlip(p = 0.5), # 水平旋转 P概率
            transforms.RandomVerticalFlip(p = 0.5), # 垂直旋转 P概率
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1), #亮度、对比度、饱和度、色相
            transforms.RandomGrayscale(p = 0.025), # 概率转化为灰度
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
        ]),
    'valid' :
        transforms.Compose({
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        }),
}

batch_size = 8
image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x : torch.utils.data.Dataloader(image_datasets[x], batch_size = batch_size, shuffle = True) for x in ['train', 'valid']}
dataset_sizes = {x : len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

