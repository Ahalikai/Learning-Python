
# import
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
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
import cv2


from torch.utils.data import Dataset, DataLoader
class Dataset_flower(Dataset):
    def __init__(self, root_dir, ann_file, transform=None):
        self.ann_file = ann_file
        self.root_dir = root_dir
        self.img_label = self.load_annotation()
        self.img = [os.path.join(self.root_dir, img) for img in list(self.img_label.keys())]
        print(self.img[0])

        #img_cv = cv2.imread(self.img[0])
        #cv2.imshow('name', img_cv)
        #cv2.waitKey(0)

        self.label = [label for label in list(self.img_label.values())]
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        image = Image.open(self.img[item])
        label = self.label[item]
        if self.transform is not None:
            print(self)
            image = self.transform(image)
        label = torch.from_numpy(np.array(label))
        return image, label

    def load_annotation(self):
        data_infos = {}
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                data_infos[filename] = np.array(gt_label, dtype=np.int64)
        return data_infos

data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(224),#从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ]),
    'valid': transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = Dataset_flower(root_dir='./flower_data/', ann_file='flower_data/train.txt', transform=data_transforms['train'])
print(len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

image, label = iter(train_loader).next()
print(image.shape)

sample = image[0].squeeze()
sample = sample.permute((1, 2, 0)).numpy()
sample = sample * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
plt.imshow(sample)
plt.show()
print('Label is : {}'.format(label[0].numpy()))

