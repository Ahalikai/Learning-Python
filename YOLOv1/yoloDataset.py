
import torch
import cv2
import os
import os.path
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image

CLASS_NUM = 4

class yoloDataset(Dataset):
    image_size = 448

    def __init__(self, img_root, list_file, train, transform): #list_file is .txt
        self.root = img_root
        self.train = train
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []

        #parameter
        self.S = 7
        self.B = 2
        self.C = CLASS_NUM
        self.mean = (123, 117, 104) # RGB
        file_txt = open(list_file)
        lines = file_txt.readlines()

        for line in lines:
            splited = line.strip().split() #移除首位换行符，再生成列表
            self.fnames.append(splited[0]) #image name
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x = float(splited[1 + 5 * i])
                y = float(splited[2 + 5 * i])
                x2 = float(splited[3 + 5 * i])
                y2 = float(splited[4 + 5 * i])
                c = splited[5 + 5 * i] # classes
                box.append([x, y, x2, y2])
                label.append(int(c))

            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.Tensor(label))

        self.num_samples = len(self.boxes)

    def __getitem__(self, item):
        fname = self.fnames[item]
        img = cv2.imread(os.path.join(self.root + fname))
        boxes = self.boxes[item].clone()
        labels = self.labels[item].clone()
        if self.train:
            pass
        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes) # 归一化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # pytorch pretrained use RGB
        img = self.subMean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size)) #resize 448*448
        target = self.encoder(boxes, labels) # encoder 7*7*30

        for t in self.transform:
            img = t(img)

        return img, target

    def encoder(self, boxes, labels):
        pass




    def __len__(self):
        return self.num_samples

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr




def main():
    file_root = 'D:\pycharm\yolov5-7.0\mydata\JPG/'
    train_dataset = yoloDataset(
        img_root=file_root,
        list_file='train.txt',
        train=True,
        transform=[ToTensor()]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        drop_last=True,
        shuffle=False,
        num_workers=0
    )
    train_iter = iter(train_loader)
    for i in range(100):
        img, target = next(train_iter)
        print(img.shape)

if __name__ == '__main__':
    main()