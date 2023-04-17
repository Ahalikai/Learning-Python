
# link: https://blog.csdn.net/ing100/article/details/125155065

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


    def encoder(self, boxes, labels): # input_boxes (x1, y1, x2, y2), output is ground truth (7*7*30)
        target = torch.zeros((self.S, self.S, int(CLASS_NUM + self.B * 5))) # 7*7*30
        cell_size = 1. / self.S # 1/7

        wh = boxes[:, 2:] - boxes[:, :2] # wh = [w, h] 1*1
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2 # 归一化center of boxes

        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1 # left-top角 (7*7)

            target[int(ij[1]), int(ij[0]), 4] = 1 # first box's confidence
            target[int(ij[1]), int(ij[0]), 9] = 1 # second box's confidence
            target[int(ij[1]), int(ij[0]), int(labels[i]) + self.B * 5] = 1 # 20 classes's confidence

            xy = ij * cell_size # 归一化left-top角 （1*1）

            delta_xy = (cxcy_sample - xy) / cell_size # center - left-top （7*7）

            # w, h is box's width & hight for image's w&h proportion
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i] # w1, h1
            target[int(ij[1]), int(ij[0]), :2] = delta_xy # x1, y1

            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]  # w2, h2
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy  # x2, y2

        return target #(xc, yc) = 7*7 (w, h) = 1*1


    def __getitem__(self, item):
        fname = self.fnames[item]
        img = cv2.imread(os.path.join(self.root + fname))
        boxes = self.boxes[item].clone()
        labels = self.labels[item].clone()
        if self.train:
            img, boxes = self.random_filp(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)

            #img, boxes, labels = self.randomShift(img, boxes, labels)

        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes) # 归一化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # pytorch pretrained use RGB
        img = self.subMean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size)) #resize 448*448
        target = self.encoder(boxes, labels) # encoder 7*7*30

        for t in self.transform:
            img = t(img)

        return img, target


    def __len__(self):
        return self.num_samples

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_filp(self, img, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(img).copy()
            h, w, _ = img.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return img, boxes

    def randomScale(self, img, boxes):
        # width = 0.8 ~ 1.2
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            h, w, c = img.shape
            img = cv2.resize(img, (int(w * scale), h))
            scale_tensor = torch.FloatTensor(
                [[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
        return img, boxes

    def randomBlur(self, img):
        if random.random() < 0.5:
            img = cv2.blur(img, (5, 5))
        return img

    def RandomBrightness(self, img):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def randomShift(self, img, boxex, labels):
        pass


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