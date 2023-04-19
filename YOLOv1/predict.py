
# link: https://blog.csdn.net/ing100/article/details/125155065

import numpy as np
import torch
import cv2
from torchvision.transforms import ToTensor
from resnet50 import resnet50

model = resnet50()
model.load_state_dict(torch.load("yolo.pth"))
model.eval()

confident = 0.35
iou_con = 0.5

VOC_CLASSES = (
    'cscn', 'lkbn', 'tkbe', 'person'
)
CLASS_NUM = len(VOC_CLASSES)

class Pred():
    def __init__(self, model, img_root):
        self.model = model
        self.img_root = img_root

    def result(self):
        img = cv2.imread(self.img_root)
        h, w, _ = img.shape
        print(h, w)
        image = cv2.resize(img, (448, 448))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean = (123, 117, 104)
        img = img - np.array(mean, dtype=np.float32)
        transform = ToTensor()
        img = transform(img)
        img = img.unsqueeze(0)
        Result = self.model(img)
        bbox = self.Decode(Result)
        bboxes = self.NMS(bbox)

        if len(bboxes) == 0:
            print("No detect ang things")
        for i in range(0, len(bboxes)):
            for j in range(0, 4):
                bboxes[i][j] = bboxes[i][j] * 64

            x1 = bboxes[i][0].item()
            x2 = bboxes[i][1].item()
            y1 = bboxes[i][2].item()
            y2 = bboxes[i][3].item()
            class_num = bboxes[i][5].item()
            print(x1, x2, y1, y2, VOC_CLASSES[int(class_num)])

            cv2.rectangle(image, (int(x1), int(y1), int(x2), int(y2)), (144, 144, 255))

        cv2.imshow('img', image)
        cv2.waitKey(0)

    def Decode(self):
        pass

if __name__ == "__main__":
    img_root = "D:\pycharm\yolov7-main\cscn/test-images/1-1.jpg"
    Pred(model, img_root)
    Pred.result()









