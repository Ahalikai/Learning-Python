
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

    def Decode(self, result): #1*7*30
        result = result.squeeze() #7*7*30
        grid_ceil1 = result[:, :, 4].unsqueeze(2) # 7*7*1
        grid_ceil2 = result[:, :, 9].unsqueeze(2)
        grid_ceil_con = torch.cat((grid_ceil1, grid_ceil2), 2) # 7*7*2
        grid_ceil_con, grid_ceil_index = grid_ceil_con.max(2)
        class_p, class_index = result[:, :, 10:].max(2) # 找出单个grid ceil预测的物体类别最大值
        class_confidence = class_p * grid_ceil_con # 真实类别概率
        bbox_info = torch.zeros(7, 7, 6)
        for i in range(0, 7):
            for j in range(0, 7):
                bbox_index = grid_ceil_index[i, j]
                bbox_info[i, j, :5] = result[i, j, (bbox_index * 5):(bbox_index + 1) * 5] # 筛选
        bbox_info[:, :, 4] = class_confidence
        bbox_info[:, :, 5] = class_index
        print(bbox_info[1, 5, :])
        return bbox_info # 7*7*6    6 = bbox4 + confience + class

    def NMS(self, bbox, iou_con = iou_con):
        for i in range(0, 7):
            for j in range(0, 7):
                xmin = j + bbox[i, j, 0] - bbox[i, j, 2] * 7 / 2
                xmax = j + bbox[i, j, 0] + bbox[i, j, 2] * 7 / 2
                ymin = i + bbox[i, j, 1] - bbox[i, j, 3] * 7 / 2
                ymax = i + bbox[i, j, 1] + bbox[i, j, 3] * 7 / 2

                bbox[i, j, 0] = xmin
                bbox[i, j, 1] = xmax
                bbox[i, j, 2] = ymin
                bbox[i, j, 3] = ymax

        bbox = bbox.view(-1, 6) # 49 * 6
        bboxes = []
        ori_class_index = bbox[:, 5]
        class_index, class_order = ori_class_index.sort(dim=0, descending=False)
        class_index = class_index.tolist() #从0->7排序
        bbox = bbox[class_order, :] # 更改bbox排序顺序
        a = 0
        for i in range(0, CLASS_NUM):
            num = class_index.count(i)
            if num == 0:
                continue
            x = bbox[a : a+num, :] # 同一类别的所有信息
            score = x[:, 4]
            score_index, score_order = score.sort(dim=0, descending=True)
            y = x[score_order, :] # t同一类别按置信度排序
            if y[0, 4] >= confident:
                for k in range(0, num):
                    y_score = y[:, 4]
                    _, y_score_order = y[:, 4]
                    _, y_score_order = y_score.sort(dim=0, descending=True)
                    y = y[y_score_order, :]
                    if y[k, 4] > 0:
                        area0 = (y[k, 1] - y[k, 0]) * (y[k, 3]- y[k, 2])
                        for j in range(k+1, num):
                            area1 = (y[j, 1] - y[j, 0]) * (y[j, 3]- y[j, 2])
                            x1 = max(y[k, 0], y[j, 0])
                            x2 = min(y[k, 1], y[j, 1])
                            y1 = max(y[k, 2], y[j, 2])
                            y2 = min(y[k, 3], y[j, 3])
                            w = x2 - x1
                            h = y2 - y1
                            if w < 0 or h < 0:
                                w = h = 0
                            inter = w * h
                            iou = inter / (area0 + area1 - inter)
                            if iou >= iou_con or y[j, 4] < confident:
                                y[j, 4] = 0

                for mask in range(0, num):
                    if y[mask, 4] > 0:
                        bboxes.append(y[mask])

            a = num + a

        return bboxes

if __name__ == "__main__":
    img_root = "D:\pycharm\yolov7-main\cscn/test-images/1-1.jpg"
    Pred = Pred(model, img_root)
    Pred.result()









