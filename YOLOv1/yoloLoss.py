
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings

warnings.filterwarnings('ignore') # ignore warning
CLASS_NUM = 4


class yoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        # l_coord = 5, l_noobj = 0.5
        super(yoloLoss, self).__init__()
        self.S = S # B = 7
        self.B = B # B = 2
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2): #box1(2, 4) box2(1, 4)
        N = box1.size(0) # 2
        M = box2.size(0) # 1

        lt = torch.max( # return torch's max
            # [N,2] -> [N,1,2] -> [N,M,2]
            box1[:, :2].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),
        )

        rb = torch.min(  # return torch's min
            # [N,2] -> [N,1,2] -> [N,M,2]
            box1[:, :2].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),
        )

        wh = rb - lt # [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, N] 重复面积

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]) # [N, ]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1]) # [M, ]

        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [N,] -> [N, 1] -> [N, M]

        iou = inter / (area1 + area2 - inter)
        return iou # [2, 1]

    def forward(self, pred_tensor, target_tensor):
        '''
        :param pred_tensor: (tensor) [batchsize, 7, 7, 30]
        :param target_tensor: (tensor) [batchsize, 7, 7, 30] --ground truth
        :return:
        '''
        N = pred_tensor.size()[0] # batchsize
        coo_mask = target_tensor[:, :, :, 4] > 0 # 有目标标签的索引值 true batchsize * 7 * 7
        noo_mask = target_tensor[:, :, :, 4] == 0 # 无目标标签的索引值 false batchsize * 7 * 7
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor) # get含物体的坐标信息，batchsize * 7 * 7 * 30
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor) # get不含物体的坐标信息，batchsize * 7 * 7 * 30

        coo_pred = pred_tensor[coo_mask].view(-1, int(CLASS_NUM + self.B * 5)) # view 类似于 reshape
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5) # X行5列 （-1 为自动计算
        class_pred = coo_pred[:, 10:] # [n_coord, 20]

        coo_target = target_tensor[coo_mask].view(-1, int(CLASS_NUM + self.B * 5))
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        # 不包含物体的grid ceil的置信度损失
        noo_pred = pred_tensor[noo_mask].view(-1, int(CLASS_NUM + self.B * 5))
        noo_target = target_tensor[noo_mask].view(-1, int(CLASS_NUM + self.B * 5))
        noo_pred_mask = torch.cuda.ByteTeneor(noo_pred.size()).bool()
        noo_pred_mask.zero_()
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        noo_pred_c = noo_pred[noo_pred_mask] # noo_pred只计算c的损失size[-1, 2]
        noo_target_c = noo_target[noo_pred_mask]





















