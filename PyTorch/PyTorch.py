
import torch
import torch.nn.functional as F

loss_func = F.cross_entropy()

def model(xb):
    pass


bs = 64

print(torch.cuda.is_available())