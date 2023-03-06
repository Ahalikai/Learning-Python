
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
    'train' : 1
}