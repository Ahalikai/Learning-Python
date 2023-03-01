import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle

batch_size = 4

def load_cifar_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
        Y = np.array(Y)
        return list(zip(X, Y))

def load_cifar(ROOT):
    dataset = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        batch = load_cifar_batch(f)
        dataset.extend(batch)
    data_train = np.concatenate(dataset)
    del batch
    data_test = load_cifar_batch(os.path.join(ROOT, 'test_batch'))
    return data_train, data_test

class cifar(Dataset):
    def __init__(self, root, segmentation='train', transforms=None):
        if segmentation == 'train':
            self.data = load_cifar(root)[0]
        elif segmentation == 'test':
            self.data = load_cifar(root)[1]
        self.transform = transforms

    def __getitem__(self, item):
        data = self.data[item][0]
        if(self.transform):
            data = (self.transform(data))
        else:
            data = (torch.from_numpy(data))
        label = self.data[item][1]
        return data, label

    def __len__(self):
        return len(self.data)
