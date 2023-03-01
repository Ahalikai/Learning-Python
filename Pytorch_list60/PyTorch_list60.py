

# 60题PyTorch简易入门
# https://zhuanlan.zhihu.com/p/102492108
# LeNet5


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms
import cifar_load


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # image 32 * 32 * 3, conv 5 * 5 * 6
        self.conv1 = nn.Conv2d(3, 6, 5)
        # in_clannel 3, out 16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # in 16 * 5 * 5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # conv1 + relu + max_pool
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # conv2 + relu + max_pool
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x -> 列向量
        x = x.view(-1, 16 *  5 * 5)
        # fc1 + relu ...
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

def imshow(img):
    img = img / 2 + 0.5     # 把数据退回(0,1)区间
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

trainset = cifar_load.cifar(root='D:/Chrome/cifar-10', segmentation='train', transforms=transform)
testset = cifar_load.cifar(root='D:/Chrome/cifar-10', segmentation='test', transforms=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


'''

net = Net()
print(net)

params = list(net.parameters())
print(len(params))

for i in range(len(params)):
    print(params[i].size())

in_rand = torch.randn(1, 3, 32, 32)

criterion = nn.MSELoss()
tatget = torch.randn(10).view(1, -1)


net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
out = net(in_rand)

loss = criterion(tatget, out)
optimizer.step()
print(loss)


import torch
import numpy as np

a = input().split(',')
print(a)

# 2.2 梯度
x = torch.randn(3, requires_grad = True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

# 2.1 张量的自动微分
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
print(y, y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)

out.backward()
print(x.grad)

# 1.2 Numpy的操作
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out = a)
print(a, b)

a = torch.ones(5)
a.add_(1)
print(a)
b = a.numpy()
print(b)

# 1.1 张量 : 1 - 13
x = torch.empty(5, 3)
x = torch.rand(5, 3)
x = torch.zeros(5, 3, dtype=torch.long)
x = torch.tensor([5.5, 3])
x = torch.ones(5, 3, dtype=torch.double)
x = torch.randn_like(x, dtype=float)
print(x)
print(x.size())

y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

print(x[:, 1])

y = x.view(15)
y = x.view(3, 5)
y = x.view(5, -1)
print(x.size(), y.size())

x = torch.randn(1)
print(x)
print(x.item())

'''