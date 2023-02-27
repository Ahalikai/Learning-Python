

# 60题PyTorch简易入门
# https://zhuanlan.zhihu.com/p/102492108

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