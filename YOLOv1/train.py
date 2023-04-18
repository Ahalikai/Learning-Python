
# link: https://blog.csdn.net/ing100/article/details/125155065

from yoloDataset import yoloDataset
from yoloLoss import yoloLoss
from  resnet50 import resnet50
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

device = 'cuda'
file_root = 'D:\pycharm\yolov5-7.0\mydata\JPG/'
batch_size = 6
learning_rate = 0.001
num_epochs = 100

train_dataset = yoloDataset(img_root=file_root, list_file='train.txt', train=True, transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataset = yoloDataset(img_root=file_root, list_file='test.txt', train=False, transform=[transforms.ToTensor()])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print('the dataset has %d images' % (len(train_dataset)))

net = resnet50()
net = net.cuda()
resnet = models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()
op = net.state_dict()

for new_state_dict_num, new_state_dict_value in enumerate(new_state_dict.values()):
    for op_num, op_key in enumerate(op.keys()):
        if op_num == new_state_dict_num and op_num <= 317:
            op[op_key] = new_state_dict_value
net.load_state_dict(op)

print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

criterion = yoloLoss(7, 2, 5, 0.5)
criterion = criterion.to(device)
net.train()

params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    params += [{'params' : [value], 'lr' : learning_rate}]

optimizer = torch.optim.SGD(
    params,
    lr=learning_rate,
    momentum=0.9,
    weight_decay=5e-4
)

for epoch in range(num_epochs):
    net.train()
    if epoch == 60:
        learning_rate = 0.0001
    if epoch == 80:
        learning_rate = 0.00001
    for params_group in optimizer.param_groups:
        params_group['lr'] = learning_rate
    print('\n\nStaring epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    total_loss = 0
    for i, (images, target) in enumerate(train_loader):
        images, target = images.cuda(), target.cuda()
        pred = net(images)
        loss = criterion(pred, target)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (
                epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), total_loss / (i + 1)
            ))

    validation_loss = 0.0
    net.eval()
    for i, (images, target) in enumerate(test_loader):
        images, target = images.cuda(), target.cuda()
        pred = net(images)
        loss = criterion(pred, target)
        validation_loss += loss.item()
    validation_loss /= len(test_loader)

    best_test_loss = validation_loss
    print('get best test loss %.5f' % best_test_loss)
    torch.save(net.state_dict(), 'yolo.pth')
