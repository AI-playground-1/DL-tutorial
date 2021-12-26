import torch
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
# from torchvision.transforms import transforms
import torch.nn.functional as F


class MyCNN(nn.Module):
    def __init__(self, num_channels, classes):
        # call the parent constructor
        super(MyCNN, self).__init__()

        # create CNN with batch norm, otherwise based on LeNet
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=(5, 5))
        self.bn1 = nn.BatchNorm2d(num_features=20)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.bn2 = nn.BatchNorm2d(num_features=50)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = x.view(-1 , 800)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.logsoftmax(x)

        return x


# code below deeply inspired by https://github.com/Atcold/pytorch-Deep-Learning/blob/master/06-convnet.ipynb
def train(model, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # send to device
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, accuracy_list, device):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    input_size = 28*28   # images are 28x28 pixels
    classes = 10      # there are 10 classes    
    cnn = MyCNN(1, classes)
    optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.5)

    accuracy_list = []

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/piotr_gnome/Data', train=True, download=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/piotr_gnome/Data', train=False, download=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=1000, shuffle=True)
    
    for epoch in range(10):
        train(cnn, optimizer, epoch, device)
        test(cnn, accuracy_list, device)

    test = 1