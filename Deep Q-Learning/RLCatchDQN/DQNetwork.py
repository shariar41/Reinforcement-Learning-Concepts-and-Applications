
import torch
import torch.nn as nn
import torch.nn.functional as F
class DQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) # input is cwith 4 channels
        # 32 is the number of feature maps and the kernel size is 5x5
        self.pool = nn.MaxPool2d(2,2)
        # maxpool will be used multiple times, but defined once
        # in_channels = 32 because self.conv1 output is 32 channels
        self.conv2 = nn.Conv2d(32,6,5)
        # 22*22 comes from the dimension of the last conv layer
        self.fc1 = nn.Linear(6*22*22, 100)
        self.fc2 = nn.Linear(100, 3)
        self.sm = nn.Softmax(dim=1)
    def forward(self, x):
        #x2 = x.reshape(x.shape[0],1,100,100)
        x2 = self.pool(F.relu(self.conv1(x)))
        x2 = self.pool(F.relu(self.conv2(x2)))
        x2 = x2.view(-1, 6*22*22)
        x2 = F.relu(self.fc1(x2))
        x2 = self.sm(self.fc2(x2)) # softmax activation on final layer
        return x2