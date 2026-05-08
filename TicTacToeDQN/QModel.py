import torch

class QModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(3, 3)
        self.layer1 = torch.nn.Linear(30, 100)
        self.layer2 = torch.nn.Linear(100, 100)
        self.layer3 = torch.nn.Linear(100, 9)
        self.relu = torch.nn.ReLU()
        self.softm = torch.nn.Softmax(dim=1)

    def forward(self, states2d, turns):
        x = torch.cat([states2d.flatten(1), turns[:,None]], 1)
        x = self.relu(self.embedding(x)).flatten(1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        x = self.softm(x)
        return x
