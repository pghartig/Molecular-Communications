import torch
import torch.nn as nn
import torch.nn.functional as F

class viterbiNet(nn.Module):

    def __init__(self, D_in, H1, H2, D_out):
        # initialize inheretence
        super(viterbiNet, self).__init__()

        # initialize weight layers of the network

        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, D_out)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)  # Note that the cross entropy function in PyTorch automatically takes softmax
        x = F.softmax(self.fc3(x))

        return x
