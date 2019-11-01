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

        # self.fc1 = torch.randn(D_in, H1, device=device, requires_grad=False)
        # self.fc2 = torch.randn(H1, H2, device=device, requires_grad=False)
        # self.fc3 = torch.randn(H2, D_out, device=device, requires_grad=False)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))

        return x
