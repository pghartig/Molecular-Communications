import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The various Neural Networks used in this work. 
"""


class ViterbiNet(nn.Module):
    """
    The neural network used in the ViterbiNet paper
    """
    def __init__(self, D_in, H1, H2, D_out):
        # initialize inheretence
        super(ViterbiNet, self).__init__()

        # initialize weight layers of the network
        self.name = "viterbiNet"
        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, D_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)  # Note that the cross entropy function in PyTorch automatically takes softmax
        # x = self.fc3(x)
        x = F.log_softmax(self.fc3(x))
        return x


class ViterbiNetDropout(nn.Module):
    """
    The neural network used in the ViterbiNet paper with dropout during training
    """
    def __init__(self, D_in, H1, H2, D_out, dropout_probability):
        # initialize inheretence
        super(ViterbiNetDropout, self).__init__()

        # initialize weight layers of the network
        self.drop_out = nn.Dropout(dropout_probability)
        self.name = "viterbiNet"
        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, D_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.drop_out(x)
        x = F.relu(self.fc2(x))
        x = self.drop_out(x)
        # x = self.fc3(x)
        # x = self.fc3(x)  # Note that the cross entropy function in PyTorch automatically takes softmax
        x = F.log_softmax(self.fc3(x))
        return x


class ViterbiNetDeeper(nn.Module):
    """
    The neural network used in the ViterbiNet paper with additional layers
    """
    def __init__(self, D_in, H1, H2, H3, D_out, dropout_probability):
        # initialize inheretence
        super(ViterbiNetDeeper, self).__init__()
        self.name = "deeper_viterbiNet"

        # initialize weight layers of the network
        self.drop_out = nn.Dropout(dropout_probability)
        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, H3)
        self.fc4 = nn.Linear(H3, D_out)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.drop_out(x)
        x = F.relu(self.fc2(x))
        x = self.drop_out(x)
        x = F.relu(self.fc3(x))
        x = self.drop_out(x)
        # x = self.fc4(x)
        # Note that the cross entropy function in PyTorch automatically takes softmax
        x = F.log_softmax(self.fc4(x))


        return x
