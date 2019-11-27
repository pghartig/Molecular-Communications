import numpy as np
import torch
import torch.nn.functional as F

class gaussian_channel_metric_working():
    """
    returns vector of metrics for incoming state of viterbi with a gaussian channel
    :param survivor_paths:
    :param index:
    :param transmit_alphabet:
    :param channel_output:
    :param cir:
    :return:
    """
    def __init__(self, csi, received):
        self.parameters = csi
        self.received = np.flip(received)

    def metric(self, index, states):
        costs = []
        for state in states:
            channel_output = self.received[0, index]
            predicted = np.dot(np.asarray(state), np.flip(self.parameters).T)
            cost = np.linalg.norm((predicted - channel_output))
            costs.append(cost)
        return np.asarray(costs)

class nn_mm_metric():

    """
    returns vector of metrics for incoming state of viterbi with a gaussian channel
    :param survivor_paths:
    :param index:
    :param transmit_alphabet:
    :param channel_output:
    :param cir:
    :return:
    """
    def __init__(self, nn, mm, received):
        self.nn = nn
        self.mm = mm
        self.received = np.flip(received)

    def metric(self, index, state=None):
        torch_input = torch.tensor([self.received[0, index]])   # Be careful using the PyTorch parser with scalars
        nn = self.nn(torch_input)
        mm = self.mm(self.received[0, index])
        # return -nn*mm  # Provides metrics for entire column of states
        return - nn    # Need to change sign to align with argmin used in viterbi


