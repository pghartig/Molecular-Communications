import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractclassmethod

#TODO import ABC
class metric(ABC):
    def __init__(self,received):
        self.received = np.flip(received)

    @classmethod
    def metric(self, index):
        pass


class gaussian_channel_metric_working(metric):
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
        metric.__init__(self, received)
        self.parameters = csi

    def metric(self, index, states):
        costs = []
        for state in states:
            channel_output = self.received[0, index]
            predicted = np.dot(np.asarray(state), np.flip(self.parameters).T)
            cost = np.linalg.norm((predicted - channel_output))
            costs.append(cost)
        return np.asarray(costs)

class nn_mm_metric(metric):

    """
    returns vector of metrics for incoming state of viterbi with a gaussian channel
    :param survivor_paths:
    :param index:
    :param transmit_alphabet:
    :param channel_output:
    :param cir:
    :return:
    """
    def __init__(self, nn, mm, received, input_length=1):
        metric.__init__(self, received)
        self.nn = nn
        self.mm = mm
        self.received = np.flip(received)
        self.nn_input_size = input_length-1

    def metric(self, index, state=None):
        # Be careful using the PyTorch parser with scalars
        torch_input = torch.tensor([self.received[0, index-self.nn_input_size:index+1]]).float()
        nn = self.nn(torch_input).flatten()
        mm = self.mm(self.received[0, index])
        # return -nn*mm  # Provides metrics for entire column of states
        return - nn    # Need to change sign to align with argmin used in viterbi

    def llr_metric(self, index, states):
        # Be careful using the PyTorch parser with scalars
        torch_input = torch.tensor([self.received[0, index-self.nn_input_size:index+1]]).float()
        nn = self.nn(torch_input).flatten()
        mm = self.mm(self.received[0, index])


