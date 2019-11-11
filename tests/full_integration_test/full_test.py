"""
Full Test for using the mixture model and neural network to provide metrics to be used in the viterbi algorithm
"""


from communication_util.em_algorithm import mixture_model
from nn_utilities import *
from communication_util.error_rates import *
import numpy as np
from viterbi.viterbi import *
from communication_util.data_gen import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.error_rates import *

def test_full_integration():

    error_tolerance = np.power(10.0, -3)

    """
    Load in Trained Neural Network
    """

    saved_network_path = '/Users/peterhartig/Documents/Projects/moco_project/molecular-communications-project/Output/nn.pt'
    neural_net = torch.load(saved_network_path)

    """
    Load Trained Mixture Model
    """
    mm_pickle_in = open(
        "/Users/peterhartig/Documents/Projects/moco_project/molecular-communications-project/Output/mm.pickle", "rb")
    model = pickle.load(mm_pickle_in)
    mm = mixture_model(mu=model[0], sigma_square=model[1],alpha=model[2])
    mm = mm.get_probability
    mm_pickle_in.close()

    """
    Generated Testing Data using the same channel as was used for training the mixture model and the nn
    """
    number_symbols = 60

    #TODO look at handling complex channels

    # channel = 1j * np.zeros((1, 5))
    channel =  np.zeros((1, 5))
    channel[0, [0, 3, 4]] = 1, 0.5, 0.4
    # channel[0, [0, 3]] = 1, 0.7
    # channel[0, [0]] = 1

    # TODO make consolidate this part
    data_gen = \
        training_data_generator(SNR=1, symbol_stream_shape=(1, number_symbols), channel=channel, plot=True)
    # data_gen = training_data_generator(plot=True)

    data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()

    """
    After sending through channel, symbol detection should be performed using something like a matched filter
    """


    metric = nn_mm_metric(neural_net, mm, data_gen.channel_output)
    detected = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                        metric.metric)



    """
    Analyze SER performance
    """

    ser = symbol_error_rate(detected, data_gen.symbol_stream_matrix)
    assert error_tolerance >= ser
