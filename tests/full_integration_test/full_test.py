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

    saved_network_path = "tests/NN_related/nn.pt"
    neural_net = torch.load(saved_network_path)

    """
    Load Trained Mixture Model
    """
    mm_pickle_in = open("tests/Mixture_Model/Output/mm.pickle", "rb")
    model = pickle.load(mm_pickle_in)
    mm = mixture_model(mu=model[0], sigma_square=model[1],alpha=model[2])
    mm = mm.get_probability
    mm_pickle_in.close()

    """
    Generated Testing Data using the same channel as was used for training the mixture model and the nn
    """
    number_symbols = 5000

    # channel = np.zeros((1, 5))
    # channel[0, [0, 3, 4]] = 1, 0.5, 0.4

    channel = np.zeros((1, 3))
    channel[0, [0, 1, 2]] = 1, 0.6, 0.3

    # channel = np.zeros((1, 1))
    # channel[0, 0] = 1

    data_gen = training_data_generator(
        symbol_stream_shape=(1, number_symbols),
        SNR=5,
        plot=True,
        channel=channel,
    )
    # data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    """
    After sending through channel, symbol detection should be performed using something like a matched filter
    """


    metric = nn_mm_metric(neural_net, mm, data_gen.channel_output)
    detected_nn = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                        metric.metric)
    ser_nn = symbol_error_rate(detected_nn, data_gen.symbol_stream_matrix)


    """
    Compare to Classical Viterbi with full CSI
    """

    metric = gaussian_channel_metric_working(channel, data_gen.channel_output)
    detected_classic = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                        metric.metric)
    ser_classic = symbol_error_rate(detected_classic, data_gen.symbol_stream_matrix)

    """
    Analyze SER performance
    """

    assert error_tolerance >= ser_nn
