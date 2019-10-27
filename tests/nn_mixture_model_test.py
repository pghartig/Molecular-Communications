from communication_util.em_algorithm import *
import numpy as np
import matplotlib.pyplot as plt
from communication_util.data_gen import training_data_generator

#TODO implement this test

def nn_mm_test():
    """
    This tests incorporates the neural network output with the mixture model

    :return:
    """
    tolerance = np.power(10.0, -3)

    """
    generate a set of training data using a static channel    
    """
    channel = 1j*np.zeros((1, 5))
    channel[0, [0, 3, 4]] = 1, 0.5, 0.4
    data_gen = training_data_generator(SNR=10, plot=True, channel=channel)
    data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()

    """
    create mixture model from training data
    """
    # TODO should this be the number of non-zero (or nearly non-zero) elements in the channel
    num_sources = np.power(data_gen.alphabet.size, data_gen.CIR_matrix.shape[1])
    mu, variance, alpha = em_gausian(num_sources, data_gen.channel_output.T, 30)

    """
    Generate new stream of symbols on which to evaluate performance
    """

    channel = 1j * np.zeros((1, 5))
    channel[0, [0, 3, 4]] = 1, 0.5, 0.4
    data_gen = training_data_generator(SNR=10, plot=True, channel=channel)
    data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()

    """ 
    Decoding section
    """
    for symbol in data_gen.channel_output:

        """
        Get Neural Network (may have to store weights of the network somewhere to prevent training every time)
        """

        """
        Using the mixture model created from the training data find the probability of incoming symbol y[i]
        """
        p_yi = alpha*probability_from_gaussian_sources(mu, variance, symbol)


        """
        Perform decoding using neural network and mixture model as inputs to the viterbi algorithm
        """



    assert False

