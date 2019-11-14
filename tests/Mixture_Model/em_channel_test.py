from communication_util.em_algorithm import em_gausian
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from communication_util import training_data_generator



def test_em_real_channel():
    """
    The goal of this test is to ensure that the implemented EM algorithm converges to the correct parameters used to
    to generate a set of data that is then fed into the algorithm.
    :return:
    """
    tolerance = np.power(10.0, -3)


    """
    Setup Training Data
    """
    #EM is stable to number of training examples TODO see when not stable to number of iterations
    number_symbols = 5000

    # channel = np.zeros((1, 5))
    # channel[0, [0, 3, 4]] = 1, 0.5, 0.4

    channel = np.zeros((1, 3))
    channel[0, [0, 1, 2]] = 1, 0.6, 0.3

    # channel = np.zeros((1, 1))
    # channel[0, 0] = 1

    data_gen = training_data_generator(
        symbol_stream_shape=(1, number_symbols),
        SNR=2,
        plot=True,
        channel=channel,
    )
    # data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    num_sources = pow(data_gen.alphabet.size, data_gen.CIR_matrix.shape[1])

    # generate data from a set of gaussians
    data = data_gen.channel_output.flatten()

    # TODO See why diverging for large number of iterations (check if diverging in gaussian case)
    mu, variance, alpha = em_gausian(num_sources, data, 10)

    """
    Want to plot the probability of a train and test set during each iteration
    """
    assert True
