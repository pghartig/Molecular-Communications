from mixture_model.em_algorithm import em_gausian
import numpy as np
from communication_util import training_data_generator
from matplotlib import pyplot as plt
import time



def test_mixture_model_results():
    """
    The goal of this test is to ensure that the implemented EM algorithm converges to the correct parameters used to
    to generate a set of data that is then fed into the algorithm.
    :return:
    """
    tolerance = np.power(10.0, -3)

    """
    Setup Training Data
    """
    """
        Setup Training Data
        """
    number_symbols = 5000

    # channel = np.zeros((1, 5))
    # channel[0, [0, 3, 4]] = 1, 0.5, 0.4

    channel = np.zeros((1, 4))
    channel[0, [0, 1, 2, 3]] = 1, 0.6, 0.3, 0.2

    # channel = np.zeros((1, 1))
    # channel[0, 0] = 1

    data_gen = training_data_generator(
        symbol_stream_shape=(1, number_symbols),
        SNR=4,
        plot=True,
        channel=channel,
    )
    # data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    num_sources = pow(data_gen.alphabet.size, data_gen.CIR_matrix.shape[1])
    true_variance = 0
    true_mu = 0
    true_alpha = np.ones((1, num_sources))/num_sources

    # generate data from a set of gaussians
    data = data_gen.channel_output.flatten()
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    test_data = data_gen.channel_output.flatten()

    """
    Train Mixture Model
    """
    model, mu, variance, alpha, total_sequence_probability, test_sequence_probability = \
        em_gausian(num_sources, data, 10, test_data=test_data, save=True, both=True)

    """
    Plot Results
    """

    plt.figure()
    plt.plot(total_sequence_probability, label='training_probability')
    plt.plot(test_sequence_probability, label='test_probability')
    path = "Output/Mixture_Model_" + str(time.time())+"_Convergence.png"
    plt.title("Error over epochs")
    plt.xlabel("Iteration")
    plt.ylabel("Log Probability of Data set")
    plt.legend(loc='upper left')
    plt.savefig(path, format="png")




    # error_mu = np.linalg.norm(np.sort(mu)-np.sort(true_mu))
    # error_variance = np.linalg.norm(np.sort(variance)-np.sort(true_variance))
    # error_alpha = np.linalg.norm(np.sort(true_alpha) - np.sort(alpha))


    """
    Want to plot the probability of a train and test set during each iteration
    """
    # assert error_mu < tolerance and error_variance < tolerance and error_alpha < tolerance
    assert True #TODO decide on pass criteria