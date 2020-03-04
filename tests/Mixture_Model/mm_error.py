from mixture_model.em_algorithm import *
import matplotlib.pyplot as plt
import numpy as np
from communication_util.data_gen import *
def test_mm():
    vec = np.linspace(-3, 3, 100)
    SNRs_dB = np.linspace(20, 20, 1)
    # SNRs_dB = np.linspace(6, 10,3)
    SNRs = np.power(10, SNRs_dB / 10)
    number_symbols = 1000
    channel = np.zeros((1, 2))
    channel[0, [0, 1]] = 1, .1
    channel = np.flip(channel)
    data_gen = CommunicationDataGenerator(symbol_stream_shape=(1, number_symbols), SNR=SNRs, plot=True, channel=channel)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    mixture_model_training_data = data_gen.channel_output.flatten()
    num_sources = 4
    num_iterations = 30
    mu, sigma_square, alpha, likelihood_vector, test_set_probability = em_gausian(num_sources, mixture_model_training_data, num_iterations)

    channel = np.zeros((1, 2))
    channel[0, [0, 1]] = 0.4, 0.8
    data_gen2 = CommunicationDataGenerator(symbol_stream_shape=(1, number_symbols), SNR=SNRs, plot=True, channel=channel)
    data_gen2.random_symbol_stream()
    data_gen2.send_through_channel()
    mixture_model_training_data = data_gen2.channel_output.flatten()
    mu2, sigma_square2, alpha2, likelihood_vector2, test_set_probability = em_gausian(num_sources,
                                                                                  mixture_model_training_data,
                                                                                  num_iterations)

    plt.figure()
    plt.scatter(data_gen.channel_output.flatten(), data_gen2.channel_output.flatten(), label="Channel Ouput")
    for ind, mus in enumerate(mu):
        if ind ==0:
            plt.scatter(mus * np.ones(mu.size), mu2, c='Orange', label="Gaussian Mean - Mixture Model ")
        plt.scatter(mus*np.ones(mu.size), mu2, c='Orange')
    plt.xlabel("Real")
    plt.ylabel("Img")
    plt.legend(loc='upper left')
    plt.show()
    # plt.scatter(self.channel_output, self.channel_output, label="Quantized Output")
    # plt.axvline(c='grey', lw=1)
    # plt.axhline(c='grey', lw=1)
    # plt.legend(loc='upper left')
    # plt.show()
    # plt.plot(likelihood_vector)
    # plt.show()
    #
    # samples = []
    # for i in vec:
    #     samples.append(probability_from_gaussian_sources(i,0,1))
    # plt.plot(vec)
    # plt.show()

