from mixture_model.em_algorithm import *
import matplotlib.pyplot as plt
import numpy as np
from communication_util.data_gen import *
def test_mm():
    SNRs_dB = np.linspace(30, 30, 1)
    # SNRs_dB = np.linspace(6, 10,3)
    SNRs = np.power(10, SNRs_dB / 10)
    number_symbols = 1000
    channel = 1j*np.zeros((1, 4))
    # channel[0, [0, 1, 2, 3]] = 1+1j*.5, -1, -0.8 - 1j*.5, 0.8
    channel = np.zeros((1, 4))
    channel[0, [0, 1, 2, 3]] = 1, -1, .2, -.5

    channel = np.flip(channel)
    data_gen = CommunicationDataGenerator(symbol_stream_shape=(1, number_symbols), SNR=SNRs, plot=True, channel=channel)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    mixture_model_training_data = data_gen.channel_output.flatten()
    num_sources = 16
    num_iterations = 20
    mu, sigma_square, alpha, likelihood_vector, test_set_probability =\
        em_gausian(num_sources, mixture_model_training_data, num_iterations, mixture_model_training_data)

    plt.figure()

    plt.scatter(np.real(data_gen.channel_output.flatten()), np.imag(data_gen.channel_output.flatten()), label="Channel Output")
    for ind, mus in enumerate(mu):
        if ind ==0:
            plt.scatter(np.real(mus), np.imag(mus), c='Orange', label="Gaussian Mean - Mixture Model ")
        plt.scatter(np.real(mus), np.imag(mus), c='Orange')

    # plt.scatter(data_gen.channel_output.flatten(), data_gen.channel_output.flatten(), label="Channel Output")
    # for ind, mus in enumerate(mu):
    #     if ind ==0:
    #         plt.scatter(mus, mus, c='Orange', label="Gaussian Mean - Mixture Model ")
    #     plt.scatter(mus, mus, c='Orange')


    plt.xlabel("Real")
    plt.ylabel("Img")
    plt.legend(loc='lower left', prop={"size":12})
    plt.show()
    path = f"Output/mixture_model.png"
    plt.savefig(path, format="png")