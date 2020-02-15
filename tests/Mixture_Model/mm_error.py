from mixture_model.em_algorithm import *
import matplotlib.pyplot as plt
import numpy as np
from communication_util.data_gen import *

vec = np.linspace(-3,3,100)
SNRs_dB = np.linspace(10, 15, 1)
# SNRs_dB = np.linspace(6, 10,3)
SNRs = np.power(10, SNRs_dB / 10)
number_symbols = 1000
channel = np.zeros((1, 5))
channel[0, [0, 1, 2, 3, 4]] = 0.227, 0.460, 0.688, 0.460, 0.227
data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SNRs, plot=True, channel=channel)
data_gen.random_symbol_stream()
data_gen.send_through_channel()
mixture_model_training_data = data_gen.channel_output.flatten()
num_sources = pow(data_gen.alphabet.size, data_gen.CIR_matrix.shape[1])
num_iterations = 20
mu, sigma_square, alpha, likelihood_vector, test_set_probability = em_gausian(num_sources, mixture_model_training_data, num_iterations)
plt.plot(likelihood_vector)
plt.show()
#
# samples = []
# for i in vec:
#     samples.append(probability_from_gaussian_sources(i,0,1))
# plt.plot(vec)
# plt.show()

