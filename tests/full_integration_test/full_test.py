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
mm = mixture_model(mu=model[0], sigma_square=model[0])

mm_pickle_in.close()

"""
Generated Testing Data using the same channel as was used for training the mixture model and the nn
"""
number_symbols = 60

channel = 1j * np.zeros((1, 5))
channel[0, [0, 3, 4]] = 1, 0.5, 0.4
data_gen = training_data_generator(
    symbol_stream_shape=(1, number_symbols + 2 * channel.size),
    SNR=10,
    plot=True,
    channel=channel,
)
data_gen.setup_channel(shape=None)
data_gen.random_symbol_stream()
data_gen.send_through_channel()

"""
After sending through channel, symbol detection should be performed using something like a matched filter
"""

x, y = data_gen.get_labeled_data()
x = torch.Tensor(x)
y = torch.Tensor(y)

"""
Detect symbols using Viterbi Algorithm
"""
detected = viterbi_NN_MM_output(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix, mm.get_probability,neural_net)
"""
Analyze SER performance
"""
ser = symbol_error_rate(detected, data_gen.symbol_stream_matrix)

