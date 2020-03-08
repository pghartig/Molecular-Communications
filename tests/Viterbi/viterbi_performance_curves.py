"""
This test generates the symbol error rate curves over various SNR for comparing the performance of different decoding scemes.
"""

import torch.nn as nn
from mixture_model.em_algorithm import MixtureModel
from mixture_model.em_algorithm import em_gausian
import pickle
from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.general_tools import *
from communication_util.Equalization.supervise_equalization import *
from nn_utilities import models
import torch.optim as optim
import os
import time
import pandas as pd

def test_viterbi():

    viterbi_net_performance = []
    linear_mmse_performance = []
    classic_performance = []
    SNRs_dB = np.linspace(33, 33, 10)
    # SNRs_dB = np.linspace(6, 10,3)
    SNRs =  np.power(10, SNRs_dB/10)
    seed_generator = 0
    data_gen = None
    channel = None
    number_symbols = 5000
    channel = np.zeros((1, 5))
    # Method used in comparison on MATLAB
    # channel[0, [0, 1, 2, 3, 4]] = 0.9, 0.7, 0.3, 0.5, 0.1
    channel[0, [0, 1, 2, 3, 4]] = 1, 0, .2, .2, .4

    # channel = np.zeros((1, 2))
    # channel[0, [0]] = 1
    # Method used in ViterbiNet Paper
    # channel[0, :] = np.random.randn(channel.size)
    for SNR in SNRs:
        """
        Generated Testing Data using the same channel as was used for training the mixture model and the nn
        """

        """
        Create new set of test data. 
        """
        data_gen = CommunicationDataGenerator(symbol_stream_shape=(1, number_symbols), SNR=SNR, plot=True, channel=channel)
        data_gen.random_symbol_stream()
        data_gen.send_through_channel(plot=True)
        channel_length = data_gen.CIR_matrix.shape[1]


        """
        Compare to Classical Viterbi with full CSI
        """
        metric = GaussianChannelMetric(channel, np.flip(data_gen.channel_output))  # This is a function to be used in the viterbi
        detected_classic = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                            metric.metric)
        ser_classic = symbol_error_rate(detected_classic, data_gen.symbol_stream_matrix, channel_length)

        """
        Evaluate performance with linear MMSE
        """


        """
        Analyze SER performance
        """
        classic_performance.append(ser_classic)


    figure, dictionary = plot_symbol_error_rates(SNRs_dB, [classic_performance], data_gen.get_info_for_plot())
    time_path = "Output/SER_"+f"{time.time()}"+"curves.png"
    figure.savefig(time_path, format="png")
    text_path = "Output/SER_"+f"{time.time()}"+"curves.csv"
    pd.DataFrame.from_dict(dictionary).to_csv(text_path)
    figure.savefig(time_path, format="png")


