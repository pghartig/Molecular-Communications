"""
This test generates the symbol error rate curves over various SNR for comparing the performance of different decoding scemes.
"""

import torch.nn as nn
from mixture_model.em_algorithm import mixture_model
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
    SNRs_dB = np.linspace(0, 16, 10)
    # SNRs_dB = np.linspace(6, 10,3)
    SNRs =  np.power(10, SNRs_dB/10)
    seed_generator = 0
    data_gen = None
    channel = None
    for SNR in SNRs:
        """
        Generated Testing Data using the same channel as was used for training the mixture model and the nn
        """
        number_symbols = 5000
        channel = np.zeros((1, 5))
        channel[0, [0, 1, 2, 3, 4]] = 0.227, 0.460, 0.688, 0.460, 0.227

        """
        Create new set of test data. 
        """
        del data_gen
        data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SNR, plot=True, channel=channel)
        data_gen.random_symbol_stream()
        data_gen.send_through_channel()


        """
        Compare to Classical Viterbi with full CSI
        """
        # channel= np.round(channel*10)
        metric = gaussian_channel_metric_working(channel, data_gen.channel_output)  # This is a function to be used in the viterbi
        detected_classic = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                            metric.metric)
        ser_classic = symbol_error_rate(detected_classic, data_gen.symbol_stream_matrix, channel.shape[1])

        """
        Evaluate performance with linear MMSE
        """


        """
        Analyze SER performance
        """
        linear_mmse_performance.append(linear_mmse(data_gen.symbol_stream_matrix, data_gen.channel_output, data_gen.symbol_stream_matrix,channel.size))
        classic_performance.append(ser_classic)

    viterbi_net_performance = linear_mmse_performance

    figure, dictionary = plot_symbol_error_rates(SNRs_dB, [classic_performance, linear_mmse_performance, viterbi_net_performance], data_gen.get_info_for_plot())
    time_path = "Output/SER_"+f"{time.time()}"+"curves.png"
    figure.savefig(time_path, format="png")
    text_path = "Output/SER_"+f"{time.time()}"+"curves.csv"
    pd.DataFrame.from_dict(dictionary).to_csv(text_path)
    figure.savefig(time_path, format="png")


