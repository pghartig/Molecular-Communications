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
from nn_utilities import models
import torch.optim as optim
import os
import time

def test_viterbi_quantization():


    classic_performance = []
    SNRs_dB = np.linspace(-5, 10, 10)
    # SNRs_dB = np.linspace(6, 10,3)
    SNRs =  np.power(10, SNRs_dB/10)
    data_gen = None
    channel = None
    for SNR in SNRs:
        """
        Generated Testing Data using the same channel as was used for training the mixture model and the nn
        """
        number_symbols = 2000
        channel = np.zeros((1, 5))
        channel[0, [0, 1, 2, 3, 4]] = 1, .1, .01, .1, .04
        # channel[0, [0, 1, 2, 3, 4]] = 1, .1, .1, .1, .4
        # channel[0, [0, 1, 2, 3, 4]] = 1, .4, .7, .1, .3
        # channel = np.zeros((1, 2))
        # channel[0, [0]] = 1
        data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SNR, plot=True, channel=channel)
        data_gen.random_symbol_stream()
        data_gen.send_through_channel()

        """
        After sending through channel, symbol detection should be performed using something like a matched filter.
        Create new set of test data. 
        """

        data_gen = training_data_generator(symbol_stream_shape=(1, 2000), SNR=SNR, plot=True, channel=channel)
        data_gen.random_symbol_stream()
        data_gen.send_through_channel()

        """
        Compare to Classical Viterbi with full CSI
        """
        channel_length = data_gen.CIR_matrix.shape[1]
        quant_ser = []
        for level in range(4):
            quantized_output =  np.round(data_gen.channel_output*(pow(10,level)))
            metric = gaussian_channel_metric_working(channel, quantized_output)
            detected_classic = viterbi_setup_with_nodes(data_gen.alphabet, quantized_output, data_gen.CIR_matrix.shape[1],
                                                metric.metric)
            ser_classic = symbol_error_rate(detected_classic, data_gen.symbol_stream_matrix, channel_length)
            quant_ser.append(ser_classic)

        """
        Analyze SER performance
        """
        classic_performance.append(quant_ser)
    classic_performance = np.asarray(classic_performance).T
    figure = quant_symbol_error_rates(SNRs_dB, classic_performance)
    time_path = "Output/SER_"+f"{time.time()}"+"curves.png"

    figure.savefig(time_path, format="png")


    assert True
