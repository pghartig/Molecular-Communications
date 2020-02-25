"""
This test generates the symbol error rate curves over various SNR for comparing the performance of different decoding scemes.
"""

import pickle
from communication_util.data_gen import *
from communication_util.general_tools import *
from communication_util.Equalization.supervise_equalization import *
import time
import pandas as pd

def test_analytic():
    classic_performance = []
    SNRs_dB = np.linspace(0, 10, 15)
    SNRs = np.power(10, SNRs_dB/10)
    data_gen = None
    number_symbols = 100000
    channel = np.zeros((1, 1))
    channel[0, [0]] = 1
    for SNR in SNRs:
        """
        Generated Testing Data using the same channel as was used for training the mixture model and the nn
        """

        data_gen = CommunicationDataGenerator(symbol_stream_shape=(1, number_symbols), SNR=SNR, plot=True, channel=channel)
        data_gen.random_symbol_stream()
        data_gen.send_through_channel()


        """
        Compare to Classical Viterbi with full CSI
        """
        # channel= np.round(channel*10)
        detected_thresholder = threshold_detector(data_gen.alphabet, np.flip(data_gen.channel_output))
        channel_length = 1
        ser_classic = symbol_error_rate(detected_thresholder, data_gen.symbol_stream_matrix, channel_length)

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



