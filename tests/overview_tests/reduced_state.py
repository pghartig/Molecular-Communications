from mixture_model.em_algorithm import em_gausian
import numpy as np
from matplotlib import pyplot as plt
from communication_util.general_tools import quantizer
from communication_util.data_gen import *
import time

def test_reduced_state():

    number_symbols = 100
    channel = np.zeros((1, 4))
    channel[0, [0, 1, 2, 3]] = 1, 1, 0.15, 0.1
    # channel[0, [0, 1, 2, 3]] = 1, .7, 0.4, 0.1
    SNR_db = 15
    SNRs = np.power(10, SNR_db/10)
    data_gen = CommunicationDataGenerator(symbol_stream_shape=(1, number_symbols), SNR=SNRs, plot=True, channel=channel)
    # data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()

    plt.figure(2)
    plt.scatter(data_gen.channel_output, data_gen.channel_output)

    plt.xlabel("Channel Output")
    plt.ylabel("Channel Output")
    plt.show()
    # path = f"Output/Quantizer.png"
    # plt.savefig(path, format="png")