from mixture_model.em_algorithm import em_gausian
import numpy as np
from matplotlib import pyplot as plt
from communication_util.general_tools import quantizer
from communication_util.data_gen import *
import time

def test_quantizer():

    number_symbols = 100
    channel = np.zeros((1, 5))
    channel[0, [0, 1, 2, 3, 4]] = 0.9, 0.7, 0.3, 0.5, 0.1
    SNR_db = 10
    SNRs = np.power(10, SNR_db/10)
    data_gen = CommunicationDataGenerator(symbol_stream_shape=(1, number_symbols), SNR=SNRs, plot=True, channel=channel)
    # data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()


    input_array = np.linspace(-3, 3, num=1000)
    quantized = quantizer(input_array, 1, -5, 5)
    plt.figure(2)
    # plt.scatter(data_gen.channel_output, data_gen.channel_output)
    plt.scatter(data_gen.channel_output, quantizer(data_gen.channel_output, 1, -5, 5), c='orange')
    plt.plot(input_array, quantized, label='Test Error')
    plt.xlabel("Quantizer Input")
    plt.ylabel("Quantizer Ouput")
    plt.show()
    # path = f"Output/Quantizer.png"
    # plt.savefig(path, format="png")