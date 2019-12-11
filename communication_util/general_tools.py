import numpy as np
import matplotlib.pyplot as plt
import time


def get_combinatoric_list(alpabet, item_length, item_list, item):
    for i in range(alpabet.size):
        new = list(item)
        new.append(alpabet[i])
        if item_length > 1:
            get_combinatoric_list(alpabet, item_length - 1, item_list, new)
        if item_length == 1:
            item_list.append(new)

def symbol_error_rate(detected_symbols, input_symbols):
    # ignore last symbols since there is extra from the convolution
    array = np.asarray(detected_symbols)
    detected = np.flip(array[:input_symbols.shape[1]])  #This is a very key step to ensuring the detected symbols are aligned properly
    return np.sum(np.logical_not(np.equal(detected, input_symbols))) / detected.size

def symbol_error_rate_channel_compensated(detected_symbols, input_symbols,channel_length):
    # ignore last symbols since there is extra from the convolution
    channel_length -= 1
    detected_array = np.flip(np.asarray(detected_symbols))
    # This is a key step to ensuring the detected symbols are aligned properly
    detected = np.flip(detected_array[channel_length:input_symbols.shape[1]])
    input = np.flip(input_symbols)
    return np.sum(np.logical_not(np.equal(detected,  input[0, channel_length::]))) / detected.size

def symbol_error_rate_sampled(detected_symbols, input_symbols):
    # ignore last symbols since there is extra from the convolution
    array = np.asarray(detected_symbols)
    return np.sum(np.logical_not(np.equal(array, input_symbols))) / array.size

def random_channel():
    return np.random.randn()

def plot_symbol_error_rates(SNRs_dB, SER_list,info):
    fig = plt.figure(1)
    for SER in SER_list:
        plt.plot(SNRs_dB, SER, label='standard viterbi')
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.xscale('linear')
    # plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title(str(info), fontdict={'fontsize': 10})
    plt.title("SER vs SNR Curve")
    # plt.show()
    return fig