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

def symbol_error_rate(detected_symbols, input_symbols,channel_length):
    # ignore last symbols since there is extra from the convolution
    detected = np.flip(np.asarray(detected_symbols))
    input = input_symbols.flatten()
    input = input[1:detected.size]
    test = np.sum(np.logical_not(np.equal(detected, input_symbols))) / detected.size
    return np.sum(np.logical_not(np.equal(detected, input_symbols))) / detected.size

def symbol_error_rate_channel_compensated(detected_symbols, input_symbols,channel_length):
    channel_length -= 1
    detected_array = np.flip(np.asarray(detected_symbols))
    # This is a key step to ensuring the detected symbols are aligned properly
    detected = np.flip(detected_array[channel_length:input_symbols.shape[1]])
    input = np.flip(input_symbols)
    test = np.sum(np.logical_not(np.equal(detected,  input[0, channel_length::]))) / detected.size
    return np.sum(np.logical_not(np.equal(detected,  input[0, channel_length::]))) / detected.size


def symbol_error_rate_channel_compensated_NN(detected_symbols, input_symbols,channel_length):
    """
    The first symbol the viterbi detects is the l-1 symbol in the stream. where l is the channel impulse response length.
    :param detected_symbols:
    :param input_symbols:
    :param channel_length:
    :return:
    """
    #TODO Notes The returned survivor path should be this long if channel is longer than 1.
    detected_array = np.asarray(detected_symbols)
    # This is a key step to ensuring the detected symbols are aligned properly
    t = input_symbols.flatten()
    # test1 = np.max(np.convolve(detected_array,t))
    # test2 = np.argmax(np.convolve(detected_array,t))
    detected_array = np.flip(detected_array)
    check1 = detected_array[(channel_length):]
    check2 = t[:check1.size]
    ser = np.sum(np.not_equal(check2, check1)) / check1.size
    return ser




def symbol_error_rate_sampled(detected_symbols, input_symbols):
    # ignore last symbols since there is extra from the convolution
    array = np.asarray(detected_symbols)
    return np.sum(np.logical_not(np.equal(array, input_symbols))) / array.size

def random_channel():
    return np.random.randn()

def plot_symbol_error_rates(SNRs_dB, SER_list,info):
    fig = plt.figure(1)
    names =["Classic Viterbi", "Nerual Net"]
    for ind, SER in enumerate (SER_list):
        plt.plot(SNRs_dB, SER, label=f'curve: {names[ind]}')
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title(str(info), fontdict={'fontsize': 10})
    plt.title("Symbol Error Rate vs SNR")
    # plt.show()
    return fig


def threshold_detector(alphabet, output):
    detected_symbols = []
    for stream in range(output.shape[0]):
        for received_symbol in output[stream, :]:
            detected = alphabet[np.argmin(np.abs(alphabet - received_symbol))]
            detected_symbols.append(detected)
    return detected_symbols