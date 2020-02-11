import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

def slicer(received_sequence, true_sequence, alphabet, channel_length):
    channel_length += -1
    detected = []
    for symbol in received_sequence:
        detected.append(np.sign(symbol))
    detected = np.asarray(detected[channel_length:])
    ser = np.sum(np.not_equal(detected, true_sequence)) / true_sequence.size
    return ser
def get_combinatoric_list(alpabet, item_length, item_list, item):
    for i in range(alpabet.size):
        new = list(item)
        new.append(alpabet[i])
        if item_length > 1:
            get_combinatoric_list(alpabet, item_length - 1, item_list, new)
        if item_length == 1:
            item_list.append(new)

def symbol_error_rate(detected_symbols, input_symbols,channel_length):
    detected_array = np.asarray(detected_symbols)
    # This is a key step to ensuring the detected symbols are aligned properly
    t = input_symbols.flatten().astype('int32')
    test1 = np.max(np.convolve(detected_array,t))
    test3 = np.max(np.convolve(np.flip(detected_array),t))
    check2 = t[(channel_length-1):]
    check1 = detected_array[:check2.size]
    ser = np.sum(np.not_equal(check2[:check1.size], check1)) / check1.size
    return ser

def symbol_error_rate_mc_data(detected_symbols, input_symbols, pre_pad = 0):
    detected_array = np.asarray(detected_symbols[pre_pad::])
    input = input_symbols.astype('int32').flatten()[:detected_array.size]
    test1 = np.max(np.convolve(detected_array,input))
    test2 = np.max(np.convolve(detected_array,np.flip(input)))
    ser = np.sum(np.not_equal(detected_array, input)) / detected_array.size
    return ser

def symbol_error_rate_channel_compensated(detected_symbols, input_symbols,channel_length):
    channel_length -= 1
    detected_array = np.flip(np.asarray(detected_symbols))
    # This is a key step to ensuring the detected symbols are aligned properly
    detected = np.flip(detected_array[channel_length:input_symbols.shape[1]])
    input = np.flip(input_symbols)
    return np.sum(np.logical_not(np.equal(detected,  input[0, channel_length::]))) / detected.size

def symbol_error_rate_channel_compensated_NN(detected_symbols, input_symbols,channel_length):
    """
    The first symbol the viterbi detects is the l-1 symbol in the stream. where l is the channel impulse response length.
    :param detected_symbols:
    :param input_symbols:
    :param channel_length:
    :return:
    """
    detected_array = np.asarray(detected_symbols)
    ratio_test2 = np.sum(detected_array)
    # This is a key step to ensuring the detected symbols are aligned properly
    t = input_symbols.flatten().astype('int32')
    test1 = np.max(np.convolve(detected_array, t))
    test2 = np.max(np.convolve(np.flip(detected_array), t))
    check2 = t[(channel_length-1):]
    check1 = detected_array[:check2.size]
    ser = np.sum(np.not_equal(check2, check1)) / check1.size
    return ser

def symbol_error_rate_channel_compensated_NN_reduced(detected_symbols, input_symbols,channel_length):
    """
    The first symbol the viterbi detects is the l-1 symbol in the stream. where l is the channel impulse response length.
    :param detected_symbols:
    :param input_symbols:
    :param channel_length:
    :return:
    """
    detected_array = np.asarray(detected_symbols)
    ratio_test2 = np.sum(detected_array)
    # This is a key step to ensuring the detected symbols are aligned properly
    flat_input = input_symbols.flatten().astype('int32')
    test1 = np.max(np.convolve(detected_array, flat_input))
    test2 = np.max(np.convolve(np.flip(detected_array), flat_input))
    flat_input = flat_input[(flat_input.size-detected_array.size)::]
    ser = np.sum(np.not_equal(flat_input, detected_array)) /detected_array.size
    return ser

def symbol_error_rate_sampled(detected_symbols, input_symbols):
    # ignore last symbols since there is extra from the convolution
    array = np.asarray(detected_symbols)
    ser = np.sum(np.logical_not(np.equal(array, input_symbols))) / array.size
    return ser

def random_channel():
    return np.random.randn()

def plot_symbol_error_rates(SNRs_dB, SER_list,info, analytic_ser=True):
    fig = plt.figure(1)
    names = ["Classic Viterbi", "Linear MMSE", "Neural Net"]
    data_dict = dict()
    data_dict["SNRs_dB"] = SNRs_dB
    for ind, SER in enumerate(SER_list):
        plt.plot(SNRs_dB, SER, label=f'{names[ind]}')
        data_dict[names[ind]]= np.asarray(SER)
    if analytic_ser==True:
        #TODO general to other pam schemes
        SNRs_dB = np.linspace(-5, 10, 100)
        snrs = np.power(10, SNRs_dB / 10)
        analytic = 1- norm.cdf(np.sqrt(2 * snrs))
        plt.plot(SNRs_dB, analytic, label='analytic_ml')
    plt.xlabel(r'$10log(E[x]/\sigma^2_n$) [dB]')
    plt.ylabel("SER")
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.title(str(info), fontdict={'fontsize': 10})
    plt.title("Symbol Error Rate vs SNR")
    # plt.show()
    return fig, data_dict

def plot_quantized_symbol_error_rates_nn_compare(SNRs_dB, SER_list,info, analytic_ser=True):
    fig = plt.figure(1)
    names =["Classic Viterbi", "Linear MMSE", "Neural Net Reduced", "Neural Net"]
    data_dict = dict()
    data_dict["SNRs_dB"] = SNRs_dB
    for ind, SER in enumerate(SER_list):
        plt.plot(SNRs_dB, SER, label=f'{names[ind]}')
        data_dict[names[ind]]= np.asarray(SER)
    if analytic_ser==True:
        #TODO general to other pam schemes
        SNRs_dB = np.linspace(-5, 10, 100)
        snrs = np.power(10, SNRs_dB / 10)
        analytic = 1- norm.cdf(np.sqrt(2 * snrs))
        plt.plot(SNRs_dB, analytic, label='analytic_ml')
    plt.xlabel(r'$10log(E[x]/\sigma^2_n$) [dB]')
    plt.ylabel("SER")
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.title(str(info), fontdict={'fontsize': 10})
    plt.title("Symbol Error Rate vs SNR")
    # plt.show()
    return fig, data_dict

def plot_quantized_symbol_error_rates(quantization_levels, SNRs_dB, SER_list,info, analytic_ser=True):
    fig = plt.figure()
    names =["Classic Viterbi", "Linear MMSE", "Neural Net"]
    data_dict = dict()
    data_dict["SNRs_dB"] = SNRs_dB
    for ind, SER in enumerate(SER_list):
        for level in range(quantization_levels):
            plt.plot(SNRs_dB, SER[level], label=f'{names[ind]}_q{level}')
            data_dict[f"{names[ind]}_{level}"] = np.asarray(SER[level])
    if analytic_ser==True:
        #TODO general to other pam schemes
        SNRs_dB = np.linspace(-5, 10, 100)
        snrs = np.power(10, SNRs_dB / 10)
        q_function = norm.cdf
        SER = 1- q_function(np.sqrt(2 * snrs))
        plt.plot(SNRs_dB, SER, label='analytic_ml')
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.title(str(info), fontdict={'fontsize': 10})
    plt.title("Symbol Error Rate vs SNR")
    # plt.show()
    return fig, data_dict

def quant_symbol_error_rates(SNRs_dB, SER_list):
    fig = plt.figure(1)
    for ind in range(SER_list.shape[0]):
        plt.plot(SNRs_dB, SER_list[ind,:], label=f" {ind}")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='upper right')
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

def quantizer(input, level):
    """
    Range should be based on expected std of noise and the largest value due to the estimate channel and the known transmit alphabet
    :param input:
    :param range:
    :param bits_available:
    :return:
    """
    check = np.around(input, decimals=level)
    return np.around(input, decimals=level)

def base_2_quantizer(input, level, clip_low = None, clip_high = None):
    """
    base 2 quantizer that also introduces clipping
    :param input:
    :param range:
    :param bits_available:
    :return:
    """
    test = np.int_(10000*input)
    # check = test & 0b11111100
    test2 = test/10000
    if clip_high == None or clip_low == None:
        return np.round(input * (pow(10, level)))
    return np.round(input * (pow(10, level)))