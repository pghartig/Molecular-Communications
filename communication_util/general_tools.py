import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm


def get_combinatoric_list(alpabet, item_length, item_list, item):
    """
    TODO Consider refactor before final commit
    Used to generate a list of potential states for a given length and symbol alphabet
    :param alpabet:
    :param item_length:
    :param item_list:
    :param item:
    :return:
    """
    for i in range(alpabet.size):
        new = list(item)
        new.append(alpabet[i])
        if item_length > 1:
            get_combinatoric_list(alpabet, item_length - 1, item_list, new)
        if item_length == 1:
            item_list.append(new)


def symbol_error_rate(detected_symbols, input_symbols, channel_length):
    """
    A function to return the symbol error rate. Note that there a many version of this functionality to
    accomodate the different equalizers which may require that the alignment is setup differently.
    :param detected_symbols:
    :param input_symbols:
    :param channel_length:
    :return:
    """
    channel_length -= 1
    detected_array = np.flip(np.asarray(detected_symbols))
    # This is a key step to ensuring the detected symbols are aligned properly
    input_symbols = input_symbols.flatten().astype('int32')
    # This is a useful way of checking equalizer performance before alignment is correct
    test1 = np.max(np.convolve(detected_array, input_symbols))
    test3 = np.max(np.convolve(np.flip(detected_array), input_symbols))
    check2 = input_symbols[:(input_symbols.size-channel_length)]
    check1 = detected_array[:check2.size]
    ser = np.sum(np.not_equal(check2[:check1.size], check1)) / check1.size
    return ser



def symbol_error_rate_mc_data(detected_symbols, input_symbols, channel_length, pre_pad=0):
    """
    A function to return the symbol error rate. Note that there a many version of this functionality to
    accomodate the different equalizers which may require that the alignment is setup differently.
    :param detected_symbols:
    :param input_symbols:
    :param channel_length:
    :param pre_pad:
    :return:
    """
    detected_array = np.asarray(detected_symbols[pre_pad::])
    detected_array += (detected_array == 0)*-1
    input = input_symbols.flatten()
    input += (input == 0)*-1
    test1 = np.argmax(np.max(np.convolve(detected_array, input)))
    test3 = np.max(np.convolve(np.flip(detected_array), input))
    test2 = np.max(np.convolve(detected_array, input))
    input = input[(channel_length+1)::]
    input = input[:detected_array.size]
    ser = np.sum(np.not_equal(detected_array, input)) / detected_array.size
    return ser


def symbol_error_rate_channel_compensated(detected_symbols, input_symbols,channel_length):
    """
    A function to return the symbol error rate. Note that there a many version of this functionality to
    accomodate the different equalizers which may require that the alignment is setup differently.
    :param detected_symbols:
    :param input_symbols:
    :param channel_length:
    :return:
    """
    channel_length -= 1
    detected_array = np.flip(np.asarray(detected_symbols))
    # This is a key step to ensuring the detected symbols are aligned properly
    detected = np.flip(detected_array[channel_length:input_symbols.shape[1]])
    input = np.flip(input_symbols)
    return np.sum(np.logical_not(np.equal(detected,  input[0, channel_length::]))) / detected.size


def symbol_error_rate_channel_compensated_NN(detected_symbols, input_symbols,channel_length):
    """
    A function to return the symbol error rate. Note that there a many version of this functionality to
    accomodate the different equalizers which may require that the alignment is setup differently.     :param detected_symbols:
    :param input_symbols:
    :param channel_length:
    :return:
    """
    detected_array = np.asarray(detected_symbols)
    ratio_test2 = np.sum(detected_array)
    t = input_symbols.flatten().astype('int32')
    test1 = np.max(np.convolve(detected_array, t))
    test2 = np.max(np.convolve(np.flip(detected_array), t))
    check2 = t[(channel_length-1):]
    check1 = detected_array[:check2.size]
    ser = np.sum(np.not_equal(check2, check1)) / check1.size
    return ser


def symbol_error_rate_channel_compensated_NN_reduced(detected_symbols, input_symbols, channel_length):
    """
    A function to return the symbol error rate. Note that there a many version of this functionality to
    accommodate the different equalizers which may require that the alignment is setup differently.
    :param detected_symbols:
    :param input_symbols:
    :param channel_length:
    :return:
    """
    channel_length -= 1
    detected_array = np.asarray(detected_symbols)
    estimate_ratio = np.sum(detected_array)
    input_ratio = np.sum(input_symbols)
    flat_input = input_symbols.flatten().astype('int32')
    test1 = np.max(np.convolve(detected_array, flat_input))
    test2 = np.max(np.convolve(np.flip(detected_array), flat_input))
    flat_input = flat_input[:(flat_input.size-channel_length)]
    ser = np.sum(np.not_equal(flat_input, detected_array)) /detected_array.size
    return ser


def symbol_error_rate_sampled(detected_symbols, input_symbols):
    """
    A function to return the symbol error rate. Note that there a many version of this functionality to
    accomodate the different equalizers which may require that the alignment is setup differently.
    :param detected_symbols:
    :param input_symbols:
    :return:
    """
    # ignore last symbols since there is extra from the convolution
    array = np.asarray(detected_symbols)
    ser = np.sum(np.logical_not(np.equal(array, input_symbols))) / array.size
    return ser


def random_channel():
    return np.random.randn()


def plot_symbol_error_rates(SNRs_dB, SER_list, info, analytic_ser=True):
    """
    Variations of plotting the symbol error rates.
    Because there was enough variety in the types of testing, just allowing for configuration via parameters would have
    been more effort and unwieldy so instead there are one-offs for the different types of results plotted.
    :param SNRs_dB:
    :param SER_list:
    :param info:
    :param analytic_ser:
    :return:
    """
    fig = plt.figure(1)
    names = ["Classic Viterbi", "Linear MMSE", "Neural Net"]
    data_dict = dict()
    data_dict["SNRs_dB"] = SNRs_dB
    for ind, SER in enumerate(SER_list):
        plt.plot(SNRs_dB, SER, label=f'{names[ind]}')
        data_dict[names[ind]]= np.asarray(SER)
    if analytic_ser==True:
        #TODO general to other pam schemes
        SNRs_dB = np.linspace(0, 10, 500)
        snrs = np.power(10, SNRs_dB / 10)
        analytic = 1 - norm.cdf(np.sqrt(2 * snrs))
        plt.plot(SNRs_dB, analytic, label='analytic_ml')
    plt.xlabel(r'$10log(E[x]/\sigma^2_n$) [dB]')
    plt.ylabel("SER")
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='lower left')
    # plt.title(str(info), fontdict={'fontsize': 10})
    # plt.title("Symbol Error Rate vs SNR")
    # plt.show()
    return fig, data_dict


def plot_quantized_symbol_error_rates_nn_compare(SNRs_dB, SER_list,info, analytic_ser=True):
    """
    Variations of plotting the symbol error rates.
    Because there was enough variety in the types of testing, just allowing for configuration via parameters would have
    been more effort and unwieldy so instead there are one-offs for the different types of results plotted.
    :param SNRs_dB:
    :param SER_list:
    :param info:
    :param analytic_ser:
    :return:
    """
    fig = plt.figure(1)
    names =["Classic Viterbi", "Linear MMSE", "Neural Net Reduced", "Neural Net"]
    data_dict = dict()
    data_dict["SNRs_dB"] = SNRs_dB
    for ind, SER in enumerate(SER_list):
        plt.plot(SNRs_dB, SER, label=f'{names[ind]}')
        data_dict[names[ind]]= np.asarray(SER)
    if analytic_ser==True:
        #TODO general to other pam schemes
        SNRs_dB = np.linspace(0, 10, 100)
        snrs = np.power(10, SNRs_dB / 10)
        analytic = 1- norm.cdf(np.sqrt(2 * snrs))
        plt.plot(SNRs_dB, analytic, label='analytic_ml')
    plt.xlabel(r'$10log(E[x]/\sigma^2_n$) [dB]')
    plt.ylabel("SER")
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='lower left')
    # plt.title(str(info), fontdict={'fontsize': 10})
    # plt.title("Symbol Error Rate vs SNR")
    # plt.show()
    return fig, data_dict


def plot_quantized_symbol_error_rates(quantization_levels, SNRs_dB, SER_list,info, analytic_ser=False):
    """
    Variations of plotting the symbol error rates.
    Because there was enough variety in the types of testing, just allowing for configuration via parameters would have
    been more effort and unwieldy so instead there are one-offs for the different types of results plotted.
    :param SNRs_dB:
    :param SER_list:
    :param info:
    :param analytic_ser:
    :return:
    """
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
        SNRs_dB = np.linspace(0, 10, 100)
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
    # plt.title(str(info), fontdict={'fontsize': 10})
    # plt.title("Symbol Error Rate vs SNR")
    # plt.show()
    return fig, data_dict


def quant_symbol_error_rates(SNRs_dB, SER_list):
    """
    Variations of plotting the symbol error rates.
    Because there was enough variety in the types of testing, just allowing for configuration via parameters would have
    been more effort and unwieldy so instead there are one-offs for the different types of results plotted.
    :param SNRs_dB:
    :param SER_list:
    :param info:
    :param analytic_ser:
    :return:
    """
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
    """
    To use for memoryless channels with Gaussian noise
    :param alphabet:
    :param output:
    :return:
    """
    detected_symbols = []
    for stream in range(output.shape[0]):
        for received_symbol in output[stream, :]:
            detected = alphabet[np.argmin(np.abs(alphabet - received_symbol))]
            detected_symbols.append(detected)
    return detected_symbols


def quantizer(input, level):
    """
    An easy implementation of quantization in which the decimals are truncated to a selection place.
    :param input:
    :param level: The number of decimal positions to round to.
    :return:
    """
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