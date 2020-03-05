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
    accommodate the different equalizers which may require that the alignment is setup differently.
    :param detected_symbols:
    :param input_symbols:
    :param channel_length:
    :return:
    """
    channel_length-=1
    detected_array = np.flip(np.asarray(detected_symbols)).astype('int32')
    input_symbols = input_symbols.flatten().astype('int32')
    test1 = np.max(np.convolve(detected_array, input_symbols))
    test2 = np.max(np.convolve(np.flip(detected_array), input_symbols))
    input_symbols = input_symbols[:detected_array.size]
    # check = np.not_equal(input_symbols, detected_array)*1
    # check1 = np.argmax(check)
    ser = np.sum(np.not_equal(input_symbols, detected_array)) / input_symbols.size
    # correct for indexing problem
    # ser = 1-test3 / input_symbols.size
    return ser


def symbol_error_rate_mc_data(detected_symbols, input_symbols, channel_length, pre_pad=0):
    """
    A function to return the symbol error rate. Note that there a many version of this functionality to
    accommodate the different equalizers which may require that the alignment is setup differently.
    :param detected_symbols:
    :param input_symbols:
    :param channel_length:
    :param pre_pad:
    :return:
    """
    detected_array = np.flip(np.asarray(detected_symbols))
    detected_array = detected_array[pre_pad::]
    detected_array += (detected_array == 0)*-1
    input_symbols = input_symbols.flatten()
    input_symbols += (input_symbols == 0)*-1

    test3 = np.max(np.convolve(np.flip(detected_array), input_symbols))
    test2 = np.max(np.convolve(detected_array, input_symbols))
    input_symbols = input_symbols[(channel_length+1)::]
    input_symbols = input_symbols[:detected_array.size]
    ser = np.sum(np.not_equal(detected_array[:input_symbols.size], input_symbols)) / detected_array.size
    return ser


def symbol_error_rate_channel_compensated(detected_symbols, input_symbols,channel_length):
    """
    A function to return the symbol error rate. Note that there a many version of this functionality to
    accommodate the different equalizers which may require that the alignment is setup differently.
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


def symbol_error_rate_channel_compensated_NN(detected_symbols, input_symbols, channel_length):
    """
    A function to return the symbol error rate. Note that there a many version of this functionality to
    accommodate the different equalizers which may require that the alignment is setup differently.     :param detected_symbols:
    :param input_symbols:
    :param channel_length:
    :return:
    """
    channel_length-=1
    detected_array = np.flip(np.asarray(detected_symbols)).astype('int32')
    ratio_test2 = np.sum(detected_array)
    input_symbols = input_symbols.flatten().astype('int32')
    test1 = np.max(np.convolve(detected_array, input_symbols))
    test2 = np.max(np.convolve(np.flip(detected_array), input_symbols))
    input_symbols = input_symbols[:detected_array.size]
    ser = np.sum(np.not_equal(input_symbols, detected_array)) / input_symbols.size
    # correct for indexing problem
    # ser = 1-test2 / input_symbols.size
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
    detected_array = np.flip(np.asarray(detected_symbols)).astype('int32')
    estimate_ratio = np.sum(detected_array)
    input_ratio = np.sum(input_symbols)
    input_symbols = input_symbols.flatten().astype('int32')
    test1 = np.max(np.convolve(detected_array, input_symbols))
    test2 = np.max(np.convolve(np.flip(detected_array), input_symbols))
    detected_array = detected_array[:input_symbols.size]
    input_symbols = input_symbols[:detected_array.size]
    ser = np.sum(np.not_equal(input_symbols, detected_array)) / input_symbols.size
    # correct for indexing problem
    ser = 1-test2 / detected_array.size
    return ser


def symbol_error_rate_sampled(detected_symbols, input_symbols):
    """
    A function to return the symbol error rate. Note that there a many version of this functionality to
    accommodate the different equalizers which may require that the alignment is setup differently.
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
        data_dict[names[ind]] = np.asarray(SER)
    if analytic_ser==True:
        #TODO general to other pam schemes
        SNRs_dB = np.linspace(0, 10, 500)
        snrs = np.power(10, SNRs_dB / 10)
        analytic = 1 - norm.cdf(np.sqrt(2*snrs))
        plt.plot(SNRs_dB, analytic, label='Analytic')
    plt.xlabel(r'10log$(E[x]/\sigma^2_n$) [dB]')
    plt.ylabel("SER")
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='lower left')
    # plt.title(str(info), fontdict={'fontsize': 10})
    # plt.title("Symbol Error Rate vs SNR")
    # plt.show()
    return fig, data_dict


def plot_quantized_symbol_error_rates_nn_compare(SNRs_dB, SER_list, info, analytic_ser=True):
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
    names =["Classic Viterbi", "Linear MMSE",  "Neural Net", "Neural Net Reduced"]
    data_dict = dict()
    data_dict["SNRs_dB"] = SNRs_dB
    for ind, SER in enumerate(SER_list):
        plt.plot(SNRs_dB, SER, label=f'{names[ind]}')
        data_dict[names[ind]]= np.asarray(SER)
    if analytic_ser==True:
        #TODO general to other pam schemes
        SNRs_dB = np.linspace(0, 10, 100)
        snrs = np.power(10, SNRs_dB / 10)
        analytic = 1 - norm.cdf(np.sqrt(2*snrs))
        plt.plot(SNRs_dB, analytic, label='Analytic')
    plt.xlabel(r'10log$(E[x]/\sigma^2_n$) [dB]')
    plt.ylabel("SER")
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='lower left')
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
        SER = 1- q_function(np.sqrt(snrs))
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


def quantizer(input, level, min=None, max=None):
    """
    An easy implementation of quantization in which the decimals are truncated to a selection place.
    :param input:
    :param level: The number of decimal positions to round to.
    :return:
    """
    if min is not None:
        return np.clip(np.around(input, decimals=level), min, max)
    else:
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


def get_symbol_probabilities(totals, states, symbol_alphabet):
    """

    :param totals:
    :param states:
    :param symbol_alphabet:
    :return:
    """
    #   For each of the reduced states, find the probability of a previous transmissions being a certain symbol
    output = np.zeros((totals.shape[0], states.shape[1], symbol_alphabet.size))
    for reduced_state_ind in range(totals.shape[0]):
        state_labels = totals[reduced_state_ind, :]
        percentages = state_labels/np.sum(state_labels)
        for original_state_ind in range(totals.shape[1]):
            if percentages[original_state_ind]>0:
                original_state = states[original_state_ind, :]
                for memory_index, original_state_symbol in enumerate(original_state):
                    for symbol_ind, symbol in enumerate(symbol_alphabet):
                        if original_state_symbol == symbol:
                            output[reduced_state_ind, memory_index, symbol_ind] += percentages[original_state_ind]
    return output


def get_symbols_from_probabilities(state_path, probabilities, alphabet):
    output = np.zeros((alphabet.size, (len(state_path)+probabilities.shape[1])))
    output = np.ones((alphabet.size, (len(state_path) + probabilities.shape[1])))
    for ind, state in enumerate(state_path):
        #   Get symbol memory probabilities
        probability_matrix = probabilities[state, :, :].T
        probability_matrix = np.flip(probability_matrix, axis=1)
        output[:, ind:(ind + probability_matrix.shape[1])] *= probability_matrix
        #   Now insert this into a final array
    #   Now find most probable symbol
    most_likely_symbol_ind = alphabet[np.argmax(output, axis=0)]
    return most_likely_symbol_ind