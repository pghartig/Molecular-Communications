import numpy as np


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
    detected = np.flip(array[:input_symbols.shape[1]])
    return np.sum(np.logical_not(np.equal(detected, input_symbols))) / detected.size

def symbol_error_rate_sampled(detected_symbols, input_symbols):
    # ignore last symbols since there is extra from the convolution
    array = np.asarray(detected_symbols)
    return np.sum(np.logical_not(np.equal(array, input_symbols))) / array.size

def random_channel():
    return np.random.randn()