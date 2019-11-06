import numpy as np


def symbol_error_rate(detected_symbols, input_symbols):
    # ignore last symbols since there is extra from the convolution
    array = np.asarray(detected_symbols)
    detected = np.flip(array[:input_symbols.shape[1]])
    return np.sum(np.logical_not(np.equal(detected, input_symbols))) / detected.size
