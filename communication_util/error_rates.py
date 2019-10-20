import numpy as np


def symbol_error_rate(detected_symbols, input_symbols):
    # TODO need to get rid of zero padding
    return (
        np.sum(np.logical_not(np.equal(detected_symbols, input_symbols)))
        / detected_symbols.size
    )
