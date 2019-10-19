import numpy as np
from communication_util.model_metrics import *


def viterbi_output(transmit_alphabet, channel_output, channel_information):
    alphabet_size = transmit_alphabet.size
    num_states = np.power(alphabet_size, np.size(channel_information, axis=1))
    survivor_paths = np.zeros(
        (num_states, channel_output.shape[1]), dtype=np.int8
    )

    # iterate through the metrics
    i = 0
    for detected_symbol in channel_output:
        # This may not be the most memory efficient way to do this below task but it is implemented in this way to
        # accomodate what I predict to be the output of the Neural Net
        metric_vector = gaussian_channel_metric(
            survivor_paths, i, transmit_alphabet, detected_symbol, channel_information
        )
        for state in range(num_states):
            survivor_paths[i, state] = transmit_alphabet[
                np.argmin(
                    (metric_vector[state * alphabet_size : (state + 1) * alphabet_size])
                )
            ]
        i += 1

    final_path_ind = np.argmin(np.sum(survivor_paths, axis=1))
    return np.flip(survivor_paths[final_path_ind, :], axis=1)
