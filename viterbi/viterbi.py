import numpy as np
from communication_util.model_metrics import *


def viterbi_output(transmit_alphabet, channel_output, channel_information):
    alphabet_size = transmit_alphabet.size
    # number of states is alphabet size raised to the power of the number of channel taps minus one.
    num_states = np.power(alphabet_size, channel_information.shape[1] - 1)
    survivor_paths = 1j*np.zeros((num_states, channel_output.shape[1]))

    # iterate through the metrics
    for i in range(channel_output.shape[1]):
        # assume zero-padding at beginning of word so set survivor path portions to zeros automatically
        if i < channel_information.shape[1] - 1:
            for state in range(num_states):
                survivor_paths[state, i] = transmit_alphabet[0]
            continue
        else:
            # TODO don't pass entire channel_output and survivor_paths
            metric_vector = gaussian_channel_metric(
                survivor_paths,
                i,
                transmit_alphabet,
                channel_output,
                channel_information,
            )
            for state in range(num_states):
                symbol = np.argmin(
                    (metric_vector[state * alphabet_size : (state + 1) * alphabet_size])
                )
                survivor_paths[state, i] = transmit_alphabet[symbol]

    final_path_ind = np.argmin(np.sum(survivor_paths, axis=1))
    return survivor_paths[final_path_ind, channel_information.size - 1 :]
