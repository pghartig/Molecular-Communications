import numpy as np
from itertools import permutations


def gaussian_channel_metric(survivor_paths, index, transmit_alphabet, channel_output, cir):
    """
    returns vector of metrics for incoming state of viterbi with a gaussian channel
    :param survivor_paths:
    :param index:
    :param transmit_alphabet:
    :param channel_output:
    :param cir:
    :return:
    """
    num_states = np.power(np.size(transmit_alphabet), np.size(cir, axis=1))
    alphabet_cardinality = np.size(transmit_alphabet)
    metric_vector = np.zeros(alphabet_cardinality * num_states)
    for path in range(survivor_paths.shape[0]):
        for symbol in transmit_alphabet:
            metric_vector[path*alphabet_cardinality] = np.dot(np.concatenate((survivor_paths[path, :], symbol)).T, cir)

