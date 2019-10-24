import numpy as np
from itertools import permutations
from communication_util.em_algorithm import em_gausian


def gaussian_channel_metric(
    survivor_paths, index, transmit_alphabet, channel_output, cir
):
    """
    returns vector of metrics for incoming state of viterbi with a gaussian channel
    :param survivor_paths:
    :param index:
    :param transmit_alphabet:
    :param channel_output:
    :param cir:
    :return:
    """
    num_states = np.power(np.size(transmit_alphabet), np.size(cir, axis=1) - 1)
    alphabet_cardinality = np.size(transmit_alphabet)
    metric_vector = np.zeros(alphabet_cardinality * num_states)
    for path in range(survivor_paths.shape[0]):
        for i in range(transmit_alphabet.size):
            candidate = np.append(
                survivor_paths[path, index - cir.size + 1: index], transmit_alphabet[i]
            ).T
            received = channel_output[:, index - cir.size + 1: index + 1]
            metric_vector[path * alphabet_cardinality + i] = np.linalg.norm(
                (candidate * np.flip(cir) - received)
            )
    return metric_vector


def autoencoder_channel_metric(survivor_paths, index, transmit_alphabet, channel_output, cir):
    """
    returns vector of metrics for incoming state of viterbi with a gaussian channel
    :param survivor_paths:
    :param index:
    :param transmit_alphabet:
    :param channel_output:
    :param cir:
    :return:
    """
    mixture_model_probability = em_gausian()
    return None