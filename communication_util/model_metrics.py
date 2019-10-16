import numpy as np
from itertools import permutations

def gaussian_channel_metric(transmit_alphabet, channel_output, CIR):
    """
    goal is to return a matrix with all metrics
    :param transmit_alphabet:
    :param channel_output:
    :param CIR:
    :return:
    """
    # first create a matrix with all possible states of previous symbols to be used in metric calculations next
    num_states = np.power(np.size(transmit_alphabet), np.size(CIR, axis=1))
    alphabet_cardinality = np.size(transmit_alphabet)
    metric_vector = np.zeros(alphabet_cardinality * num_states)
    sequence_states = np.zeros((CIR.shape[1],np.size(transmit_alphabet)))

    for i in range(np.size(CIR, axis=1)):
        for symbol in range(alphabet_cardinality):
            sequence_states[i, symbol] = 1
            metric_vector[alphabet_cardinality*i+symbol,0] = channel_output - np.dot(sequence_states,CIR)


