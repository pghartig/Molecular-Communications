import numpy as np
from communication_util.model_metrics import *

def viterbi_output(transmit_alphabet, channel_output, channel_information):
    num_states = np.power(np.size(transmit_alphabet), np.size(channel_information, axis=1))
    survivor_paths = np.zeros((num_states.shape[0],channel_output.shape[1]), dtype=np.int8)
    #iterate through the metrics
    for detected_symbol in channel_output:
        metric_vector = gaussian_channel_metric(transmit_alphabet,detected_symbol, channel_information)

    final_path_ind = np.argmin(np.sum(survivor_paths,axis=1))
    return np.flip(survivor_paths[final_path_ind, :],axis=1)