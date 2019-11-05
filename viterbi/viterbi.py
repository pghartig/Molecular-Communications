import numpy as np
from communication_util.model_metrics import *
from communication_util.general_tools import get_combinatoric_list


def viterbi_output(transmit_alphabet, channel_output, channel_information):
    alphabet_size = transmit_alphabet.size
    # number of states is alphabet size raised to the power of the number of channel taps minus one.
    num_states = np.power(alphabet_size, channel_information.shape[1] - 1)
    survivor_paths = 1j * np.zeros((num_states, channel_output.shape[1]))
    survivor_paths_costs = np.zeros((num_states,1))


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
                survivor_paths_costs[state, 0] += metric_vector[symbol]


    final_path_ind = np.argmin(np.sum(survivor_paths, axis=1))
    return survivor_paths[final_path_ind, channel_information.size - 1:]


def viterbi_reimplemented(transmit_alphabet, channel_output, channel_information):
    alphabet_size = transmit_alphabet.size
    # number of states is alphabet size raised to the power of the number of channel taps minus one.
    num_states = np.power(alphabet_size, channel_information.shape[1])
    survivor_paths = 1j * np.zeros((num_states, channel_output.shape[1]))
    survivor_paths_costs = np.zeros((num_states, 1))

    # iterate through the metrics
    for i in range(channel_output.shape[1]):
        # assume zero-padding at beginning of word so set survivor path portions to zeros automatically
        if i < channel_information.shape[1] - 1:
            for state in range(num_states):
                survivor_paths[state, i] = transmit_alphabet[0]
                # no cost added to paths since symbols are known
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
                survivor_costs = survivor_paths_costs[state * alphabet_size: (state + 1) * alphabet_size]
                probability_metrics = metric_vector[state * alphabet_size: (state + 1) * alphabet_size]
                symbol = np.argmin(survivor_costs.flatten() + probability_metrics)
                survivor_paths[state, i] = transmit_alphabet[symbol]
                survivor_paths_costs[state, 0] += metric_vector[symbol]

    final_path_ind = np.argmin(np.sum(survivor_paths, axis=1))
    return survivor_paths[final_path_ind, channel_information.size - 1:]


def viterbi_NN_MM_output(transmit_alphabet, channel_output, channel_length,mm,net):
    """

    :param transmit_alphabet:
    :param channel_output:
    :param channel_length:
    :param mm:
    :param net:
    :return:
    """
    alphabet_size = transmit_alphabet.size
    # number of states is alphabet size raised to the power of the number of channel taps minus one.
    num_states = alphabet_size**(channel_length)
    survivor_paths = 1j * np.zeros((num_states, channel_output.shape[1]))
    survivor_paths_costs = np.zeros((num_states,1))

    # iterate through the metrics
    for i in range(channel_output.shape[1]):
        # assume zero-padding at beginning of word so set survivor path portions to zeros automatically
        if i < channel_length - 1:
            for state in range(num_states):
                survivor_paths[state, i] = transmit_alphabet[0]
            continue
        else:
            metric_vector =\
                autoencoder_channel_metric(net, mm, transmit_alphabet, channel_output[0, i], channel_length)
            for state in range(num_states):
                candidates = metric_vector[state]+survivor_paths_costs[state * alphabet_size: (state + 1) * alphabet_size,0]
                symbol = np.argmin(candidates)
                survivor_paths[state, i] = transmit_alphabet[symbol]
                survivor_paths_costs[state, 0] += metric_vector[symbol]

    final_path_ind = np.argmin(np.sum(survivor_paths, axis=1))
    return survivor_paths[final_path_ind, channel_length - 1:]


def viterbi_setup_with_nodes(transmit_alphabet, channel_output, channel_length):
    """

    :param transmit_alphabet:
    :param channel_output:
    :param channel_length:
    :param mm:
    :param net:
    :return:
    """
    alphabet_size = transmit_alphabet.size
    # number of states is alphabet size raised to the power of the number of channel taps minus one.
    num_states = alphabet_size**channel_length

    states = []
    item = []
    get_combinatoric_list(transmit_alphabet, channel_length, states, item)  # Generate states used below
    tellis = viterbi_trellis(np.add, num_states, transmit_alphabet, states)
    survivor_paths_costs = np.zeros((num_states, 1))

    # iterate through the metrics

    return False


class viterbi_trellis():
    def __init__(self, num_states, alphabet,states):
        self.states = states
        self.alphabet = alphabet
        self.survivor_paths = []
        self.next_states = []
        self.setup_trellis()

    def setup_trellis(self):
        for state in self.states:
            self.survivor_paths.append(viterbi_node(state))
            self.next_states.append(viterbi_node(state))

    def step_trellis(self,metrics):
        for index, state in self.next_states:
            state.check_smallest_incoming(metrics[index-1])
        #need to then move next_steps to previous and zero out next step costs

class viterbi_node():
    def __init__(self, state):
        self.incoming_nodes = []
        self.outgoing_nodes = []
        self.survivor_path = 0
        self.survivor_path_cost = 0
        self.state = state

    def check_smallest_incoming(self):
        incoming_metrics = []
        for incoming in self.incoming_nodes:
            metric = 0
            cost = incoming.survivor_path_cost + metric
            incoming_metrics.append(incoming.survivor_path_cost)
            # look through all incoming nodes and select the one with the




