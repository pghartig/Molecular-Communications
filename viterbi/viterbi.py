import numpy as np
from communication_util.model_metrics import *
from communication_util.general_tools import get_combinatoric_list


def viterbi_setup_with_nodes(transmit_alphabet, channel_output, channel_length, metric_function):
    """

    :param transmit_alphabet:
    :param channel_output:
    :param channel_length:
    :param mm:
    :param net:
    :return:
    """
    # number of states is alphabet size raised to the power of the number of channel taps minus one.
    states = []
    item = []
    get_combinatoric_list(transmit_alphabet, channel_length, states, item)  # Generate states used below
    trellis = viterbi_trellis(transmit_alphabet, states, metric_function)
    # step through channel output
    for index in range(channel_output.shape[1]):
        trellis.step_trellis(index)
    return trellis.return_survivor()


class viterbi_trellis():
    def __init__(self, alphabet, states, metric_function):
        self.states = states
        self.alphabet = alphabet
        self.survivor_paths = []
        self.next_states = []
        self.setup_trellis(metric_function)

    def setup_trellis(self, metric_function):
        # create the trellis structure for a single step in the trellis
        for state in self.states:
            self.survivor_paths.append(viterbi_node(state, metric_function))
            self.next_states.append(viterbi_node(state, metric_function))
        # make connections between the nodes in the trellis
        for node in self.next_states:
            for previous_state in self.survivor_paths:
                check1 = node.state[:-1]
                check2 = previous_state.state[1:]
                if check1 == check2:
                    node.incoming_nodes.append(previous_state)

    def step_trellis(self, index):
        for node in self.next_states:
            node.check_smallest_incoming(index)
        # Move next_steps to previous and zero out next step costs
        i = 0
        for node in self.next_states:
            self.survivor_paths[i] = node
            i+=1

    def return_survivor(self):
        costs = []
        for path in self.survivor_paths:
            costs.append(path.survivor_path_cost)
        survivor_index = np.argmin(np.asarray(costs))
        return self.survivor_paths[survivor_index]


class viterbi_node():
    def __init__(self, state, metric_function):
        self.incoming_nodes = []
        self.survivor_path = []
        self.survivor_path_cost = 0
        self.state = state
        self.metric_function = metric_function

    def check_smallest_incoming(self, index):
        """
        Based on a metric, find the new surviving path going into the next step in the trellis
        :return:
        """
        incoming_costs = []
        for incoming in self.incoming_nodes:
            incoming_metric = self.metric_function(index, self.state)
            incoming_costs.append(incoming.survivor_path_cost+incoming_metric)
        survivor_index = np.argmin(np.asarray(incoming_costs))
        survivor_node = self.incoming_nodes[survivor_index]
        # add symbol to survivor path
        self.survivor_path = [survivor_node.survivor_path] + [self.state[-1] ] + [3]
        self.survivor_path_cost = incoming_costs[survivor_index]






