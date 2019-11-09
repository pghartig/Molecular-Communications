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
        self.previous_states = []
        self.next_states = []
        self.setup_trellis(metric_function)
        self.metric_function = metric_function


    def setup_trellis(self, metric_function):
        # create the trellis structure for a single step in the trellis
        for state in self.states:
            self.previous_states.append(viterbi_node(state, metric_function))
            self.next_states.append(viterbi_node(state, metric_function))
        # make connections between the nodes in the trellis
        for node in self.next_states:
            for previous_state in self.previous_states:
                check1 = node.state[:-1]
                check2 = previous_state.state[1:]
                if check1 == check2:
                    node.incoming_nodes.append(previous_state)

    def step_trellis(self, index):
        i = 0
        metrics = self.metric_function(index, self.states)
        for node in self.next_states:
            node.check_smallest_incoming(index, metrics[i])
            i+=1
        i = 0
        for node in self.next_states:
            self.previous_states[i].survivor_path = node.survivor_path
            self.previous_states[i].survivor_path_cost = node.survivor_path_cost
            i+=1

    def return_survivor(self):
        costs = []
        for path in self.previous_states:
            costs.append(path.survivor_path_cost)
        survivor_index = np.argmin(np.asarray(costs))
        survivor = self.previous_states[survivor_index]
        return survivor.survivor_path


class viterbi_node():
    def __init__(self, state, metric_function):
        self.incoming_nodes = []
        self.survivor_path = []
        self.survivor_path_cost = 0
        self.state = state
        self.state_metric = 0
        self.metric_function = metric_function #TODO get rid of after testing

    def check_smallest_incoming(self, index, state_metric):
        """
        Based on a metric, find the new surviving path going into the next step in the trellis
        :return:
        """
        incoming_costs = []
        for incoming in self.incoming_nodes:
            incoming_costs.append(incoming.survivor_path_cost+state_metric)
        survivor_index = np.argmin(np.asarray(incoming_costs))
        survivor_node = self.incoming_nodes[survivor_index]
        # add symbol to survivor path
        self.survivor_path = survivor_node.survivor_path + [self.state[-1]]
        self.survivor_path_cost = incoming_costs[survivor_index]






