import numpy as np
from communication_util.model_metrics import *
from communication_util.general_tools import get_combinatoric_list
import time


def viterbi_setup_with_nodes(transmit_alphabet, channel_output, channel_length, metric_function, reduced_length=None, reduced=False):

    # number of states is alphabet size raised to the power of the number of channel taps minus one.
    if reduced_length == None:
        reduced_length = channel_length
    else:
        reduced_length = int(np.log2(reduced_length))
    states = []
    item = []
    get_combinatoric_list(transmit_alphabet, reduced_length, states, item)
    if reduced is False:
        trellis = viterbi_trellis(transmit_alphabet, states, metric_function)
    else:
        trellis = viterbi_trellis(transmit_alphabet, states, metric_function, reduced=True)

    # step through channel output
    times_list = []
    t0 = time.clock()
    for index in range(channel_output.shape[1]):
        # Need to prevent stepping until there are sufficient metrics for the input to the NN
        if index>=channel_length-1 and index<= channel_output.shape[1] - (channel_length):
            t1 = time.clock()
            trellis.step_trellis(index)
            times_list.append(time.clock()-t1)
    check1 = np.average(np.asarray(times_list[::900]))
    check = time.clock()-t0
    return trellis.return_survivor()


class viterbi_trellis():
    """
    Note that in order to prevent keeping a large trellis in memory, the nodes are reused in each step.
    """
    def __init__(self, alphabet, states, metric_function, reduced=False):
        self.states = states
        self.alphabet = alphabet
        self.previous_states = []
        self.next_states = []
        self.metric_function = metric_function
        self.setup_trellis(reduced)

    def setup_trellis(self, reduced=False):
        # create the trellis structure for a single step in the trellis
        for state in self.states:
            self.previous_states.append(viterbi_node(state))
            self.next_states.append(viterbi_node(state))
        # make connections between the nodes in the trellis
        for node in self.next_states:
            for previous_state in self.previous_states:
                check1 = node.state[:-1]
                check1 = node.state[1:]
                check2 = previous_state.state[1:]
                check2 = previous_state.state[:-1]
                #test for reduced state
                if reduced == False:
                    if check1 == check2:
                        node.incoming_nodes.append(previous_state)
                if reduced == True:
                    node.incoming_nodes.append(previous_state)

    def step_trellis(self, index):
        metrics = self.metric_function(index, self.states)
        #TODO replace with lighter weight version not requiring whole output
        for ind, node in enumerate(self.next_states):
            node.check_smallest_incoming(metrics[ind])
        for ind, node in enumerate(self.next_states):
            self.previous_states[ind].survivor_path = node.survivor_path
            self.previous_states[ind].survivor_path_cost = node.survivor_path_cost

    def return_survivor(self):
        costs = []
        for path in self.previous_states:
            costs.append(path.survivor_path_cost)
        survivor_index = np.argmin(np.asarray(costs))
        survivor = self.previous_states[survivor_index]
        return survivor.survivor_path


class viterbi_node():
    def __init__(self, state):
        self.incoming_nodes = []
        self.survivor_path = []
        self.survivor_path_cost = 0
        self.state = state

    def check_smallest_incoming(self, state_metric):
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
        self.survivor_path = survivor_node.survivor_path + [self.state[0]]
        self.survivor_path_cost = incoming_costs[survivor_index]






