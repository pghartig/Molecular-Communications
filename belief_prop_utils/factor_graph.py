import numpy as np
from communication_util.model_metrics import *
# import ABC, abstract_method

class factor_graph():
    """
    The factor graph should be determined by the nodes and thus no inheritence for specific factor graph algorithms
    should be needed here.
    """
    def __init__(self, metric : metric, code_rules = None):
        self.variables = []
        self.functions = []
        self.setup_varible_nodes(metric)
        self.setup_function_nodes(code_rules)

    def setup_varible_nodes(self, metric):
        # First setup variables nodes
        for index , symbol in enumerate(metric.received):
            self.variables.append(variable_node(metric.metric(index)))

    def setup_function_nodes(self, code_rules=None):
        """
        Code Rules should come as a dict with the associated parity checks for each bit as the values for the symbrol
        index key.
        :param code_rules:
        :return:
        """
        if code_rules is None:
            # In this case just attach adjacent variable nodes
            #TODO turn into a code rule
            previous = None
            for ind, variable_node in enumerate(self.variables):
                if previous is not None:
                    new_func_node = function_node(np.sum)
                    self.functions.append(new_func_node)
                    variable_node.add_connection(new_func_node)
                    previous.add_connection(new_func_node)
                previous = variable_node
        else:
            #TODO
            for symbol in code_rules:
                self.factors.get("variables")[symbol]

    def iterate_message_passing(self):
        pass

#TODO make ABC
class graph_node():
    """
    may want to enforce bitartite creation first
    """
    def __init__(self):
        self.neighbors = []
        # Rule for processing incoming message (this will be a function)
        self.rule = None
        self.incoming_message_queue = []
        self.outgoing_message = None

    def add_connection(self, neighbor):
        self.neighbors.append(neighbor)

    def add_message(self, message):
        self.incoming_message_queue.append(message)

    #@abstract_method
    def collect_messages(self):
        while self.incoming_message_queue is not None:
            self.rule(self.outgoing_message, self.incoming_message_queue.pop())

    #@abstract_method
    def map_message(self):
        for neighbor in self.neighbors:
            neighbor.add_message(self.outgoing_message)

class variable_node(graph_node):
    def __init__(self, metric):
        graph_node.__init__(self)
        self.metric = metric


class function_node(graph_node):
    def __init__(self, operation = np.sum):
        graph_node.__init__(self)
        self.operation = operation
