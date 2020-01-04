import numpy as np
from communication_util.model_metrics import *
from abc import ABC, abstractmethod

class factor_graph():
    """
    The factor graph should be determined by the nodes and thus no inheritence for specific factor graph algorithms
    should be needed here.
    """
    def __init__(self, metric : metric, code_rules = None):
        self.variables = []
        self.functions = []
        self.setup_variable_nodes(metric)
        self.setup_function_nodes(code_rules)

    def setup_variable_nodes(self, variable_metric):
        # First setup variables nodes
        for index, symbol in enumerate(variable_metric.received):
            self.variables.append(variable_node(variable_metric.metric(index)))

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
            #TODO impelement factor node setup for general code
            for symbol in code_rules:
                self.factors.get("variables")[symbol]

    def iterate_message_passing(self):
        """
        First all variables nodes will passing their message to the queue of adjacent function nodes.
        Then function nodes will pass the message back to the queue of variable nodes.
        In this way, potential decoding can be done after each iteration.
        Each set of nodes will empty their message queue following an iteration.
        :return:
        """
        self.map_messages(self.variables)
        self.collect(self.functions)
        self.map_messages(self.functions)
        self.collect(self.variables)

    def map_messages(self, nodes):
        for node in nodes:
            node.map_message()
        pass

    def collect(self, nodes):
        for node in nodes:
            node.collect_messages()
        pass

class graph_node(ABC):
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

    #@abstractmethod
    def collect_messages(self):
        while self.incoming_message_queue is not None:
            self.rule(self.outgoing_message, self.incoming_message_queue.pop())
        self.incoming_message_queue = []

    #@abstractmethod
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