import numpy as np

# import ABC, abstract_method

class factor_graph():
    """
    The factor graph should be determined by the nodes and thus no inheritence for specific factor graph algorithms
    should be needed here.
    """
    def __init__(self, received_sequence, code_rules = None):
        self.variables = []
        self.functions = []
        self.setup_varible_nodes(received_sequence)
        self.setup_function_nodes(code_rules)

    def setup_varible_nodes(self, received):
        # First setup variables nodes
        for symbol in received:
            self.variables.append(graph_node(symbol))

    def setup_function_nodes(self, code_rules):
        """
        Code Rules should come as a dict with the associated parity checks for each bit as the values for the symbrol
        index key.
        :param code_rules:
        :return:
        """
        if code_rules is None:
            # In this case just attach adjacent variable nodes
            previous = None
            for ind, variable_node in enumerate(self.variables):
                if previous is not None:
                    variable_node.add_connection(previous)
                    previous.add_connection(variable_node)
        else:
            #TODO
            for symbol in code_rules:
                self.factors.get("variables")[symbol]


#TODO make ABC
class graph_node():
    """
    may want to enforce bitartite creation first
    """
    def __init__(self, metric):
        self.neighbors  = []
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

class function_node():
    def __init__(self):
        pass


class function_node():
    def __init__(self):
        pass
