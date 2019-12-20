import numpy as np

# import ABC, abstract_method

class factor_graph():
    def __init__(self):
        factor = dict()

#TODO make ABC
class graph_node():
    """
    may want to enforce bitartite creation first
    """
    def __init__(self):
        self.neighbors  = []
        # Rule for processing incoming message (this will be a function)
        self.rule = None
        self.incoming_message_queue = []
        self.outgoing_message = None

    def add_message(self, message):
        self.incoming_message_queue.append(message)

    #@abstract_method
    def collect_messages(self):
        while self.incoming_message_queue is not None:
            self.rule(self.outgoing_message, self.incoming_message_queue.pop())

    #@abstract_method
    def map_message(self):
        return None



class function_noe():
    def __init__(self):
        pass
