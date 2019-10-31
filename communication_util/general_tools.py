import numpy as np


def get_combinatoric_list(alpabet, item_length, item_list, item):
    for i in range(alpabet.size):
        new = list(item)
        new.append(alpabet[i])
        if item_length > 0:
            get_combinatoric_list(alpabet, item_length - 1, item_list, new)
        if item_length == 0:
            item_list.append(new)
