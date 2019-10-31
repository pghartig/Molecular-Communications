import numpy as np
from communication_util.general_tools import get_combinatoric_list

def test_combinatoric_list_generation():
    alphabet = np.array([1, -1])
    list = []
    item = []
    np.asarray(get_combinatoric_list(alphabet, 5, list, item))
    list = np.asarray(list)
    for i in range(list.shape[0]):
        for j in range(list.shape[0]):
            if i != j:
                assert not np.array_equal(list[i, :], list[j, :])