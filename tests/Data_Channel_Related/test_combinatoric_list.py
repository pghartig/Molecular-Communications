import numpy as np
from communication_util.general_tools import get_combinatoric_list


def test_combinatoric_list_generation():
    channel = np.zeros((1, 5))
    channel[0, [0, 3, 4]] = 1, 0.5, 0.4
    alphabet = np.array([-1,1])
    string_list = []
    item = []
    get_combinatoric_list(alphabet, 5, string_list, item)
    new_string_list = np.array(string_list)
    for i in range(new_string_list.shape[0]):
        for j in range(new_string_list.shape[0]):
            if i != j:
                # assert not np.array_equal(list[i, :], list[j, :])
                check1 = new_string_list[i, :]
                check2 = new_string_list[j, :]
                assert not np.equal(check1, check2).all()
    print("finished")