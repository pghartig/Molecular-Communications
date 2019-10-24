from communication_util.em_algorithm import em_gausian
import numpy as np
import matplotlib.pyplot as plt

def test_em_gaussian():
    """
    The goal of this test is to ensure that the implemented EM algorithm converges to the correct parameters used to
    to generate a set of data that is then fed into the algorithm.
    :return:
    """
    tolerance = np.power(10.0, -3)

    # generate data from a set of gaussians
    num_sources = 4
    data = []
    for i in range(num_sources):
        data.append(np.random.normal(loc=(i+1), scale=1/(i+1), size=(1, 200)))
    data = np.asarray(data).flatten().T
    data = np.random.permutation(np.asarray(data).flatten().T)
    mu, variance, alpha = em_gausian(num_sources, data, 30)
    assert False

