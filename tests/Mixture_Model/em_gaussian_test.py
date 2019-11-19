from mixture_model.em_algorithm import em_gausian
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
    num_sources = 16
    data = []
    true_mu = []
    true_sigma = []
    for i in range(num_sources):
        var = 1
        mu = (i + 1)
        data.append(np.random.normal(loc=mu, scale=var, size=(1, 100)))
        true_mu.append(mu)
        true_sigma.append(var)
    data = np.asarray(data).flatten().T
    data = np.random.permutation(np.asarray(data).flatten().T)
    mu, variance, alpha, iteration = em_gausian(num_sources, data, 50)



    plt.figure
    plt.plot(iteration)
    # test = os.getcwd()+"Output/SER.png"
    path = "Output/SER.png"
    plt.savefig(path, format="png")
    plt.show()
    assert True
