"""
This is testing the basic Nueral Net infrastructure to be used in Viterbi decoding. This can be used to get an
expectation for how well this neural network will perform.
"""

import torch.nn as nn
from mixture_model.em_algorithm import MixtureModel
from mixture_model.em_algorithm import em_gausian
import pickle
from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.general_tools import *
from nn_utilities import models
import torch.optim as optim
import os
import time

def test_auto_encoder():

    """
    Generated Testing Data using the same channel as was used for training the mixture model and the nn
    """
    number_symbols = 5000
    sources = 32
    x = np.random.randint(sources,size = (number_symbols, 1))
    y = x.flatten()
    x = x + np.random.standard_normal((number_symbols, 1))*1
    # plt.scatter(x, x)
    # plt.show()
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    train_size = int(.6 * number_symbols)
    x_train = x[0:train_size]
    x_test = x[train_size::]
    y_train = y[0:train_size]
    y_test = y[train_size::]

    """
    Setup NN and optimizer
    """

    N, D_in, H1, H2, D_out = number_symbols, 1, 100, 50, sources
    net = models.ViterbiNet(D_in, H1, H2, D_out)
    # N, D_in, H1, H2, H3, D_out = number_symbols, 1, 50, 50, 50, sources
    # net = models.deeper_viterbiNet(D_in, H1, H2, H3, D_out, dropout_probability)
    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    # optimizer = optim.SGD(net.parameters(), lr=1e-1)

    """
    Train NN
    """
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    train_cost_over_epoch = []
    test_cost_over_epoch = []
    accuracy = []
    batch_size = 1000

    # If training is perfect, then NN should be able to perfectly predict the class to which a test set belongs and thus the loss (KL Divergence) should be zero
    for t in range(900):
        batch_indices = np.random.randint(len(y_train), size=(1, batch_size))
        x_batch = x_train[(batch_indices)]
        y_batch = y_train[(batch_indices)]
        output = net(x_batch)
        loss = criterion(output, y_batch.long())
        train_cost_over_epoch.append(loss)
        net.zero_grad()
        # print(loss)
        loss.backward()
        optimizer.step()
        test_batch_indices = np.random.randint(len(y_test), size=(1, batch_size))
        x_batch_test = x_test[(test_batch_indices)]
        y_batch_test = y_test[(test_batch_indices)]
        # check = np.argmax(net(x_batch_test))
        # accuracy.append(np.sum(np.equal(net(x_batch_test), y_batch_test)))
        test_cost_over_epoch.append(criterion(net(x_batch_test), y_batch_test.long()))


    x  = np.random.randint(sources,size = (100000,1))
    y = x.flatten()
    x = torch.Tensor(x)
    check = net(x).detach().numpy()
    output = np.argmax(net(x).detach().numpy(),axis=1)
    test = np.not_equal(y, output)
    error = np.sum(np.not_equal(y, output))
    print(f"number of errors: {error/100000}")

    #Plots for NN training information
    # plt.figure(1)
    # plt.plot(accuracy, label='Test Error')
    plt.figure(2)
    plt.plot(test_cost_over_epoch, label='Test Error')
    plt.plot(train_cost_over_epoch, label='Train Error')
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='upper left')
    plt.show()
    assert True
