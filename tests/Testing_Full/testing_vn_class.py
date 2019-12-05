from nn_utilities import models
from communication_util import training_data_generator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import pickle



def test_viterbi_net_class():

    """
    Train and save the neural network
    :return:
    """

    """
    Choose platform
    """
    device = torch.device("cpu")
    # device = torch.device('cuda') # Uncomment this to run on GPU

    """
    Setup Training Data
    """
    number_symbols = 1000

    # channel = np.zeros((1, 5))
    # channel[0, [0, 3, 4]] = 1, 0.5, 0.4

    channel = np.zeros((1, 3))
    channel[0, [0, 1, 2]] = 1, 0.6, 0.3

    # channel = np.zeros((1, 1))
    # channel[0, 0] = 1

    data_gen = training_data_generator(
        symbol_stream_shape=(1, number_symbols),
        SNR=2,
        plot=True,
        channel=channel,
    )
    # data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()

    """
    After sending through channel, symbol detection should be performed using something like a matched filter
    """
    num_inputs_for_training_data = 1
    x, y = data_gen.get_labeled_data(inputs=num_inputs_for_training_data)
    y = np.argmax(y, axis=1)  # Fix for how the pytorch Cross Entropy expects class labels to be shown
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    train_size = int(.6*number_symbols)
    x_train = x[0:train_size, :]
    x_test = x[train_size::, :]
    y_train = y[0:train_size]
    y_test = y[train_size::]

    """
    Setup NN and optimizer
    """
    m = data_gen.alphabet.size
    channel_length = data_gen.CIR_matrix.shape[1]

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H1, H2, D_out = number_symbols, num_inputs_for_training_data, 100, 50, np.power(m, channel_length)


    net = models.viterbiNet(D_in, H1, H2, D_out)
    #TODO use better optimizer
    optimizer = optim.SGD(net.parameters(), lr=1e-2)
    optimizer = optim.Adam(net.parameters(), lr=1e-2)

    # optimizer = optim.SGD(net.parameters(), lr=5)


    """
    Train NN
    """
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    train_cost_over_epoch = []
    test_cost_over_epoch = []

    # If training is perfect, then NN should be able to perfectly predict the class to which a test set belongs and thus the loss (KL Divergence) should be zero
    for t in range(500):
        output = net(x_train)
        loss = criterion(output, y_train.long())
        train_cost_over_epoch.append(loss)
        net.zero_grad()
        print(loss)
        loss.backward()
        optimizer.step()
        test_cost_over_epoch.append(criterion(net(x_test), y_test.long()))

    path = "Output/nn.pt"
    torch.save(net, path)

    """
    Test NN
    """
    plt.figure()
    plt.plot(test_cost_over_epoch, label='Test Error')
    plt.plot(train_cost_over_epoch, label='Train Error')
    plt.title(str(data_gen.get_info_for_plot()), fontdict={'fontsize': 10})
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='upper left')
    path = "Output/Neural_Network_Convergence.png"
    path = "Output/NN_ERROR" + str(number_symbols) + " symbols " \
           + str(num_inputs_for_training_data) + " inputs " + str(time.time())+"curves.png"
    plt.savefig(path, format="png")
    plt.show()
    assert True