"""
This test generates the symbol error rate curves over various SNR for comparing the performance of different decoding scemes.
"""

import torch.nn as nn
from mixture_model.em_algorithm import mixture_model
from mixture_model.em_algorithm import em_gausian
import pickle
from communication_util.basic_detectors import *
from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.general_tools import *
from nn_utilities import models
import torch.optim as optim
import os
import time

def test_nn_validation():

    viterbi_net_performance = []
    threshold_performance = []
    classic_performance = []
    SNRs_dB = 2
    SNRs =  np.power(10, SNRs_dB/10)
    seed_generator = 0
    data_gen = None

    """
    Generated Testing Data using the same channel as was used for training the mixture model and the nn
    """
    number_symbols = 2000
    channel = np.zeros((1, 3))
    channel[0, [0, 1, 2]] = 1, 0.1, 0.1
    data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SNRs, plot=True, channel=channel)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()

    """
    Load in Trained Neural Network and verify that it is acceptable performance
    """
    device = torch.device("cpu")
    num_inputs_for_nn = 1
    x, y = data_gen.get_labeled_data(inputs=num_inputs_for_nn)
    y = np.argmax(y, axis=1)  # Fix for how the pytorch Cross Entropy expects class labels to be shown
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    train_size = int(.6 * number_symbols)
    x_train = x[0:train_size, :]
    x_test = x[train_size::, :]
    y_train = y[0:train_size]
    y_test = y[train_size::]

    """
    Setup NN and optimizer
    """
    m = data_gen.alphabet.size
    channel_length = data_gen.CIR_matrix.shape[1]

    N, D_in, H1, H2, D_out = number_symbols, 1, 100, 50, np.power(m, channel_length)
    net = models.viterbiNet(D_in, H1, H2, D_out)
    # N, D_in, H1, H2, H3, D_out = number_symbols, num_inputs_for_nn, 20, 10, 10, np.power(m, channel_length)
    # net = models.deeper_viterbiNet(D_in, H1, H2, H3, D_out)
    # optimizer = optim.Adam(net.parameters(), lr=1e-4)
    optimizer = optim.SGD(net.parameters(), lr=1e-2)

    """
    Train NN
    """
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    train_cost_over_epoch = []
    test_cost_over_epoch = []
    batch_size = 500

    # If training is perfect, then NN should be able to perfectly predict the class to which a test set belongs and thus the loss (KL Divergence) should be zero
    for t in range(1000):
        batch_indices = np.random.randint(len(y_train), size=(1, batch_size))
        x_batch = x_train[(batch_indices)]
        y_batch = y_train[(batch_indices)]
        output = net(x_batch)
        loss = criterion(output, y_batch.long())
        train_cost_over_epoch.append(loss)
        net.zero_grad()
        print(loss)
        loss.backward()
        optimizer.step()
        test_batch_indices = np.random.randint(len(y_test), size=(1, batch_size))
        x_batch_test = x_test[(test_batch_indices)]
        y_batch_test = y_test[(test_batch_indices)]
        test_cost_over_epoch.append(criterion(net(x_batch_test), y_batch_test.long()))



    #Plots for NN training information
    plt.figure(2)
    plt.plot(test_cost_over_epoch, label='Test Error')
    plt.plot(train_cost_over_epoch, label='Train Error')
    plt.title(str(data_gen.get_info_for_plot()), fontdict={'fontsize': 10})
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='upper left')
    plt.show()
    assert True
