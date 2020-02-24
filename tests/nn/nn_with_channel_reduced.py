"""
This test generates the symbol error rate curves over various SNR for comparing the performance of different decoding scemes.
"""

import torch.nn as nn
from mixture_model.em_algorithm import MixtureModel
from mixture_model.em_algorithm import em_gausian
import pickle
from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.general_tools import *
from communication_util.Equalization.supervise_equalization import *
from nn_utilities import models
import torch.optim as optim
import os
import time
import pandas as pd

def test_nn_channel_reduced():

    viterbi_net_performance = []
    linear_mmse_performance = []
    classic_performance = []
    SNRs_dB = np.linspace(15, 15, 1)
    # SNRs_dB = np.linspace(6, 10,3)
    SNRs = np.power(10, SNRs_dB/10)
    seed_generator = 0
    data_gen = None
    channel = None

    number_symbols = 5000
    channel = np.zeros((1, 5))
    # channel[0, [0, 1, 2, 3, 4]] = 0.227, 0.460, 0.688, 0.460, 0.227
    # Method used in ViterbiNet Paper
    channel[0, :] = np.random.randn(channel.size)
    # channel = np.zeros((1, 5))
    # channel[0, [0, 1, 2, 3, 4]] = 1, 0, .2, .2, .4
    # channel[0, [0, 1, 2, 3]] = .8, 0, .02, .4

    # channel[0, [0, 1, 2, 3, 4]] = 1, .7, .3, .1, .4
    # channel[0, [0, 1, 2, 3, 4]] = 1, .4, .7, .1, .3
    # channel = np.zeros((1, 3))
    # channel[0, [0]] = 1

    """
    Generated Testing Data using the same channel as was used for training the mixture model and the nn
    """

    data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SNRs, plot=True, channel=channel)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()

    # plt.scatter(data_gen.channel_output.flatten(),data_gen.channel_output.flatten())
    # plt.show()
    """
    Load in Trained Neural Network and verify that it is acceptable performance
    """
    device = torch.device("cpu")
    reduced_state = 32
    x, y = data_gen.get_labeled_data_reduced_state(reduced_state)
    # x, y = data_gen.get_labeled_data()
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
    output_layer_size = np.power(m, channel_length)
    output_layer_size = reduced_state
    N, D_in, H1, H2, D_out = number_symbols, 1, 100, 50, output_layer_size
    # N, D_in, H1, H2, H3, D_out = number_symbols, 1, 20, 20, 20, output_layer_size


    net = models.ViterbiNet(D_in, H1, H2, D_out)
    # net = models.viterbiNet(D_in, H1, H2, D_out, dropout_probability)
    # net = models.deeper_viterbiNet(D_in, H1, H2, H3, D_out, dropout_probability)


    # N, D_in, H1, H2, H3, D_out = number_symbols, num_inputs_for_nn, 20, 10, 10, np.power(m, channel_length)
    # net = models.deeper_viterbiNet(D_in, H1, H2, H3, D_out)
    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    # optimizer = optim.SGD(net.parameters(), lr=1e-1)

    """
    Train NN
    """
    # This loss function looks at only the true class and takes the NLL of that.
    criterion = nn.NLLLoss()
    train_cost_over_epoch = []
    test_cost_over_epoch = []
    batch_size = 1000

    # If training is perfect, then NN should be able to perfectly predict the class to which a test set belongs and thus the loss (KL Divergence) should be zero
    epochs = 900
    for t in range(epochs):
        batch_indices = np.random.randint(len(y_train), size=(1, batch_size))
        x_batch = x_train[(batch_indices)]
        y_batch = y_train[(batch_indices)]
        # Add "dropout to prevent overfitting data"
        net.zero_grad()
        output = net(x_batch)
        loss = criterion(output, y_batch.long())
        loss.backward()
        optimizer.step()

        test_batch_indices = np.random.randint(len(y_test), size=(1, batch_size))
        x_batch_test = x_test[(test_batch_indices)]
        y_batch_test = y_test[(test_batch_indices)]
        # Setup Accuracy test
        # detached_ouput = output.
        max_state_train = np.argmax(output.detach().numpy(), axis=1)
        check = np.not_equal(max_state_train, y_batch.detach().numpy())
        max_state_test = np.argmax(net(x_batch_test).detach().numpy(), axis=1)
        train_cost_over_epoch.append(np.sum(np.not_equal(max_state_train, y_batch.detach().numpy()))/y_batch.size())
        test_cost_over_epoch.append(np.sum(np.not_equal(max_state_test, y_batch_test.detach().numpy()))/y_batch_test.size())
        # test_cost_over_epoch.append(criterion(net(x_batch_test), y_batch_test.long()))


    #Plots for NN training information
    plt.figure(2)
    plt.plot(test_cost_over_epoch, label='Test Error')
    plt.plot(train_cost_over_epoch, label='Train Error')
    plt.title(str(data_gen.get_info_for_plot()), fontdict={'fontsize': 10})
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='upper right')
    path = f"Output/Neural_Network{time.time()}_Convergence.png"
    plt.savefig(path, format="png")


