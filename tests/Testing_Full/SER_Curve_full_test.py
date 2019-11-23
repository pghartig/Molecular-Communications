"""
Full Test for using the mixture model and neural network to provide metrics to be used in the viterbi algorithm
"""

import torch.nn as nn
from mixture_model.em_algorithm import mixture_model
from mixture_model.em_algorithm import em_gausian
import pickle
from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.error_rates import *
from nn_utilities import models
import torch.optim as optim
import os

def test_full_integration():

    viterbi_net_performance = []
    classic_performance = []
    SNRs = np.linspace(1, 2, 10)
    seed_generator = 0
    for SRN in SNRs:

        error_tolerance = np.power(10.0, -3)

        """
        Generated Testing Data using the same channel as was used for training the mixture model and the nn
        """
        number_symbols = 5000
        channel = np.zeros((1, 3))
        channel[0, [0, 1, 2]] = 1, 0.6, 0.3
        data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SRN, plot=True, channel=channel)
        data_gen.random_symbol_stream()
        data_gen.send_through_channel()

        """
        Load in Trained Neural Network and verify that it is acceptable performance
        """
        device = torch.device("cpu")
        x, y = data_gen.get_labeled_data()
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

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        N, D_in, H1, H2, D_out = number_symbols, 1, 100, 50, np.power(m, channel_length)

        net = models.viterbiNet(D_in, H1, H2, D_out)
        # TODO use better optimizer
        optimizer = optim.SGD(net.parameters(), lr=1e-2)
        optimizer = optim.Adam(net.parameters(), lr=1e-2)

        # optimizer = optim.SGD(net.parameters(), lr=5)

        """
        Train NN
        """
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.NLLLoss()
        train_cost_over_epoch = []
        test_cost_over_epoch = []

        # If training is perfect, then NN should be able to perfectly predict the class to which a test set belongs and thus the loss (KL Divergence) should be zero
        for t in range(200):
            output = net(x_train)
            loss = criterion(output, y_train.long())
            train_cost_over_epoch.append(loss)
            net.zero_grad()
            print(loss)
            loss.backward()
            optimizer.step()
            test_cost_over_epoch.append(criterion(net(x_test), y_test.long()))


        # Test NN
        x, y = data_gen.get_labeled_data()
        y = np.argmax(y, axis=1)  # Fix for how the pytorch Cross Entropy expects class labels to be shown
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        criterion = nn.NLLLoss()
        # criterion = nn.CrossEntropyLoss()
        cost = criterion(net(x), y.long())
        threshold = 1e-2
        assert cost < threshold

        """
        Train Mixture Model
        """
        mixture_model_training_data = data_gen.channel_output.flatten()

        num_sources = pow(data_gen.alphabet.size, data_gen.CIR_matrix.shape[1])
        mm = em_gausian(num_sources, mixture_model_training_data, 20, save=True, model=True)
        mm = mm.get_probability


        """
        After sending through channel, symbol detection should be performed using something like a matched filter
        """

        #   !! Make sure channel output gets flipped here!!
        metric = nn_mm_metric(net, mm, data_gen.channel_output)  # This is a function to be used in the viterbi
        detected_nn = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                            metric.metric)
        ser_nn = symbol_error_rate(detected_nn, data_gen.symbol_stream_matrix)


        """
        Compare to Classical Viterbi with full CSI
        """

        metric = gaussian_channel_metric_working(channel, data_gen.channel_output)  # This is a function to be used in the viterbi
        detected_classic = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                            metric.metric)
        ser_classic = symbol_error_rate(detected_classic, data_gen.symbol_stream_matrix)

        """
        Analyze SER performance
        """
        viterbi_net_performance.append(ser_nn)
        classic_performance.append(ser_classic)

    assert True