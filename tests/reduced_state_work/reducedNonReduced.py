"""
This test generates the symbol error rate curves over various SNR for comparing the performance of different decoding scemes.
"""

import torch.nn as nn
from mixture_model.em_algorithm import mixture_model
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

def test_full_integration():

    viterbi_net_performance = []
    viterbi_net_reduced_performance = []
    linear_mmse_performance = []
    classic_performance = []
    SNRs_dB = np.linspace(10, 15, 2)
    # SNRs_dB = np.linspace(6, 10,3)
    SNRs = np.power(10, SNRs_dB/10)
    seed_generator = 0
    data_gen = None
    channel = None

    number_symbols = 5000
    channel = np.zeros((1, 5))
    channel[0, [0, 1, 2, 3, 4]] = 0.227, 0.460, 0.688, 0.460, 0.227
    # Method used in ViterbiNet Paper
    # channel[0, :] = np.random.randn(channel.size)
    for SNR in SNRs:
        """
        Generated Testing Data using the same channel as was used for training the mixture model and the nn
        """

        data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SNR, plot=True, channel=channel)
        data_gen.random_symbol_stream()
        data_gen.send_through_channel()

        """
        Load in Trained Neural Network and verify that it is acceptable performance
        """
        device = torch.device("cpu")
        reduced_state = 8
        x, y = data_gen.get_labeled_data_reduced_state(reduced_state)
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
        # test_length = channel_length-1
        output_layer_size = reduced_state
        # output_layer_size = np.power(m, channel_length)
        N, D_in, H1, H2, D_out = number_symbols, 1, 100, 50, output_layer_size
        reduced_net = models.viterbiNet(D_in, H1, H2, D_out)
        optimizer_reduced = optim.Adam(reduced_net.parameters(), lr=1e-2)

        channel_length = data_gen.CIR_matrix.shape[1]
        output_layer_size = np.power(m, channel_length)
        N, D_in, H1, H2, D_out = number_symbols, 1, 100, 50, output_layer_size


        net = models.viterbiNet(D_in, H1, H2, D_out)
        optimizer = optim.Adam(net.parameters(), lr=1e-2)

        """
        Train NN
        """
        # This loss function looks at only the true class and takes the NLL of that.
        criterion_reduced = nn.NLLLoss()
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
            reduced_net.zero_grad()
            reduced_output = reduced_net(x_batch)
            loss_reduced = criterion(reduced_output, y_batch.long())
            loss_reduced.backward()
            optimizer_reduced.step()

            net.zero_grad()
            output = net(x_batch)
            loss = criterion_reduced(output, y_batch.long())
            loss.backward()
            optimizer.step()

            test_batch_indices = np.random.randint(len(y_test), size=(1, batch_size))

        """
        Train Mixture Model
        """
        mixture_model_training_data_reduced = data_gen.channel_output.flatten()[0:train_size]
        num_sources = reduced_state
        mm_reduced = em_gausian(num_sources, mixture_model_training_data_reduced, 10, save=True, model=True)
        mm_reduced = mm_reduced.get_probability


        mixture_model_training_data = data_gen.channel_output.flatten()[0:train_size]
        num_sources = pow(data_gen.alphabet.size, data_gen.CIR_matrix.shape[1])
        mm = em_gausian(num_sources, mixture_model_training_data, 10, save=True, model=True)
        mm = mm.get_probability


        """
        User training data to train MMSE equalizer to then use on the test data
        """
        mmse_equalizer = linear_mmse()
        mmse_equalizer.train_equalizer(data_gen.symbol_stream_matrix, data_gen.channel_output, data_gen.symbol_stream_matrix, channel.size)

        """
        Create new set of test data. 
        """
        del data_gen
        data_gen = training_data_generator(symbol_stream_shape=(1, 2000), SNR=SNR, plot=True, channel=channel)
        data_gen.random_symbol_stream()
        data_gen.send_through_channel()

        """
        Evaluate Neural Net Performance
        """
        metric = NeuralNetworkMixtureModelMetric(reduced_net, mm_reduced, data_gen.channel_output)
        detected_nn = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                            metric.metric, reduced_length=reduced_state, reduced=True)
        ser_nn_reduced = symbol_error_rate_channel_compensated_NN_reduced(detected_nn, data_gen.symbol_stream_matrix, channel_length)


        metric = NeuralNetworkMixtureModelMetric(net, mm, data_gen.channel_output)
        detected_nn = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                            metric.metric)
        ser_nn = symbol_error_rate_channel_compensated_NN(detected_nn, data_gen.symbol_stream_matrix, channel_length)

        """
        Compare to Classical Viterbi with full CSI
        """
        # channel= np.round(channel*10)
        metric = GaussianChannelMetric(channel, data_gen.channel_output)  # This is a function to be used in the viterbi
        detected_classic = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                            metric.metric)

        ser_classic = symbol_error_rate(detected_classic, data_gen.symbol_stream_matrix, channel_length)

        """
        Evaluate performance with linear MMSE
        """


        """
        Analyze SER performance
        """
        linear_mmse_performance.append(mmse_equalizer.test_equalizer(data_gen.symbol_stream_matrix, data_gen.channel_output))
        viterbi_net_reduced_performance.append(ser_nn_reduced)
        viterbi_net_performance.append(ser_nn)
        classic_performance.append(ser_classic)


    figure, dictionary = plot_quantized_symbol_error_rates_nn_compare(SNRs_dB, [classic_performance,
                                                                                linear_mmse_performance, viterbi_net_reduced_performance,
                                                                                viterbi_net_performance], data_gen.get_info_for_plot())
    time_path = "Output/SER_"+f"{time.time()}"+"curves.png"

    figure.savefig(time_path, format="png")

    text_path = "Output/SER_"+f"{time.time()}"+"curves.csv"
    pd.DataFrame.from_dict(dictionary).to_csv(text_path)
    figure.savefig(time_path, format="png")




