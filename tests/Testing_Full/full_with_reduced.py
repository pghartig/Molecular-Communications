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

def test_reduced_full():

    viterbi_net_performance = []
    viterbi_net_reduced_performance = []
    linear_mmse_performance = []
    classic_performance = []
    SNRs_dB = np.linspace(30, 30, 2)
    SNRs = np.power(10, SNRs_dB/10)
    seed_generator = 0
    data_gen = None
    channel = None
    quantization_level = None
    # quantization_level = 1
    noise_levels = None
    # noise_levels = 2

    number_symbols = 5000
    channel = np.zeros((1, 5))
    channel[0, [0, 1, 2, 3, 4]] = 0.227, 0.460, 0.688, 0.460, 0.227
    # channel[0, [0, 1, 2, 3, 4]] = 0.9, 0.7, 0.3, 0.5, 0.1

      # Channel to use for redundancy testing
    # Method used in ViterbiNet Paper
    # channel[0, :] = np.random.randn(channel.size)
    # channel = np.zeros((1, 5))
    # channel[0, [0, 1, 2, 3, 4]] = .9, 0, .0, .8, .7
    # channel = np.zeros((1, 3))
    # channel[0, [0, 1, 2]] = .9, .8, .7
    # channel = np.zeros((1, 3))
    # channel[0, [0]] = 1
    # channel = np.flip(channel)

    for SNR in SNRs:
        """
        Generated Testing Data using the same channel as was used for training the mixture model and the nn
        """

        data_gen = CommunicationDataGenerator(symbol_stream_shape=(1, number_symbols), SNR=SNR, plot=True, channel=channel)
        data_gen.random_symbol_stream()
        data_gen.send_through_channel(quantization_level, noise_levels=noise_levels)

        # plt.scatter(data_gen.channel_output.flatten(),data_gen.channel_output.flatten())
        # plt.show()
        """
        Setup Reduced ViterbiNet training data. 
        """
        reduced_state = 32
        x_reduced, y_reduced, states_reduced, states_original, totals = data_gen.get_labeled_data_reduced_state(reduced_state)
        y_reduced = np.argmax(y_reduced, axis=1)  # Fix for how the pytorch Cross Entropy expects class labels to be shown
        x_reduced = torch.Tensor(x_reduced)
        y_reduced = torch.Tensor(y_reduced)
        train_size = int(.6 * number_symbols)
        x_train_reduced = x_reduced[0:train_size, :]
        x_test_reduced = x_reduced[train_size::, :]
        y_train_reduced = y_reduced[0:train_size]
        y_test_reduced = y_reduced[train_size::]

        """
        Setup Standard ViterbiNet training data. 
        """
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
        Setup reduced NN and optimizer
        """
        m = data_gen.alphabet.size
        channel_length = data_gen.CIR_matrix.shape[1]
        output_layer_size = reduced_state
        N, D_in, H1, H2, D_out = number_symbols, 1, 100, 50, output_layer_size
        net_reduced = models.ViterbiNet(D_in, H1, H2, D_out)
        optimizer_reduced = optim.Adam(net_reduced.parameters(), lr=1e-2)
        criterion_reduced = nn.NLLLoss()
        batch_size = 1000
        train_cost_over_epoch_reduced = []
        test_cost_over_epoch_reduced = []


        # If training is perfect, then NN should be able to perfectly predict the class to which a test set belongs and thus the loss (KL Divergence) should be zero
        epochs = 50
        for t in range(epochs):
            batch_indices = np.random.randint(len(y_train), size=(1, batch_size))
            x_batch_reduced = x_train_reduced[(batch_indices)]
            y_batch_reduced = y_train_reduced[(batch_indices)]
            # Add "dropout to prevent overfitting data"
            net_reduced.zero_grad()

            output_reduced = net_reduced(x_batch_reduced)
            loss_reduced = criterion_reduced(output_reduced, y_batch_reduced.long())
            loss_reduced.backward()
            optimizer_reduced.step()


            test_batch_indices = np.random.randint(len(y_test), size=(1, batch_size))
            x_batch_test_reduced = x_test_reduced[(test_batch_indices)]
            y_batch_test_reduced = y_test_reduced[(test_batch_indices)]
            # Setup Accuracy test
            max_state_train_reduced = np.argmax(output_reduced.detach().numpy(), axis=1)
            max_state_test_reduced = np.argmax(net_reduced(x_batch_test_reduced).detach().numpy(), axis=1)
            train_cost_over_epoch_reduced.append(np.sum(np.not_equal(max_state_train_reduced, y_batch_reduced.detach().numpy()))/y_batch_reduced.size())
            test_cost_over_epoch_reduced.append(np.sum(np.not_equal(max_state_test_reduced, y_batch_test_reduced.detach().numpy()))/y_batch_test_reduced.size())


        """
        Setup standard NN and optimizer
        """
        m = data_gen.alphabet.size
        channel_length = data_gen.CIR_matrix.shape[1]
        # channel_length = channel_length-2
        output_layer_size = np.power(m, channel_length)
        num_states = output_layer_size
        N, D_in, H1, H2, D_out = number_symbols, 1, 100, 50, output_layer_size
        net = models.ViterbiNet(D_in, H1, H2, D_out)
        optimizer = optim.Adam(net.parameters(), lr=1e-2)



        """
        Train NN
        """
        # This loss function looks at only the true class and takes the NLL of that.
        criterion = nn.NLLLoss()
        train_cost_over_epoch = []
        test_cost_over_epoch = []

        epochs = 500
        for t in range(epochs):
            batch_indices = np.random.randint(len(y_train), size=(1, batch_size))
            x_batch = x_train[(batch_indices)]
            y_batch = y_train[(batch_indices)]

            # Add "dropout to prevent overfitting data"
            net.zero_grad()
            net_reduced.zero_grad()
            output = net(x_batch)

            loss = criterion(output, y_batch.long())
            loss.backward()
            optimizer.step()

            test_batch_indices = np.random.randint(len(y_test), size=(1, batch_size))
            x_batch_test = x_test[(test_batch_indices)]
            y_batch_test = y_test[(test_batch_indices)]

            # Setup Accuracy test
            max_state_train = np.argmax(output.detach().numpy(), axis=1)
            max_state_test = np.argmax(net(x_batch_test).detach().numpy(), axis=1)
            train_cost_over_epoch.append(np.sum(np.not_equal(max_state_train, y_batch.detach().numpy()))/y_batch.size())
            test_cost_over_epoch.append(np.sum(np.not_equal(max_state_test, y_batch_test.detach().numpy()))/y_batch_test.size())


        """
        Train Mixture Model
        """
        mixture_model_training_data = data_gen.channel_output.flatten()[0:train_size]
        num_sources = reduced_state
        mm = em_gausian(num_sources, mixture_model_training_data, 10, save=True, model=True)
        mm = mm.get_probability

        """
        Train Reduced Mixture Model
        """
        mixture_model_training_data_reduced = data_gen.channel_output.flatten()[0:train_size]
        num_sources = reduced_state
        mm_reduced = em_gausian(num_sources, mixture_model_training_data_reduced, 10, save=True, model=True)
        mm_reduced = mm_reduced.get_probability


        """
        User training data to train MMSE equalizer to then use on the test data
        """
        mmse_equalizer = LinearMMSE()
        mmse_equalizer.train_equalizer(data_gen.symbol_stream_matrix, data_gen.channel_output, data_gen.symbol_stream_matrix, channel.size)

        """
        Create new set of test data. 
        """
        ser_nn = []
        ser_nn_reduced = []
        ser_classic = []
        ser_lmmse = []

        for reps in range(2):

            del data_gen
            data_gen = CommunicationDataGenerator(symbol_stream_shape=(1, 1000), SNR=SNR, plot=True, channel=channel)
            data_gen.random_symbol_stream()
            data_gen.send_through_channel(quantization_level, noise_levels=noise_levels)

            """
            Evaluate Reduced Neural Net Performance
            """
            metric = NeuralNetworkMixtureModelMetric(net_reduced, mm_reduced, data_gen.channel_output)
            # metric = NeuralNetworkMixtureModelMetric(net_reduced, mm_reduced, np.flip(data_gen.channel_output))
            detected_nn = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                                metric.metric, reduced_length=reduced_state, reduced=True)
            symbol_probabilities = get_symbol_probabilities(totals, states_original, data_gen.alphabet)
            survivor_state_path = get_symbols_from_probabilities(detected_nn, symbol_probabilities, data_gen.alphabet)

            ser_nn_reduced = symbol_error_rate_channel_compensated_NN_reduced(survivor_state_path, data_gen.symbol_stream_matrix,
                                                                      channel_length)

            """
            Evaluate Neural Net Performance
            """

            # metric = NeuralNetworkMixtureModelMetric(net, mm, np.flip(data_gen.channel_output))
            metric = NeuralNetworkMixtureModelMetric(net, mm, data_gen.channel_output)

            detected_nn = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                                metric.metric)
            ser_nn = symbol_error_rate_channel_compensated_NN(detected_nn, data_gen.symbol_stream_matrix, channel_length)



            """
            Compare to Classical Viterbi with full CSI
            """
            metric = GaussianChannelMetric(channel,  np.flip(data_gen.channel_output), quantization_level)  # This is a function to be used in the viterbi
            # metric = GaussianChannelMetric(channel, data_gen.channel_output, quantization_level)  # This is a function to be used in the viterbi
            detected_classic = viterbi_setup_with_nodes(data_gen.alphabet,
                                                        data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                                        metric.metric)
            ser_classic = symbol_error_rate(detected_classic, data_gen.symbol_stream_matrix, channel_length)

            """
            Evaluate LMMSE
            """
            ser_lmmse.append(mmse_equalizer.test_equalizer(data_gen.symbol_stream_matrix, data_gen.channel_output))


        ser_nn = np.average(ser_nn)
        ser_nn_reduced = np.average(ser_nn_reduced)
        ser_classic = np.average(ser_classic)
        ser_lmmse = np.average(ser_lmmse)

        """
        Analyze SER performance
        """
        linear_mmse_performance.append(ser_lmmse)
        viterbi_net_performance.append(ser_nn)
        classic_performance.append(ser_classic)
        viterbi_net_reduced_performance.append(ser_nn_reduced)


    figure, dictionary = plot_quantized_symbol_error_rates_nn_compare(SNRs_dB, [classic_performance,
                                                                                linear_mmse_performance,
                                                                                viterbi_net_performance,
                                                                                viterbi_net_reduced_performance],
                                                                      data_gen.get_info_for_plot())
    time_path = "Output/SER_"+f"{time.time()}"+"curves.png"
    figure.savefig(time_path, format="png")
    text_path = "Output/SER_"+f"{time.time()}"+"curves.csv"
    pd.DataFrame.from_dict(dictionary).to_csv(text_path)
    figure.savefig(time_path, format="png")



