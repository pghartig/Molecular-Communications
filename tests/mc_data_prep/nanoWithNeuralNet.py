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
from communication_util.load_mc_data import *

def test_nano_data_nerual_net():

    viterbi_net_performance = []

    """
    Generated Testing Data using the same channel as was used for training the mixture model and the nn
    """
    #   Load data from path
    train_path = 'mc_data/5_cm_train.csv'
    test_path = 'mc_data/5_cm_test.csv'
    true_path = 'mc_data/input_string.csv'
    # train_path = 'mc_data/20_cm_train.csv'
    # test_path = 'mc_data/20_cm_test.csv'
    test_input_sequence = 'mc_data/input_string.csv'
    test_input_sequence = np.loadtxt(test_input_sequence, delimiter=",")
    true_input_string = np.loadtxt(true_path, delimiter=",")
    # For now just making a channel that represents some estimated memory length of the true channel
    SNRs_dB = np.linspace(5, 5, 1)
    # SNRs_dB = np.linspace(6, 10,3)
    SNRs = np.power(10, SNRs_dB/10)

    channel = np.zeros((1, 8))
    channel[0, [0]] = 1
    train_time, train_measurement = load_file(train_path)
    test_time, test_measurement = load_file(test_path)
    # plt.plot(test_measurement)
    # plt.show()
    pulse_shape = get_pulse(train_time, train_measurement)
    #   Train with a random symbol stream generated from the training set pulse
    number_symbols = 10000
    data_gen = CommunicationDataGenerator(SNR=SNRs, symbol_stream_shape=(1, number_symbols), constellation="onOffKey", channel=channel)
    data_gen.random_symbol_stream()
    # 5 is true perdiod
    symbol_period = 5
    data_gen.modulate_sampled_pulse(pulse_shape, symbol_period)
    data_gen.filter_sample_modulated_pulse(pulse_shape, symbol_period)

    # Test for making sure alignment is correct
    # ser = symbol_error_rate_mc_data(data_gen.symbol_stream_matrix.flatten(), data_gen.channel_output.flatten(), channel.size)

    """
    Load in Trained Neural Network and verify that it is acceptable performance
    """
    reduced_state = 4
    # x, y, states_reduced, states_original, totals = data_gen.get_labeled_data_reduced_state(reduced_state)
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
    channel_length = data_gen.CIR_matrix.shape[1]
    # test_length = channel_length-1
    output_layer_size = reduced_state
    output_layer_size = np.power(data_gen.alphabet.size, channel_length)
    N, D_in, H1, H2, D_out = number_symbols, 1, 100, 50, output_layer_size
    net = models.ViterbiNet(D_in, H1, H2, D_out)
    optimizer = optim.Adam(net.parameters(), lr=1e-2)

    """
    Train NN
    """
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
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
        output = net(x_batch)
        loss = criterion(output, y_batch.long())
        net.zero_grad()
        loss.backward()
        optimizer.step()
        test_batch_indices = np.random.randint(len(y_test), size=(1, batch_size))
        x_batch_test = x_test[(test_batch_indices)]
        y_batch_test = y_test[(test_batch_indices)]
        max_state_train = np.argmax(output.detach().numpy(), axis=1)
        max_state_test = np.argmax(net(x_batch_test).detach().numpy(), axis=1)
        train_cost_over_epoch.append(np.sum(np.not_equal(max_state_train, y_batch.detach().numpy())) / y_batch.size())
        test_cost_over_epoch.append(
            np.sum(np.not_equal(max_state_test, y_batch_test.detach().numpy())) / y_batch_test.size())



    """
    Train Mixture Model
    """
    # num_sources = pow(data_gen.alphabet.size, data_gen.CIR_matrix.shape[1])
    num_sources = reduced_state
    mixture_model_training_data = data_gen.channel_output.flatten()[0:train_size]
    mm = em_gausian(num_sources, mixture_model_training_data, 10, save=True, model=True)
    mm = mm.get_probability


    """
    Create new set of test data. 
    """
    # For comparing generated data to the true test data
    # del data_gen
    # number_symbols = 2000
    # data_gen = CommunicationDataGenerator(SNR=SNRs, symbol_stream_shape=(1, number_symbols), constellation="onOffKey", channel= channel)
    # data_gen.random_symbol_stream()
    # data_gen.modulate_sampled_pulse(pulse_shape, symbol_period)
    # data_gen.filter_sample_modulated_pulse(pulse_shape, symbol_period)
    # generated_output = data_gen.channel_output

    del data_gen
    data_gen = CommunicationDataGenerator(SNR=SNRs, symbol_stream_shape=(1, number_symbols), constellation="onOffKey", channel= channel)
    data_gen.random_symbol_stream(true_input_string)
    data_gen.provide_transmitted_matrix(test_measurement)
    data_gen.filter_sample_modulated_pulse(pulse_shape, symbol_period)
    # # plt.scatter(data_gen.channel_output, data_gen.channel_output)
    # plt.scatter(generated_output, generated_output)
    # plt.show()

    """
    Use Test Data 
    """


    """
    Evaluate Neural Net Performance
    """
    # metric = NeuralNetworkMixtureModelMetric(net, mm, data_gen.channel_output)
    metric = NeuralNetworkMixtureModelMetric(net, mm, np.flip(data_gen.channel_output))
    # detected_nn = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
    #                                        metric.metric, reduced_length=reduced_state, reduced=True)
    detected_nn = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                           metric.metric)
    # symbol_probabilities = get_symbol_probabilities(totals, states_original, data_gen.alphabet)
    # detected_nn = get_symbols_from_probabilities(detected_nn, symbol_probabilities, data_gen.alphabet)
    ser_nn = symbol_error_rate_mc_data(detected_nn, data_gen.symbol_stream_matrix, channel_length)
    # ser_nn = symbol_error_rate_channel_compensated_NN(detected_nn, data_gen.symbol_stream_matrix, channel_length)

    viterbi_net_performance.append(ser_nn)

    print(viterbi_net_performance)

    path = "Output/SER.pickle"
    pickle_out = open(path, "wb")
    pickle.dump([viterbi_net_performance], pickle_out)
    pickle_out.close()

    # figure, dictionary = plot_symbol_error_rates(SNRs_dB, [classic_performance, linear_mmse_performance, viterbi_net_performance], data_gen.get_info_for_plot())
    # time_path = "Output/SER_"+f"{time.time()}"+"curves.png"
    # figure.savefig(time_path, format="png")
    # text_path = "Output/SER_"+f"{time.time()}"+"curves.csv"
    # pd.DataFrame.from_dict(dictionary).to_csv(text_path)
    # figure.savefig(time_path, format="png")
    #

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


