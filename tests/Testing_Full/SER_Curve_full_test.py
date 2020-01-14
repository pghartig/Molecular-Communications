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

def test_full_integration():

    viterbi_net_performance = []
    linear_mmse_performance = []
    classic_performance = []
    SNRs_dB = np.linspace(-5, 10, 10)
    # SNRs_dB = np.linspace(6, 10,3)
    SNRs =  np.power(10, SNRs_dB/10)
    seed_generator = 0
    data_gen = None
    channel = None
    for SNR in SNRs:
        """
        Generated Testing Data using the same channel as was used for training the mixture model and the nn
        """
        number_symbols = 2000
        channel = np.zeros((1, 5))
        channel[0, [0, 1, 2, 3, 4]] = 1, .1, .01, .1, .04
        # channel[0, [0, 1, 2, 3, 4]] = 1, .1, .3, .1, .4
        # channel[0, [0, 1, 2, 3, 4]] = 1, .4, .7, .1, .3
        # channel = np.zeros((1, 1))
        # channel[0, [0]] = 1
        data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SNR, plot=True, channel=channel)
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
        # test_length = channel_length-1
        # output_layer_size = reduced_state
        output_layer_size = np.power(m, channel_length)
        N, D_in, H1, H2, D_out = number_symbols, num_inputs_for_nn, 100, 50, output_layer_size

        # net = models.viterbiNet(D_in, H1, H2, D_out)
        dropout_probability = .3
        net = models.viterbiNet_dropout(D_in, H1, H2, D_out, dropout_probability)

        # N, D_in, H1, H2, H3, D_out = number_symbols, num_inputs_for_nn, 20, 10, 10, np.power(m, channel_length)
        # net = models.deeper_viterbiNet(D_in, H1, H2, H3, D_out)
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        # optimizer = optim.SGD(net.parameters(), lr=1e-1)

        """
        Train NN
        """
        criterion = nn.NLLLoss()
        # criterion = nn.CrossEntropyLoss()
        train_cost_over_epoch = []
        test_cost_over_epoch = []
        batch_size = 30

        # If training is perfect, then NN should be able to perfectly predict the class to which a test set belongs and thus the loss (KL Divergence) should be zero
        epochs = 300
        for t in range(epochs):
            batch_indices = np.random.randint(len(y_train), size=(1, batch_size))
            x_batch = x_train[(batch_indices)]
            y_batch = y_train[(batch_indices)]
            # Add "dropout to prevent overfitting data"

            output = net(x_batch)
            loss = criterion(output, y_batch.long())
            train_cost_over_epoch.append(loss)
            net.zero_grad()
            loss.backward()
            optimizer.step()
            test_batch_indices = np.random.randint(len(y_test), size=(1, batch_size))
            x_batch_test = x_test[(test_batch_indices)]
            y_batch_test = y_test[(test_batch_indices)]
            test_cost_over_epoch.append(criterion(net(x_batch_test), y_batch_test.long()))


        """
        Train Mixture Model
        """
        mixture_model_training_data = data_gen.channel_output.flatten()[0:train_size]
        num_sources = pow(data_gen.alphabet.size, data_gen.CIR_matrix.shape[1])
        mm = em_gausian(num_sources, mixture_model_training_data, 10, save=True, model=True)
        mm = mm.get_probability


        """
        Create new set of test data. 
        """
        del data_gen
        data_gen = training_data_generator(symbol_stream_shape=(1, 1000), SNR=SNR, plot=True, channel=channel)
        data_gen.random_symbol_stream()
        data_gen.send_through_channel()

        """
        Evaluate Neural Net Performance
        """
        metric = nn_mm_metric(net, mm, data_gen.channel_output, input_length=num_inputs_for_nn)
        detected_nn = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                            metric.metric)
        ser_nn = symbol_error_rate_channel_compensated_NN(detected_nn, data_gen.symbol_stream_matrix, channel_length)


        """
        Compare to Classical Viterbi with full CSI
        """
        # channel= np.round(channel*10)
        metric = gaussian_channel_metric_working(channel, data_gen.channel_output)  # This is a function to be used in the viterbi
        detected_classic = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                            metric.metric)
        ser_classic = symbol_error_rate(detected_classic, data_gen.symbol_stream_matrix, channel_length)

        """
        Evaluate performance with linear MMSE
        """


        """
        Analyze SER performance
        """
        linear_mmse_performance.append(linear_mmse(data_gen.symbol_stream_matrix, data_gen.channel_output, data_gen.symbol_stream_matrix,channel.size))
        viterbi_net_performance.append(ser_nn)
        classic_performance.append(ser_classic)


    path = "Output/SER.pickle"
    pickle_out = open(path, "wb")
    pickle.dump([classic_performance, viterbi_net_performance], pickle_out)
    pickle_out.close()

    figure = plot_symbol_error_rates(SNRs_dB, [classic_performance,linear_mmse_performance, viterbi_net_performance], data_gen.get_info_for_plot())
    time_path = "Output/SER_"+f"{time.time()}"+"curves.png"

    figure.savefig(time_path, format="png")

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

    #Plots for channel
    plt.figure(3)
    plt.plot(channel)
    plt.title("channel", fontdict={'fontsize': 10})
    plt.ylabel("tap gain")
    path = f"Output/Channel{time.time()}.png"
    plt.savefig(path, format="png")


    assert True


def test_plt_ser():
    SNRs_dB = np.linspace(-5, 10, 100)
    snrs = np.power(10, SNRs_dB / 10)
    q_function = norm.sf
    SER = q_function(2 * np.sqrt(snrs))
    plt.plot(SNRs_dB, SER, label='analytic_ml')
    plt.xscale('linear')
    plt.yscale('log')
    plt.show()