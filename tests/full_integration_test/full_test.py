"""
Full Test for using the mixture model and neural network to provide metrics to be used in the viterbi algorithm
"""

import torch.nn as nn
from mixture_model.em_algorithm import mixture_model
import pickle
from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.error_rates import *

def test_full_integration():

    error_tolerance = np.power(10.0, -3)

    """
       Generated Testing Data using the same channel as was used for training the mixture model and the nn
       """
    number_symbols = 5000

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
    Load in Trained Neural Network and verify that it is acceptable performance
    """
    #Load NN
    saved_network_path = "nn.pt"
    neural_net = torch.load(saved_network_path)
    # Test NN
    x, y = data_gen.get_labeled_data()
    y = np.argmax(y, axis=1)  # Fix for how the pytorch Cross Entropy expects class labels to be shown
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    cost = criterion(neural_net(x), y.long())
    threshold = 1e-2
    assert cost < threshold

    """
    Load Trained Mixture Model
    """
    mm_pickle_in = open("tests/Mixture_Model/Output/mm.pickle", "rb")
    model = pickle.load(mm_pickle_in)
    mm = mixture_model(mu=model[0], sigma_square=model[1], alpha=model[2])
    mm = mm.get_probability
    mm_pickle_in.close()


    """
    After sending through channel, symbol detection should be performed using something like a matched filter
    """


    metric = nn_mm_metric(neural_net, mm, data_gen.channel_output)  # This is a function to be used in the viterbi
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
    print("viterbi SER" + str(ser_classic))
    print("viterbiNet" + str(ser_nn))

    assert error_tolerance >= ser_nn
