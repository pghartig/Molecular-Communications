from nn_utilities import models
from communication_util import training_data_generator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



def test_viterbi_net_class():

    """
    Testing the setup and training of a neural network using the viterbiNet Architecture
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
    number_symbols = 60

    channel = 1j * np.zeros((1, 5))
    channel[0, [0, 3, 4]] = 1, 0.5, 0.4
    data_gen = training_data_generator(
        symbol_stream_shape=(1, number_symbols + 2 * channel.size),
        SNR=10,
        plot=True,
        channel=channel,
    )
    data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()

    """
    After sending through channel, symbol detection should be performed using something like a matched filter
    """

    x, y = data_gen.get_labeled_data()
    y = np.argmax(y, axis=1)  # Fix for how the pytorch Cross Entropy expects class labels to be shown
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    """
    Setup NN and optimizer
    """
    m = data_gen.alphabet.size
    channel_length = data_gen.CIR_matrix.shape[1]

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H1, H2, D_out = number_symbols, 1, 100, 50, np.power(m, channel_length)


    net = models.viterbiNet(D_in, H1, H2, D_out)
    optimizer = optim.SGD(net.parameters(), lr=1e-6)

    """
    Train NN
    """
    criterion = nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss(reduction='sum')

    for t in range(50):
        output = net(x)
        loss = criterion(output, y.long())
        net.zero_grad()
        print(loss)
        loss.backward()
        optimizer.step()

    path = '/Users/peterhartig/Documents/Projects/moco_project/molecular-communications-project/Output/weights.pt'
    torch.save(net, path)

    assert False