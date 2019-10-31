import torch
import numpy as np
from communication_util import training_data_generator


def test_nn_load_test():
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

    input, output = data_gen.get_labeled_data()
    test_set = torch.Tensor(input)
    test_results = torch.Tensor(output)


    SER_threshold = np.power(10.0, -3)
    path = '/Users/peterhartig/Documents/Projects/moco_project/molecular-communications-project/Output/weights.pt'
    w1, w2, w3 = torch.load(path)

    outputs = []

    i = 0
    for x in test_set:
        h1 = x.mm(w1)
        h1_relu = h1.clamp(min=0)

        h2 = h1_relu.mm(w2)
        h2_relu = h2.clamp(min=0)

        y_pred = h2_relu.mm(w3)
        outputs.append(np.power(y_pred-test_results[i]), 2)
        i+=1

    predictions = 0
    test_SER = 0
    assert test_SER <= SER_threshold