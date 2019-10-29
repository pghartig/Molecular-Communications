import numpy as np
from communication_util import training_data_generator


def test_fundamental_pulse():
    channel = np.zeros((1, 8))
    channel[0, [0, 3, 4, 5]] = 1, 0.5, 0.1, 0.2
    # TODO make consolidate this part
    data_gen = training_data_generator(SNR=2, channel=channel, plot=True)
    # data_gen = training_data_generator(plot=True)
    data_gen.setup_channel()
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    data_gen.plot_setup()