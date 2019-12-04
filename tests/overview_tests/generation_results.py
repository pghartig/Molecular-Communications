import numpy as np
from communication_util.data_gen import *


def test_data_create():

    channel = np.zeros((1, 8))
    channel[0, [0, 3, 4, 5]] = 1, 0.5, 0.1, 0.2
    data_gen = training_data_generator(SNR=2, channel=channel, plot=True)
    # data_gen = training_data_generator(plot=True)
    data_gen.setup_channel()
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    data_gen.plot_setup(save=True)


def test_modulate():

    data_gen = training_data_generator(SNR=2,  plot=True, sampling_period=1, symbol_period= 12)
    data_gen.setup_channel()
    data_gen.random_symbol_stream()

    fundamental_pulse = rect_function_class     #Note the passed function here cannot be a lambda function
    parameters = 1/12
    data_gen.modulate_version3(fundamental_pulse, parameters)
    data_gen.sample_modulated_function()

    data_gen.plot_setup(save=True)