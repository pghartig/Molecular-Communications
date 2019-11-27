import numpy as np
from communication_util import training_data_generator
from communication_util import pulse_shapes
from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.general_tools import *
import matplotlib.pyplot as plt

def test_dynamic_pulse():
    """

    :return:
    """
    """
    Generate symbol stream
    """
    tolerance = 1e-3

    number_symbols = 1000
    channel = np.zeros((1, 2))
    channel[0, [0, 1]] = 1, 1
    channel = np.ones((1, 1))
    data_gen = training_data_generator(
        symbol_stream_shape=(1, number_symbols), SNR=4, plot=True, sampling_period=1, symbol_period= 10
    )
    data_gen.random_symbol_stream()
    channel_length = channel.shape[1]

    """
    Setup channel to convolve with
    """
    # channel_real = pulse_shapes.rect_receiver_class(1/10) #channel length 1 for dirac
    channel_real = pulse_shapes.dirac_channel()     #channel length 1 for dirac

    data_gen.setup_real_channel(channel_real, channel_length)

    """
    Modulate symbols onto fundamental pulse
    """
    fundamental_pulse = rect_function_class     #Note the passed function here cannot be a lambda function
    parameters = 1/10
    data_gen.modulate_version3(fundamental_pulse, parameters)
    # fundamental_pulse = dynamic_pulse     #Note the passed function here cannot be a lambda function
    # parameters = 1/10
    # data_gen.modulate_version3(fundamental_pulse, parameters)

    """
     Send modulated signal through channel
     """
    data_gen.transmit_modulated_signal2()
    data_gen.sample_modulated_function()

    """
    Viterbi Performance with demodulated symbols from sampled transmission
    Note that in this case, the receiver is not needed as the viterbi metrics are taken against the over-sampled version
    of the impulse response.
    """
    fundamental_pulse = dynamic_pulse     #Note the passed function here cannot be a lambda function
    parameters = 1/10
    states = []
    item = []
    get_combinatoric_list(data_gen.alphabet, channel_length, states, item)
    data_gen.gaussian_channel_metric_sampled(fundamental_pulse, parameters, states)
    metric = data_gen.metric_cost_sampled
    detected = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.demodulated_symbols, channel_length,
                                        metric)
    ser_sampled_symbols = symbol_error_rate_sampled(detected, data_gen.symbol_stream_matrix)

    assert ser_sampled_symbols < tolerance

