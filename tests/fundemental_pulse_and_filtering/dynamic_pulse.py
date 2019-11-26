from communication_util.pulse_shapes import *



import numpy as np
from communication_util import training_data_generator
from communication_util import pulse_shapes
from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.error_rates import *
import matplotlib.pyplot as plt

def test_dynamic_pulse():
    """

    :return:
    """
    """
    Generate symbol stream
    """
    tolerance = 1e-3

    number_symbols = 10
    channel = np.zeros((1, 2))
    channel[0, [0, 1]] = 1, 1
    data_gen = training_data_generator(
        symbol_stream_shape=(1, number_symbols), SNR=2, plot=True, sampling_period=1, symbol_period= 11
    )
    data_gen.random_symbol_stream()
    channel_length = 2
    channel_real = pulse_shapes.rect_receiver_class(1/10) #channel length 1 for dirac
    data_gen.setup_real_channel(channel_real, channel_length)

    """
    Modulate symbols onto fundamental pulse
    """
    fundamental_pulse = dynamic_pulse     #Note the passed function here cannot be a lambda function
    parameters = 1/10
    data_gen.modulate_version3(fundamental_pulse, parameters)

    """
     Send modulated signal through channel
     """
    data_gen.transmit_modulated_signal2()
    data_gen.sample_modulated_function(500)
    data_gen.plot_setup()


    """
    Setup Receive Filter
    """
    receive_filter = pulse_shapes.rect_receiver_class(1 / 10)
    data_gen.setup_receive_filter(receive_filter)
    data_gen.filter_received_modulated_signal()

    """
    Viterbi Performance with demodulated symbols from sampled transmission
    """
    metric = gaussian_channel_metric_working(channel, data_gen.demodulated_symbols)
    detected = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.demodulated_symbols, channel_length,
                                        metric.metric)
    ser_sampled_symbols = symbol_error_rate(detected, data_gen.symbol_stream_matrix)

    assert ser_sampled_symbols < tolerance

