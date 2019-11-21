import numpy as np
from communication_util import training_data_generator
from communication_util import pulse_shapes
from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.error_rates import *
import matplotlib.pyplot as plt

def test_pulse_with_viterbi():
    """

    :return:
    """
    """
    Generate symbol stream
    """
    number_symbols = 20
    channel = np.zeros((1, 8))
    channel[0, [0, 3, 4, 5]] = 1, 0.5, 0.1, 0.2
    # TODO consolidate this part
    data_gen = training_data_generator(
        symbol_stream_shape=(1, number_symbols), SNR=2, channel=channel, plot=True, sampling_period=1, symbol_period= 11
    )
    data_gen.random_symbol_stream()

    """
    Modulate symbols onto fundamental pulse
    """
    fundamental_pulse = pulse_shapes.rect_function_class(1/10)
    data_gen.modulate_fundamental_pulse(fundamental_pulse)

    """
    Setup Channel function
    """
    # channel = pulse_shapes.rect_function_class(1/10)
    # data_gen.setup_real_channel(channel, 5)
    channel_real = pulse_shapes.dirac_channel()
    data_gen.setup_real_channel(channel_real, 1)
    corresponding_channel = np.zeros((1, 8))
    channel[0, [0, 3, 4, 5]] = 1, 0.5, 0.1, 0.2


    """
     Send modulated signal through channel
     """
    data_gen.send_through_channel()
    data_gen.transmit_modulated_signal()

    """
    Setup Receive Filter
    """
    receive_filter = pulse_shapes.rect_function_class(1/10)
    data_gen.setup_receive_filter(receive_filter)
    data_gen.filter_received_modulated_signal()
    # data_gen.plot_setup()

    """
    Viterbi Performance with demodulated symbols from sampled transmission
    """
    metric = gaussian_channel_metric_working(channel, data_gen.demodulated_symbols)
    detected = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.demodulated_symbols, data_gen.CIR_matrix.shape[1],
                                        metric.metric)
    ser_sampled_symbols = symbol_error_rate(detected, data_gen.symbol_stream_matrix)

    """
    Viterbi Performance with demodulated symbols from sampled transmission
    """

    metric = gaussian_channel_metric_working(channel, data_gen.channel_output)
    detected = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                        metric.metric)
    ser_symbols = symbol_error_rate(detected, data_gen.symbol_stream_matrix)

    assert True

