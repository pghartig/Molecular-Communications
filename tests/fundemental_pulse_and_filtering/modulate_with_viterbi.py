import numpy as np
from communication_util import training_data_generator
from communication_util import pulse_shapes
import matplotlib.pyplot as plt

def test_pulse_with_viterbi():
    """

    :return:
    """

    number_symbols = 1000
    channel = np.zeros((1, 8))
    channel[0, [0, 3, 4, 5]] = 1, 0.5, 0.1, 0.2
    # TODO consolidate this part
    data_gen = training_data_generator(
        symbol_stream_shape=(1, number_symbols), SNR=1, channel=channel, plot=True
    )
    data_gen.random_symbol_stream()
    data_gen.modulate_fundamental_pulse(pulse_shapes.rectangle)
    # data_gen.modulate_fundemental_pulse(pulse_shapes.root_raise_cosine)

    """
    Setup Channel function
    """
    sample_period = 1 / 10
    filter = pulse_shapes.rect_function_class(1/5).return_samples
    data_gen.setup_real_channel(filter, 5)
    """
    Setup Receive Filter
    """
    sample_period = 1 / 10
    filter = pulse_shapes.rect_function_class(1/5).return_samples
    data_gen.setup_receive_filter(filter)
    """
     Send modulated signal through channel
     """
    data_gen.send_through_channel()
    data_gen.transmit_modulated_signal()

