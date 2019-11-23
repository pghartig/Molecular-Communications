import numpy as np
from communication_util import training_data_generator
from communication_util import pulse_shapes
import matplotlib.pyplot as plt


def test_fundamental_pulse():
    plt.figure(1)
    sample = np.linspace(-1, 1, num=500)
    plt.plot(sample, pulse_shapes.root_raise_cosine(sample, 1 / 100, 1 / 10, 1, 0.5))

    # plt.figure(1)
    # sample = np.linspace(-1, 1, num=1000)
    # plt.plot(sample, pulse_shapes.rectangle(sample, 1/100, 1/10, 1, .5))

    channel = np.zeros((1, 8))
    channel[0, [0, 3, 4, 5]] = 1, 0.5, 0.1, 0.2
    # TODO consolidate this part
    data_gen = training_data_generator(
        symbol_stream_shape=(1, 30), SNR=1, channel=channel, plot=True
    )
    # data_gen = training_data_generator(plot=True)
    # data_gen.setup_channel()
    data_gen.random_symbol_stream()
    data_gen.modulate_fundamental_pulse(pulse_shapes.rectangle)
    # data_gen.modulate_fundemental_pulse(pulse_shapes.root_raise_cosine)
    data_gen.send_through_channel()
    data_gen.plot_setup()