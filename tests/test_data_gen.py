import numpy as np
from communication_util.data_gen import *


def test_data_create():
    data_gen = training_data_generator(plot=True)
    data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    data_gen.plot_setup()
