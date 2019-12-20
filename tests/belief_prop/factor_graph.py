from belief_prop_utils.factor_graph import factor_graph
from communication_util import *

def test_graph_setup():
    number_symbols = 1000
    SNR = 2
    channel = np.zeros((1, 5))
    channel[0, [0, 1, 2, 3, 4]] = 1, .3, .1, .2, .4
    data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SNR, plot=True, channel=channel)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    graph = factor_graph(data_gen.channel_output.flatten())