from communication_util.data_gen import *
from communication_util.Equalization.supervise_equalization import *

def test_linear_mmse_equalization():
    SNR=5
    number_symbols = 1000
    channel = np.zeros((1, 5))
    # channel[0, [0, 1, 2, 3, 4]] = 1, .1, .01, .1, .04
    channel[0, [0, 1, 2, 3, 4]] = 1, .3, .1, .2, .4
    data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SNR, plot=True, channel=channel)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    equalizer = linear_mmse(data_gen.symbol_stream_matrix, data_gen.channel_output, data_gen.symbol_stream_matrix, channel.size)
    equalizer1 = linear_mmse(data_gen.symbol_stream_matrix, data_gen.channel_output, data_gen.symbol_stream_matrix, channel.size-1)
    equalizer2 = linear_mmse(data_gen.symbol_stream_matrix, data_gen.channel_output, data_gen.symbol_stream_matrix, channel.size-2)



    pass