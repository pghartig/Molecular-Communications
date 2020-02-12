from communication_util.data_gen import *
from communication_util.Equalization.supervise_equalization import *

def test_linear_mmse_equalization():
    SNRs_dB = 15
    SNR = np.power(10, SNRs_dB / 10)
    number_symbols = 5000
    channel = np.zeros((1, 5))
    # channel[0, [0, 1, 2, 3, 4]] = 1, .1, .01, .1, .04
    channel[0, [0, 1, 2, 3, 4]] = 0.227, 0.460, 0.688, 0.460, 0.227
    data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SNR, plot=True, channel=channel)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    equalizer = linear_mmse()
    equalizer.train_equalizer(data_gen.symbol_stream_matrix, data_gen.channel_output, data_gen.symbol_stream_matrix, channel.size)
    del data_gen
    data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SNR, plot=True, channel=channel)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    equalizer.test_equalizer(data_gen.symbol_stream_matrix, data_gen.channel_output)

    # equalizer1 = linear_mmse.train_equalizer(data_gen.symbol_stream_matrix, data_gen.channel_output, data_gen.symbol_stream_matrix, channel.size-1)
    # equalizer2 = linear_mmse.train_equalizer(data_gen.symbol_stream_matrix, data_gen.channel_output, data_gen.symbol_stream_matrix, channel.size-2)