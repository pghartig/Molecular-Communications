from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.general_tools import *

def test_viterbi_new_gaussian():
    error_tolerance = np.power(10.0, -3)
    # setup data
    channel = np.zeros((1, 8))
    channel[0, [0, 3, 4, 5]] = 1, 0.5, 0.1, 0.2

    # TODO make consolidate this part
    data_gen = training_data_generator(SNR=2, channel=channel, plot=True)
    data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()
    channel_length = data_gen.CIR_matrix.shape[1]

    # detect with Viterbi
    # notice that this assumes perfect CSI at receiver
    metric = GaussianChannelMetric(channel, data_gen.channel_output)
    detected = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                        metric.metric)
    ser = symbol_error_rate(detected, data_gen.symbol_stream_matrix)
    assert error_tolerance >= ser
