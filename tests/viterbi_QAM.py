from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.error_rates import *


def test_viterbi_QAM():
    error_tolerance = np.power(10.0, -3)
    # setup data
    channel = 1j*np.zeros((1, 5))
    channel[0, [0, 3, 4]] = 1, 1j*0.5, 0.4
    # TODO make consolidate this part
    data_gen = training_data_generator(SNR=10, constellation='QAM', constellation_size=4, channel=channel, plot=True)
    # data_gen = training_data_generator(plot=True)

    data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()

    # detect with Viterbi
    detected = viterbi_output(
        data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix
    )
    # check WER
    ser = symbol_error_rate(detected, data_gen.symbol_stream_matrix)
    assert error_tolerance >= ser
