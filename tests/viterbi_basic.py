from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.error_rates import *


def test_viterbi_gaussian():
    error_tolerance = np.power(10.0, -5)
    # setup data
    # TODO make consolidate this part
    data_gen = training_data_generator(plot=True)
    data_gen.setup_channel(shape=None)
    data_gen.random_bit_stream()
    data_gen.send_through_channel()

    # detect with Viterbi
    detected = viterbi_output(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix)
    # check WER
    WER = symbol_error_rate(detected, data_gen.symbol_stream_matrix)
    assert error_tolerance >= WER
