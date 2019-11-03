from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.error_rates import *


def test_viterbi_gaussian():
    error_tolerance = np.power(10.0, -3)
    # setup data
c

    # detect with Viterbi
    # notice that this assumes perfect CSI at receiver
    detected = viterbi_output(
        data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix
    )
    # check WER
    ser = symbol_error_rate(detected, data_gen.symbol_stream_matrix)
    assert error_tolerance >= ser
