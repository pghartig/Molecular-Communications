from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.error_rates import *
import matplotlib.pyplot as plt


def test_perforamcen_plot():
    performance = []
    SNRs = np.linspace(.1, 10, 20)
    for SRN in SNRs:
        # setup data
        channel = 1j*np.zeros((1, 4))
        channel[0, [0, 3]] = 1, 1j*0.5
        # TODO make consolidate this part
        data_gen = training_data_generator(SNR=SRN, constellation='QAM', constellation_size=4, channel=channel, plot=False)
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
        performance.append(ser)
    plt.figure
    plt.plot(SNRs,performance)
    plt.show()
    # plt.savefig("performance_curve")
