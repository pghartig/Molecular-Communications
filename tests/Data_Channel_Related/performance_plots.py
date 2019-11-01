from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.error_rates import *
import matplotlib.pyplot as plt


def test_performance_plot():
    """
    TODO add logs here for plotting in dB
    :return:
    """
    performance = []
    SNRs = np.linspace(1, 2, 10)
    seed_generator = 0
    for SRN in SNRs:
        channel = np.zeros((1, 8))
        channel[0, [0, 3, 4, 5]] = 1, 0.5, 0.1, 0.2
        data_gen = training_data_generator(
            seed=seed_generator, SNR=SRN, channel=channel, plot=False
        )
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
    plt.plot(SNRs, performance)
    plt.savefig(
        "/Users/peterhartig/Documents/Projects/moco_project/molecular-communications-project/Output/SER.png",
        format="png",
    )
    plt.show()
