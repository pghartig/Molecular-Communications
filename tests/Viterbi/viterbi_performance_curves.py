from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.general_tools import *
from communication_util.basic_detectors import *
import time

def test_viterbi_performance_curve():
    error_tolerance = np.power(10.0, -3)
    threshold_performance = []
    classic_performance = []
    SNRs = np.linspace(0.1, 4, 10)
    seed_generator = 0
    data_gen=None
    for SRN in SNRs:
        number_symbols = 1000
        channel = np.zeros((1, 8))
        channel[0, [0, 3, 4, 5]] = 1, 0.5, 0.1, 0.2
        # channel = np.ones((1, 1))
        data_gen = training_data_generator(
            symbol_stream_shape=(1, number_symbols), SNR=SRN, channel=channel, plot=True, sampling_period=1, symbol_period=10
        )
        # data_gen = training_data_generator(plot=True)

        data_gen.setup_channel(shape=None)
        data_gen.random_symbol_stream()
        data_gen.send_through_channel()

        # detect with Viterbi
        # notice that this assumes perfect CSI at receiver
        metric = gaussian_channel_metric_working(channel, data_gen.channel_output)
        detected = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                            metric.metric)
        threshold_detected = threshold_detector(data_gen.alphabet, data_gen.channel_output)
        threshold_ser = symbol_error_rate(threshold_detected, data_gen.symbol_stream_matrix)
        ser = symbol_error_rate(detected, data_gen.symbol_stream_matrix)
        classic_performance.append(ser)
        threshold_performance.append(threshold_ser)

    plt.figure(1)
    plt.plot(SNRs, threshold_performance, label='basic threshold')
    plt.plot(SNRs, classic_performance, label='standard viterbi')
    plt.title(str(data_gen.get_info_for_plot()),fontdict={'fontsize':10} )
    plt.title("SER vs SNR Curve")
    plt.xlabel("SNR")
    plt.ylabel("SER")
    plt.legend(loc='upper left')
    time_path = "Output/SER_" + str(time.time())+"curves.png"
    plt.savefig(time_path, format="png")