from communication_util.data_gen import *
from viterbi.viterbi import *
from communication_util.general_tools import *
from communication_util.basic_detectors import *
import time

def test_viterbi_performance_curve():
    error_tolerance = np.power(10.0, -3)
    threshold_performance = []
    classic_performance = []
    SNRs_dB = np.linspace(0, 10, 10)
    SNRs =  np.power(10, SNRs_dB/10)
    data_gen=None
    for SNR in SNRs:
        number_symbols = 2000
        channel = np.zeros((1, 3))
        channel[0, [0, 1, 2]] = 1, 0.5, 0.1
        channel = np.zeros((1, 1))
        channel[0, [0]] = 1
        data_gen = training_data_generator(
            symbol_stream_shape=(1, number_symbols), SNR=SNR, channel=channel, plot=True, sampling_period=1, symbol_period=10
        )
        channel_length = data_gen.CIR_matrix.shape[1]
        data_gen.setup_channel(shape=None)
        data_gen.random_symbol_stream()
        data_gen.send_through_channel()
        metric = gaussian_channel_metric_working(channel, data_gen.channel_output)  # This is a function to be used in the viterbi
        detected_classic = viterbi_setup_with_nodes(data_gen.alphabet, data_gen.channel_output, data_gen.CIR_matrix.shape[1],
                                            metric.metric)
        ser_classic = symbol_error_rate_channel_compensated(detected_classic, data_gen.symbol_stream_matrix, channel_length)
        classic_performance.append(ser_classic)


    figure = plot_symbol_error_rates(SNRs_dB,[classic_performance],data_gen.get_info_for_plot())
    time_path = "Output/SER_" + str(time.time())+"curves.png"
    figure.savefig(time_path, format="png")