from communication_util.load_mc_data import *
import matplotlib.pyplot as plt

def test_load():
    train_path = 'mc_data/5_cm_train.csv'
    test_path = 'mc_data/5_cm_test.csv'
    # train_path = 'mc_data/20_cm_train.csv'
    # test_path = 'mc_data/20_cm_test.csv'
    test_input_sequence = 'mc_data/input_string.txt'
    test_input_sequence = np.loadtxt(test_input_sequence, delimiter=",")
    train_time, train_measurement = load_file(train_path)
    test_time, test_measurement = load_file(test_path)
    pulse_shape = get_pulse(train_time, train_measurement)
    symbol_period = 10
    # stream = np.random.randint(0, 2, (50))
    stream = np.array((1,1,1,0,0,1,0,1,1,0))
    transmit_signal = send_pulses(pulse_shape, stream, symbol_period)
    plt.plot(transmit_signal)
    plt.plot(transmit_signal)
    plt.show()
    symbols = match_filter(transmit_signal, pulse_shape, symbol_period, stream.size)
    impulse_response = impulse_response_from_oversamples(pulse_shape, symbol_period)
    #TODO now use this as input to the viterbi algo
    plt.figure(1)
    plt.plot(pulse_shape, label='Matched Filter')
    plt.title("matched filter", fontdict={'fontsize': 10})
    plt.legend(loc='upper right')
    path = f"Output/Matched_filter.png"
    plt.savefig(path, format="png")
