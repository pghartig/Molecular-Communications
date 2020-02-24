import pandas as pd
import numpy as np
from communication_util.data_gen import normalize_vector2
import matplotlib.pyplot as plt
"""
This is a set of tools to use for loading data from real transmissions. 
"""


def load_file(path):
    """
    Load in a sampled received data stream. This should be passed to some kind of filtering in order to produce
    detected symbols that can be used in an equalization scheme.
    :param path: Path from which to load csv file.
    :return:
    """
    raw = pd.read_csv(path, sep=",")
    numpy_raw = raw.as_matrix()
    time = numpy_raw[:, 0]
    susceptability = numpy_raw[:, 1]
    susceptability = susceptability
    return time, susceptability


def get_pulse(time_vec, measurement):
    """
    There are a couple of commented lines left for producing figures and debugging for loading different types of data.
    This algorithm is used to identify a fundamental pulse with which to subsequently match filter a signal and works
    as follows:
    1. Find all points in training data exceeding some threshold.
    2. Using this to find the length (time) between the training pulses.
    3. Use the length from previous step to sample impulse responses in the training data and place in matrix.
    4. Using the covariance matrix of the above training examples and find eigenvector for largest eigenvalue.
    5. Averaging the impulse responses yields a nearly identical model pulse.
    :param time_vec:
    :param measurement:
    :return:
    """
    truncate = 490
    for ind, time_point in enumerate(time_vec):
        if time_point > truncate:
            time = time_vec[:ind]
            break
    measurement = measurement[:time.size]
    # plt.plot(time,measurement)
    # plt.show()
    threshold = max(measurement)/2
    # get index of values exceeding threshold
    exceed_threshold = measurement > threshold
    # plt.plot(exceed_threshold)
    # plt.show()
    # now find average distance in samples between these clusters
    previous = False
    cur_length = 0
    lengths = []
    for index, value in enumerate(exceed_threshold):
        if value and not previous:
            lengths.append(cur_length)
            cur_length = 0
        else:
            cur_length += 1
        previous = value
    symbol_period_estimate = np.average(lengths)
    # symbol_period_estimate = np.median(lengths)
    impulse_responses = []
    training_data_size = 10
    for index, value in enumerate(exceed_threshold):
        if value and not previous and len(impulse_responses)<training_data_size:
            if exceed_threshold.size> index+symbol_period_estimate:
                impulse = measurement[index-30:index+int(symbol_period_estimate)]
                impulse_responses.append(impulse)
                # plt.plot(impulse)
        previous = value
    # plt.show()
    impulse_responses = np.vstack(impulse_responses)
    Rxx = impulse_responses.T@impulse_responses
    eigen_values, eigen_vectors = np.linalg.eigh(Rxx)
    max_eigen_vector = eigen_vectors[:,eigen_values.size-1]
    max_eigen_vector_normalized = normalize_vector2(max_eigen_vector)
    # plt.plot(max_eigen_vector_normalized, "g")
    ave_impulse_response = normalize_vector2(np.average(impulse_responses, 0).flatten())
    # plt.plot(ave_impulse_response,"r")
    # plt.show()
    return max_eigen_vector_normalized


def match_filter(measurements: np.ndarray, receive_filter: np.ndarray, symbol_period: int, number_symbols: int):
    """

    :param measurements: Data to be filtered
    :param receive_filter: Filter to apply
    :param symbol_period: Sampling period for detected symbols
    :param number_symbols: The number of samples to take from the detected, oversampled stream.
    :return:
    """
    # check = np.convolve(measurements, np.flip(receive_filter))
    # plt.plot(check,'r')
    # plt.title("filtered")
    # plt.show()
    detected_symbols = []
    for symbol_ind in range(number_symbols):
        incoming_samples = measurements[symbol_ind*symbol_period:symbol_ind*symbol_period + receive_filter.size]
        sample = receive_filter@incoming_samples
        detected_symbols.append(sample)
    detected_symbols = np.asarray(detected_symbols)
    return detected_symbols


def normalize_vector(vector):
    return (vector - np.average(vector))/np.std(vector)


def send_pulses(modulation_pulse: np.ndarray, symbols: np.ndarray, symbol_period: int):
    """
    IN WORK
    Will move to data gen eventually.
    :param modulation_pulse:
    :param symbols:
    :return:
    """
    #   Create resulting stream matrix in advance
    transmitted = np.zeros((modulation_pulse.size+(symbols.size-1)*symbol_period))
    for ind, symbol in enumerate(symbols):
        transmitted[ind*symbol_period:ind*symbol_period+modulation_pulse.size] += symbol*modulation_pulse
    return transmitted
