import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_file(path):
    #TODO remove nans from data
    raw = pd.read_csv(path, sep=",")
    numpy_raw = raw.as_matrix()
    time = numpy_raw[:, 0]
    susceptability = numpy_raw[:, 1]
    susceptability = normalize_vector(susceptability)
    return time, susceptability

def get_pulse(time_vec, measurement):
    """
    This algorithm is used to identify a fundamental pulse with which to subsequently match filter a signal and works
    as follows:
    1. Find all points in training data exceeding some threshold.
    2. Using this to find the length (time) between the training pulses.
    3. Use the length from previous step to sample impulse responses in the training data and place in matrix.
    4. Using the covariance matrix of the above training examples and find eigenvector for largest eigenvalue.
    5. Averaging the impulse responses yields a nearly identical model pulse.
    :param time:
    :param measurement:
    :return:
    """
    truncate = 490
    for ind ,time_point in enumerate(time_vec):
        if time_point > truncate:
            time = time_vec[:ind]
            break
    measurement = measurement[:time.size]
    plt.plot(time,measurement)
    plt.show()
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
    symbol_period_estimate = np.median(lengths)
    impulse_responses = []
    training_data_size = 10
    for index, value in enumerate(exceed_threshold):
        if value and not previous and len(impulse_responses)<training_data_size:
            if exceed_threshold.size> index+symbol_period_estimate:
                impulse = measurement[index-30:index+int(symbol_period_estimate)]
                impulse_responses.append(impulse)
                plt.plot(impulse)
        previous = value
    plt.show()
    impulse_responses = np.vstack(impulse_responses)
    Rxx = impulse_responses.T@impulse_responses
    eigen_values, eigen_vectors = np.linalg.eigh(Rxx)
    max_eigen_vector = - normalize_vector(eigen_vectors[:,eigen_values.size-1])
    plt.plot(max_eigen_vector,"g")
    ave_impulse_response = normalize_vector(np.average(impulse_responses, 0).flatten())
    plt.plot(ave_impulse_response,"r")
    plt.show()
    return ave_impulse_response

def match_filter(measurements, receive_filter):
    check = np.convolve(measurements,np.flip(receive_filter))
    plt.plot(check)
    plt.show()


def normalize_vector(vector):
    return (vector - np.average(vector))/np.std(vector)