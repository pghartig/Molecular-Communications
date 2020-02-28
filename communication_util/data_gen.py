import numpy as np
from scipy.cluster.vq import kmeans, vq
import math
import matplotlib.pyplot as plt
import logging as log
from communication_util.general_tools import get_combinatoric_list
from communication_util.pulse_shapes import  *
from communication_util.general_tools import quantizer as quantizer


def constellation_selection(constellation_type, size):
    """
    Allows for selecting the transmit symbol constellation.
    :param constellation_type:
    :param size:
    :return:
    """
    if constellation_type is "QAM":
        points = np.linspace(-1, 1, np.floor(math.log(size, 2)))
        constellation = 1j * np.zeros((points.size, points.size))
        for i in range(points.size):
            for k in range(points.size):
                constellation[i, k] = points[i] + 1j * points[k]
        return constellation.flatten()
    elif constellation_type is "ASK":
        return np.linspace(-1, +1, size)
    elif constellation_type is "onOffKey":
        return np.array((0, 1))
    elif constellation_type is "PSK":
        return np.exp(1j * np.linspace(0, 2 * np.pi, size))
    else:
        return np.array([1, -1])


def get_probability_vector(input, states):
    """
    For LTI channel. The probability of the given states corresponds to a one-hot vector.
    :return:
    """
    metrics = []
    for state in states:
        if np.array_equal(input, state):
            metrics.append(1)
        else:
            metrics.append(0)
    return np.asarray(metrics)


def visualize(data, c, name=None):
    for channels in range(data.shape[0]):
        plt.title(str(name), fontdict={'fontsize': 10})
        plt.stem(data[channels, :], linefmt=c)


def normalize_vector2(vector):
    return vector/np.linalg.norm(vector)


class CommunicationDataGenerator:
    def __init__(
        self,
        SNR,
        symbol_stream_shape=(1, 100),
        channel=None,
        channel_shape=None,
        plot=False,
        constellation="ASK",
        constellation_size=2,
        noise_parameter=np.array((0.0, 1.0)),
        seed=None,
        sampling_period = 1 / 10,
        symbol_period = 1 / 1
    ):

        """
        Basic parameters of the data generations
        """
        self.symbol_stream_matrix = np.zeros(symbol_stream_shape)
        # self.CIR_matrix = channel
        self.setup_channel(channel)
        self.channel_shape = channel_shape
        self.zero_pad = False
        self.terminated = False
        self.plot = plot
        self.channel_output = []
        self.alphabet = constellation_selection(constellation, constellation_size)
        self.seed = seed

        """
        Parameters and Variables Related to the communication channel
        """
        self.SNR = SNR
        self.noise_parameter = noise_parameter

        """
        Modulation and pulse related parameters and variables
        """
        self.modulated_CIR_matrix = None
        self.transmit_signal_matrix = []
        self.modulated_signal_function = []
        self.modulated_signal_function_sampled = []
        self.modulated_convolved_signal_function_sampled = []
        self.modulated_channel_output = []
        self.receive_filter = None
        self.sampling_period = sampling_period
        self.symbol_period = symbol_period
        self.samples_per_symbol_period = int(np.floor(self.symbol_period / self.sampling_period))

        """
        Decoding metrics
        """
        self.metrics = None

    def setup_channel(self, channel):
        """
        Setup for LTI channels. Note that this will normalize the input to ensure that power is not added
        via the channel.
        :param channel: The LTI channel to use in transmitting the SYMBOLS.
        """
        if channel is not None:
            self.CIR_matrix = np.zeros((channel.shape))
            self.channel_shape = channel.shape
            for ind in range(channel.shape[0]):
                # self.CIR_matrix[ind, :] = channel
                self.CIR_matrix[ind, :] = channel/np.linalg.norm(channel)
        else:
            self.CIR_matrix = np.ones((1, 1))
            self.channel_shape = self.CIR_matrix.shape

    def add_channel_uncertainty(self):
        """
        Used to added channel uncertainty to analyze performance for imperfect channel knownledge of LTI channels.
        """
        self.CIR_matrix += self.noise_parameter[1] * np.random.randn(self.CIR_matrix[0], self.CIR_matrix[1])

    def setup_real_channel(self, function: sampled_function, symbol_length):
        """
        IN WORK
        This has not been used for real tests but it should allow for modulated symbols onto a provided function
        that will in theory be filtered at the receiver to obtain received symbols.
        :param function: The function for the fundamental pulse and a number of transmission symbols over which the
        function should be extended.
        :param symbol_length: Describes the number of symbols over which the channel should be extended.
        :return: a  channel according to the provided function and the symbol mixing length
        *Note that the provided function should be scaled appropriate to the current sampling period.
        """
        samples_per_symbol_period = int(np.floor(self.symbol_period / self.sampling_period))
        self.samples_per_symbol_period = samples_per_symbol_period
        num_samples = symbol_length*samples_per_symbol_period
        self.modulated_CIR_matrix = function.return_samples(num_samples, self.sampling_period)

    def setup_receive_filter(self, receive_filter: sampled_function):
        """
        A function with which to filter an oversampled receive signal with a filter in order to find detected symbols.
        In the ideal case, this will be the matched filter to the modulated fundemental pulse.
        :param receive_filter:
        :return:
        """
        samples_per_symbol_period = int(np.floor(self.symbol_period / self.sampling_period))
        self.receive_filter = \
            receive_filter.return_samples(samples_per_symbol_period, self.sampling_period, start_index=0)

    def random_symbol_stream(self, predefined_input=None):
        """
        Based on the selected constellation, this function generates a stream of symbols of the pre-selected size.
        Note a uniform distribution is used to selected the transmitted symbols.
        :param predefined_input: A pre-selected symbol stream
        """
        if predefined_input is not None:
            self.symbol_stream_matrix = np.reshape(predefined_input, (self.symbol_stream_matrix.shape[0], predefined_input.size))
            return
        shape = self.symbol_stream_matrix.shape
        if self.seed is not None:
            np.random.seed(self.seed)
        if (self.zero_pad is True and self.terminated is True and self.CIR_matrix is not None):
            shape = self.symbol_stream_matrix.shape
            self.symbol_stream_matrix = np.random.random_integers(0, self.alphabet.size - 1, shape)
            self.symbol_stream_matrix = self.alphabet[self.symbol_stream_matrix]
            self.symbol_stream_matrix[:, 1 - self.CIR_matrix.shape[1]:] = self.alphabet[0]
            self.symbol_stream_matrix[:, 0: self.CIR_matrix.shape[1] - 1] = self.alphabet[0]
        elif (self.zero_pad is True and self.terminated is not True and self.CIR_matrix is not None):
            self.symbol_stream_matrix = np.random.random_integers(0, self.alphabet.size - 1, shape)
            self.symbol_stream_matrix = self.alphabet[self.symbol_stream_matrix]
            self.symbol_stream_matrix[:, 1 - self.CIR_matrix.shape[1]:] = self.alphabet[0]
        else:
            self.symbol_stream_matrix = np.random.random_integers(0, self.alphabet.size - 1, shape)
            self.symbol_stream_matrix = self.alphabet[self.symbol_stream_matrix]

    def modulate_sampled_pulse(self, modulation_pulse: np.ndarray, symbol_period: int):
        """
        The function allows for a generated symbol stream to be modulated onto a provided pulse. The pulse in this case
        is provided as a vector of samples from the pulse
        :param modulation_pulse:
        :param symbol_period:
        :return:
        """
        self.noise_parameter[1] = np.sqrt(np.var(self.alphabet) * (1 / self.SNR))
        for stream in range(self.symbol_stream_matrix.shape[0]):
            symbol_stream = self.symbol_stream_matrix[stream,:]
            transmitted = np.zeros((modulation_pulse.size + (symbol_stream.size - 1) * symbol_period))
            for ind, symbol in enumerate(symbol_stream):
                transmitted[ind * symbol_period:ind * symbol_period + modulation_pulse.size] += \
                    symbol * modulation_pulse
            transmitted += self.noise_parameter[0] + self.noise_parameter[1] * np.random.standard_normal(
                transmitted.shape)
            self.transmit_signal_matrix.append(transmitted)

    def provide_transmitted_matrix(self, provided_received):
        """
        In the case of data for which a symbols have already be modulated and oversampled, use this to create
        a transmit signal that can then be sent through a channel and detected at a receiver.
        :param provided_received:
        :return:
        """
        self.transmit_signal_matrix.append(provided_received)

    def filter_sample_modulated_pulse(self, receive_filter: np.ndarray, symbol_period: int,  quantization_level=None):
        """
        The complement to modulate_sampled_pulse or provide_transmitted_matrix, this function will provide a sampling
        of a modulated, discrete signal. Note that the next step after this will typically be to perform some sort of
        equalization of the detected symbols.
        :param receive_filter: The filter with which to correlate the channel output
        :param symbol_period: The frequency with which samples of the correlation will be detected.
        :param quantization_level: Add a quantization to the detected symbols
        :return:
        """
        if self.transmit_signal_matrix is not None:
            sampled_received_streams = []
            for stream in self.transmit_signal_matrix:
                number_symbols = self.symbol_stream_matrix.shape[1]
                detected_symbols = []
                for symbol_ind in range(number_symbols):
                    incoming_samples = stream[symbol_ind * symbol_period:
                                              symbol_ind * symbol_period + receive_filter.size]
                    incoming_samples = normalize_vector2(incoming_samples)
                    sample = receive_filter[:incoming_samples.size] @ incoming_samples
                    detected_symbols.append(sample)
                sampled_received_streams.append(np.asarray(detected_symbols))
            self.channel_output = np.reshape(np.asarray(detected_symbols), self.symbol_stream_matrix.shape)
            if quantization_level is not None:
                #   TODO allow for selecting the quantizer of the system.
                self.channel_output = quantizer(self.channel_output, quantization_level)

    def modulate_fundamental_pulse_continous(self, fundamental_pulse: sampled_function):
        """
        IN WORK
        The goal of this function was to allow for the sampling to be postponed until necessary. The current version
        still needs work as generating a reasonable length stream of symbols takes too much memory.
        :return:
        """
        """
        First look at pulse and determine where to cut off
        will just keep making samples until some percentage of max is reached.
        Notice that this assumes a symmetric pulse shape
        """
        sample_number = 0
        peak_energy = energy = fundamental_pulse.evaluate(sample_number * self.sampling_period)
        # TODO verify this threshold for where to cutoff fundamental pulse
        #  TODO (asymetric case)
        while energy >= 0.05 * peak_energy:
            sample_number += 1
            energy = fundamental_pulse.evaluate((sample_number) * self.sampling_period)
        sample_number -=1
        num_samples = 2*sample_number+1
        start_index = - sample_number
        self.start_index = start_index
        pulse_sample_vector = fundamental_pulse.return_samples(num_samples, self.sampling_period, start_index)
        sampling_width = int(np.floor(pulse_sample_vector.size / 2))  #TODO change for asymetric case

        """
        In order to allow for adding components from multiple symbols into a single sample, the array for the
        sampled, modulated signal must be pre-allocated.
        """
        samples_per_symbol_period = int(np.floor(self.symbol_period / self.sampling_period))
        overlap = max(int(sampling_width - samples_per_symbol_period / 2), 0)  #TODO change for asymetric case
        #TODO add symbol dependent modulation pulse
        self.transmit_signal_matrix = np.zeros((1, samples_per_symbol_period * self.symbol_stream_matrix.shape[1]
                + 2 * overlap))
        try:
            for symbol_ind in range(self.symbol_stream_matrix.shape[1]):
                center = symbol_ind * samples_per_symbol_period + sampling_width
                ind1= center - sample_number
                ind2= center + sample_number+1
                self.transmit_signal_matrix[:, ind1 : ind2] = \
                    (pulse_sample_vector * self.symbol_stream_matrix[:, symbol_ind])/samples_per_symbol_period
        except:
            log.log("problem")

    def modulate_version3(self, modulation_function: sampled_function(), parameters):
        """
        IN WORK
        This is the beginning of the work to allow for a dynamic modulation pulse.
        :param modulation_function:
        :return:
        """
        for stream_ind in range(self.symbol_stream_matrix.shape[0]):
            stream = list(self.symbol_stream_matrix[stream_ind, :])
            self.modulated_signal_function.append(self._modulate_stream_on_function(stream, modulation_function, parameters))

    def sample_modulated_function(self):
        """
        IN WORK
        The complementing function to modulate_fundamental_pulse, this function is what will actually sample the
        previously continuous valued transmit signal.
        :return:
        """
        num_samples = self.samples_per_symbol_period*self.symbol_stream_matrix.shape[1]
        self.modulated_signal_function_sampled = self._sample_function(num_samples, self.modulated_signal_function, noise=False)

    def send_through_channel(self, quantization_level=None, plot=False):
        """
        This function simulates passing the transmitted signal through an LTI channel.
        IMPORTANT: The output of this function is in the sample order as input. This is key for proper detection.
        Note that given the numpy convolution default, the impulse response should be provided with the longest delay
        tap on the left most index.
        :param quantization_level: If the channel includes quantization apply this level of the quantizer
        :param plot: Allow for plotting for generating figures to visualize channel output
        :return:
        """
        self.channel_output = []
        #   Apply LTI Channel
        for bit_streams in range(self.symbol_stream_matrix.shape[0]):
            # Note that Numpy's Convolve automatically flips second argument to the function.
            self.channel_output.append(np.convolve(self.symbol_stream_matrix[bit_streams,:], np.flip(self.CIR_matrix[bit_streams,:]), mode="full"))
        # self.channel_output = np.flip(np.asarray(self.channel_output))
        self.channel_output = np.asarray(self.channel_output)


        if plot==True:
            plt.figure()
            fig_main = plt.figure()
            no_noise = fig_main.add_subplot(1, 3, 1)
            no_noise.set_title("Channel Output")
            no_noise.scatter(self.channel_output, self.channel_output)



        #   Quantize before adding noise to ensure noise profile is not changed
        if quantization_level is not None:
            self.channel_output = quantizer(self.channel_output, quantization_level)
            quantized_input = np.linspace(-4, 4, num=200)
            # plt.figure()
            # plt.plot(quantized_input, quantizer(quantized_input, quantization_level), 'r', label='Quantization Function')
            # plt.scatter(self.channel_output, self.channel_output, label="Quantized Output")
            # plt.axvline(c='grey', lw=1)
            # plt.axhline(c='grey', lw=1)
            # plt.legend(loc='upper left')
            # plt.show()
        if plot==True:
            quantized = fig_main.add_subplot(1, 3, 2)
            quantized.set_title("Quantized")
            quantized.scatter(self.channel_output, self.channel_output)

        #   Add AWGN with pre-selected noise power
        self.noise_parameter[1] = np.sqrt(np.var(self.alphabet) * (1 / (self.SNR*2)))
        self.channel_output += self.noise_parameter[0] + self.noise_parameter[1]*np.random.standard_normal(self.channel_output.shape)
        if plot==True:
            noised = fig_main.add_subplot(1, 3, 3)
            noised.set_title("Noise Added")
            noised.scatter(self.channel_output, self.channel_output)
            plt.show()

    def transmit_modulated_signal2(self):
        """
        IN WORK
        A companion function to modulate_fundamental_pulse_continous.
        :return:
        """
        for bit_streams in range(self.symbol_stream_matrix.shape[0]):
            self.modulated_signal_function[bit_streams].virtual_convole_functions(self.modulated_CIR_matrix)

    def convolve_sampled_modulated(self):
        """
        IN WORK
        A companion function to modulate_fundamental_pulse_continous.
        """
        for bit_streams in range(self.symbol_stream_matrix.shape[0]):
            self.modulated_convolved_signal_function_sampled.append(
                np.convolve(np.flip(self.modulated_signal_function_sampled[bit_streams,:]), self.CIR_matrix[bit_streams,:], mode="full"))
        self.modulated_convolved_signal_function_sampled = np.asarray(self.modulated_convolved_signal_function_sampled)
        self.noise_parameter[1] *= np.sqrt(np.var(self.alphabet) * (1 / self.SNR))
        self.modulated_convolved_signal_function_sampled += \
            self.noise_parameter[0] + self.noise_parameter[1] * \
            np.random.randn(self.modulated_convolved_signal_function_sampled.shape[0],
                            self.modulated_convolved_signal_function_sampled.shape[1])

    def filter_received_modulated_signal(self):
        """
        Use the pre-established filter to detect received symbols (e.g. matched filtering).
        The output of this function is in the sample order as the original transmit symbol stream so if
        a memoryless channel is used, the detected symbols should be in the same order.
        :return:
        """
        if self.modulated_channel_output is not None:
            samples_per_symbol_period = int(np.floor(self.symbol_period / self.sampling_period))
            offset = int(np.ceil(self.symbol_period / 2))
            #   Sample the received, modulated signal every samples_per_symbol_period samples
            for stream_ind in range(self.symbol_stream_matrix.shape[0]):
                stream = []
                for ind in range(self.symbol_stream_matrix.shape[1]):
                    ind1 = offset+samples_per_symbol_period*ind - offset
                    ind2 = offset+int(samples_per_symbol_period/2) + samples_per_symbol_period * ind
                    samples_filtered = self.modulated_convolved_signal_function_sampled[stream_ind, ind1:ind2]
                    stream.append(np.dot(self.receive_filter, samples_filtered))
                self.demodulated_symbols[stream_ind, :] = np.asarray(stream)

    def gaussian_channel_metric_sampled(self, modulation_function, parameters, states):
        """
        IN WORK
        More work for the continuous valued function concept.
        :param modulation_function:
        :param parameters:
        :param states:
        :return:
        """
        self.metrics = []
        for state in states:
            # For each state create the modulated sym
            stream = list(state)
            # returns a function for the modulated version fo the state
            modulated_state = self._modulate_stream_on_function(stream, modulation_function, parameters)
            # imitate this state running through the known channel
            modulated_state.virtual_convole_functions(self.modulated_CIR_matrix)
            # returns samples of above function
            sampled = self._sample_function(self.modulated_CIR_matrix.size, modulated_state)
            predicted = sampled*self.modulated_CIR_matrix
            self.metrics.append(predicted)

    def metric_cost_sampled(self, index, states):
        """
        IN WORK
        More work for the continuous valued function concept.
        :param index:
        :return:
        """
        costs = []
        sampled_index = self.samples_per_symbol_period*index + self.modulated_CIR_matrix.size
        received = self.modulated_signal_function_sampled[0,sampled_index - self.modulated_CIR_matrix.size:sampled_index]
        for ind, state in enumerate(states):
            """
            Note that the line below is sensitive to ordering but is like this to prevent recreating the metrics for
            each state transition.
            """
            costs.append(np.linalg.norm(received - self.metrics[ind]))
        return np.asarray(costs)

    def get_labeled_data(self):
        """
        This function returns the labeled training data for learning the distribution for a channel which has memory.
        Currently, the memory of the channel is defined by the length of the LTI channel impulse response but this can
        be easily defined in other ways.
        :return: x_list is a list of the received signals with corresponding state classification contained in y_list.
        """
        #   Generate states used below based on known memory length
        x_list = []
        y_list = []
        states = []
        item = []
        get_combinatoric_list(self.alphabet, self.CIR_matrix.shape[1], states, item)
        states = np.asarray(states)
        if self.channel_output is not None:
            j = 0
            for i in range(self.channel_output.shape[1]):
                if (self.CIR_matrix.shape[1]-1 <= i
                        < self.symbol_stream_matrix.shape[1] - self.CIR_matrix.shape[1] + 1):
                    state = self.symbol_stream_matrix[:, j: j+self.CIR_matrix.shape[1]].flatten()
                    probability_vec = get_probability_vector(state, states)
                    x_list.append(self.channel_output[:, i].flatten())
                    y_list.append(probability_vec)
                    j += 1
        return x_list, y_list

    def get_labeled_data_reduced_state(self, outputs):
        """
        This function returns the labeled training data for learning the distribution for a channel which has memory.
        This is similar to get_labeled_data except that the number of potential channel states is reduced using
        an unsupervised k-means clustering. This clustering could be done in other ways without changing the overall
        idea of this function.
        :param outputs:
        :return:
        """
        x_list = []
        y_list = []
        base_states = int(np.ceil(np.log2(outputs)))
        states = []
        item = []
        get_combinatoric_list(self.alphabet, self.CIR_matrix.shape[1], states, item)
        states = np.asarray(states)
        #   Compress states using training data
        num_clusters = int(pow(2, base_states))
        labels, totals = self.reduced_state_mapping(num_clusters)
        states_reduced = []
        item_reduced = []
        get_combinatoric_list(self.alphabet, base_states, states_reduced, item_reduced)
        states_reduced = np.asarray(states_reduced)
        states_final = []
        item_final = []
        get_combinatoric_list(self.alphabet, base_states, states_final, item_final)
        states_reduced = np.asarray(states_reduced)
        if self.channel_output is not None:
            j = 0
            #   Go create training example from each channel output
            for i in range(self.channel_output.shape[1]):
                #   Take care of causality of creating training sets and unused output symbols
                if (i >= self.CIR_matrix.shape[1]-1 and i < self.symbol_stream_matrix.shape[1] - self.CIR_matrix.shape[1] + 1):
                    #   Get true state of the system
                    symbol = self.symbol_stream_matrix[:,j].flatten()
                    true_state = self.symbol_stream_matrix[:, j: j+self.CIR_matrix.shape[1]].flatten().flatten()
                    probability_vec = get_probability_vector(true_state, states)
                    #   Now find corresponding reduced state cluster number for the true state
                    state = labels[np.argmax(probability_vec)]
                    new_state = states_reduced[state]
                    reduced_state = new_state
                    # reduced_state = np.append(symbol, new_state)
                    probability_vec_reduced = get_probability_vector(reduced_state, states_final)
                    y_list.append(probability_vec_reduced)
                    #   y_list.append(probability_vec)
                    x_list.append(self.channel_output[:, i].flatten())
                    j += 1
        #   Please leave commented line below. It is a useful check for detected symbol frequency.
        # totals = np.sum(np.asarray(y_list), axis=0)
        return x_list, y_list, states_reduced, states, totals

    def reduced_state_mapping(self, num_clusters):
        """
        A function to provide a reduction of channel states.
        There is a lot of commented code below but this is useful for testing so leave in for now.
        :param num_clusters:
        :return: a vector indicating a compression of some number of states into a set of clusters
        """
        #   Take training data and cluster the output points
        x_list, y_list = self.get_labeled_data()
        clusters = kmeans(x_list, num_clusters)
        labels = vq(x_list, clusters[0])[0]
        #   Now go through each of the original number of states and assign this to
        # centroids = clusters[0]
        # plt.scatter(x_list, x_list)
        # plt.scatter(centroids, centroids)
        # plt.show()
        totals = np.zeros((num_clusters, y_list[0].size))
        for ind, label in enumerate(labels):
            for cluster in range(num_clusters):
                if label ==cluster:
                    totals[label, :] += y_list[ind]
                    break
        #   Now find the cluster into which each of the original states has the majority of labels in
        return np.argmax(totals, axis=0), totals

    def plot_setup(self, save=False):
        """
        A function to display the different stages of the communication process.
        :param save:
        :return:
        """
        num = 3
        figure = plt.figure()
        figure.add_subplot(num, 1,1)
        visualize(self.CIR_matrix, "C0-", name="CIR_matrix")
        figure.add_subplot(num, 1,2)
        visualize(self.symbol_stream_matrix, "C1-", name="symbol_stream_matrix")
        figure.add_subplot(num, 1,3)
        visualize(self.modulated_signal_function_sampled, "C4-", name="modulated_signal_function_sampled")

        plt.show()
        if save==True:
            time_path = "modulated_rect.png"
            figure.savefig(time_path, format="png")

    def _modulate_stream_on_function(self, stream, modulation_function: sampled_function(), parameters):
        """
        IN WORK
        A helper function for the continuous signal tranmission concept.
        :param stream:
        :param modulation_function:
        :param parameters:
        :return:
        """
        function = combined_function()
        for ind, symbol in enumerate(stream):
            to_add = modulation_function(parameters)
            center = ind*self.symbol_period + np.ceil(self.symbol_period/2)
            to_add.setup(center, symbol)
            function.add_function(to_add)
        return function

    def _sample_function(self, num_samples, function, noise = False):
        """
        IN WORK
        A helper function for sampling the continuous transmission concept.
        :param num_samples:
        :param function:
        :param noise:
        :return:
        """
        self.noise_parameter[1] *= np.sqrt(np.var(self.alphabet) * (1 / self.SNR))
        if noise == False:
            if type(function) == list:
                total_samples = []
                for function_ind in function:
                    samples = []
                    for sample_index in range(num_samples):
                        samples.append(function_ind.evaluate(sample_index*self.sampling_period))
                total_samples.append(np.asarray(samples))
                return np.asarray(total_samples)
            else:
                samples = []
                for sample_index in range(num_samples):
                    samples.append(function.evaluate(sample_index*self.sampling_period))
                return np.asarray(samples)
        if noise == True:
            if type(function) == list:
                total_samples = []
                for function_ind in function:
                    samples = []
                    for sample_index in range(num_samples):
                        samples.append(function_ind.evaluate(sample_index * self.sampling_period)+
                                       self.noise_parameter[0] + self.noise_parameter[1] * np.random.randn())
                total_samples.append(np.asarray(samples))
                return np.asarray(total_samples)
            else:
                samples = []
                for sample_index in range(num_samples):
                    samples.append(function.evaluate(sample_index * self.sampling_period) +
                                   self.noise_parameter[0] + self.noise_parameter[1] *  np.random.randn)
                return np.asarray(samples)

    def get_info_for_plot(self):
        """
        :return: Relevant data regarding the various paramters used in the communication.
        """
        information = []
        information.append("Alphabet Size: " + str(len(self.alphabet)))
        information.append("Channel Length (wrt symbols): " + str(self.CIR_matrix.shape[1]))
        information.append("# transmitted symbols: " + str(self.symbol_stream_matrix.shape[1]))
        return information