import numpy as np
from scipy.cluster.vq import kmeans, vq
import math
import matplotlib.pyplot as plt
import logging as log
from communication_util.general_tools import get_combinatoric_list
from communication_util.pulse_shapes import  *
from communication_util.general_tools import quantizer as quantizer
from communication_util.load_mc_data import normalize_vector


class training_data_generator:
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
        self.alphabet = self.constellation(constellation, constellation_size)
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
        self.demodulated_symbols = np.zeros(symbol_stream_shape)
        self.sampling_period = sampling_period
        self.symbol_period = symbol_period
        self.samples_per_symbol_period = int(np.floor(self.symbol_period / self.sampling_period))

        """
        Decoding metrics
        """
        self.metrics = None

    def setup_channel(self, channel):
        """
        Setup channel matrix such that the channel is normalized
        :param shape:
        :return:
        """
        if channel is not None:
            self.CIR_matrix = np.zeros((channel.shape))
            self.channel_shape = channel.shape
            for ind in range(channel.shape[0]):
                self.CIR_matrix[ind, :] = channel/np.linalg.norm(channel)
            # plt.stem(np.arange(0, self.CIR_matrix.size), self.CIR_matrix.flatten())
            # plt.show()
        else:
            self.CIR_matrix = np.ones((1, 1))
            self.channel_shape = self.CIR_matrix.shape

    def add_channel_uncertainty(self):
        self.CIR_matrix += self.noise_parameter[1] * np.random.randn(self.CIR_matrix[0], self.CIR_matrix[1])

    def setup_real_channel(self, function: sampled_function, symbol_length):
        """
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
        return None

    def setup_receive_filter(self, filter: sampled_function):
        samples_per_symbol_period = int(np.floor(self.symbol_period / self.sampling_period))
        test = filter.return_samples(samples_per_symbol_period, self.sampling_period, start_index=0)
        self.receive_filter = \
            filter.return_samples(samples_per_symbol_period, self.sampling_period, start_index=0)

    def constellation(self, type, size):
        # TODO for large tests may want to select dtype
        if type is "QAM":
            points = np.linspace(-1, 1, np.floor(math.log(size, 2)))
            constellation = 1j * np.zeros((points.size, points.size))
            for i in range(points.size):
                for k in range(points.size):
                    constellation[i, k] = points[i] + 1j * points[k]
            return constellation.flatten()
        elif type is "ASK":
            return np.linspace(-1, +1, size)
        elif type is "PSK":
            return np.exp(1j * np.linspace(0, 2 * np.pi, size))
        else:
            return np.array([1, -1])

    def random_symbol_stream(self):
        """
        TODO allow for loading in a signal stream
        :return:
        """
        shape = self.symbol_stream_matrix.shape
        if self.seed is not None:
            np.random.seed(self.seed)

        if (
            self.zero_pad is True
            and self.terminated is True
            and self.CIR_matrix is not None
        ):
            # TODO fix zero padding
            shape = self.symbol_stream_matrix.shape
            test = shape[1]
            self.symbol_stream_matrix = np.random.random_integers(
                0, self.alphabet.size - 1, shape
            )
            self.symbol_stream_matrix = self.alphabet[self.symbol_stream_matrix]
            self.symbol_stream_matrix[
                :, 1 - self.CIR_matrix.shape[1] :
            ] = self.alphabet[0]
            self.symbol_stream_matrix[
                :, 0 : self.CIR_matrix.shape[1] - 1
            ] = self.alphabet[0]
        elif (
            self.zero_pad is True
            and self.terminated is not True
            and self.CIR_matrix is not None
        ):
            self.symbol_stream_matrix = np.random.random_integers(
                0, self.alphabet.size - 1, shape
            )
            self.symbol_stream_matrix = self.alphabet[self.symbol_stream_matrix]
            self.symbol_stream_matrix[
                :, 1 - self.CIR_matrix.shape[1] :
            ] = self.alphabet[0]
        else:
            self.symbol_stream_matrix = np.random.random_integers(
                0, self.alphabet.size - 1, shape
            )
            self.symbol_stream_matrix = self.alphabet[self.symbol_stream_matrix]

    def modulate_fundamental_pulse(self, fundamental_pulse: sampled_function):
        """
        The purpose of this funcion is to take a symbol stream and use it to modulate the fundamental pulse
        on which it will be send over the channel.
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

    def modulate_version2(self, modulation_function: sampled_function):
        for stream_ind in range(self.symbol_stream_matrix.shape[0]):
            stream = list(self.symbol_stream_matrix[stream_ind, :])
            self.modulated_signal_function.append(lambda:
                                                  sum([modulation_function(- ind*self.symbol_period, symbol)
                                                       for ind, symbol in enumerate(stream)]))
        print('works')

    def modulate_version3(self, modulation_function: sampled_function(), parameters):
        """
        Some further refinements on the second verions
        :param modulation_function:
        :return:
        """
        for stream_ind in range(self.symbol_stream_matrix.shape[0]):
            stream = list(self.symbol_stream_matrix[stream_ind, :])
            self.modulated_signal_function.append(self._modulate_stream_on_function(stream, modulation_function, parameters))

    def sample_modulated_function(self):
        num_samples = self.samples_per_symbol_period*self.symbol_stream_matrix.shape[1]
        self.modulated_signal_function_sampled = self._sample_function(num_samples, self.modulated_signal_function, noise=False)

    def send_through_channel(self, quantization_level=None):
        """
        Note that given the numpy convolution default, the impulse response should be provided with the longest delay
        tap on the left most index.
        :return:
        """
        # plt.figure()
        self.channel_output = []
        for bit_streams in range(self.symbol_stream_matrix.shape[0]):
            self.channel_output.append(
                np.convolve(np.flip(self.symbol_stream_matrix[bit_streams,:]), self.CIR_matrix[bit_streams,:], mode="full"))
        self.channel_output = np.flip(np.asarray(self.channel_output))
        # plt.subplot(1,3,1)
        # plt.scatter(self.channel_output,self.channel_output)

        #   Quantize before adding noise to ensure noise profile is not changed
        if quantization_level is not None:
            self.channel_output = quantizer(self.channel_output, quantization_level)

        # plt.subplot(1,3,2)
        # plt.scatter(self.channel_output,self.channel_output)

        #   adjust noise power to provided SNR parameter. Note symbols should always be normalized to unit power.
        self.noise_parameter[1] = np.sqrt(np.var(self.alphabet) * (1 / self.SNR))
        self.channel_output += self.noise_parameter[0] + self.noise_parameter[1]*np.random.standard_normal(self.channel_output.shape)
        # plt.subplot(1,3,3)
        # plt.scatter(self.channel_output,self.channel_output)
        # plt.show()
        pass

    def transmit_modulated_signal2(self):
        """
        For this version, it is assumed that a impulse response is obtained. Using this, the linearity property of
        convolution is exploited in order to imitate the mixing of symbols in the channel.
        :return:
        """
        for bit_streams in range(self.symbol_stream_matrix.shape[0]):
            self.modulated_signal_function[bit_streams].virtual_convole_functions(self.modulated_CIR_matrix)

        #TODO make sure the signal is properly flipped if convolution flips output.

    def convolve_sampled_modulated(self):
        """
        For this version, it is assumed that a impulse response is obtained. Using this, the linearity property of
        convolution is exploited in order to imitate the mixing of symbols in the channel.
        :return:
        """
        for bit_streams in range(self.symbol_stream_matrix.shape[0]):
            self.modulated_convolved_signal_function_sampled.append(
                np.convolve(np.flip(self.modulated_signal_function_sampled[bit_streams,:]), self.CIR_matrix[bit_streams,:], mode="full"))
        self.modulated_convolved_signal_function_sampled = np.asarray(self.modulated_convolved_signal_function_sampled)
        # self.modulated_convolved_signal_function_sampled = np.flip(np.asarray(self.modulated_convolved_signal_function_sampled))

        self.noise_parameter[1] *= np.sqrt(np.var(self.alphabet) * (1 / self.SNR))
        self.modulated_convolved_signal_function_sampled += self.noise_parameter[0] + self.noise_parameter[1] * \
                               np.random.randn(self.modulated_convolved_signal_function_sampled.shape[0],
                                               self.modulated_convolved_signal_function_sampled.shape[1])


        #TODO make sure the signal is properly flipped if convolution flips output.

    def transmit_modulated_signal(self):
        """
        Transmit the signal that has been modulated on a fundamental pulse through the channel.
        :return:
        """
        for bit_streams in range(self.symbol_stream_matrix.shape[0]):
            self.modulated_channel_output.append(
                np.convolve(np.flip(self.transmit_signal_matrix[bit_streams, :]), self.modulated_CIR_matrix,
                            mode="full"))
        self.modulated_channel_output = np.flip(np.asarray(self.modulated_channel_output))

    def filter_received_modulated_signal(self):
        """
        Use a provided filter to identify received symbols (e.g. matched filtering)
        :return:
        """
        # First check that there is a received signal
        if self.modulated_channel_output is not None:
            samples_per_symbol_period = int(np.floor(self.symbol_period / self.sampling_period))
            offset = int(np.ceil(self.symbol_period / 2))
            """
            Sample/filter the received, modulated signal every 
            """
            for stream_ind in range(self.symbol_stream_matrix.shape[0]):
                stream = []
                for ind in range(self.symbol_stream_matrix.shape[1]):
                    ind1 = offset+samples_per_symbol_period*ind - offset
                    ind2 = offset+int(samples_per_symbol_period/2) + samples_per_symbol_period * ind
                    samples_filtered = self.modulated_convolved_signal_function_sampled[stream_ind, ind1:ind2]
                    stream.append(np.dot(self.receive_filter, samples_filtered))
                self.demodulated_symbols[stream_ind,:] = np.asarray(stream)
            return None

    def gaussian_channel_metric_sampled(self, modulation_function, parameters, states):
        """
        Needs to return the costs of the
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
        Like all metrics used by the Viterbi class. This should take a set of states and index and output the metrics
        for the various states possible given the sampled input.
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
        x_list = []
        y_list = []
        states = []
        item = []
        get_combinatoric_list(self.alphabet, self.CIR_matrix.shape[1], states, item)  # Generate states used below
        states = np.asarray(states)
        if self.channel_output is not None:
            j=0
            for i in range(self.channel_output.shape[1]):
                if (i >= self.CIR_matrix.shape[1]-1 and i < self.symbol_stream_matrix.shape[1] - self.CIR_matrix.shape[1] + 1):
                    state = self.symbol_stream_matrix[:, j: j+self.CIR_matrix.shape[1]].flatten()
                    probability_vec = self.get_probability(state, states)
                    x_list.append(self.channel_output[:, i].flatten())
                    y_list.append(probability_vec)
                    j+=1
        return x_list, y_list

    def get_labeled_data_reduced_state(self, outputs, quantization_level=None):
        x_list = []
        y_list = []
        base_states = int(np.ceil(np.log2(outputs)))
        states = []
        item = []
        get_combinatoric_list(self.alphabet, self.CIR_matrix.shape[1] - 1, states, item)
        # get_combinatoric_list(self.alphabet, self.CIR_matrix.shape[1], states, item)

        states = np.asarray(states)
        # reduced = np.asarray(states)@self.CIR_matrix.T
        normalized_channel =self.CIR_matrix[:, 1::]/ np.linalg.norm(self.CIR_matrix[:, 1::])
        reduced = np.asarray(states)@normalized_channel.T

        if quantization_level is not None:
            reduced = quantizer(reduced,quantization_level)
        num_clusters = int(pow(2, base_states-1))
        clusters = kmeans(reduced, num_clusters)
        labels = vq(reduced, clusters[0])[0]
        centroids = clusters[0]
        # plt.scatter(reduced, reduced)
        # plt.scatter(centroids, centroids)
        # plt.show()
        states_reduced = []
        item_reduced = []
        get_combinatoric_list(self.alphabet, base_states-1, states_reduced, item_reduced)
        states_reduced = np.asarray(states_reduced)


        states_final = []
        item_final = []
        get_combinatoric_list(self.alphabet, base_states, states_final, item_final)
        states_reduced = np.asarray(states_reduced)

        if self.channel_output is not None:
            j=0
            #   Go create training example from each channel output
            for i in range(self.channel_output.shape[1]):
                #   Take care of causality of creating training sets and unused output symbols
                if (i >= self.CIR_matrix.shape[1]-1 and i < self.symbol_stream_matrix.shape[1] - self.CIR_matrix.shape[1] + 1):
                    #   Get true state of the system
                    symbol = self.symbol_stream_matrix[:,j].flatten()
                    true_state = self.symbol_stream_matrix[:, j+1: j+self.CIR_matrix.shape[1]].flatten()
                    probability_vec = self.get_probability(true_state, states)
                    # Now find corresponding reduced state cluster number for the true state
                    state = labels[np.argmax(probability_vec)]
                    new_state = states_reduced[state]
                    # reduced_state = np.append(symbol, new_state)
                    reduced_state = np.append(new_state, symbol)
                    probability_vec_reduced = self.get_probability(reduced_state, states_final)
                    y_list.append(probability_vec_reduced)
                    # y_list.append(probability_vec)
                    x_list.append(self.channel_output[:, i].flatten())
                    j+=1
        # below check should at least be symmetric
        totals = np.sum(np.asarray(y_list), axis=0)
        return x_list, y_list

    def get_probability(self, input, states):
        """

        :return:
        """
        metrics = []
        for state in states:
            if np.array_equal(input, state):
                metrics.append(1)
                #TODO break out of loop after finding match
            else:
                metrics.append(0)
        return np.asarray(metrics)

    def plot_setup(self, save=False):
        num = 3
        figure = plt.figure()
        figure.add_subplot(num, 1,1)
        self.visualize(self.CIR_matrix, "C0-", name="CIR_matrix")
        figure.add_subplot(num, 1,2)
        self.visualize(self.symbol_stream_matrix, "C1-", name="symbol_stream_matrix")
        figure.add_subplot(num, 1,3)
        self.visualize(self.modulated_signal_function_sampled, "C4-", name="modulated_signal_function_sampled")

        plt.show()
        if save==True:
            time_path = "modulated_rect.png"
            figure.savefig(time_path, format="png")

    def visualize(self, data, c, name = None):
        for channels in range(data.shape[0]):
            plt.title(str(name), fontdict={'fontsize': 10})
            plt.stem(data[channels, :], linefmt=c)

    def _modulate_stream_on_function(self, stream, modulation_function: sampled_function(), parameters):
        function = combined_function()
        for ind, symbol in enumerate(stream):
            to_add = modulation_function(parameters)
            center = ind*self.symbol_period + np.ceil(self.symbol_period/2)
            to_add.setup(center, symbol)
            function.add_function(to_add)
        return function

    def _sample_function(self, num_samples, function, noise = False):
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
        information = []
        information.append("Alphabet Size: " + str(len(self.alphabet)))
        information.append("Channel Length (wrt symbols): " + str(self.CIR_matrix.shape[1]))
        information.append("# transmitted symbols: " + str(self.symbol_stream_matrix.shape[1]))
        return information