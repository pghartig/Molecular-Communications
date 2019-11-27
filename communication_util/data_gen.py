import numpy as np
import math
import matplotlib.pyplot as plt
import logging as log
from communication_util.general_tools import get_combinatoric_list
from communication_util.pulse_shapes import  *


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
        noise_parameter=np.array((0, 1)),
        seed=None,
        sampling_period = 1 / 10,
        symbol_period = 1 / 1
    ):

        """
        Basic parameters of the data generations
        """
        self.symbol_stream_matrix = np.zeros(symbol_stream_shape)
        self.CIR_matrix = channel
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
        self.modulated_channel_output = []
        self.receive_filter = None
        self.demodulated_symbols = np.zeros(symbol_stream_shape)
        self.sampling_period = sampling_period
        self.symbol_period = symbol_period

        """
        Decoding metrics
        """
        self.metrics = None

    def setup_channel(self, shape=(1, 1)):
        if self.CIR_matrix is not None:
            self.channel_shape = self.CIR_matrix.shape
        elif self.channel_shape is not None:
            self.CIR_matrix = np.random.randn(shape[0], shape[1])
        else:
            self.CIR_matrix = np.ones((1, 1))
            self.channel_shape = self.CIR_matrix.shape

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

    def sample_modulated_function(self, num_samples):
        self.modulated_signal_function_sampled = self._sample_function(num_samples, self.modulated_signal_function)
        return None

    def send_through_channel(self):
        """
        Note that given the numpy convolution default, the impulse response should be provided with the longest delay
        tap on the left most index.
        :return:
        """
        for bit_streams in range(self.symbol_stream_matrix.shape[0]):
            self.channel_output.append(
                np.convolve(np.flip(self.symbol_stream_matrix[bit_streams,:]), self.CIR_matrix[bit_streams,:], mode="full"))
        self.channel_output = np.flip(np.asarray(self.channel_output))
        # self.channel_output = np.asarray(self.channel_output)   #Test to correct problem with viterbi

        # add noise
        # adjust noise power to provided SNR parameter
        self.noise_parameter[1] *= np.sqrt(np.var(self.alphabet) * (1 / self.SNR))
        self.channel_output += self.noise_parameter[0] + self.noise_parameter[
            1
        ] * np.random.randn(self.channel_output.shape[0], self.channel_output.shape[1])

    def transmit_modulated_signal2(self):
        """
        For this version, it is assumed that a impulse response is obtained. Using this, the linearity property of
        convolution is exploited in order to imitate the mixing of symbols in the channel.
        :return:
        """
        for bit_streams in range(self.symbol_stream_matrix.shape[0]):
            self.modulated_signal_function[bit_streams].virtual_convole_functions(self.modulated_CIR_matrix)

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
            offset = int(1+samples_per_symbol_period/2)
            """
            Sample/filter the received, modulated signal every 
            """
            for stream_ind in range(self.symbol_stream_matrix.shape[0]):
                stream = []
                for ind in range(self.symbol_stream_matrix.shape[1]):
                    ind1 = offset+samples_per_symbol_period*ind - offset
                    ind2 = offset+int(samples_per_symbol_period/2) + samples_per_symbol_period * ind
                    samples_filtered = self.modulated_signal_function_sampled[stream_ind, ind1:ind2]
                    stream.append(np.dot(self.receive_filter,samples_filtered))
                self.demodulated_symbols[stream_ind,:]= np.asarray(stream)
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
            #returns samples of above function
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
        num_possible_states = np.power(self.alphabet, self.CIR_matrix.shape[1])
        states = []
        item = []
        get_combinatoric_list(self.alphabet, self.CIR_matrix.shape[1], states, item)  # Generate states used below
        test = np.asarray(states)  # up to here signs are preserved. Sorting is flipping signs
        # states = np.sort(np.asarray(states), 1)
        states = np.asarray(states)
        if self.channel_output is not None:
            for i in range(self.channel_output.shape[1]):
                if (i >= self.CIR_matrix.shape[1]-1 and i < self.symbol_stream_matrix.shape[1] - self.CIR_matrix.shape[1] + 1):
                    input = self.symbol_stream_matrix[:, i - self.CIR_matrix.shape[1]+1: i+1].flatten()
                    probability_vec = self.get_probability(input, states)
                    x_list.append(self.channel_output[:, i].flatten())
                    y_list.append(probability_vec)
        return x_list, y_list

    def get_probability(self, input, states):
        """

        :return:
        """
        num_possible_states = int(np.power(self.alphabet.size, self.CIR_matrix.shape[1]))
        metrics = []
        for state_ind in range(num_possible_states):
            state = states[state_ind, :]  # Look up state in table based on index (should be passed from calling loop!
            if np.array_equal(input, state):
                metrics.append(1)
            else:
                metrics.append(0)
        test1 = metrics
        test = np.asarray(metrics)
        return np.asarray(metrics)

    def plot_setup(self):
        num =1
        figure = plt.figure()
        # figure.add_subplot(num, 1,1)
        # self.visualize(self.CIR_matrix, "C0-")
        # figure.add_subplot(num, 1,2)
        # self.visualize(self.symbol_stream_matrix, "C1-")
        # figure.add_subplot(num, 1,3)
        # self.visualize(self.channel_output, "C2-")
        # figure.add_subplot(num, 1,4)
        # self.visualize(self.transmit_signal_matrix, "C3-")
        # figure.add_subplot(num, 1,5)
        # self.visualize(self.modulated_signal_function_sampled, "C4-")
        figure.add_subplot(num, 1, 1)
        self.visualize(self.modulated_signal_function_sampled, "C5-")
        plt.show()

    def visualize(self, data, c):
        for channels in range(data.shape[0]):
            plt.stem(data[channels, :], linefmt=c)

    def _modulate_stream_on_function(self, stream, modulation_function: sampled_function(), parameters):
        function = combined_function()
        for ind, symbol in enumerate(stream):
            to_add = modulation_function(parameters)
            to_add.setup(ind*self.symbol_period, symbol)
            function.add_function(to_add)
        return function

    def _sample_function(self, num_samples, function):
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



