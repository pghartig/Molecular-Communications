import numpy as np
import math
import matplotlib.pyplot as plt
import logging as log
from communication_util.general_tools import get_combinatoric_list


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
    ):
        self.SNR = SNR
        self.symbol_stream_matrix = np.zeros(symbol_stream_shape)
        self.transmit_signal_matrix = []
        self.CIR_matrix = channel
        self.channel_shape = channel_shape
        self.zero_pad = True
        self.terminated = True
        self.plot = plot
        self.noise_parameter = noise_parameter
        self.channel_output = []
        self.alphabet = self.constellation(constellation, constellation_size)
        self.seed = seed

    def setup_channel(self, shape=(1, 1)):
        if self.CIR_matrix is not None:
            self.channel_shape = self.CIR_matrix.shape
        elif self.channel_shape is not None:
            self.CIR_matrix = np.random.randn(shape[0], shape[1])
        else:
            self.CIR_matrix = np.ones((1, 1))
            self.channel_shape = self.CIR_matrix.shape
        # else:
        #     self.CIR_matrix = np.zeros((1,10))
        #     self.CIR_matrix[0, [0, 3, 5]] = 1, .5, .2

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

    def modulate_fundamental_pulse(self, fundamental_pulse):
        """
        Fundamenet
        The purpose of this funcion is to take a symbol stream and use it to modulate the fundemental pulse
        on which it will be send over the channel.
        :return:
        """
        # include parameter of samples/symbol
        t_samp = 1 / 10
        t_sym = 1 / 1

        """
        First look at pulse and determine where to cut off
        will just keep making samples until we full to some percentage of max.
        Notice that assumes a symmetric pulse shape
        """
        sample_number = 0
        peak_energy = energy = fundamental_pulse(sample_number * t_samp)
        # TODO verify this threshold for where to cutoff fundamental pulse (asymetric case)
        while energy >= 0.05 * peak_energy:
            energy = fundamental_pulse(
                sample_number * t_samp, sample_period=t_samp, symbol_period=t_sym
            )
            sample_number += 1

        sample_vector = np.arange(-sample_number, sample_number + 1) * t_samp
        vec_pulse = np.vectorize(fundamental_pulse)
        sample_vector = vec_pulse(sample_vector)
        sampling_width = int(np.floor(sample_vector.size / 2))

        """
        In order to allow for adding components from multiple symbols into a single sample, the array for the
        sampled, modulated signal must be pre-allocated.
        """
        samples_per_symbol_period = int(np.floor(t_sym / t_samp))
        overlap = max(int(sampling_width - samples_per_symbol_period / 2), 0)
        # TODO figure out why +1 is needed in line below for shape
        self.transmit_signal_matrix = np.zeros(
            (
                1,
                samples_per_symbol_period * self.symbol_stream_matrix.shape[1]
                + 2 * overlap
                + 1,
            )
        )
        try:
            for symbol_ind in range(self.symbol_stream_matrix.shape[1]):
                center = symbol_ind * samples_per_symbol_period + sampling_width
                # samples = self.symbol_stream_matrix.shape[symbol_ind]*sample_vector
                # test = self.transmit_signal_matrix[:, (- sample_number + center) : (center + sample_number +1)]
                self.transmit_signal_matrix[
                    :, (-sample_number + center) : (center + sample_number + 1)
                ] = (sample_vector * self.symbol_stream_matrix[:, symbol_ind])
        except:
            log.log("problem")

    def send_through_channel(self):
        """
        Note that given the numpy convolution default, the impulse response should be provided with the longest delay
        tap on the left most index.
        :return:
        """
        for bit_streams in range(self.symbol_stream_matrix.shape[0]):
            self.channel_output.append(
                np.convolve(
                    self.symbol_stream_matrix[bit_streams, :],
                    np.flip(self.CIR_matrix[bit_streams, :]),
                    mode="full",
                )
            )
        self.channel_output = np.asarray(self.channel_output)
        # add noise
        # adjust noise power to provided SNR parameter
        self.noise_parameter[1] *= np.sqrt(np.var(self.alphabet) * (1 / self.SNR))
        self.channel_output += self.noise_parameter[0] + self.noise_parameter[
            1
        ] * np.random.randn(self.channel_output.shape[0], self.channel_output.shape[1])

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

    def get_labeled_data(self):
        x_list = []
        y_list = []
        num_possible_states = np.power(self.alphabet, self.CIR_matrix.shape[1])
        states = []
        item = []
        get_combinatoric_list(self.alphabet, self.CIR_matrix.shape[1]-1, states, item)  # Generate states used below
        states = np.asarray(states)

        if self.channel_output is not None:
            for i in range(self.channel_output.shape[1]):
                if (
                    i >= self.CIR_matrix.shape[1] and i < self.symbol_stream_matrix.shape[1] - self.CIR_matrix.shape[1]):
                    x_list.append(self.channel_output[:, i].flatten())
                    input = self.symbol_stream_matrix[
                        :, i - self.CIR_matrix.shape[1] : i
                    ].flatten()
                    probability_vec = self.get_probability(input, states)
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
            if np.array_equal(input,state):
                metrics.append(1)
            else:
                metrics.append(0)
        return np.asarray(metrics)

    def plot_setup(self):
        figure = plt.figure()
        figure.add_subplot(411)
        self.visualize(self.CIR_matrix, "C0-")
        figure.add_subplot(412)
        self.visualize(self.symbol_stream_matrix, "C1-")
        figure.add_subplot(413)
        self.visualize(self.channel_output, "C2-")
        figure.add_subplot(414)
        self.visualize(self.transmit_signal_matrix, "C3-")
        plt.show()

    def visualize(self, data, c):
        for channels in range(data.shape[0]):
            plt.stem(data[channels, :], linefmt=c)
