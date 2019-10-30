import numpy as np
import math
import matplotlib.pyplot as plt


class training_data_generator:
    def __init__(
        self,
        SNR,
        symbol_stream_shape=(1, 100),
        channel=None,
        channel_shape=None,
        plot=False,
        constellation='ASK',
        constellation_size=2,
        noise_parameter = np.array((0, 1)),
        seed = None

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
            self.symbol_stream_matrix = \
                np.random.random_integers(0, self.alphabet.size - 1, shape)
            self.symbol_stream_matrix = self.alphabet[self.symbol_stream_matrix]
            self.symbol_stream_matrix[:, 1 - self.CIR_matrix.shape[1]:] = self.alphabet[0]
            self.symbol_stream_matrix[:, 0: self.CIR_matrix.shape[1] - 1] = self.alphabet[0]
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

    def modulate_fundemental_pulse(self, fundamental_pulse):
        """
        Fundamenet
        The purpose of this funcion is to take a symbol stream and use it to modulate the fundemental pulse
        on which it will be send over the channel.
        :return:
        """
        #include parameter of samples/symbol
        t_samp = 1/1000
        t_sym = 1/10

        """
        First look at pulse and determine where to cut off
        will just keep making samples until we full to some percentage of max.
        Notice that assumes a symmetric pulse shape
        """
        sample_number = 0
        peak_energy = energy = fundamental_pulse(sample_number * t_samp)
        #TODO verify this threshold for where to cutoff fundamental pulse
        while energy >= .05*peak_energy:
            energy = fundamental_pulse(sample_number*t_samp)
            sample_number += 1

        sample_vector = np.arange(-sample_number, sample_number+1)*t_samp
        sample_vector = fundamental_pulse(sample_vector)
        sampling_width = int(np.floor(sample_vector.size/2))


        """
        In order to allow for adding components from multiple symbols into a single sample, the array for the
        sampled, modulated signal must be pre-allocated.
        """
        samples_per_symbol = int(np.floor(t_sym/t_samp))
        overlap = max(sampling_width - samples_per_symbol/2,0)
        self.transmit_signal_matrix =\
            np.zeros((1, samples_per_symbol*self.symbol_stream_matrix.shape[1] + 2*overlap))

        for symbol_ind in range(self.symbol_stream_matrix.shape[1]):
            center = symbol_ind*samples_per_symbol + int(np.ceil(samples_per_symbol/2))
            test = self.transmit_signal_matrix[center - sampling_width: sampling_width+1]
            self.transmit_signal_matrix[:, center - sampling_width: sampling_width+1] = sample_vector

            self.transmit_signal_matrix.append()
            return None


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
        self.noise_parameter[1] *= np.sqrt(np.var(self.alphabet)*(1/self.SNR))
        self.channel_output += self.noise_parameter[0] + self.noise_parameter[
            1
        ] * np.random.randn(self.channel_output.shape[0], self.channel_output.shape[1])

    def constellation(self, type, size):
        #TODO for large tests may want to select dtype
        if type is 'QAM':
            points = np.linspace(-1, 1, np.floor(math.log(size, 2)))
            constellation = 1j*np.zeros((points.size, points.size))
            for i in range(points.size):
                for k in range(points.size):
                    constellation[i, k] = points[i] + 1j*points[k]
            return constellation.flatten()
        elif type is 'ASK':
            return np.linspace(-1, +1, size)
        elif type is 'PSK':
            return np.exp(1j*np.linspace(0, 2*np.pi, size))
        else:
            return np.array([1, -1])

    def get_labeled_data(self):
        x_list = []
        y_list = []
        if self.channel_output is not None:
            for i in range(self.channel_output.shape[1]):
                if i >= self.CIR_matrix.shape[1] and i < self.symbol_stream_matrix.shape[1] - self.CIR_matrix.shape[1]:
                    x_list.append(self.channel_output[:, i].flatten())
                    input = self.symbol_stream_matrix[:, i - self.CIR_matrix.shape[1]:i].flatten()
                    probability_vec = self.get_probability(input)
                    y_list.append(probability_vec)
        return x_list, y_list

    def get_probability(self):
        """
        TODO -> There would be ways of using a nice vector encoding here to make it faster
        Describe encoding ->
        :return:
        """
        num_possible_states = np.power(self.alphabet, self.CIR_matrix.shape[1])
        metrics = np.ones(1, num_possible_states)

        return None

    def plot_setup(self):
        figure = plt.figure()
        figure.add_subplot(311)
        self.visualize(self.CIR_matrix, "C0-")
        figure.add_subplot(312)
        self.visualize(self.symbol_stream_matrix, "C1-")
        figure.add_subplot(313)
        self.visualize(self.channel_output, "C2-")
        plt.show()

    def visualize(self, data, c):
        for channels in range(data.shape[0]):
            plt.stem(data[channels, :], linefmt=c)
