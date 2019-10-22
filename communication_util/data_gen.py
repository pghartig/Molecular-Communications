import numpy as np
import math
import matplotlib.pyplot as plt


class training_data_generator:
    def __init__(
        self,
        SNR,
        symbol_shape=np.zeros((1, 100)),
        channel=None,
        channel_shape=None,
        plot=False,
        constellation='ASK',
        constellation_size=2,
        noise_parameter = np.array((0, 1))

    ):
        self.SNR = SNR
        self.symbol_stream_matrix = symbol_shape
        self.CIR_matrix = channel
        self.channel_shape = channel_shape
        self.zero_pad = True
        self.terminated = True
        self.plot = plot
        self.noise_parameter = noise_parameter
        self.channel_output = []
        self.alphabet = self.constellation(constellation, constellation_size)

    def setup_channel(self, shape=(1, 1)):
        if self.channel_shape is not None:
            self.CIR_matrix = np.random.randn(shape[0], shape[1])
        elif self.CIR_matrix is None:
            self.CIR_matrix = np.ones((1, 1))
        # else:
        #     self.CIR_matrix = np.zeros((1,10))
        #     self.CIR_matrix[0, [0, 3, 5]] = 1, .5, .2

    def random_symbol_stream(self):
        shape = self.symbol_stream_matrix.shape
        if (
            self.zero_pad is True
            and self.terminated is True
            and self.CIR_matrix is not None
        ):
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
        if (
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
        self.noise_parameter[1] *= np.var(self.alphabet)*(1/self.SNR)
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
            return np.linspace(-1, +1, np.floor(size/2))
        elif type is 'PSK':
            return np.exp(1j*np.linspace(0, 2*np.pi, size))
        else:
            return np.array([1, -1])



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
