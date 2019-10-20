import numpy as np
import matplotlib.pyplot as plt


class training_data_generator:
    def __init__(
        self,
        symbol_shape=np.zeros((1, 100)),
        channel=None,
        channel_shape=None,
        noise_parm=[0, 0.01],
        plot=False,
        alphabet=np.array([1, -1]),
    ):
        self.symbol_stream_matrix = symbol_shape
        self.CIR_matrix = channel
        self.channel_shape = channel_shape
        self.zero_pad = True
        self.terminated = True
        self.plot = plot
        self.noise_para = noise_parm
        self.channel_output = []
        self.alphabet = alphabet

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
        self.channel_output += self.noise_para[0] + self.noise_para[
            1
        ] * np.random.randn(self.channel_output.shape[0], self.channel_output.shape[1])

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
