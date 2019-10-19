import numpy as np
import matplotlib.pyplot as plt


class training_data_generator:
    def __init__(self, amount=(1, 100), noise_parm=[0, 1], plot=False):
        self.shape = amount
        self.bit_stream_matrix = None
        self.CIR_matrix = None
        self.zero_pad = True
        self.plot = plot
        self.noise_para = noise_parm
        self.channel_output = []

    # depending on number of properties channel should be class?
    def setup_channel(self, shape=(1, 1)):
        if shape == None:
            self.CIR_matrix = np.ones((1, 1))
        else:
            self.CIR_matrix = np.random.randn(shape[0], shape[1])

        # else:
        #     self.CIR_matrix = np.zeros((1,10))
        #     self.CIR_matrix[0, 5, 9] = [1, .4, .2]

    def random_bit_stream(self):
        if self.zero_pad == True:
            self.bit_stream_matrix = np.random.random_integers(0, 1, self.shape)
            self.bit_stream_matrix = np.concatenate(
                (self.bit_stream_matrix, np.zeros((self.CIR_matrix.shape))), 1
            )
        else:
            self.bit_stream_matrix = np.random.random_integers(0, 1, self.shape)

    def send_through_channel(self):
        """
        Note that given the numpy convolution default, the impulse response should be provided with the longest delay
        tap on the left most index.
        :return:
        """
        for bit_streams in range(self.bit_stream_matrix.shape[0]):
            self.channel_output.append(
                np.convolve(
                    self.bit_stream_matrix[bit_streams, :],
                    self.CIR_matrix[bit_streams, :],
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
        self.visualize(self.bit_stream_matrix, "C1-")
        figure.add_subplot(313)
        self.visualize(self.channel_output, "C2-")
        plt.show()

    def visualize(self, data, c):
        for channels in range(data.shape[0]):
            plt.stem(data[channels, :], linefmt=c)
