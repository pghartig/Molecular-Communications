import numpy as np

class training_data():
    def __init__(self, size = (1,100)):
        self.shape = size
        self.bit_stream_matrix = None
        self.CIR_matrix = None

    def setup_channel(self,shape,mu,variance):
        self.CIR_matrix = = np.random.normal(mu,variance,shape)
        self.CIR_matrix[0, 5, 9] = [1, .4, .2]

    def random_bit_stream(self):
        self.bit_stream_matrix = np.random.random_integers(0, 1, self.shape)

    def send_through_channel(self):
        CIR = np.zeros((1,10))
        CIR[0,5,9] = [1,.4,.2]
        channel_output = np.convolve(self.bit_stream_matrix,CIR)
        return channel_output
