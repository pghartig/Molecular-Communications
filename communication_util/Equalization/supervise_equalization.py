import numpy as np


class LinearMMSE:
    """
    An implementation of the linear MMSE equalizer
    """
    def train_equalizer(self, transmit_sequence, received_sequence, true_sequence, channel_length):
        """
        Provided with training data, estimate a linear equalizer.
        :param transmit_sequence: Should be long enough to fully represent the observed sequence for a given channel length
        :param received_sequence:
        :param channel_length:
        :return:
        """
        E_R_yy = np.zeros((channel_length, channel_length))
        E_r_yx = np.zeros((channel_length, 1))
        for ind, symbol in enumerate(transmit_sequence.flatten()):
            received_sequence_block = received_sequence[:,ind:(ind+channel_length)]
            r_yx = received_sequence_block.T*symbol
            R_yy = received_sequence_block.T@np.conjugate(received_sequence_block)
            E_R_yy += R_yy
            E_r_yx += r_yx

        E_r_yx /= len(received_sequence)
        E_R_yy /= len(received_sequence)
        self.h = np.linalg.pinv(E_R_yy)@E_r_yx

        equalizer_detection = []
        for ind, symbol in enumerate(received_sequence.flatten()):
            if ind+len(self.h) <= received_sequence.size:
                one = received_sequence[:,ind:(ind+len(self.h))]
                check = received_sequence[:,ind:(ind+len(self.h))]@self.h
                equalizer_detection.append(received_sequence[:,ind:(ind+len(self.h))]@self.h)
        equalizer_detection = np.sign(np.asarray(equalizer_detection).flatten())
        true = true_sequence.flatten()
        #   Please Leave: Useful debugging tools
        # test1 = np.max(np.convolve(equalizer_detection, true))
        # test2 = np.max(np.convolve(equalizer_detection, np.flip(true)))
        ser = np.sum(np.not_equal(equalizer_detection[:true.size], true)) / true.size
        return ser

    def test_equalizer(self, true_sequence, received_sequence):
        """
        Detect symbols using the trained equalizer.
        :param true_sequence:
        :param received_sequence:
        :return:
        """
        if self.h is not None:
            equalizer_detection = []
            for ind, symbol in enumerate(received_sequence.flatten()):
                if ind + len(self.h) <= received_sequence.size:
                    one = received_sequence[:, ind:(ind + len(self.h))]
                    check = received_sequence[:, ind:(ind + len(self.h))] @ self.h
                    equalizer_detection.append(received_sequence[:, ind:(ind + len(self.h))] @ self.h)
            equalizer_detection = np.sign(np.asarray(equalizer_detection).flatten())
            true = true_sequence.flatten()
            #   Please Leave: Useful debugging tools
            # test1 = np.max(np.convolve(equalizer_detection, true))
            # test2 = np.max(np.convolve(equalizer_detection, np.flip(true)))
            ser = np.sum(np.not_equal(equalizer_detection[:true.size], true)) / true.size
            return ser
        else:
            raise Exception("Need to train before testing")
