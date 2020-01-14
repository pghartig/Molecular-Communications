import numpy as np
def linear_mmse(transmit_sequence, received_sequence, true_sequence, channel_length):
    """

    :param transmit_sequence: Should be long enough to fully represent the observed sequence for a given channel length
    :param received_sequence:
    :param channel_length:
    :return:
    """
    E_R_yy = np.zeros((channel_length,channel_length))
    E_r_yx = np.zeros((channel_length,1))
    for ind, symbol in enumerate(transmit_sequence.flatten()):
        received_sequence_block = received_sequence[:,ind:(ind+channel_length)]
        r_yx = received_sequence_block.T*symbol
        test = np.conjugate(received_sequence_block.T)
        R_yy = received_sequence_block.T@np.conjugate(received_sequence_block)
        E_R_yy += R_yy
        check = np.linalg.matrix_rank(E_R_yy)
        E_r_yx += r_yx

    E_r_yx /= len(received_sequence)
    E_R_yy /= len(received_sequence)
    h = np.linalg.pinv(E_R_yy)@E_r_yx

    equalizer_detection = []
    for ind, symbol in enumerate(received_sequence.flatten()):
        if ind+len(h) <= received_sequence.size:
            one = received_sequence[:,ind:(ind+len(h))]
            check = received_sequence[:,ind:(ind+len(h))]@h
            equalizer_detection.append(received_sequence[:,ind:(ind+len(h))]@h)
    equalizer_detection = np.sign(np.asarray(equalizer_detection).flatten())
    true = true_sequence.flatten()
    ser = np.sum(np.not_equal(equalizer_detection[:true.size], true)) / true.size
    return ser
