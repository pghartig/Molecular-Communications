from communication_util.data_gen import *


def test_viterbi_gaussian():
    #setup data
    #TODO make consolidate this part
    data_gen = training_data_generator(plot=True)
    data_gen.setup_channel(shape=None)
    data_gen.random_bit_stream()
    data_gen.send_through_channel()

    #detect with Viterbi and check WER

    assert False