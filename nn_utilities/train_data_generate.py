from communication_util import data_gen

def denoising_auto_encoding_data(len):
    bit_sequence = data_gen.random_bit_stream(len)
    channel_output = data_gen.send_through_channel()

    return