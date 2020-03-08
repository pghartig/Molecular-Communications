"""
THIS WHOLE MODULE IS IN WORK!
the purpose of this module is to provide a variety of pulse shapes onto which a desired symbol stream can be modulated.
All of the pulse shape functions should return the evaluted function at the sample point.
"""

import numpy as np


def root_raise_cosine(
    sample_point,
    sample_period=1 / 1000,
    symbol_period=1 / 10,
    pulse_energy=1,
    alpha=0.5,
):
    """
    # t_samp = 1 / 1000
    # t_sym = 1 / 10
    provide samples from root raised cosine pulse
    :param sample_point:
    :param symbol_period:
    :return:
    """

    sample = (
        np.sqrt((pulse_energy / symbol_period))
        * (
            (4 * alpha * sample_point)
            * np.cos(np.pi * (1 + alpha) * sample_point / symbol_period)
            + symbol_period * np.sin(np.pi * (1 - alpha) * sample_point / symbol_period)
        )
        / (
            np.pi
            * sample_point
            * (1 - np.power((4 * alpha * sample_point / symbol_period), 2))
        )
    )
    # for testing
    # sample = np.cos(sample_point)
    return sample


class sampled_function():

    def setup(self, center, symbol):
        self.center = center
        self.symbol = symbol

    def return_samples(self, number_samples, sampling_period, start_index=0):
        samples = []
        for i in range(number_samples):
            samples.append(self.evaluate(i*sampling_period + start_index*sampling_period))
        return np.asarray(samples)

    def virtual_convole(self, channel_impulse_response):
        self.evaluate_convolved = lambda x: \
            sum([self.evaluate(x+ind)*tap for ind, tap in enumerate(channel_impulse_response)])

    def evaluate(self, sample_points):
        pass


class combined_function():
    """
    This implementation is needed in order to exploit the linearity of convolution in the case of a known channel
    impulse response.
    """
    def __init__(self):
        self.functions = []

    def add_function(self, function):
        self.functions.append(function)

    #TODO implement faster
    def evaluate(self, sample_point, channel_length=1):
        sample = 0

        for function in self.functions:
            sample += function.evaluate(sample_point)
            # sample += function.evaluate_convolved(sample_point)
        return sample

    def virtual_convole_functions(self, impulse_response):
        for function in self.functions:
            function.virtual_convole(impulse_response)
            # now update the evaluate function
            # self.evaluate_convolved = lambda x :


class rect_function_class(sampled_function):
    def __init__(self, width):
        self.width = width
        self.center = 0

    def evaluate(self, sample_points):
        sample_points -= self.center
        return self.symbol*(0 if sample_points < -1 / (self.width * 2) or sample_points > 1 / (self.width * 2) else 1)*self.width


class rect_receiver_class(sampled_function):
    def __init__(self, width):
        self.width = width

    def evaluate(self, sample_points):
        return (0 if sample_points < -1 / (self.width * 2) or sample_points > 1 / (self.width * 2) else 1)*self.width


class dynamic_pulse(sampled_function):
    def __init__(self, width):
        self.width = width
        self.center = 0

    def evaluate(self, sample_points):
        sample_points -= self.center
        if self.symbol == -1:
            test = 3*self.symbol*(0 if sample_points < -1 / (self.width * 2) or sample_points > 1 / (self.width * 2) else 1)
            return 3*self.symbol*(0 if sample_points < -1 / (self.width * 2) or sample_points > 1 / (self.width * 2) else 1)
        elif self.symbol == 1:
            return self.symbol*(0 if sample_points < -1 / (self.width * 2) or sample_points > 1 / (self.width * 2) else 1)


class dirac_channel(sampled_function):
    def __init__(self, delay=0):
        super().__init__()
        self.delay = delay
        self.center = 0

    def evaluate(self, sample_points):
        sample_points -= self.center
        return 1 if sample_points == self.delay else 0