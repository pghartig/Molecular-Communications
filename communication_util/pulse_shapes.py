"""
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
    provide samples from root raised cosine pulse
    :param sample_point:
    :param symbol_period:
    :return:
    """
    # t_samp = 1 / 1000
    # t_sym = 1 / 10

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


def rectangle(
    sample_points,
    sample_period=1 / 100,
    symbol_period=1 / 10,
    pulse_energy=1,
    alpha=0.5,
):
    """
    provide samples from root raised cosine pulse
    :param sample_point:
    :param symbol_period:
    :return:
    """
    return (
        0
        if sample_points < -1 / (symbol_period * 2)
        or sample_points > 1 / (symbol_period * 2)
        else 1
    )
    # ret = []
    # for sample in sample_points:
    #     if sample < -symbol_period/2 or sample > symbol_period/2:
    #         ret.append(0)
    #     else:
    #         ret.append(pulse_energy/symbol_period)
    # return np.asarray(ret)

def flow_injection(time):
    """
    provide samples from  flow channel injection pulse
    :return: function value at the sample point
    """
    sample = 0

    return sample


class sampled_function():
    def return_samples(self, number_samples, sampling_period, start_index=0):
        samples = []
        for i in range(number_samples):
            samples.append(self.evaluate(i*sampling_period + start_index*sampling_period))
        return np.asarray(samples)

    def evaluate(self,sample_points):
        pass


class rect_function_class(sampled_function):
    def __init__(self, width):
        self.width = width

    def evaluate(self, sample_points):
        test2 = 1 / (self.width * 2)
        return (0 if sample_points < -1 / (self.width * 2) or sample_points > 1 / (self.width * 2) else 1)


class dirac_channel(sampled_function):
    def __init__(self, delay=0):
        self.delay = delay

    def evaluate(self, sample_points):
        return 1 if sample_points == self.delay else 0

def rect_function(
    sample_points,
    sample_period=1 / 10,
    symbol_period=1 / 1,
    pulse_energy=1,
    alpha=0.5,
):
    """
    provide samples from root raised cosine pulse
    :param sample_point:
    :param symbol_period:
    :return:
    """
    return (
        0
        if sample_points < -1 / (symbol_period * 2)
        or sample_points > 1 / (symbol_period * 2)
        else 1
    )
    # ret = []
    # for sample in sample_points:
    #     if sample < -symbol_period/2 or sample > symbol_period/2:
    #         ret.append(0)
    #     else:
    #         ret.append(pulse_energy/symbol_period)
    # return np.asarray(ret)

# class stream_function(sampled_function):

def func1(x):
    return np.abs(x)

def func2(x):
    return x


