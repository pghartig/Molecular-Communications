"""
the purpose of this module is to provide a variety of pulse shapes onto which a desired symbol stream can be modulated.
All of the pulse shape functions should return the evaluted function at the sample point.
"""

import numpy as np


def root_raise_cosine(sample_point, sample_period =  1 / 1000, symbol_period = 1 / 10, pulse_energy= 1, alpha= .5):
    """
    provide samples from root raised cosine pulse
    :param sample_point:
    :param symbol_period:
    :return:
    """
    # t_samp = 1 / 1000
    # t_sym = 1 / 10

    sample = np.sqrt((pulse_energy/symbol_period))*((4*alpha*sample_point)*np.cos(np.pi*(1+alpha)*sample_point/symbol_period)+
                                                    symbol_period*np.sin(np.pi*(1-alpha)*sample_point/symbol_period))\
             / (np.pi*sample_point*(1-np.power((4*alpha*sample_point/symbol_period), 2)))
    #for testing
    # sample = np.cos(sample_point)
    return sample

def rectangle(sample_points, sample_period = 1/1000, symbol_period = 1/10, pulse_energy= 1, alpha= .5):
    """
    provide samples from root raised cosine pulse
    :param sample_point:
    :param symbol_period:
    :return:
    """
    ret = []
    for sample in sample_points:
        if sample < -symbol_period/2 or sample > symbol_period/2:
            ret.append(0)
        else:
            ret.append(pulse_energy/symbol_period)
    return np.asarray(ret)