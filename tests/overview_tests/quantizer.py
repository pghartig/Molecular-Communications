from mixture_model.em_algorithm import em_gausian
import numpy as np
from matplotlib import pyplot as plt
from communication_util.general_tools import quantizer
import time

def test_quantizer():
    input_array = np.linspace(-1, 1, num=200)
    quantized = quantizer(input_array, 1, -.5, .5)
    plt.figure(2)
    plt.plot(input_array, quantized, label='Test Error')
    plt.xlabel("Quantizer Input")
    plt.ylabel("Quantizer Ouput")
    plt.show()
    # path = f"Output/Quantizer.png"
    # plt.savefig(path, format="png")