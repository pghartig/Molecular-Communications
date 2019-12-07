from mixture_model.em_algorithm import *
import matplotlib.pyplot as plt
import numpy as np

vec = np.linspace(-3,3,100)
samples = []
for i in vec:
    samples.append(probability_from_gaussian_sources(i,0,1))
plt.plot(vec)
plt.show()

