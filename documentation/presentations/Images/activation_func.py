import numpy as np
import matplotlib.pyplot as plt

range = np.arange(-5,5,.001)
sigfig = plt.figure()
sigmoid = 1/(1+np.exp(-range))
plt.plot(range,sigmoid)
#sigfig.title("Sigmoid")
plt.savefig('Output/sigmoid')

tanfig = plt.figure()
tanh = (np.exp(range)-np.exp(-range))/(np.exp(range)+np.exp(-range))
plt.plot(range,tanh)
#tanfig.title("Sigmoid")
plt.savefig('Output/tanh')

relu = plt.figure()
relu_func = np.max((0*range,1*range),0)
plt.plot(range,relu_func)
#tanfig.title("relu")
plt.savefig('Output/relu')