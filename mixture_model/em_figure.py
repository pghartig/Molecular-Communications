import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(1000)*1
data2 = np.random.randn(1000)*.5 +4
data3 = np.random.randn(1000)*2 +8
data4 = np.random.randn(1000)*.25 + 10

plt.figure(1)
plt.scatter(data3,data, label='viterbi net')
plt.scatter(data2,data, label='viterbi net')
plt.scatter(data2,data3, label='viterbi net')
plt.scatter(data4,data, label='viterbi net')

plt.title("Gaussian Mixture Model", fontdict={'fontsize':10} )
path = "Output/SER_curves.png"
plt.savefig(path, format="png")
time_path = "Output/em_figure.png"
plt.savefig(time_path, format="png")
plt.show()