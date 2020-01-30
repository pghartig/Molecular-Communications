import numpy as np
import matplotlib.pyplot as plt

#impulse and frequency response
impulse_response = np.zeros(20)
impulse_response[0] = 1
impulse_response[3] = .5
impulse_response[15] = .2
freq_response = np.fft.fft(impulse_response)

fig = plt.figure('impulse_response')
fig.suptitle('Multipath Channel', fontsize=16)
i = plt.subplot("121")
i.stem(impulse_response)
i.set_title("Impulse Reponse")
f = plt.subplot("122")
f.stem(freq_response)
f.set_title("Frequency Reponse")
plt.savefig('Output/Multipath_Channel')
#plt.show()
