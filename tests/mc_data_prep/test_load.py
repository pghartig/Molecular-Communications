from communication_util.load_mc_data import *
import matplotlib.pyplot as plt

def test_load():
    path = 'mc_data/test_data.csv'
    raw = load_file(path)

    plt.plot(raw[0],raw[1])
    plt.show()
