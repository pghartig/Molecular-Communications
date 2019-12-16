import pandas as pd
import numpy as np
import mc_data


def load_file(path):
    raw = pd.read_csv(path, sep=",")
    numpy_raw = raw.as_matrix()
    time = numpy_raw[:,0]
    susceptability = numpy_raw[:,1]
    return (time,susceptability)