import numpy as np


def viterbi_output(metrics):
    survivor_paths = np.zeros((metrics.shape[0]/2,metrics.shape[1]), dtype=np.int8)

    final_path_ind = np.argmin(np.sum(survivor_paths,axis=1))
    return np.flip(survivor_paths[final_path_ind, :],axis=1)