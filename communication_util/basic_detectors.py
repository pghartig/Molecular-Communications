import numpy as np

def threshold_detector(alphabet, output):
    detected_symbols = []
    for stream in range(output.shape[0]):
        for received_symbol in output[stream, :]:
            detected = alphabet[np.argmin(np.abs(alphabet - received_symbol))]
            detected_symbols.append(detected)
    return detected_symbols