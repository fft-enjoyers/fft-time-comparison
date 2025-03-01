import numpy as np
from scipy.io import wavfile
from scipy import signal
import time



if __name__ == '__main__':
    fs, x = wavfile.read('input.wav')
    h = wavfile.read('filter.wav')[1]

    if x.dtype != np.float32 and x.dtype != np.float64:
        max_val = np.iinfo(x.dtype).max
        x = x.astype(np.float64) / max_val

    if h.dtype != np.float32 and h.dtype != np.float64:
        max_val = np.iinfo(h.dtype).max
        h = h.astype(np.float64) / max_val

    times = ""

    for i in range(50):
        start_time = time.time()
        filtered_signal = signal.fftconvolve(x, h, mode='same')  # same - same length as input signal
        elapsed_time = time.time() - start_time
        times += (elapsed_time.__str__() + "\n")

    f = open(f"times{len(h)}.txt", "w")
    f.write(times)
    f.close()

    filtered_signal_int16 = np.int16(np.clip(filtered_signal, -1, 1) * 32767)  # scaling to [–32767,32767]
    wavfile.write('output_filtered.wav', fs, filtered_signal_int16)

    print("Output signal saved to 'output_filtered.wav'")
