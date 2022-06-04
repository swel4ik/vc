from scipy.io.wavfile import read


def load_wav(file_descriptor):
    sampling_rate, data = read(file_descriptor)
    return data, sampling_rate
