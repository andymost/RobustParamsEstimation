import numpy as np


def generate_logist(size, mu, lmbd, noize=0, mu_n=0, lmbd_n=0):
    if noize == 0 :
        return np.random.logistic(mu, lmbd, size)
    else:
        noize_size = int(size * noize)
        true_size = size - noize_size

        noize_sample = np.random.logistic(mu_n, lmbd_n, noize_size)
        true_sample = np.random.logistic(mu, lmbd, true_size)

        sample = np.append(noize_sample, [true_sample])
        np.random.shuffle(sample)

        return sample

