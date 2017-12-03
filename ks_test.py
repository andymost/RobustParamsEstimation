from scipy.stats import kstest


def ks_around_logist(sample, mu, lmbd):
    return kstest(sample, 'logistic', (mu, lmbd))[1]