from scipy.stats import logistic
import numpy as np


def radical_est_logist_val(mu, lmbd, delta, sample):
    i = 0
    sample_size = len(sample)

    if delta == 0:
        result = 1
        while i < sample_size:
            result = result * logistic.pdf(sample[i], mu, lmbd)
            i = i + 1
        return result

    tmp_1 = pow(delta, 2.0) / (1.0 + delta)
    left_multiplier = pow(lmbd, tmp_1) / delta
    right_multiplier = 0

    while i < sample_size:
        right_multiplier = right_multiplier + pow(logistic.pdf(sample[i], mu, lmbd), delta)
        i = i + 1

    return left_multiplier * right_multiplier


def gradient_radical_est_logis(mu, lmbd, delta, sample, diff):
    q = radical_est_logist_val(mu, lmbd, delta, sample)
    q_increased_mu = radical_est_logist_val(mu + mu * diff, lmbd, delta, sample)
    q_increased_lmbd = radical_est_logist_val(mu, lmbd + lmbd * diff, delta, sample)

    dq_by_dmu = (q_increased_mu - q) / (mu * diff)
    dq_by_dlmbd = (q_increased_lmbd - q) / (lmbd * diff)

    return np.array([dq_by_dmu, dq_by_dlmbd])


def radical_est_logist_step(mu, lmbd, delta, sample, diff):
    gradient = gradient_radical_est_logis(mu, lmbd, delta, sample, diff)

    beta = 1
    params = np.array([mu, lmbd])
    q_val = radical_est_logist_val(mu, lmbd, delta, sample)
    iteration_count = 10
    i = 0

    while i < iteration_count:
        new_params = params + beta * gradient
        q_new_val = radical_est_logist_val(new_params[0], new_params[1], delta, sample)
        if q_new_val > q_val:
            return new_params
        else:
            i = i + 1
            beta = beta / 2.0

    return params
