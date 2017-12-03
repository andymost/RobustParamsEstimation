import numpy as np

from generation import generate_logist
from radical_estimation import radical_est_logist_val, radical_est_logist_step

np.random.seed(1)   # init random state for reproductional

# MonteCarlo params
tau_mu = 0
tau_lmbd = 0
M = 100
j = 0

# Estimation params
step_diff = 1 / pow(10, 15)
deriative_diff = 0.05
delta = 2
start_params = [0.1, 1.1]

# Generation params
noize_level = 0.4
mu_n = 2.0
lmbd_n = 1.2
mu_t = 0.0
lmbd_t = 1.0
sample_size = 100

while j < M:
    sample = generate_logist(sample_size, mu_t, lmbd_t, noize_level, mu_n, lmbd_n)

    old_params = np.array(start_params)
    current_q = radical_est_logist_val(old_params[0], old_params[1], delta, sample)
    current_step_diff = 1

    while current_step_diff > step_diff:
        step_params = radical_est_logist_step(old_params[0], old_params[1], delta, sample, deriative_diff)
        current_step_diff = np.linalg.norm(old_params - step_params)
        old_params = step_params

    tau_mu = tau_mu + abs(old_params[0] - mu_t)/M
    tau_lmbd = tau_lmbd + abs(old_params[1] - lmbd_t)/M

    j = j + 1
    print j

print tau_mu
print tau_lmbd