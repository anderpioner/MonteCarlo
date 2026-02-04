
import numpy as np
from simulation import get_beta_params, sample_beta_dist

wr_avg = 0.28
wr_vol = 0.06
wr_min_p = 0.143
wr_max_p = 0.447

wr_alpha, wr_beta = get_beta_params(wr_avg, wr_vol)
print(f"Alpha: {wr_alpha}, Beta: {wr_beta}")

samples = sample_beta_dist(wr_alpha, wr_beta, 100000, clip_min=wr_min_p, clip_max=wr_max_p)
print(f"Min sample: {np.min(samples)}")
print(f"Max sample: {np.max(samples)}")
print(f"Mean sample: {np.mean(samples)}")

# Check if any exceed wr_max_p
over = np.sum(samples > wr_max_p)
print(f"Number of samples > wr_max_p: {over}")
