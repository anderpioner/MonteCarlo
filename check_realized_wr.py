
import numpy as np
from simulation import get_beta_params, sample_beta_dist, run_monte_carlo

wr_avg = 0.28
wr_vol = 0.06
wr_min_p = 0.143
wr_max_p = 0.447
trades_per_sim = 50
num_sims = 1000

wr_alpha, wr_beta = get_beta_params(wr_avg, wr_vol)
wr_vector = sample_beta_dist(wr_alpha, wr_beta, num_sims, clip_min=wr_min_p, clip_max=wr_max_p)

# Run simulations
results, outcomes_matrix = run_monte_carlo(10000, trades_per_sim, wr_vector, 5.0, 0.003, num_sims)

# Calculate realized win rates
realized_wr = np.sum(outcomes_matrix == 1, axis=1) / trades_per_sim

print(f"Max p (parameter): {np.max(wr_vector)}")
print(f"Max Realized WR: {np.max(realized_wr)}")
print(f"Number of paths where Realized WR > wr_max_p: {np.sum(realized_wr > wr_max_p)}")
print(f"Average path Realized WR: {np.mean(realized_wr)}")
