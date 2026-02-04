
import numpy as np
from simulation import run_monte_carlo

# Constants
start_balance = 10000
trades_per_sim = 10
risk_per_trade = 0.01

# Case 1: All wins have 0 R:R
win_rate = 1.0 # 100% win rate
rr_ratio = np.zeros((1, trades_per_sim)) # All R:R are 0

results, outcomes_matrix = run_monte_carlo(start_balance, trades_per_sim, win_rate, rr_ratio, risk_per_trade, num_simulations=1)

print(f"Outcomes: {outcomes_matrix}")
print(f"Final Balance: {results[0, -1]}")

# Calculate Win Average metric (as done in app)
set_win_rate = (np.sum(outcomes_matrix == 1) / trades_per_sim) * 100
print(f"Win Average (%): {set_win_rate}%")

# Calculate Loss Streak (as done in app)
path = results[0]
pnl = np.diff(path)
is_loss = (pnl < 0).astype(int)
print(f"Is Loss Sequence: {is_loss}")
print(f"Consecutive Losses: {np.sum(is_loss)}")
