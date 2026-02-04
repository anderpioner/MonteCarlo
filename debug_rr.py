import numpy as np
from simulation import get_lognormal_params, lognormal_clipped_mean

def test(median, base_avg, outlier_freq):
    mu, sigma = get_lognormal_params(median, base_avg, outlier_freq/100.0)
    sim_mean = lognormal_clipped_mean(mu, sigma, 60.0)
    print(f"Median: {median}, BaseAvg: {base_avg}, OutlierFreq: {outlier_freq}%")
    print(f"Result -> mu: {mu:.3f}, sigma: {sigma:.3f}, Simulated Mean: {sim_mean:.2f}")
    print("-" * 30)

print("Test 1: Default values")
test(5.0, 5.17, 10.0)

print("Test 2: Increase Outlier Freq to 20%")
test(5.0, 5.17, 20.0)

print("Test 3: Decrease Outlier Freq to 2%")
test(5.0, 5.17, 2.0)

print("Test 4: High Base Avg (Overrides Outlier Freq?)")
test(5.0, 6.5, 10.0)
test(5.0, 6.5, 2.0)
