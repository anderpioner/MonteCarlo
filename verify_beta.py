import numpy as np
import sys
import os

# Add parent directory to path to import simulation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', 'D', 'Python', 'Montecarlo')))
from simulation import get_beta_params, sample_beta_dist

def test_beta_params():
    print("Testing get_beta_params...")
    
    # Example 1: Mean 22%, Std 6%
    a, b = get_beta_params(0.22, 0.06)
    print(f"Mean 22%, Std 6% -> Alpha: {a:.2f}, Beta: {b:.2f}")
    assert a is not None and a > 0
    
    # Example 2: Impossible Std
    a_imp, b_imp = get_beta_params(0.20, 0.50)
    print(f"Mean 20%, Std 50% -> Alpha: {a_imp}, Beta: {b_imp} (Expected: None)")
    assert a_imp is None
    
    # Example 3: Zero Std
    a_zero, b_zero = get_beta_params(0.5, 0)
    print(f"Mean 50%, Std 0% -> Alpha: {a_zero}, Beta: {b_zero} (Expected: High values)")
    assert a_zero == 100
    
    print("Testing sample_beta_dist...")
    samples = sample_beta_dist(a, b, 10000)
    print(f"Sample Mean: {np.mean(samples):.4f} (Expected ~0.22)")
    print(f"Sample Std: {np.std(samples):.4f} (Expected ~0.06)")
    assert np.all(samples >= 0) and np.all(samples <= 1)
    
    print("\nVerification successful!")

if __name__ == "__main__":
    test_beta_params()
