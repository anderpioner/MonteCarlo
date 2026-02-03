import numpy as np
from simulation import sample_beta_dist

def test_clipping():
    alpha, beta = 2, 5  # Mean approx 0.28
    size = 10000
    clip_min = 0.25
    clip_max = 0.30
    
    samples = sample_beta_dist(alpha, beta, size, clip_min=clip_min, clip_max=clip_max)
    
    actual_min = np.min(samples)
    actual_max = np.max(samples)
    
    print(f"Target Constraints: [{clip_min}, {clip_max}]")
    print(f"Actual Range: [{actual_min:.4f}, {actual_max:.4f}]")
    
    assert actual_min >= clip_min, f"Error: Min value {actual_min} is below {clip_min}"
    assert actual_max <= clip_max, f"Error: Max value {actual_max} is above {clip_max}"
    print("Verification SUCCESS: Clipping is working correctly.")

if __name__ == "__main__":
    test_clipping()
