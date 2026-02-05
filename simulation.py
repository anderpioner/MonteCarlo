import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar

def run_monte_carlo(starting_balance, trades_per_sim, win_rate, rr_ratio, risk_per_trade, num_simulations=1000):
    """
    Runs Monte Carlo simulations. 
    win_rate can be a scalar, a vector (num_simulations,), or a matrix (num_simulations, trades_per_sim)
    rr_ratio can be a scalar OR a matrix (num_simulations, trades_per_sim)
    Returns: (results, outcomes_matrix)
    """
    results = np.zeros((num_simulations, trades_per_sim + 1))
    outcomes_matrix = np.zeros((num_simulations, trades_per_sim))
    results[:, 0] = starting_balance
    
    # Check dimensions for win_rate
    if isinstance(win_rate, np.ndarray):
        if win_rate.ndim == 1:
            # One win rate per path (p fixed for each simulation run)
            wr_vector = win_rate.reshape(-1, 1)
        else:
            # Dynamic win rate per trade (matrix)
            wr_vector = None
    else:
        # Scalar win rate
        wr_vector = win_rate

    is_rr_dynamic = isinstance(rr_ratio, np.ndarray)

    for i in range(1, trades_per_sim + 1):
        if isinstance(win_rate, np.ndarray) and win_rate.ndim == 2:
            random_vals = np.random.random(size=num_simulations)
            outcomes = np.where(random_vals < win_rate[:, i-1], 1, -1)
        elif wr_vector is not None:
            random_vals = np.random.random(size=num_simulations)
            current_p = wr_vector if not isinstance(wr_vector, np.ndarray) else wr_vector.flatten()
            outcomes = np.where(random_vals < current_p, 1, -1)
        else:
            outcomes = np.random.choice([1, -1], size=num_simulations, p=[win_rate, 1 - win_rate])
        
        outcomes_matrix[:, i-1] = outcomes

        if is_rr_dynamic:
            current_rr = rr_ratio[:, i-1]
            multipliers = np.where(outcomes == 1, 1 + (risk_per_trade * current_rr), 1 - risk_per_trade)
        else:
            multipliers = np.where(outcomes == 1, 1 + (risk_per_trade * rr_ratio), 1 - risk_per_trade)
        
        results[:, i] = results[:, i-1] * multipliers
        
    return results, outcomes_matrix

def get_beta_params(mean, std):
    """
    Converts Mean (mu) and Std Dev (sigma) to Beta parameters Alpha and Beta.
    Formula:
    alpha = (mu^2 * (1-mu) / sigma^2) - mu
    beta = alpha * (1-mu) / mu
    """
    if std <= 0:
        return 100, 100 # Very high alpha/beta = almost fixed value
    
    # Validation: sigma^2 must be < mu * (1 - mu)
    max_var = mean * (1 - mean)
    if (std ** 2) >= max_var:
        return None, None
        
    alpha = ( (mean**2 * (1 - mean)) / (std**2) ) - mean
    beta = alpha * (1 - mean) / mean
    
    # Clip to avoid degenerate distributions (rule 6)
    alpha = max(alpha, 0.5)
    beta = max(beta, 0.5)
    
    return alpha, beta

def get_beta_params_from_pert(min_val, mode_val, max_val):
    """
    Derived using PERT (Program Evaluation and Review Technique) estimation.
    Mean = (Min + 4*Mode + Max) / 6
    Std = (Max - Min) / 6
    
    Then converts Mean/Std to Beta Alpha/Beta.
    """
    if not (min_val <= mode_val <= max_val):
        # Fallback to mean if invalid ordering
        mean = mode_val
        std = (max_val - min_val) / 6 if max_val > min_val else 0.01
    else:
        mean = (min_val + 4 * mode_val + max_val) / 6
        std = (max_val - min_val) / 6
        
    return get_beta_params(mean, std)

def sample_beta_dist(alpha, beta, size, clip_min=None, clip_max=None):
    """
    Samples p ~ Beta(alpha, beta).
    Each sample is in (0, 1).
    """
    samples = np.random.beta(alpha, beta, size)
    if clip_min is not None or clip_max is not None:
        samples = np.clip(samples, clip_min if clip_min is not None else 0.0, clip_max if clip_max is not None else 1.0)
    return samples

def sample_gamma_dist(shape, scale, loc, size, clip_min=None, clip_max=None):
    """
    Samples from a Gamma distribution.
    shape (k) controls 'skew', scale (theta) controls 'spread', loc is shift.
    Useful for both R:R (infinite tail) and Win Rate (clipped to 1.0).
    """
    samples = np.random.gamma(shape, scale, size) + loc
    
    if clip_min is not None or clip_max is not None:
        samples = np.clip(samples, clip_min if clip_min is not None else -np.inf, clip_max if clip_max is not None else np.inf)
        
    return samples

def lognormal_cond_mean(mu, sigma, T=10):
    """
    Calculates E[X | X < T] for X ~ LogNormal(mu, sigma).
    Formula: E[X | X < T] = E[X] * Phi((ln(T) - mu - sigma^2) / sigma) / Phi((ln(T) - mu) / sigma)
    where Phi is the standard normal CDF.
    """
    if sigma <= 0: return np.exp(mu)
    
    phi1 = norm.cdf((np.log(T) - mu - sigma**2) / sigma)
    phi2 = norm.cdf((np.log(T) - mu) / sigma)
    
    if phi2 < 1e-10: return np.exp(mu) # Fallback for extreme cases
    
    total_mean = np.exp(mu + (sigma**2 / 2))
    return total_mean * (phi1 / phi2)

def lognormal_clipped_mean(mu, sigma, U):
    """
    Calculates E[clip(X, 0, U)] for X ~ LogNormal(mu, sigma).
    This handles the impact of 'rr_max_cap' on the expectancy.
    """
    if sigma <= 0: return min(np.exp(mu), U)
    
    phi_d1 = norm.cdf((np.log(U) - mu - sigma**2) / sigma)
    phi_d0 = norm.cdf((np.log(U) - mu) / sigma)
    
    unclipped_mean = np.exp(mu + (sigma**2 / 2))
    
    return unclipped_mean * phi_d1 + U * (1 - phi_d0)

def get_cond_mean_bounds(mu, T=10):
    """
    Finds the mathematical range [min, max] of E[X | X < T] for a fixed mu.
    """
    
    # Maximize E[X | X < T]
    res_max = minimize_scalar(lambda s: -lognormal_cond_mean(mu, s, T), bounds=(0.01, 10.0), method='bounded')
    max_val = -res_max.fun
    
    # For Log-Normal, as sigma -> 0, E[X|X<T] -> Median (if Median < T)
    # As sigma -> infinity, E[X|X<T] -> 0
    # So the range is effectively (0, max_val] if we consider all sigmas.
    # However, for very small sigma, the mean is Median.
    min_at_tiny_sigma = lognormal_cond_mean(mu, 0.001, T)
    
    return min_at_tiny_sigma, max_val

def get_lognormal_params(median, mean_no_outliers, prob_gt_threshold=None, threshold=10.0):
    """
    Converts Median and Conditional Mean (mean of values < threshold) to Log-Normal mu and sigma.
    If prob_gt_threshold (probability R:R > threshold) is provided, it adjusts sigma to ensure the tail is fat enough.
    """
    mu = np.log(median)
    
    # 1. Sigma derived from Probability > threshold:
    # P(X > threshold) = 1 - Phi((ln(threshold) - mu) / sigma) = prob_gt_threshold
    sigma_calib = 0
    if prob_gt_threshold is not None and prob_gt_threshold > 0:
        # Cap prob to avoid extreme math errors if Median < threshold
        safe_prob = min(0.499, prob_gt_threshold) if median < threshold else prob_gt_threshold
        target_z = norm.ppf(1 - safe_prob)
        if abs(target_z) > 1e-9:
            sigma_calib = max(0.001, (np.log(threshold) - mu) / target_z)
    
    # 2. Sigma derived from Mean of normal trades (< threshold):
    # We need to find sigma such that lognormal_cond_mean(mu, sigma, threshold) == mean_no_outliers
    # The function is not monotonic: it starts at Median (sigma~0), goes to a peak, then to 0.
    
    sigma_mean = 0.5 
    # Find the peak to split the search range
    res_max = minimize_scalar(lambda s: -lognormal_cond_mean(mu, s, threshold), bounds=(0.01, 10.0), method='bounded')
    sigma_peak = res_max.x
    max_p = -res_max.fun
    min_p = lognormal_cond_mean(mu, 0.001, threshold)
    
    def objective(s):
        return lognormal_cond_mean(mu, s, threshold) - mean_no_outliers

    try:
        if mean_no_outliers >= max_p:
            # Target is at or above the peak, use the peak sigma
            sigma_mean = sigma_peak
        elif mean_no_outliers <= min_p:
            # Target is below the median, search in the fat-tail region (downward slope)
            # objective(sigma_peak) is positive (target - max_p), objective(20.0) is negative
            sigma_mean = brentq(objective, sigma_peak, 20.0)
        else:
            # Target is between Median and Peak, search in the thin-tail region (upward slope)
            # objective(0.001) is negative (5.0 - target), objective(sigma_peak) is positive
            sigma_mean = brentq(objective, 0.001, sigma_peak)
    except Exception:
        # Fallback if solver fails
        sigma_mean = 0.5 if mean_no_outliers > min_p else 2.0
        
    # Pick the most conservative (fattest tail) sigma
    sigma = max(sigma_mean, sigma_calib)

    return mu, sigma

def sample_lognormal_dist(mu, sigma, size, clip_min=0.00001, clip_max=200.0):
    """
    Samples from a Log-Normal distribution.
    """
    samples = np.random.lognormal(mu, sigma, size)
    return np.clip(samples, clip_min, clip_max)

def calculate_metrics(win_rate, rr_ratio, risk_per_trade):
    """
    Calculates EV and Optimal Kelly.
    """
    # EV = (Win Rate * Reward) - (Loss Rate * Risk)
    # Since Risk is 1 unit, EV = (Win Rate * RR) - (Loss Rate * 1)
    ev = (win_rate * rr_ratio) - ((1 - win_rate) * 1)
    
    # Kelly % = (Win Rate / Loss Rate) - (1 / RR)
    # or Kelly % = (p*b - q) / b where p=win_rate, q=loss_rate, b=rr_ratio
    if rr_ratio > 0:
        kelly = win_rate - ((1 - win_rate) / rr_ratio)
    else:
        kelly = 0
        
    return ev, max(0, kelly)
