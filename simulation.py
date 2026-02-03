import numpy as np
import pandas as pd

def run_monte_carlo(starting_balance, trades_per_sim, win_rate, rr_ratio, risk_per_trade, num_simulations=1000):
    """
    Runs Monte Carlo simulations. 
    win_rate can be a scalar, a vector (num_simulations,), or a matrix (num_simulations, trades_per_sim)
    rr_ratio can be a scalar OR a matrix (num_simulations, trades_per_sim)
    """
    results = np.zeros((num_simulations, trades_per_sim + 1))
    results[:, 0] = starting_balance
    
    # Check dimensions for win_rate
    if isinstance(win_rate, np.ndarray):
        if win_rate.ndim == 1:
            # One win rate per path (p fixed for each simulation run)
            # Expand to (num_simulations, 1) to use broadcasting
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
            # win_rate is (num_simulations, trades_per_sim)
            random_vals = np.random.random(size=num_simulations)
            outcomes = np.where(random_vals < win_rate[:, i-1], 1, -1)
        elif wr_vector is not None:
            # win_rate is a scalar or (num_simulations, 1) vector
            random_vals = np.random.random(size=num_simulations)
            current_p = wr_vector if not isinstance(wr_vector, np.ndarray) else wr_vector.flatten()
            outcomes = np.where(random_vals < current_p, 1, -1)
        else:
            # Fallback (should not happen with new logic)
            outcomes = np.random.choice([1, -1], size=num_simulations, p=[win_rate, 1 - win_rate])
        
        if is_rr_dynamic:
            current_rr = rr_ratio[:, i-1]
            multipliers = np.where(outcomes == 1, 1 + (risk_per_trade * current_rr), 1 - risk_per_trade)
        else:
            multipliers = np.where(outcomes == 1, 1 + (risk_per_trade * rr_ratio), 1 - risk_per_trade)
        
        results[:, i] = results[:, i-1] * multipliers
        
    return results

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

def sample_beta_dist(alpha, beta, size):
    """
    Samples p ~ Beta(alpha, beta).
    Each sample is in (0, 1).
    """
    return np.random.beta(alpha, beta, size)

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
    from scipy.stats import norm
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
    from scipy.stats import norm
    if sigma <= 0: return min(np.exp(mu), U)
    
    phi_d1 = norm.cdf((np.log(U) - mu - sigma**2) / sigma)
    phi_d0 = norm.cdf((np.log(U) - mu) / sigma)
    
    unclipped_mean = np.exp(mu + (sigma**2 / 2))
    
    return unclipped_mean * phi_d1 + U * (1 - phi_d0)

def get_cond_mean_bounds(mu, T=10):
    """
    Finds the mathematical range [min, max] of E[X | X < T] for a fixed mu.
    """
    from scipy.optimize import minimize_scalar
    
    # Maximize E[X | X < T]
    res_max = minimize_scalar(lambda s: -lognormal_cond_mean(mu, s, T), bounds=(0.01, 10.0), method='bounded')
    max_val = -res_max.fun
    
    # For Log-Normal, as sigma -> 0, E[X|X<T] -> Median (if Median < T)
    # As sigma -> infinity, E[X|X<T] -> 0
    # So the range is effectively (0, max_val] if we consider all sigmas.
    # However, for very small sigma, the mean is Median.
    min_at_tiny_sigma = lognormal_cond_mean(mu, 0.001, T)
    
    return min_at_tiny_sigma, max_val

def get_lognormal_params(median, mean_no_outliers, prob_gt_10=None):
    """
    Converts Median and Conditional Mean (mean of values < 10) to Log-Normal mu and sigma.
    If prob_gt_10 (probability R:R > 10) is provided, it adjusts sigma to ensure the tail is fat enough.
    """
    mu = np.log(median)
    
    # 1. Sigma derived from Mean WITHOUT outliers (X < 10)
    # We solve: lognormal_cond_mean(mu, sigma, 10) = mean_no_outliers
    from scipy.optimize import brentq
    
    # Check mathematical limits
    min_p, max_p = get_cond_mean_bounds(mu, 10)
    
    def objective(s):
        return lognormal_cond_mean(mu, s, 10) - mean_no_outliers
    
    try:
        if mean_no_outliers >= max_p:
            # If requested mean is too high, use sigma that gets closest to max
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(lambda s: -lognormal_cond_mean(mu, s, 10), bounds=(0.01, 10.0), method='bounded')
            sigma_mean = res.x
        elif mean_no_outliers <= min_p:
            # If requested mean is too low for small sigma, it must be because sigma is LARGE (tail hollowing)
            # OR requested mean is simply very small. We search in the large sigma range.
            if mean_no_outliers <= 0.01: 
                sigma_mean = 10.0
            else:
                # Search for sigma > sigma_at_max to pull the mean down
                sigma_mean = brentq(objective, 1.0, 20.0)
        else:
            # Solution exists between 0.001 and the peak
            sigma_mean = brentq(objective, 0.001, 10.0)
    except Exception:
        sigma_mean = 0.5 # Fallback
        
    # 2. Sigma derived from Probability > 10:
    # P(X > 10) = 1 - Phi((ln(10) - mu) / sigma) = prob_gt_10
    if prob_gt_10 is not None and prob_gt_10 > 0:
        from scipy.stats import norm
        target_z = norm.ppf(1 - prob_gt_10)
        if target_z != 0:
            sigma_calib = (np.log(10) - mu) / target_z
        else:
            sigma_calib = 0
    else:
        sigma_calib = 0
        
    # Pick the most conservative (fattest tail) sigma
    sigma = max(sigma_mean, sigma_calib)

    return mu, sigma

def sample_lognormal_dist(mu, sigma, size, clip_min=0.1, clip_max=200.0):
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
