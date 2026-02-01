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
    Fórmula:
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

def get_lognormal_params(median, mean, prob_gt_10=None):
    """
    Converts Median and Mean to Log-Normal mu and sigma.
    If prob_gt_10 (probability R:R > 10) is provided, it adjusts sigma to ensure the tail is fat enough.
    """
    mu = np.log(median)
    
    # 1. Sigma derived from Mean: Mean = Median * exp(sigma^2 / 2)
    if mean > median:
        sigma_mean = np.sqrt(2 * np.log(mean / median))
    else:
        sigma_mean = 0.5
        
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
        
    # Pick the most conservative (fattest tail) sigma that satisfies the "outlier capture" spirit
    # Usually, if user says "Agressive", they want a very fat tail.
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
