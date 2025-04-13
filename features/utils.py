# Helper functions: z-score, rolling stats, ranking, feature engineering utilities
"""
utils.py
-----------------------------------------------------
High-performance feature engineering utilities for
rolling statistics, normalization, ranking, and 
cross-symbol operations. All functions are NumPy/
Numba/Bottleneck powered — optimized for real-time
and backtest-safe feature generation.
-----------------------------------------------------
"""

import numpy as np
import bottleneck as bn
from scipy.stats import rankdata
from numba import njit, prange
import pandas as pd
from typing import Tuple, Optional, Union, List
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("utils")

# -------------------------------
# Z-Score (rolling)
# -------------------------------

def rolling_zscore(x: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    """
    Rolling z-score (mean/std normalized) using bottleneck
    
    Args:
        x: Input array
        window: Rolling window size
        min_periods: Minimum number of observations required
    
    Returns:
        Z-score normalized array of same size as input
    """
    # Handle NaN values
    has_nan = np.isnan(x).any()
    if has_nan:
        # Create a copy so we don't modify the input
        x_clean = np.copy(x)
        nan_indices = np.isnan(x_clean)
        x_clean[nan_indices] = 0  # Temporarily replace NaNs with 0
        
        # Compute moving mean and std
        mean = bn.move_mean(x_clean, window=window, min_count=min_periods)
        std = bn.move_std(x_clean, window=window, min_count=min_periods)
        
        # Replace zeros with tiny epsilon to avoid divide-by-zero
        std = np.where(std < 1e-8, 1e-8, std)
        
        # Compute the z-score
        z = (x_clean - mean) / std
        
        # Restore NaN where they were
        z[nan_indices] = np.nan
    else:
        # Standard case without NaN values
        mean = bn.move_mean(x, window=window, min_count=min_periods)
        std = bn.move_std(x, window=window, min_count=min_periods)
        
        # Replace zeros with tiny epsilon to avoid divide-by-zero
        std = np.where(std < 1e-8, 1e-8, std)
        z = (x - mean) / std
    
    return z

# -------------------------------
# Rolling Percentile Rank
# -------------------------------

def rolling_percentile_rank(x: np.ndarray, window: int) -> np.ndarray:
    """
    Percentile rank within a rolling window
    
    Args:
        x: Input array
        window: Rolling window size
    
    Returns:
        Percentile rank (0-1) of each point relative to its window
    """
    out = np.full_like(x, fill_value=np.nan, dtype=np.float32)
    
    # Skip computation for windows with insufficient data
    if len(x) < window:
        return out
    
    for i in range(window - 1, len(x)):
        window_slice = x[i - window + 1:i + 1]
        
        # Skip windows with NaN values
        if np.isnan(window_slice).any():
            continue
            
        # Calculate percentile rank
        out[i] = rankdata(window_slice, method='average')[-1] / window
    
    return out

# -------------------------------
# Rolling Correlation (Numba)
# -------------------------------

@njit
def rolling_correlation(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling Pearson correlation between two arrays
    
    Args:
        a: First array
        b: Second array
        window: Rolling window size
    
    Returns:
        Rolling correlation coefficient (-1 to 1)
    """
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float32)
    
    # Fast path for short arrays
    if n < window:
        return out
    
    for i in range(window - 1, n):
        x = a[i - window + 1:i + 1]
        y = b[i - window + 1:i + 1]
        
        # Skip if we have NaN values
        has_nan_x = False
        has_nan_y = False
        
        for j in range(len(x)):
            if np.isnan(x[j]):
                has_nan_x = True
                break
            if np.isnan(y[j]):
                has_nan_y = True
                break
                
        if has_nan_x or has_nan_y:
            continue
            
        # Calculate means
        x_mean = 0.0
        y_mean = 0.0
        
        for j in range(len(x)):
            x_mean += x[j]
            y_mean += y[j]
            
        x_mean /= len(x)
        y_mean /= len(y)
        
        # Calculate correlation components
        numerator = 0.0
        x_variance = 0.0
        y_variance = 0.0
        
        for j in range(len(x)):
            x_diff = x[j] - x_mean
            y_diff = y[j] - y_mean
            
            numerator += x_diff * y_diff
            x_variance += x_diff * x_diff
            y_variance += y_diff * y_diff
        
        # Check for valid variances
        if x_variance < 1e-8 or y_variance < 1e-8:
            out[i] = 0.0
        else:
            out[i] = numerator / np.sqrt(x_variance * y_variance)
    
    return out

# -------------------------------
# Log Returns
# -------------------------------

def log_return(x: np.ndarray) -> np.ndarray:
    """
    Compute log returns: log(price_t / price_{t-1})
    
    Args:
        x: Array of prices
    
    Returns:
        Array of log returns (length = len(x) - 1)
    """
    # Handle NaN values
    result = np.full(len(x) - 1, np.nan, dtype=np.float32)
    
    # Process each pair of consecutive values
    for i in range(1, len(x)):
        if not np.isnan(x[i]) and not np.isnan(x[i-1]) and x[i-1] > 0:
            result[i-1] = np.log(x[i] / x[i-1])
    
    return result

def padded_log_return(x: np.ndarray) -> np.ndarray:
    """
    Same as log_return, but pads to keep same length as input
    
    Args:
        x: Array of prices
    
    Returns:
        Array of log returns (length = len(x))
    """
    result = np.full_like(x, fill_value=0, dtype=np.float32)
    
    # Skip computation if too short
    if len(x) <= 1:
        return result
    
    # Process each pair of consecutive values
    for i in range(1, len(x)):
        if not np.isnan(x[i]) and not np.isnan(x[i-1]) and x[i-1] > 0:
            result[i] = np.log(x[i] / x[i-1])
    
    return result

# -------------------------------
# Safe Ratio
# -------------------------------

def safe_divide(numerator: np.ndarray, denominator: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Avoid divide-by-zero when computing ratios
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        epsilon: Small value to add to denominator
    
    Returns:
        Safe division result
    """
    # Handle scalar inputs
    if np.isscalar(numerator) and np.isscalar(denominator):
        return numerator / (denominator + epsilon) if denominator != 0 else 0
    
    # Handle NaN values
    result = np.zeros_like(numerator, dtype=np.float32)
    
    # Handle arrays of different shapes
    if np.isscalar(numerator):
        numerator_array = np.full_like(denominator, numerator)
    else:
        numerator_array = numerator
        
    if np.isscalar(denominator):
        denominator_array = np.full_like(numerator, denominator)
    else:
        denominator_array = denominator
    
    # Perform safe division element-wise
    for i in range(len(result)):
        num = numerator_array[i] if i < len(numerator_array) else 0
        den = denominator_array[i] if i < len(denominator_array) else 0
        
        # Handle NaN
        if np.isnan(num) or np.isnan(den):
            result[i] = 0
        else:
            result[i] = num / (den + epsilon) if abs(den) < epsilon else num / den
    
    return result

# -------------------------------
# Candle Wick/Body Ratios
# -------------------------------

def candle_ratios(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute common candlestick shape ratios
    
    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices
    
    Returns:
        Tuple of (body_ratio, upper_wick_ratio, lower_wick_ratio, range_ratio)
    """
    # Initialize output arrays
    n = len(open_)
    body_ratio = np.zeros(n, dtype=np.float32)
    upper_wick_ratio = np.zeros(n, dtype=np.float32)
    lower_wick_ratio = np.zeros(n, dtype=np.float32)
    range_ratio = np.zeros(n, dtype=np.float32)
    
    # Process each candle
    for i in range(n):
        # Skip if any component is NaN
        if (np.isnan(open_[i]) or np.isnan(high[i]) or 
            np.isnan(low[i]) or np.isnan(close[i])):
            continue
            
        # Calculate ratios
        body = abs(close[i] - open_[i])
        candle_range = high[i] - low[i]
        
        if candle_range > 0:
            body_ratio[i] = body / candle_range
            upper_wick = high[i] - max(open_[i], close[i])
            lower_wick = min(open_[i], close[i]) - low[i]
            
            upper_wick_ratio[i] = upper_wick / candle_range
            lower_wick_ratio[i] = lower_wick / candle_range
        
        # For range_ratio, need previous close
        if i > 0 and not np.isnan(close[i-1]) and close[i-1] > 0:
            range_ratio[i] = candle_range / close[i-1]

    return body_ratio, upper_wick_ratio, lower_wick_ratio, range_ratio

# -------------------------------
# Rolling Volatility
# -------------------------------

def rolling_volatility(returns: np.ndarray, window: int = 20, annualize: bool = False, min_periods: int = 1) -> np.ndarray:
    """
    Standard deviation of returns over a rolling window
    
    Args:
        returns: Array of returns
        window: Rolling window size
        annualize: Whether to annualize (multiply by sqrt(365*24*4) for 15m bars)
        min_periods: Minimum number of observations required
    
    Returns:
        Rolling volatility
    """
    # Create a copy to avoid modifying the input
    returns_clean = np.copy(returns)
    
    # Handle NaN values
    nan_mask = np.isnan(returns_clean)
    if nan_mask.any():
        returns_clean[nan_mask] = 0
    
    # Compute rolling standard deviation
    vol = bn.move_std(returns_clean, window=window, min_count=min_periods)
    
    # Annualize if requested
    if annualize:
        # For 15-min bars, there are 4*24*365 = 35040 bars per year
        vol = vol * np.sqrt(35040 / window)
    
    # Restore NaN values
    vol[nan_mask] = np.nan
        
    return vol

# -------------------------------
# Exponential Moving Average
# -------------------------------

def ema(x: np.ndarray, span: int) -> np.ndarray:
    """
    Exponential moving average
    
    Args:
        x: Input array
        span: EMA span parameter
    
    Returns:
        EMA values
    """
    alpha = 2 / (span + 1)
    ema_values = np.full_like(x, fill_value=np.nan, dtype=np.float32)
    
    # Find first non-NaN value
    first_valid = 0
    while first_valid < len(x) and np.isnan(x[first_valid]):
        first_valid += 1
    
    if first_valid >= len(x):
        return ema_values  # All values are NaN
    
    ema_values[first_valid] = x[first_valid]
    
    for i in range(first_valid + 1, len(x)):
        if np.isnan(x[i]):
            ema_values[i] = ema_values[i-1]  # Maintain previous EMA value
        else:
            ema_values[i] = x[i] * alpha + ema_values[i-1] * (1 - alpha)
    
    return ema_values

# -------------------------------
# Cross-Correlation Matrix
# -------------------------------

def compute_correlation_matrix(returns_dict: dict, window: int = 96) -> pd.DataFrame:
    """
    Compute correlation matrix for multiple instruments
    
    Args:
        returns_dict: Dictionary of {instrument: returns_array}
        window: Rolling window for correlation
    
    Returns:
        DataFrame with correlation matrix
    """
    instruments = list(returns_dict.keys())
    n = len(instruments)
    
    # Create empty correlation matrix
    corr_matrix = pd.DataFrame(np.eye(n), index=instruments, columns=instruments)
    
    # Compute correlations between each pair
    for i in range(n):
        for j in range(i+1, n):
            instr_i = instruments[i]
            instr_j = instruments[j]
            
            ret_i = returns_dict[instr_i]
            ret_j = returns_dict[instr_j]
            
            # Only compute if we have enough data
            min_len = min(len(ret_i), len(ret_j))
            if min_len >= window:
                # Take the last window elements for correlation
                i_slice = ret_i[-window:]
                j_slice = ret_j[-window:]
                
                # Skip if we have NaN values
                if np.isnan(i_slice).any() or np.isnan(j_slice).any():
                    continue
                    
                corr = np.corrcoef(i_slice, j_slice)[0, 1]
                corr_matrix.loc[instr_i, instr_j] = corr
                corr_matrix.loc[instr_j, instr_i] = corr
    
    return corr_matrix

# -------------------------------
# Market-Neutral Returns
# -------------------------------

def compute_residual_returns(returns: np.ndarray, market_returns: np.ndarray, window: int = 96) -> np.ndarray:
    """
    Compute market-neutral (residual) returns
    
    Args:
        returns: Instrument returns
        market_returns: Market (e.g., BTC) returns
        window: Window for beta estimation
    
    Returns:
        Residual returns after removing market factor
    """
    residual_returns = np.full_like(returns, fill_value=0, dtype=np.float32)
    
    # Skip if we don't have enough data
    if len(returns) < window or len(market_returns) < window:
        return residual_returns
    
    for i in range(window, len(returns)):
        # Extract window
        r_window = returns[i-window:i]
        m_window = market_returns[i-window:i]
        
        # Skip windows with NaN
        if np.isnan(r_window).any() or np.isnan(m_window).any():
            continue
        
        # Compute beta using covariance and variance
        cov = np.cov(r_window, m_window)[0, 1]
        var = np.var(m_window)
        
        if var > 1e-8:
            beta = cov / var
            residual_returns[i] = returns[i] - beta * market_returns[i]
        else:
            residual_returns[i] = returns[i]
    
    return residual_returns

# -------------------------------
# Crossover Signals
# -------------------------------

def crossover_signal(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """
    Generate crossover signals (0=bearish, 1=neutral, 2=bullish)
    
    Args:
        fast: Fast line (e.g., price, short MA)
        slow: Slow line (e.g., long MA)
    
    Returns:
        Crossover signals
    """
    if len(fast) != len(slow):
        raise ValueError(f"Arrays must be same length: fast={len(fast)}, slow={len(slow)}")
        
    signal = np.ones(len(fast), dtype=np.int8)  # Default is 1 (neutral)
    
    # Skip if too short
    if len(fast) <= 1:
        return signal
    
    # Handle NaN values
    fast_clean = np.copy(fast)
    slow_clean = np.copy(slow)
    
    # Replace NaN with 0 to avoid comparison issues
    fast_clean[np.isnan(fast_clean)] = 0
    slow_clean[np.isnan(slow_clean)] = 0
    
    # Previous state: fast > slow?
    prev_state = fast_clean[:-1] > slow_clean[:-1]
    # Current state: fast > slow?
    curr_state = fast_clean[1:] > slow_clean[1:]
    
    # Bullish crossover: previous fast <= slow and current fast > slow
    bullish = (~prev_state) & curr_state
    # Bearish crossover: previous fast >= slow and current fast < slow
    bearish = prev_state & (~curr_state)
    
    # Set signals (shift by 1 to align with current bar)
    signal[1:][bullish] = 2  # Bullish (was 1, now 2)
    signal[1:][bearish] = 0  # Bearish (was -1, now 0)
    
    return signal

# -------------------------------
# Safe execution helper
# -------------------------------

def safe_execute(func, *args, **kwargs):
    """
    Safely execute a function with proper error tracking
    
    Args:
        func: Function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple of (result, None) if successful, (None, (error_message, error_traceback)) if failed
    """
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        error_msg = f"Error executing {func.__name__}: {str(e)}"
        error_trace = traceback.format_exc()
        logger.error(error_msg)
        logger.error(error_trace)
        return None, (error_msg, error_trace)