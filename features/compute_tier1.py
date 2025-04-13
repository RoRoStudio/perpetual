# Compute model_features_15m_tier1: real-time safe feature matrix
#!/usr/bin/env python3
"""
compute_tier1.py
-----------------------------------------------------
Builds real-time-safe features for 15m bars for use
in model_features_15m_tier1. This table powers both
live inference and supervised training. All features
are computed from past-only data — no leakage.
-----------------------------------------------------
"""

# python -m features.compute_tier1 --threads 8
# python -m features.compute_tier1 --threads 16

import numpy as np
import pandas as pd
import logging
import argparse
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from configparser import ConfigParser
from concurrent.futures import ThreadPoolExecutor
import os
import traceback

from data.database import get_connection
from features.utils import (
    padded_log_return,
    rolling_zscore,
    rolling_volatility,
    rolling_correlation,
    candle_ratios,
    safe_divide,
    rolling_percentile_rank,
)

# --- Config
ROLLING_WINDOW_VOL = 4 * 4  # 1h window for 15m bars
ROLLING_WINDOW_ZSCORE = 96  # 24h window for z-score (96 x 15m bars)
# Read compute mode from config
config = ConfigParser()
config.read("/mnt/p/perpetual/config/config.ini")
COMPUTE_MODE = config.get("FEATURES", "COMPUTE_MODE", fallback="full_backfill")
ROLLING_WINDOW = int(config.get("FEATURES", "ROLLING_WINDOW", fallback="6000"))

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("compute_tier1")

# --- Helper: Normalize calendar time
def encode_calendar_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic encoding for time features and funding metadata"""
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # Fixed calculation for minutes to next funding time (00:00, 08:00, 16:00 UTC)
    # Handle each timestamp individually to avoid negative values
    funding_times = [0, 8, 16]  # Hours when funding occurs
    
    # Define a function to calculate minutes to next funding
    def get_mins_to_next_funding(row):
        hour = row['timestamp'].hour
        minute = row['timestamp'].minute
        
        # Find the next funding hour
        next_funding = next((h for h in funding_times if h > hour), funding_times[0] + 24)
        
        # Calculate minutes until next funding
        return (next_funding - hour) * 60 - minute
    
    # Apply the function to each row
    df['mins_to_next_funding'] = df.apply(get_mins_to_next_funding, axis=1).astype('int16')
    
    return df

# --- Fetch historical candles with cursor
def load_ohlcv_data(conn, instrument_name: str, limit=None) -> pd.DataFrame:
    """Load OHLCV data for a specific instrument, with optional limit"""
    try:
        with conn.cursor() as cur:
            sql = """
            SELECT timestamp, open, high, low, close, volume
            FROM historical_ohlcv_data
            WHERE instrument_name = %s
            ORDER BY timestamp
            """
            
            if limit and COMPUTE_MODE == 'rolling_update':
                sql = """
                SELECT timestamp, open, high, low, close, volume
                FROM historical_ohlcv_data
                WHERE instrument_name = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """
                cur.execute(sql, (instrument_name, limit))
            else:
                cur.execute(sql, (instrument_name,))
                
            # Get column names from cursor description
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchall()
            
            # Create dataframe
            df = pd.DataFrame(data, columns=columns)
            
            # Sort by timestamp if needed (for DESC queries)
            if limit and COMPUTE_MODE == 'rolling_update':
                df = df.sort_values('timestamp')
                
            return df
    except Exception as e:
        logger.error(f"Error loading OHLCV data for {instrument_name}: {e}")
        return pd.DataFrame()

# --- Fetch funding data with cursor
def load_funding_data(conn, instrument_name: str, limit=None) -> pd.DataFrame:
    """Load funding rate data for a specific instrument using cursor"""
    try:
        with conn.cursor() as cur:
            sql = """
            SELECT timestamp, interest_1h, interest_8h, index_price, prev_index_price
            FROM historical_funding_rates
            WHERE instrument_name = %s
            ORDER BY timestamp
            """
            
            if limit and COMPUTE_MODE == 'rolling_update':
                sql = """
                SELECT timestamp, interest_1h, interest_8h, index_price, prev_index_price
                FROM historical_funding_rates
                WHERE instrument_name = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """
                cur.execute(sql, (instrument_name, limit))
            else:
                cur.execute(sql, (instrument_name,))
                
            # Get column names from cursor description
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchall()
            
            # Create dataframe
            df = pd.DataFrame(data, columns=columns)
            
            # Sort by timestamp if needed (for DESC queries)
            if limit and COMPUTE_MODE == 'rolling_update':
                df = df.sort_values('timestamp')
                
            return df
    except Exception as e:
        logger.error(f"Error loading funding data for {instrument_name}: {e}")
        return pd.DataFrame()

# --- Fetch volatility index with cursor
def load_vol_index_data(conn, currency: str, limit=None) -> pd.DataFrame:
    """Load volatility index data for BTC or ETH using cursor"""
    try:
        with conn.cursor() as cur:
            sql = """
            SELECT timestamp, open, high, low, close
            FROM historical_volatility_index
            WHERE currency = %s
            ORDER BY timestamp
            """
            
            if limit and COMPUTE_MODE == 'rolling_update':
                sql = """
                SELECT timestamp, open, high, low, close
                FROM historical_volatility_index
                WHERE currency = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """
                cur.execute(sql, (currency, limit))
            else:
                cur.execute(sql, (currency,))
                
            # Get column names from cursor description
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchall()
            
            # Create dataframe
            df = pd.DataFrame(data, columns=columns)
            
            # Sort by timestamp if needed (for DESC queries)
            if limit and COMPUTE_MODE == 'rolling_update':
                df = df.sort_values('timestamp')
                
            return df
    except Exception as e:
        logger.error(f"Error loading volatility data for {currency}: {e}")
        return pd.DataFrame()

# --- Load BTC/ETH data for cross-asset features
def load_btc_eth_data(conn, limit=None):
    """Load BTC and ETH data for cross-asset features (real-time safe)"""
    try:
        # Load BTC and ETH OHLCV
        try:
            btc_ohlcv = load_ohlcv_data(conn, 'BTC_USDC-PERPETUAL', limit)
        except Exception as e:
            logger.warning(f"⚠️ Could not load BTC data: {e}")
            btc_ohlcv = pd.DataFrame()
        
        try:
            eth_ohlcv = load_ohlcv_data(conn, 'ETH_USDC-PERPETUAL', limit)
        except Exception as e:
            logger.warning(f"⚠️ Could not load ETH data: {e}")
            eth_ohlcv = pd.DataFrame()

        # BTC returns
        if not btc_ohlcv.empty:
            btc_returns = btc_ohlcv[['timestamp', 'close']].copy()
            btc_returns['btc_return_1bar'] = padded_log_return(btc_returns['close'].values)
            btc_returns = btc_returns[['timestamp', 'btc_return_1bar']]
        else:
            btc_returns = pd.DataFrame(columns=['timestamp', 'btc_return_1bar'])

        # ETH returns
        if not eth_ohlcv.empty:
            eth_returns = eth_ohlcv[['timestamp', 'close']].copy()
            eth_returns['eth_return_1bar'] = padded_log_return(eth_returns['close'].values)
            eth_returns = eth_returns[['timestamp', 'eth_return_1bar']]
        else:
            eth_returns = pd.DataFrame(columns=['timestamp', 'eth_return_1bar'])

        # Load vol index
        try:
            btc_vol = load_vol_index_data(conn, 'BTC', limit)
        except Exception as e:
            logger.warning(f"⚠️ Could not load BTC vol index: {e}")
            btc_vol = pd.DataFrame()

        try:
            eth_vol = load_vol_index_data(conn, 'ETH', limit)
        except Exception as e:
            logger.warning(f"⚠️ Could not load ETH vol index: {e}")
            eth_vol = pd.DataFrame()

        # BTC vol features
        if not btc_vol.empty:
            btc_vol = btc_vol.rename(columns={'close': 'btc_vol_index'})
            btc_vol['btc_vol_change_1h'] = btc_vol['btc_vol_index'].pct_change(1).fillna(0)
            btc_vol['btc_vol_change_4h'] = btc_vol['btc_vol_index'].pct_change(4).fillna(0)
            btc_vol['btc_vol_zscore'] = rolling_zscore(btc_vol['btc_vol_index'].values, window=30 * 24)
            btc_vol = btc_vol[['timestamp', 'btc_vol_index', 'btc_vol_change_1h', 'btc_vol_change_4h', 'btc_vol_zscore']]
        else:
            btc_vol = pd.DataFrame(columns=['timestamp', 'btc_vol_index', 'btc_vol_change_1h', 'btc_vol_change_4h', 'btc_vol_zscore'])

        # ETH vol features — restricted to ETH OHLCV timestamps only
        if not eth_vol.empty and not eth_ohlcv.empty:
            valid_timestamps = set(eth_ohlcv['timestamp'])
            eth_vol = eth_vol[eth_vol['timestamp'].isin(valid_timestamps)].copy()
            eth_vol = eth_vol.rename(columns={'close': 'eth_vol_index'})
            eth_vol['eth_vol_change_1h'] = eth_vol['eth_vol_index'].pct_change(1).fillna(0)
            eth_vol['eth_vol_change_4h'] = eth_vol['eth_vol_index'].pct_change(4).fillna(0)
            eth_vol['eth_vol_zscore'] = rolling_zscore(eth_vol['eth_vol_index'].values, window=30 * 24)
            eth_vol = eth_vol[['timestamp', 'eth_vol_index', 'eth_vol_change_1h', 'eth_vol_change_4h', 'eth_vol_zscore']]
        else:
            eth_vol = pd.DataFrame(columns=['timestamp', 'eth_vol_index', 'eth_vol_change_1h', 'eth_vol_change_4h', 'eth_vol_zscore'])

        return btc_returns, eth_returns, btc_vol, eth_vol

    except Exception as e:
        logger.error(f"Error loading BTC/ETH data: {e}")
        traceback.print_exc()
        empty_df = pd.DataFrame(columns=['timestamp'])
        return empty_df, empty_df, empty_df, empty_df


# --- Get returns for all instruments (for ranking)
def load_all_instrument_returns(conn, current_timestamp, window=20):
    """Load returns for all active instruments for relative ranking"""
    try:
        # Get timestamp range for the window
        start_ts = current_timestamp - timedelta(days=1)
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT instrument_name
                FROM instruments
                WHERE used = TRUE AND quote_currency = 'USDC'
            """)
            instruments = [row[0] for row in cur.fetchall()]
        
        all_returns = {}
        all_funding = {}
        
        for instrument in instruments:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT timestamp, close
                        FROM historical_ohlcv_data
                        WHERE instrument_name = %s
                        AND timestamp BETWEEN %s AND %s
                        ORDER BY timestamp
                    """, (instrument, start_ts, current_timestamp))
                    
                    prices = pd.DataFrame(cur.fetchall(), columns=['timestamp', 'close'])
                    
                    if len(prices) > 1:
                        returns = padded_log_return(prices['close'].values)
                        # Get the most recent return
                        if len(returns) > 0:
                            all_returns[instrument] = returns[-1]
                
                # Get funding rate
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT interest_1h
                        FROM historical_funding_rates
                        WHERE instrument_name = %s
                        AND timestamp <= %s
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (instrument, current_timestamp))
                    result = cur.fetchone()
                    if result and result[0] is not None:
                        all_funding[instrument] = result[0]
                        
            except Exception as e:
                logger.error(f"Error getting data for {instrument}: {e}")
        
        return all_returns, all_funding
    except Exception as e:
        logger.error(f"Error in load_all_instrument_returns: {e}")
        return {}, {}

# --- Compute relative rankings
def calculate_relative_ranks(instrument_name, timestamp, conn):
    """Calculate where this instrument ranks relative to all others"""
    try:
        all_returns, all_funding = load_all_instrument_returns(conn, timestamp)
        
        if not all_returns or instrument_name not in all_returns:
            return None, None
        
        # Calculate return rank
        return_values = list(all_returns.values())
        if len(return_values) < 3:
            return None, None
            
        return_rank = np.searchsorted(np.sort(return_values), all_returns[instrument_name]) / len(return_values)
        
        # Calculate funding rank
        if all_funding and instrument_name in all_funding:
            funding_values = list(all_funding.values())
            if len(funding_values) >= 3:
                funding_rank = np.searchsorted(np.sort(funding_values), all_funding[instrument_name]) / len(funding_values)
            else:
                funding_rank = None
            
        return return_rank, funding_rank
    except Exception as e:
        logger.error(f"Error calculating ranks: {e}")
        return None, None

# --- NEW: Calculate future returns and dual labels with adaptive thresholds
def calculate_future_returns_and_labels(df: pd.DataFrame, instrument_name: str) -> pd.DataFrame:
    """
    Advanced dual labeling strategy with adaptive thresholds:
    1. direction_class: Perfect balance using global percentiles (for training)
    2. direction_signal: Smart dynamic thresholds (for live trading)
    3. signal_confidence: Normalized move magnitude for flexible decision making
    """
    # Calculate future returns
    df['future_return_1bar'] = df['close'].shift(-1).combine_first(df['close']) / df['close'] - 1
    
    # Calculate multi-bar future returns
    df['future_return_2bar'] = df['close'].shift(-2).combine_first(df['close']) / df['close'] - 1
    df['future_return_4bar'] = df['close'].shift(-4).combine_first(df['close']) / df['close'] - 1
    
    # Calculate future volatility
    df['future_volatility'] = df['return_1bar'].rolling(4).std().shift(-4).fillna(0)
    
    # Store original non-null future returns for global percentile calculation
    valid_returns = df['future_return_1bar'].dropna()
    
    # --------------------------------------------------------------------
    # 1. DIRECTION_CLASS: Global percentile-based labels for perfect balance
    # --------------------------------------------------------------------
    if len(valid_returns) > 0:
        quantiles = valid_returns.quantile([1/3, 2/3]).values
        
        # Assign classes based on global percentiles - ensures exact 33/33/33 split
        df['direction_class'] = 1  # Default to neutral (middle)
        df.loc[df['future_return_1bar'] <= quantiles[0], 'direction_class'] = 0  # Bottom third → down
        df.loc[df['future_return_1bar'] >= quantiles[1], 'direction_class'] = 2  # Top third → up
    else:
        df['direction_class'] = 1  # Default if no valid returns
    
    # --------------------------------------------------------------------
    # 2. DIRECTION_SIGNAL: Smart adaptive thresholds for real trading
    # --------------------------------------------------------------------
    # Calculate multiple volatility windows for adaptive thresholding
    vol_window = 20  # 20 bars = 5 hours for 15m bars
    df['volatility_20bar'] = df['return_1bar'].rolling(vol_window).std().fillna(0.001)
    
    # Calculate long-term vol for reference (80 bars = 20 hours)
    long_vol = df['return_1bar'].rolling(80).std().fillna(df['volatility_20bar'])
    
    # Calculate volatility ratio (short-term vs long-term)
    vol_ratio = df['volatility_20bar'] / long_vol
    
    # SMART ADAPTIVE THRESHOLD SCALING:
    # 1. Start with baseline threshold factor (lower than original 0.5)
    base_threshold = 0.35
    
    # 2. Adjust threshold based on instrument's volatility characteristics
    # - For major coins (BTC, ETH): Keep threshold higher
    # - For altcoins: Lower threshold to capture more signals
    instrument_adjustment = {
        'BTC_USDC-PERPETUAL': 1.1,  # Higher threshold for BTC
        'ETH_USDC-PERPETUAL': 1.0,  # Standard threshold for ETH
        'SOL_USDC-PERPETUAL': 0.9,  # Slightly lower for major altcoins
        'XRP_USDC-PERPETUAL': 0.9,
        'AVAX_USDC-PERPETUAL': 0.9,
        'DOGE_USDC-PERPETUAL': 0.8,
        'DOT_USDC-PERPETUAL': 0.8, 
        'ADA_USDC-PERPETUAL': 0.8,
        'LTC_USDC-PERPETUAL': 0.8,
        'LINK_USDC-PERPETUAL': 0.7,
        'ALGO_USDC-PERPETUAL': 0.7,
        'BCH_USDC-PERPETUAL': 0.7,
        'NEAR_USDC-PERPETUAL': 0.7,
        'UNI_USDC-PERPETUAL': 0.6,
        'TRX_USDC-PERPETUAL': 0.5,  # Much lower for stable coins
    }.get(instrument_name, 0.7)  # Default adjustment for unknown instruments
    
    # 3. Dynamic adjustment based on current market conditions
    # - In high volatility regimes (vol_ratio > 1.5): Increase threshold
    # - In low volatility regimes (vol_ratio < 0.5): Decrease threshold
    dynamic_factor = np.sqrt(vol_ratio).clip(0.7, 1.3)
    
    # 4. Compute final threshold with all adjustments
    threshold_factor = base_threshold * instrument_adjustment * dynamic_factor
    
    # Ensure threshold is at least 0.2*vol (minimum signal detection)
    threshold_factor = np.maximum(threshold_factor, 0.2)
    
    # Calculate the final threshold for signal detection
    vol_thresh = threshold_factor * df['volatility_20bar']
    
    # Generate trading signals
    df['direction_signal'] = 1  # Default to neutral
    df.loc[df['future_return_1bar'] > vol_thresh, 'direction_signal'] = 2  # Up signal
    df.loc[df['future_return_1bar'] < -vol_thresh, 'direction_signal'] = 0  # Down signal
    
    # --------------------------------------------------------------------
    # 3. SIGNAL_CONFIDENCE: Normalized magnitude for flexible decisions
    # --------------------------------------------------------------------
    # Signal confidence = absolute return / volatility (normalized move size)
    df['signal_confidence'] = (np.abs(df['future_return_1bar']) / 
                              df['volatility_20bar'].replace(0, 0.001)).clip(0, 5)
    
    # Calculate return quantile rank (useful for model calibration)
    df['return_quantile_rank'] = (valid_returns.rank(pct=True).reindex(df.index)
                                 .fillna(0.5))  # Default to middle rank if no data
    
    return df

# --- Compute features for a single instrument
def compute_features_for_instrument(conn, instrument_name: str):
    """Compute comprehensive tier 1 features for an instrument"""
    logger.info(f"🚀 Computing features for {instrument_name}")
    
    try:
        # Load data with possible limit for rolling updates
        limit = ROLLING_WINDOW if COMPUTE_MODE == 'rolling_update' else None
        ohlcv = load_ohlcv_data(conn, instrument_name, limit)
        funding = load_funding_data(conn, instrument_name, limit)
        
        # Check if we have any data at all
        if len(ohlcv) == 0:
            logger.warning(f"⚠️ No OHLCV data found for {instrument_name}")
            return
            
        if len(funding) == 0:
            logger.warning(f"⚠️ No funding data found for {instrument_name}, proceeding with empty funding features")
            
        # Load cross-asset data (BTC/ETH)
        btc_returns, eth_returns, btc_vol, eth_vol = load_btc_eth_data(conn, limit)

        if len(ohlcv) < ROLLING_WINDOW_VOL + 1:
            logger.warning(f"⚠️ Not enough data for {instrument_name} (found {len(ohlcv)} rows, need {ROLLING_WINDOW_VOL + 1})")
            return

        # Price action features
        df = ohlcv.copy()
        df['prev_close'] = df['close'].shift(1)
        df['price_range_ratio'] = safe_divide(df['high'] - df['low'], df['prev_close'])

        # Log returns
        df['return_1bar'] = padded_log_return(df['close'].values)
        df['return_2bar'] = df['return_1bar'].rolling(2).sum()
        df['return_4bar'] = df['return_1bar'].rolling(4).sum()
        df['return_acceleration'] = df['return_1bar'] - df['return_1bar'].shift(1)

        # Candle shape
        (
            df['candle_body_ratio'],
            df['wick_upper_ratio'],
            df['wick_lower_ratio'],
            df['range_ratio'],
        ) = candle_ratios(df['open'].values, df['high'].values, df['low'].values, df['close'].values)

        # Volume features
        df['volume_change_pct'] = df['volume'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        df['volume_zscore'] = rolling_zscore(df['volume'].values, window=20)
        df['volume_ratio'] = safe_divide(df['volume'], df['volume'].rolling(20).mean())

        # Volatility
        df['volatility_1h'] = rolling_volatility(df['return_1bar'].values, window=ROLLING_WINDOW_VOL)
        trend = safe_divide(df['volatility_1h'], df['volatility_1h'].shift(24))
        df['volatility_trend'] = np.minimum(trend, 50)  # Cap extreme values
        atr = (df['high'] - df['low']).rolling(14).mean()
        df['atr_normalized'] = safe_divide(atr, df['close'])

        # Merge funding features - using left join to keep all OHLCV data
        if len(funding) > 0:
            df = df.merge(funding, on='timestamp', how='left')
            df.rename(columns={
                'interest_1h': 'funding_1h',
                'interest_8h': 'funding_8h'
            }, inplace=True)
            
            # Fill missing funding data with zeros
            df['funding_1h'] = df['funding_1h'].fillna(0)
            df['funding_8h'] = df['funding_8h'].fillna(0)
            
            # Compute funding features
            df['funding_spread'] = df['funding_1h'] - (df['funding_8h'] / 8)
            
            # FIX: Handle NaN values before converting to integer type
            df['funding_direction'] = np.sign(df['funding_1h'].fillna(0)).astype('int8')
            
            df['cumulative_funding_4h'] = df['funding_1h'].rolling(4).sum().fillna(0)
            df['funding_rate_zscore'] = rolling_zscore(df['funding_1h'].fillna(0).values, window=96)
            
            # Basis = (close - index) / index, but handle missing index_price
            if 'index_price' in df.columns:
                df['basis'] = safe_divide(df['close'] - df['index_price'].fillna(df['close']), 
                                        df['index_price'].fillna(df['close']))
                df['basis_change'] = df['basis'].diff().fillna(0)
            else:
                df['basis'] = 0
                df['basis_change'] = 0
        else:
            # Add empty funding features
            df['funding_1h'] = 0
            df['funding_8h'] = 0
            df['funding_spread'] = 0
            df['funding_direction'] = 0
            df['cumulative_funding_4h'] = 0
            df['funding_rate_zscore'] = 0
            df['basis'] = 0
            df['basis_change'] = 0

        # Encode calendar
        df = encode_calendar_fields(df)
        
        # Add BTC/ETH returns via left join
        if len(btc_returns) > 0:
            df = df.merge(btc_returns, on='timestamp', how='left')
            df['btc_return_1bar'] = df['btc_return_1bar'].fillna(0)
        else:
            df['btc_return_1bar'] = 0
            
        if len(eth_returns) > 0:
            df = df.merge(eth_returns, on='timestamp', how='left')
            df['eth_return_1bar'] = df['eth_return_1bar'].fillna(0)
        else:
            df['eth_return_1bar'] = 0
        
        # Add volatility index data via left join
        if len(btc_vol) > 0:
            df = df.merge(btc_vol, on='timestamp', how='left')
            # Fill missing values with zeros
            for col in ['btc_vol_index', 'btc_vol_change_1h', 'btc_vol_change_4h', 'btc_vol_zscore']:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
        else:
            df['btc_vol_index'] = 0
            df['btc_vol_change_1h'] = 0
            df['btc_vol_change_4h'] = 0
            df['btc_vol_zscore'] = 0
            
        if len(eth_vol) > 0:
            df = df.merge(eth_vol, on='timestamp', how='left')
            # Fill missing values with zeros
            for col in ['eth_vol_index', 'eth_vol_change_1h', 'eth_vol_change_4h', 'eth_vol_zscore']:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
        else:
            df['eth_vol_index'] = 0
            df['eth_vol_change_1h'] = 0
            df['eth_vol_change_4h'] = 0
            df['eth_vol_zscore'] = 0
        
        # Compute cross-asset correlations
        if 'btc_return_1bar' in df.columns and not df['btc_return_1bar'].isna().all():
            df['correlation_with_btc'] = rolling_correlation(
                df['return_1bar'].values, 
                df['btc_return_1bar'].fillna(0).values,
                window=ROLLING_WINDOW_VOL
            )
            
            # Special case for BTC itself - should always be 1.0
            if instrument_name == 'BTC_USDC-PERPETUAL':
                df['correlation_with_btc'] = 1.0
        else:
            df['correlation_with_btc'] = 0
            
        if 'eth_return_1bar' in df.columns and not df['eth_return_1bar'].isna().all():
            df['correlation_with_eth'] = rolling_correlation(
                df['return_1bar'].values, 
                df['eth_return_1bar'].fillna(0).values,
                window=ROLLING_WINDOW_VOL
            )
            
            # Special case for ETH itself - should always be 1.0
            if instrument_name == 'ETH_USDC-PERPETUAL':
                df['correlation_with_eth'] = 1.0
        else:
            df['correlation_with_eth'] = 0
        
        # Compute relative rankings for each timestamp - only for last 100 rows to save time
        df['relative_return_rank'] = np.nan
        df['relative_funding_rank'] = np.nan
        
        # Only compute rankings in full backfill mode or for important symbols to save time
        if COMPUTE_MODE == 'full_backfill' or instrument_name in ['BTC_USDC-PERPETUAL', 'ETH_USDC-PERPETUAL']:
            for i, row in df.tail(min(len(df), 100)).iterrows():
                try:
                    return_rank, funding_rank = calculate_relative_ranks(instrument_name, row['timestamp'], conn)
                    if return_rank is not None:
                        df.loc[i, 'relative_return_rank'] = return_rank
                    if funding_rank is not None:
                        df.loc[i, 'relative_funding_rank'] = funding_rank
                except Exception as e:
                    logger.error(f"Error calculating ranks for {instrument_name} at {row['timestamp']}: {e}")
                    continue
        
        # Fill missing rank values with 0.5 (neutral)
        df['relative_return_rank'] = df['relative_return_rank'].fillna(0.5)
        df['relative_funding_rank'] = df['relative_funding_rank'].fillna(0.5)
        
        # Add instrument name
        df['instrument_name'] = instrument_name
        
        # NEW: Calculate future returns and advanced labels using the state-of-the-art approach
        df = calculate_future_returns_and_labels(df, instrument_name)
        
        # Add instrument name
        df['instrument_name'] = instrument_name
        
        # Keep only allowed columns and drop any remaining NaN rows
        keep_cols = [
            "instrument_name", "timestamp",
            "return_1bar", "return_2bar", "return_4bar", "return_acceleration",
            "price_range_ratio", "candle_body_ratio", "wick_upper_ratio", "wick_lower_ratio",
            "volume", "volume_change_pct", "volume_zscore", "volume_ratio",
            "volatility_1h", "volatility_trend", "atr_normalized",
            "funding_1h", "funding_8h", "funding_spread", "funding_direction",
            "cumulative_funding_4h", "funding_rate_zscore", "basis", "basis_change",
            "btc_vol_index", "btc_vol_change_1h", "btc_vol_change_4h", "btc_vol_zscore",
            "eth_vol_index", "eth_vol_change_1h", "eth_vol_change_4h", "eth_vol_zscore",
            "btc_return_1bar", "eth_return_1bar", 
            "correlation_with_btc", "correlation_with_eth",
            "relative_return_rank", "relative_funding_rank",
            "hour_of_day_sin", "hour_of_day_cos", "day_of_week_sin", "day_of_week_cos",
            "is_weekend", "mins_to_next_funding",
            # NEW: Add the new columns for future returns and advanced dual labels
            "future_return_1bar", "future_return_2bar", "future_return_4bar",
            "future_volatility", "direction_class", "direction_signal", "return_quantile_rank",
            "volatility_20bar", "signal_confidence"
        ]
        
        # Ensure we only keep columns that are in the keep_cols list
        final_cols = [col for col in keep_cols if col in df.columns]
        df = df[final_cols].copy()
        
        # Fill any remaining NaN values with zeros to avoid insertion errors
        df = df.fillna(0)
        
        if COMPUTE_MODE == 'rolling_update':
            # For rolling update, only keep the most recent rows that don't exist in the DB
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT MAX(timestamp) FROM model_features_15m_tier1
                    WHERE instrument_name = %s
                """, (instrument_name,))
                last_ts = cur.fetchone()[0]
            
            if last_ts:
                df = df[df['timestamp'] > last_ts].copy()
        
        # Only insert if we have data
        if len(df) > 0:
            write_features_to_db(conn, df)
        else:
            logger.info(f"No new data to insert for {instrument_name}")
            
    except Exception as e:
        logger.error(f"❌ Error computing features for {instrument_name}: {e}")
        traceback.print_exc()

# --- Write to DB
def write_features_to_db(conn, df: pd.DataFrame):
    """Write features to the database using COPY for maximum performance"""
    if len(df) == 0:
        logger.warning("No data to insert")
        return
        
    try:
        # For full backfill mode, consider deleting and reinserting
        if COMPUTE_MODE == 'full_backfill':
            with conn.cursor() as cur:
                # Delete existing data for this instrument
                cur.execute("""
                    DELETE FROM model_features_15m_tier1 
                    WHERE instrument_name = %s
                """, (df['instrument_name'].iloc[0],))
                
                # Create a temporary table
                cur.execute("CREATE TEMP TABLE tmp_features (LIKE model_features_15m_tier1 INCLUDING ALL) ON COMMIT DROP")
                
                # Convert DataFrame to CSV-like buffer
                from io import StringIO
                buffer = StringIO()
                df.to_csv(buffer, index=False, header=False, sep='\t')
                buffer.seek(0)
                
                # Use COPY command for bulk insert
                columns = df.columns.tolist()
                cur.copy_from(buffer, 'tmp_features', columns=columns)
                
                # Insert from temp table to main table
                cur.execute("""
                    INSERT INTO model_features_15m_tier1
                    SELECT * FROM tmp_features
                """)
            
            conn.commit()
            logger.info(f"✅ Successfully inserted {len(df)} rows using COPY method")
        else:
            # For rolling updates, stick with execute_batch but increase page size
            columns = df.columns.tolist()
            placeholders = ["%s"] * len(columns)
            
            insert_query = f"""
                INSERT INTO model_features_15m_tier1 ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT (instrument_name, timestamp) DO UPDATE SET
            """
            
            update_cols = [col for col in columns if col not in ['instrument_name', 'timestamp']]
            update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_cols])
            insert_query += update_clause
            
            with conn.cursor() as cur:
                values = [tuple(x) for x in df.to_numpy()]
                # Increase page size to 5000 for better performance
                psycopg2.extras.execute_batch(cur, insert_query, values, page_size=5000)
            
            conn.commit()
            logger.info(f"✅ Successfully upserted {len(df)} rows")
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Error inserting data: {e}")
        traceback.print_exc()

# --- Main Entrypoint
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instrument', type=str, help='Run for a specific instrument only')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads to use')
    args = parser.parse_args()
    
    # For single instrument mode, we don't need parallelism
    if args.instrument:
        conn = get_connection()
        try:
            logger.info(f"Processing single instrument: {args.instrument}")
            compute_features_for_instrument(conn, args.instrument)
        except Exception as e:
            logger.exception(f"❌ Error computing features for {args.instrument}: {e}")
        finally:
            conn.close()
            logger.info("🔌 Database connection closed")
        return
    
    # For multiple instruments, use parallel processing
    conn = get_connection()
    try:
        # Get list of instruments
        with conn.cursor() as cur:
            cur.execute("""
                SELECT instrument_name FROM instruments 
                WHERE used = TRUE ORDER BY instrument_name
            """)
            instruments = [row[0] for row in cur.fetchall()]
        
        logger.info(f"Found {len(instruments)} instruments to process")
        logger.info(f"Running in {COMPUTE_MODE} mode with window size {ROLLING_WINDOW}")
        logger.info(f"Using {args.threads} parallel threads")
        
        # Temporarily disable indexes for full backfill mode
        indexes_disabled = False
        if COMPUTE_MODE == 'full_backfill':
            try:
                logger.info("Temporarily disabling indexes for faster bulk loading...")
                with conn.cursor() as cur:
                    # Disable autovacuum
                    cur.execute("ALTER TABLE model_features_15m_tier1 SET (autovacuum_enabled = false)")
                    # Get all indexes except primary key
                    cur.execute("""
                        SELECT indexname FROM pg_indexes 
                        WHERE tablename = 'model_features_15m_tier1' 
                        AND indexname NOT LIKE '%pkey%'
                    """)
                    indexes = [row[0] for row in cur.fetchall()]
                    for idx in indexes:
                        logger.info(f"Dropping index: {idx}")
                        cur.execute(f"DROP INDEX IF EXISTS {idx}")
                conn.commit()
                indexes_disabled = True
                logger.info("Indexes disabled, proceeding with bulk loading")
            except Exception as e:
                logger.warning(f"Failed to disable indexes: {e}")
                # Continue even if we couldn't disable indexes
        
        # Each thread needs its own DB connection, so we create a function that opens one
        def process_instrument(symbol):
            # Get a new connection for this thread
            conn = get_connection()
            try:
                compute_features_for_instrument(conn, symbol)
                return f"✅ Completed {symbol}"
            except Exception as e:
                logger.exception(f"❌ Error computing features for {symbol}: {e}")
                return f"❌ Failed {symbol}: {str(e)}"
            finally:
                conn.close()
        
        # Process instruments in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_instrument, symbol) for symbol in instruments]
            
            # Wait for all to complete and handle any exceptions
            for future in futures:
                result = future.result()  # This will re-raise any exceptions
                logger.info(result)
                
        logger.info(f"✅ All instruments processed using {args.threads} threads")
        
        # After all processing is complete, recreate indexes if we disabled them
        if indexes_disabled:
            try:
                logger.info("Recreating indexes...")
                with conn.cursor() as cur:
                    # Recreate typical indexes for this kind of data
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_model_features_timestamp ON model_features_15m_tier1 (timestamp)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_model_features_instrument_timestamp ON model_features_15m_tier1 (instrument_name, timestamp)")
                    
                    # Analyze table and re-enable autovacuum
                    cur.execute("ANALYZE model_features_15m_tier1")
                    cur.execute("ALTER TABLE model_features_15m_tier1 SET (autovacuum_enabled = true)")
                conn.commit()
                logger.info("Indexes recreated and autovacuum re-enabled")
            except Exception as e:
                logger.error(f"Failed to recreate indexes: {e}")
                
    except Exception as e:
        logger.exception(f"❌ Error in main process: {e}")
    finally:
        conn.close()
        logger.info("🔌 Database connection closed")

if __name__ == "__main__":
    main()