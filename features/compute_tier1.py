#!/usr/bin/env python3
"""
compute_tier1.py
-----------------------------------------------------
Builds real-time-safe features for 15m bars for use
in tier1_features_15m and tier1_labels_15m tables. 
These tables power both live inference and supervised training. 
All features are computed from past-only data — no leakage.
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
from typing import Tuple

from data.database import get_connection, db_connection
from features.utils import (
    padded_log_return,
    rolling_zscore,
    rolling_volatility,
    rolling_correlation,
    candle_ratios,
    safe_divide,
    rolling_percentile_rank,
)

# Define terms that indicate future-leaking features
LEAKY_TERMS = ('future_', 'next_', 'direction_', 'signal_', 'quantile')

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
    
    # Vectorized calculation for mins_to_next_funding (funding at 00:00, 08:00, 16:00 UTC)
    dt = df['timestamp'].dt
    mins_since_funding = (dt.hour % 8) * 60 + dt.minute
    df['mins_to_next_funding'] = ((8 * 60) - mins_since_funding).clip(0, 480).astype('int16')
    
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
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
        empty_df = pd.DataFrame(columns=['timestamp'])
        return empty_df, empty_df, empty_df, empty_df


# --- Get returns for all instruments (for ranking)
def load_all_instrument_returns(conn, current_timestamp, window=20):
    """Load returns for all active instruments for relative ranking - optimized version"""
    try:
        # Get timestamp range for the window
        start_ts = current_timestamp - timedelta(days=1)
        
        # Get all instruments in a single query
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT instrument_name
                FROM instruments
                WHERE used = TRUE AND quote_currency = 'USDC'
            """)
            instruments = [row[0] for row in cur.fetchall()]
        
        all_returns = {}
        all_funding = {}
        
        # Fetch all prices in a batch
        try:
            with conn.cursor() as cur:
                # Use array_agg to get arrays of values per instrument
                cur.execute("""
                    SELECT instrument_name, 
                           array_agg(close ORDER BY timestamp) as prices,
                           array_agg(timestamp ORDER BY timestamp) as timestamps
                    FROM historical_ohlcv_data
                    WHERE instrument_name = ANY(%s)
                    AND timestamp BETWEEN %s AND %s
                    GROUP BY instrument_name
                """, (instruments, start_ts, current_timestamp))
                
                for row in cur.fetchall():
                    instrument, prices, timestamps = row
                    
                    if prices and len(prices) > 1:
                        # Calculate return from last two prices
                        last_return = np.log(prices[-1] / prices[-2]) if prices[-2] > 0 else 0
                        all_returns[instrument] = last_return
        except Exception as e:
            logger.error(f"Error fetching prices batch: {e}")
            # Fall back to individual queries if the optimized query fails
            for instrument in instruments:
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT close
                            FROM historical_ohlcv_data
                            WHERE instrument_name = %s
                            AND timestamp <= %s
                            ORDER BY timestamp DESC
                            LIMIT 2
                        """, (instrument, current_timestamp))
                        
                        prices = [row[0] for row in cur.fetchall()]
                        if len(prices) == 2 and prices[1] > 0:
                            # prices[0] is the latest, prices[1] is the previous
                            all_returns[instrument] = np.log(prices[0] / prices[1])
                except Exception as inner_e:
                    logger.error(f"Error getting prices for {instrument}: {inner_e}")
        
        # Fetch all funding rates in a single query
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    WITH latest_funding AS (
                        SELECT DISTINCT ON (instrument_name) 
                            instrument_name, interest_1h
                        FROM historical_funding_rates
                        WHERE instrument_name = ANY(%s)
                        AND timestamp <= %s
                        ORDER BY instrument_name, timestamp DESC
                    )
                    SELECT instrument_name, interest_1h FROM latest_funding
                """, (instruments, current_timestamp))
                
                for instrument, interest_1h in cur.fetchall():
                    if interest_1h is not None:
                        all_funding[instrument] = interest_1h
        except Exception as e:
            logger.error(f"Error fetching funding batch: {e}")
            # Fall back to individual queries if the optimized query fails
            for instrument in instruments:
                try:
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
                except Exception as inner_e:
                    logger.error(f"Error getting funding for {instrument}: {inner_e}")
                        
        return all_returns, all_funding
    except Exception as e:
        logger.error(f"Error in load_all_instrument_returns: {e}")
        logger.error(traceback.format_exc())
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
        funding_rank = None
        if all_funding and instrument_name in all_funding:
            funding_values = list(all_funding.values())
            if len(funding_values) >= 3:
                funding_rank = np.searchsorted(np.sort(funding_values), all_funding[instrument_name]) / len(funding_values)
            
        return return_rank, funding_rank
    except Exception as e:
        logger.error(f"Error calculating ranks: {e}")
        return None, None

# --- Calculate future returns and dual labels with adaptive thresholds
def calculate_future_returns_and_labels(df: pd.DataFrame, instrument_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Advanced dual labeling strategy with adaptive thresholds:
    1. direction_class: Perfect balance using global percentiles (for training) - Using 0,1,2
    2. direction_signal: Smart dynamic thresholds (for live trading) - Using 0,1,2
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
        # Using 0, 1, 2 for consistent labels
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
    
    # Generate trading signals using 0, 1, 2 for consistent labeling
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
    
    # Create separate feature and label DataFrames
    feature_cols = [col for col in df.columns 
                   if not any(term in col for term in LEAKY_TERMS)
                   and col not in ('timestamp', 'instrument_name')]
    
    label_cols = ['future_return_1bar', 'future_return_2bar', 'future_return_4bar',
                 'future_volatility', 'direction_class', 'direction_signal', 
                 'signal_confidence', 'return_quantile_rank']
    
    features_df = df[['instrument_name', 'timestamp'] + feature_cols].copy()
    labels_df = df[['instrument_name', 'timestamp'] + label_cols].copy()
    
    return features_df, labels_df

# --- Compute features for a single instrument
def compute_features_for_instrument(conn, instrument_name: str):
    """Compute comprehensive tier 1 features for an instrument"""
    logger.info(f"🚀 Computing features for {instrument_name}")
    
    try:
        # Load data with possible limit for rolling updates
        limit = ROLLING_WINDOW if COMPUTE_MODE == 'rolling_update' else None
        ohlcv = load_ohlcv_data(conn, instrument_name, limit)
        
        # Check if we have any data at all
        if len(ohlcv) == 0:
            logger.warning(f"⚠️ No OHLCV data found for {instrument_name}")
            return
            
        funding = load_funding_data(conn, instrument_name, limit)
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
            # Keep funding_direction as -1,0,1 since it represents sign, not a class
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
        
        # Calculate future returns and advanced labels - now returns features and labels separately
        try:
            features_df, labels_df = calculate_future_returns_and_labels(df, instrument_name)
        except Exception as e:
            logger.error(f"❌ Error calculating labels for {instrument_name}: {e}")
            logger.error(traceback.format_exc())
            return
        
        # Fill NaNs in both dataframes
        features_df = features_df.fillna(0)
        labels_df = labels_df.fillna(0)
        
        if COMPUTE_MODE == 'rolling_update':
            # For rolling update, only keep the most recent rows that don't exist in the DB
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT MAX(timestamp) FROM tier1_features_15m
                    WHERE instrument_name = %s
                """, (instrument_name,))
                last_ts = cur.fetchone()[0]
            
            if last_ts:
                features_df = features_df[features_df['timestamp'] > last_ts].copy()
                labels_df = labels_df[labels_df['timestamp'] > last_ts].copy()
        
        # Only insert if we have data
        if len(features_df) > 0:
            # First insert features, then labels
            try:
                success = write_features_to_db(conn, features_df)
                if success and len(labels_df) > 0:
                    write_labels_to_db(conn, labels_df)
            except Exception as e:
                logger.error(f"❌ Error inserting data for {instrument_name}: {e}")
                logger.error(traceback.format_exc())
                conn.rollback()
                
        else:
            logger.info(f"No new data to insert for {instrument_name}")
            
    except Exception as e:
        logger.error(f"❌ Error computing features for {instrument_name}: {e}")
        logger.error(traceback.format_exc())
        
        # Try to commit any pending transactions to prevent database locks
        try:
            conn.rollback()
        except:
            pass

# --- Write features to DB
def write_features_to_db(conn, df: pd.DataFrame) -> bool:
    """
    Write features to the database using execute_batch method.
    
    Args:
        conn: Database connection
        df: DataFrame with feature data
        
    Returns:
        True if successful, False otherwise
    """
    if len(df) == 0:
        logger.warning("No data to insert")
        return False
        
    try:
        # For full backfill mode, consider deleting and reinserting
        if COMPUTE_MODE == 'full_backfill':
            # Get all columns in the target table
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'tier1_features_15m'
                    ORDER BY ordinal_position
                """)
                table_columns = [row[0] for row in cur.fetchall()]
                
                # Delete existing data for this instrument
                cur.execute("""
                    DELETE FROM tier1_features_15m 
                    WHERE instrument_name = %s
                """, (df['instrument_name'].iloc[0],))
                
                # Only keep columns that exist in the table
                df_columns = [col for col in df.columns if col in table_columns]
                
                # Execute insert
                placeholders = ', '.join(['%s'] * len(df_columns))
                insert_query = f"""
                    INSERT INTO tier1_features_15m ({', '.join(df_columns)})
                    VALUES ({placeholders})
                """
                
                # Convert dataframe to list of tuples
                values = [tuple(row) for row in df[df_columns].values]
                
                # Use execute_batch instead of copy_from
                psycopg2.extras.execute_batch(cur, insert_query, values, page_size=1000)
                conn.commit()
                
                logger.info(f"✅ Successfully inserted {len(df)} rows to tier1_features_15m")
                return True
                
        else:
            # For rolling updates
            # Get all columns in the target table
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'tier1_features_15m'
                    ORDER BY ordinal_position
                """)
                table_columns = [row[0] for row in cur.fetchall()]
                
                # Only keep columns that exist in the table
                df_columns = [col for col in df.columns if col in table_columns]
                
                # Prepare statements
                placeholders = ', '.join(['%s'] * len(df_columns))
                insert_query = f"""
                    INSERT INTO tier1_features_15m ({', '.join(df_columns)})
                    VALUES ({placeholders})
                    ON CONFLICT (instrument_name, timestamp) DO UPDATE SET
                """
                
                update_cols = [col for col in df_columns if col not in ['instrument_name', 'timestamp']]
                update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_cols])
                insert_query += update_clause
                
                # Convert dataframe to list of tuples
                values = [tuple(row) for row in df[df_columns].values]
                
                # Use execute_batch instead of copy_from
                psycopg2.extras.execute_batch(cur, insert_query, values, page_size=1000)
                conn.commit()
                
                logger.info(f"✅ Successfully upserted {len(df)} rows to tier1_features_15m")
                return True
                
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Error inserting data to tier1_features_15m: {e}")
        logger.error(traceback.format_exc())
        return False

# --- Write labels to DB
def write_labels_to_db(conn, df: pd.DataFrame) -> bool:
    """
    Write labels to the database using execute_batch method.
    
    Args:
        conn: Database connection
        df: DataFrame with label data
        
    Returns:
        True if successful, False otherwise
    """
    if len(df) == 0:
        logger.warning("No label data to insert")
        return False
    
    try:
        # For full backfill mode, consider deleting and reinserting
        if COMPUTE_MODE == 'full_backfill':
            # Get all columns in the target table
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'tier1_labels_15m'
                    ORDER BY ordinal_position
                """)
                table_columns = [row[0] for row in cur.fetchall()]
                
                # Delete existing data for this instrument
                cur.execute("""
                    DELETE FROM tier1_labels_15m 
                    WHERE instrument_name = %s
                """, (df['instrument_name'].iloc[0],))
                
                # Only keep columns that exist in the table
                df_columns = [col for col in df.columns if col in table_columns]
                
                # Execute insert for each row to handle foreign key dependencies
                placeholders = ', '.join(['%s'] * len(df_columns))
                insert_query = f"""
                    INSERT INTO tier1_labels_15m ({', '.join(df_columns)})
                    VALUES ({placeholders})
                """
                
                # Convert dataframe to list of tuples
                values = [tuple(row) for row in df[df_columns].values]
                
                # Add ON CONFLICT clause for insert safety
                safe_insert_query = f"""
                    INSERT INTO tier1_labels_15m ({', '.join(df_columns)})
                    VALUES ({placeholders})
                    ON CONFLICT DO NOTHING
                """
                
                # Use execute_batch with smaller batches
                psycopg2.extras.execute_batch(cur, safe_insert_query, values, page_size=500)
                conn.commit()
                
                logger.info(f"✅ Successfully inserted {len(df)} rows to tier1_labels_15m")
                return True
                
        else:
            # For rolling updates
            # Get all columns in the target table
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'tier1_labels_15m'
                    ORDER BY ordinal_position
                """)
                table_columns = [row[0] for row in cur.fetchall()]
                
                # Only keep columns that exist in the table
                df_columns = [col for col in df.columns if col in table_columns]
                
                # Prepare statements
                placeholders = ', '.join(['%s'] * len(df_columns))
                insert_query = f"""
                    INSERT INTO tier1_labels_15m ({', '.join(df_columns)})
                    VALUES ({placeholders})
                    ON CONFLICT (instrument_name, timestamp) DO UPDATE SET
                """
                
                update_cols = [col for col in df_columns if col not in ['instrument_name', 'timestamp']]
                update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_cols])
                insert_query += update_clause
                
                # Convert dataframe to list of tuples
                values = [tuple(row) for row in df[df_columns].values]
                
                # Use execute_batch with smaller batches
                psycopg2.extras.execute_batch(cur, insert_query, values, page_size=500)
                conn.commit()
                
                logger.info(f"✅ Successfully upserted {len(df)} rows to tier1_labels_15m")
                return True
                
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Error inserting data to tier1_labels_15m: {e}")
        logger.error(traceback.format_exc())
        return False

# --- Main Entrypoint
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instrument', type=str, help='Run for a specific instrument only')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads to use')
    args = parser.parse_args()
    
    # For single instrument mode, we don't need parallelism
    if args.instrument:
        with db_connection() as conn:
            try:
                logger.info(f"Processing single instrument: {args.instrument}")
                compute_features_for_instrument(conn, args.instrument)
            except Exception as e:
                logger.exception(f"❌ Error computing features for {args.instrument}: {e}")
        return
    
    # For multiple instruments, use parallel processing
    with db_connection() as conn:
        try:
            # Get list of instruments - use only instruments with the USDC quote currency
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT instrument_name FROM instruments 
                    WHERE used = TRUE AND quote_currency = 'USDC'
                    ORDER BY instrument_name
                """)
                instruments = [row[0] for row in cur.fetchall()]
                
            if not instruments:
                logger.error("No active USDC-quoted instruments found in the database!")
                return
            
            logger.info(f"Found {len(instruments)} instruments to process")
            logger.info(f"Running in {COMPUTE_MODE} mode with window size {ROLLING_WINDOW}")
            logger.info(f"Using {args.threads} parallel threads")
            
            # Temporarily disable autovacuum for full backfill mode
            # Note: We don't create or drop indexes, just change autovacuum setting
            if COMPUTE_MODE == 'full_backfill':
                try:
                    with conn.cursor() as cur:
                        for table in ['tier1_features_15m', 'tier1_labels_15m']:
                            cur.execute(f"ALTER TABLE {table} SET (autovacuum_enabled = false)")
                    conn.commit()
                    logger.info("Autovacuum disabled, proceeding with bulk loading")
                except Exception as e:
                    logger.error(f"Error disabling autovacuum: {e}")
                    logger.error(traceback.format_exc())
            
            # Each thread needs its own DB connection, so we create a function that opens one
            def process_instrument(symbol):
                # Get a new connection for this thread
                with db_connection() as thread_conn:
                    try:
                        compute_features_for_instrument(thread_conn, symbol)
                        return f"✅ Completed {symbol}"
                    except Exception as e:
                        logger.exception(f"❌ Error computing features for {symbol}: {e}")
                        return f"❌ Failed {symbol}: {str(e)}"
            
            # Process instruments in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                futures = [executor.submit(process_instrument, symbol) for symbol in instruments]
                
                # Wait for all to complete and handle any exceptions
                for future in futures:
                    result = future.result()  # This will re-raise any exceptions
                    logger.info(result)
                    
            logger.info(f"✅ All instruments processed using {args.threads} threads")
            
            # Re-enable autovacuum if we disabled it
            if COMPUTE_MODE == 'full_backfill':
                try:
                    with conn.cursor() as cur:
                        for table in ['tier1_features_15m', 'tier1_labels_15m']:
                            cur.execute(f"ALTER TABLE {table} SET (autovacuum_enabled = true)")
                            cur.execute(f"ANALYZE {table}")
                    conn.commit()
                    logger.info("Autovacuum re-enabled")
                except Exception as e:
                    logger.error(f"Error re-enabling autovacuum: {e}")
                    logger.error(traceback.format_exc())
                
        except Exception as e:
            logger.exception(f"❌ Error in main process: {e}")

if __name__ == "__main__":
    main()