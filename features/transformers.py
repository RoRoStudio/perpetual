# /mnt/p/perpetual/features/transformers.py
"""
Feature transformation utilities for model training and inference.
Handles scaling, normalization, and other preprocessing steps.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional, Any
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("transformers")

# Define constants
SCALER_DIR = "/mnt/p/perpetual/models/scalers"
os.makedirs(SCALER_DIR, exist_ok=True)

# Features to scale with StandardScaler (mean=0, std=1)
STANDARD_SCALE_FEATURES = [
    'return_1bar', 'return_2bar', 'return_4bar', 'return_acceleration',
    'price_range_ratio', 'candle_body_ratio', 'wick_upper_ratio', 'wick_lower_ratio',
    'volatility_1h', 'volatility_trend', 'atr_normalized',
    'funding_1h', 'funding_8h', 'funding_spread', 'funding_rate_zscore',
    'basis', 'basis_change',
    'btc_vol_index', 'btc_vol_change_1h', 'btc_vol_change_4h', 'btc_vol_zscore',
    'eth_vol_index', 'eth_vol_change_1h', 'eth_vol_change_4h', 'eth_vol_zscore',
    'correlation_with_btc', 'correlation_with_eth'
]

# Features to scale with MinMaxScaler (0-1 range)
MINMAX_SCALE_FEATURES = [
    'volume', 'volume_change_pct', 'volume_zscore', 'volume_ratio',
    'relative_return_rank', 'relative_funding_rank'
]

# Features to leave as-is (don't scale)
CATEGORICAL_FEATURES = [
    'funding_direction', 'is_weekend'
]

# Cyclic features (already normalized by sin/cos transformation)
CYCLIC_FEATURES = [
    'hour_of_day_sin', 'hour_of_day_cos', 'day_of_week_sin', 'day_of_week_cos'
]

# Features that need special handling and winsorization
WINSORIZE_FEATURES = {
    'volatility_trend': (0, 100.0),  # Clip upper limit to 100
    'return_1bar': (-5.0, 5.0),      # Clip to +/- 5 standard deviations
    'return_2bar': (-7.0, 7.0),      # Slightly wider bounds for multi-bar returns
    'return_4bar': (-10.0, 10.0),    # Even wider for longer-term returns
    'volume_ratio': (0, 20.0),       # Volume spikes capped at 20x average
    'volume_zscore': (-10.0, 10.0)   # Extreme volume events capped
}

class FeatureTransformer:
    """
    Handles feature transformations for both training and inference.
    Maintains consistent scaling between training and real-time inference.
    """
    
    def __init__(self, instrument_name: str, recalibrate: bool = False):
        """
        Initialize the transformer for a specific instrument.
        
        Args:
            instrument_name: The name of the instrument (e.g., 'BTC_USDC-PERPETUAL')
            recalibrate: Force recalculation of scaling parameters
        """
        self.instrument_name = instrument_name
        
        # Create paths for scalers
        self.base_scaler_path = os.path.join(SCALER_DIR, self._get_clean_name())
        self.std_scaler_path = f"{self.base_scaler_path}_std_scaler.pkl"
        self.minmax_scaler_path = f"{self.base_scaler_path}_minmax_scaler.pkl"
        self.metadata_path = f"{self.base_scaler_path}_metadata.pkl"
        
        # Initialize scalers
        self.std_scaler = None
        self.minmax_scaler = None
        self.metadata = None
        
        # Try to load existing scalers, or create new ones
        if os.path.exists(self.std_scaler_path) and not recalibrate:
            self.load_scalers()
        else:
            self.std_scaler = StandardScaler()
            self.minmax_scaler = MinMaxScaler()
            self.metadata = {
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'num_training_samples': 0,
                'training_period_start': None,
                'training_period_end': None,
                'feature_means': {},
                'feature_stds': {},
                'feature_mins': {},
                'feature_maxs': {},
            }
    
    def _get_clean_name(self) -> str:
        """Get a filesystem-friendly version of the instrument name"""
        return self.instrument_name.replace('-', '_').replace('/', '_')
    
    def load_scalers(self) -> bool:
        """Load scalers from disk"""
        try:
            with open(self.std_scaler_path, 'rb') as f:
                self.std_scaler = pickle.load(f)
                
            with open(self.minmax_scaler_path, 'rb') as f:
                self.minmax_scaler = pickle.load(f)
                
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
                
            logger.info(f"Loaded scalers for {self.instrument_name}, last updated: {self.metadata['updated_at']}")
            return True
        except Exception as e:
            logger.error(f"Failed to load scalers: {e}")
            self.std_scaler = StandardScaler()
            self.minmax_scaler = MinMaxScaler()
            self.metadata = {
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'num_training_samples': 0,
                'training_period_start': None,
                'training_period_end': None,
                'feature_means': {},
                'feature_stds': {},
                'feature_mins': {},
                'feature_maxs': {},
            }
            return False
    
    def save_scalers(self) -> bool:
        """Save scalers to disk"""
        try:
            with open(self.std_scaler_path, 'wb') as f:
                pickle.dump(self.std_scaler, f)
                
            with open(self.minmax_scaler_path, 'wb') as f:
                pickle.dump(self.minmax_scaler, f)
                
            # Update metadata
            self.metadata['updated_at'] = datetime.now()
            
            # Store feature statistics
            if hasattr(self.std_scaler, 'mean_') and self.std_scaler.mean_ is not None:
                for i, feature in enumerate(STANDARD_SCALE_FEATURES):
                    if i < len(self.std_scaler.mean_):
                        self.metadata['feature_means'][feature] = float(self.std_scaler.mean_[i])
                        self.metadata['feature_stds'][feature] = float(self.std_scaler.scale_[i])
            
            if hasattr(self.minmax_scaler, 'min_') and self.minmax_scaler.min_ is not None:
                for i, feature in enumerate(MINMAX_SCALE_FEATURES):
                    if i < len(self.minmax_scaler.min_):
                        self.metadata['feature_mins'][feature] = float(self.minmax_scaler.min_[i])
                        self.metadata['feature_maxs'][feature] = float(self.minmax_scaler.data_max_[i])
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
                
            logger.info(f"Saved scalers for {self.instrument_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save scalers: {e}")
            return False
    
    def _preprocess_for_fitting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps before fitting scalers.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Apply winsorization (clipping) to specific features
        for col, (min_val, max_val) in WINSORIZE_FEATURES.items():
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].clip(lower=min_val, upper=max_val)
        
        # Handle outliers in all numeric columns
        numeric_cols = df_processed.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        for col in numeric_cols:
            # Replace infs with NaN
            df_processed[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Compute robust statistics for outlier detection
            if not df_processed[col].isna().all():
                q1 = df_processed[col].quantile(0.01)
                q3 = df_processed[col].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                # Clip extreme outliers
                if col not in WINSORIZE_FEATURES:
                    df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_processed
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit scalers on training data
        
        Args:
            df: DataFrame containing feature data
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided for fitting scalers")
            return
        
        # Apply preprocessing
        df_processed = self._preprocess_for_fitting(df)
        
        # Extract standard scale features
        std_features = [f for f in STANDARD_SCALE_FEATURES if f in df_processed.columns]
        std_data = df_processed[std_features].copy()
        
        # Handle remaining infinities and NaNs
        std_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        std_data.fillna(0, inplace=True)
        
        # Fit standard scaler
        self.std_scaler.fit(std_data)
        
        # Extract minmax scale features
        minmax_features = [f for f in MINMAX_SCALE_FEATURES if f in df_processed.columns]
        minmax_data = df_processed[minmax_features].copy()
        
        # Handle infinities and NaNs
        minmax_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        minmax_data.fillna(0, inplace=True)
        
        # Fit minmax scaler
        self.minmax_scaler.fit(minmax_data)
        
        # Update metadata
        self.metadata['num_training_samples'] = len(df)
        if 'timestamp' in df.columns:
            self.metadata['training_period_start'] = df['timestamp'].min()
            self.metadata['training_period_end'] = df['timestamp'].max()
        
        # Save the fitted scalers
        self.save_scalers()
        
        logger.info(f"Fitted scalers on {len(df)} samples of {self.instrument_name} data")

    def needs_recalibration(self, max_age_days: int = 7) -> bool:
        """
        Check if the transformer needs recalibration based on age
        
        Args:
            max_age_days: Maximum age of scalers in days
            
        Returns:
            True if recalibration is needed, False otherwise
        """
        # Check if metadata exists
        if not self.metadata:
            logger.info(f"No metadata found for {self.instrument_name}, recalibration needed")
            return True
            
        # Check if updated_at exists
        last_updated = self.metadata.get("updated_at")
        if not last_updated:
            logger.info(f"No last update timestamp for {self.instrument_name}, recalibration needed")
            return True
            
        # Check age
        age_days = (datetime.now() - last_updated).days
        if age_days > max_age_days:
            logger.info(f"Scaler for {self.instrument_name} is {age_days} days old (max {max_age_days}), recalibration needed")
            return True
            
        # Check if we have enough samples
        if self.metadata.get("num_training_samples", 0) < 1000:
            logger.info(f"Insufficient training samples for {self.instrument_name}, recalibration needed")
            return True
            
        return False

    def detect_distribution_shift(self, df: pd.DataFrame, tolerance: float = 0.5) -> bool:
        """
        Detect if there's a significant shift in the data distribution.
        
        Args:
            df: DataFrame with recent data
            tolerance: Tolerance threshold for shift detection
            
        Returns:
            True if distribution shift detected, False otherwise
        """
        if not self.metadata or not self.metadata.get('feature_means'):
            return True
            
        # Check if we have enough recent data
        if len(df) < 100:
            return False
            
        shift_detected = False
        
        # Compare distribution of key features
        for feature in ['return_1bar', 'volume_zscore', 'volatility_1h', 'funding_1h']:
            if feature not in df.columns or feature not in self.metadata['feature_means']:
                continue
                
            # Get current and stored statistics
            current_mean = df[feature].mean()
            stored_mean = self.metadata['feature_means'].get(feature, 0)
            
            current_std = df[feature].std()
            stored_std = self.metadata['feature_stds'].get(feature, 1)
            
            # Calculate normalized difference
            if stored_std != 0:
                mean_diff = abs(current_mean - stored_mean) / stored_std
                
                if mean_diff > tolerance:
                    logger.info(f"Distribution shift detected in {feature} for {self.instrument_name}: {mean_diff:.2f} > {tolerance}")
                    shift_detected = True
                    break
        
        return shift_detected

    def calibrate_recent_lookback(self, lookback_days: int = 60) -> bool:
        """
        Calibrate using recent data
        
        Args:
            lookback_days: Number of days to look back for calibration data
            
        Returns:
            True if calibration succeeded, False otherwise
        """
        from data.database import get_connection
        
        logger.info(f"Auto-calibrating scaler for {self.instrument_name} with {lookback_days} days lookback")
        
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM model_features_15m_tier1
                    WHERE instrument_name = %s
                    AND timestamp >= NOW() - INTERVAL '%s days'
                    ORDER BY timestamp
                """, (self.instrument_name, lookback_days))
                
                columns = [desc[0] for desc in cur.description]
                data = cur.fetchall()
                df = pd.DataFrame(data, columns=columns)
                
            if len(df) >= 100:
                logger.info(f"Fitting scaler on {len(df)} recent samples")
                self.fit(df)
                return True
            else:
                logger.warning(f"Insufficient recent data for {self.instrument_name}: only {len(df)} samples")
                return False
        except Exception as e:
            logger.error(f"Error calibrating scaler: {e}")
            return False
        finally:
            conn.close()

    def transform(self, df: pd.DataFrame, auto_calibrate: bool = True) -> pd.DataFrame:
        """
        Transform features using fitted scalers, with enhanced numerical stability
        
        Args:
            df: DataFrame containing raw features
            auto_calibrate: Whether to auto-calibrate if needed
                
        Returns:
            DataFrame with scaled features
        """
        # Check if calibration is needed
        if auto_calibrate:
            if self.needs_recalibration() or self.detect_distribution_shift(df):
                self.calibrate_recent_lookback()
        
        if self.std_scaler is None or self.minmax_scaler is None:
            raise ValueError("Scalers not initialized. Call fit() or load_scalers() first.")
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Apply more aggressive cleaning for numerical stability
        
        # 1. First, replace any extreme outliers
        for col in result.columns:
            if col not in ['instrument_name', 'timestamp']:
                # Replace extreme values with something more reasonable
                result[col] = result[col].clip(lower=-1e3, upper=1e3)
                
                # Replace NaN and inf more robustly
                result[col] = result[col].replace([np.inf, -np.inf, np.nan], 0)
        
        # 2. Apply more targeted winsorization (clipping) to specific features
        for col, (min_val, max_val) in WINSORIZE_FEATURES.items():
            if col in result.columns:
                result[col] = result[col].clip(lower=min_val, upper=max_val)
        
        # Extract standard scale features that exist in the dataframe
        std_features = [f for f in STANDARD_SCALE_FEATURES if f in df.columns]
        if std_features:
            std_data = result[std_features].copy()
            
            # Replace infs and NaNs
            std_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            std_data.fillna(0, inplace=True)
            
            # Transform data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    scaled_std = self.std_scaler.transform(std_data)
                    
                    # Additional safety: clip the transformed values to reasonable ranges
                    scaled_std = np.clip(scaled_std, -10.0, 10.0)
                    
                    # Check for invalid values in scaled output
                    if not np.isfinite(scaled_std).all():
                        logger.warning(f"❌ Detected NaN or Inf after standard scaling for {self.instrument_name}")
                        # Fix the problematic values
                        scaled_std = np.nan_to_num(scaled_std, nan=0.0, posinf=10.0, neginf=-10.0)
                        scaled_std = np.clip(scaled_std, -10.0, 10.0)
                        
                    # Update result dataframe
                    for i, feature in enumerate(std_features):
                        result[feature] = scaled_std[:, i]
                except Exception as e:
                    logger.error(f"Error during standard scaling: {e}")
                    # Fallback for error cases: apply simple standardization
                    for feature in std_features:
                        mean = np.nanmean(std_data[feature])
                        std = np.nanstd(std_data[feature])
                        if std > 1e-8:
                            result[feature] = (std_data[feature] - mean) / std
                        else:
                            result[feature] = 0.0

        # Extract minmax scale features that exist in the dataframe
        minmax_features = [f for f in MINMAX_SCALE_FEATURES if f in df.columns]
        if minmax_features:
            minmax_data = result[minmax_features].copy()
            
            # Replace infs
            minmax_data.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Impute relative rank NaNs with 0.5 (neutral rank)
            for rank_col in ['relative_return_rank', 'relative_funding_rank']:
                if rank_col in minmax_data.columns:
                    minmax_data[rank_col] = minmax_data[rank_col].fillna(0.5)

            # Impute other NaNs with 0
            minmax_data.fillna(0, inplace=True)
            
            # Transform data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    scaled_minmax = self.minmax_scaler.transform(minmax_data)
                    
                    # Additional safety: clip values to [0, 1] range
                    scaled_minmax = np.clip(scaled_minmax, 0.0, 1.0)

                    if not np.isfinite(scaled_minmax).all():
                        logger.warning(f"❌ Detected NaN or Inf after MinMax scaling for {self.instrument_name}")
                        # Fix the problematic values
                        scaled_minmax = np.nan_to_num(scaled_minmax, nan=0.5, posinf=1.0, neginf=0.0)
                        scaled_minmax = np.clip(scaled_minmax, 0.0, 1.0)
                    
                    # Update result dataframe
                    for i, feature in enumerate(minmax_features):
                        result[feature] = scaled_minmax[:, i]
                except Exception as e:
                    logger.error(f"Error during minmax scaling: {e}")
                    # Fallback for error cases: apply simple min-max scaling manually
                    for feature in minmax_features:
                        min_val = np.nanmin(minmax_data[feature])
                        max_val = np.nanmax(minmax_data[feature])
                        if max_val > min_val:
                            result[feature] = (minmax_data[feature] - min_val) / (max_val - min_val)
                        else:
                            result[feature] = 0.5  # Neutral value
        
        # Final safety check: replace any remaining NaN or inf values
        for col in result.columns:
            if col not in ['instrument_name', 'timestamp']:
                if result[col].isna().any() or np.isinf(result[col]).any().any():
                    logger.warning(f"Fixing remaining NaN/Inf values in {col}")
                    result[col] = result[col].replace([np.inf, -np.inf, np.nan], 0)
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scalers and transform data in one step
        
        Args:
            df: DataFrame containing raw features
            
        Returns:
            DataFrame with scaled features
        """
        self.fit(df)
        return self.transform(df, auto_calibrate=False)
    
    def get_feature_stats(self) -> Dict:
        """
        Get statistics about the scaled features
        
        Returns:
            Dictionary with feature statistics
        """
        stats = {
            'instrument': self.instrument_name,
            'num_samples': self.metadata.get('num_training_samples', 0),
            'training_start': self.metadata.get('training_period_start'),
            'training_end': self.metadata.get('training_period_end'),
            'features': {}
        }
        
        # Add standard scaled features
        for feature in STANDARD_SCALE_FEATURES:
            if feature in self.metadata.get('feature_means', {}):
                stats['features'][feature] = {
                    'scaling': 'standard',
                    'mean': self.metadata['feature_means'][feature],
                    'std': self.metadata['feature_stds'][feature]
                }
        
        # Add minmax scaled features
        for feature in MINMAX_SCALE_FEATURES:
            if feature in self.metadata.get('feature_mins', {}):
                stats['features'][feature] = {
                    'scaling': 'minmax',
                    'min': self.metadata['feature_mins'][feature],
                    'max': self.metadata['feature_maxs'][feature]
                }
        
        return stats

    def preprocess_sequences(self, df: pd.DataFrame, seq_length: int = 64, stride: int = 1) -> np.ndarray:
        """
        Create sequences for time-series model input
        
        Args:
            df: DataFrame with scaled features
            seq_length: Length of each sequence
            stride: Stride between consecutive sequences
            
        Returns:
            Array of sequences with shape [num_sequences, seq_length, num_features]
        """
        if len(df) < seq_length:
            logger.warning(f"DataFrame length ({len(df)}) is less than sequence length ({seq_length})")
            return np.array([])
        
        # Transform the dataframe
        scaled_df = self.transform(df)
        
        # Get feature columns (exclude timestamp and instrument_name)
        feature_cols = [col for col in scaled_df.columns if col not in ['timestamp', 'instrument_name']]
        
        # Convert to numpy for faster slicing
        data = scaled_df[feature_cols].values
        
        # Create sequences
        sequences = []
        for i in range(0, len(data) - seq_length + 1, stride):
            sequences.append(data[i:i+seq_length])
        
        return np.array(sequences)


class ScalerManager:
    """
    Manage calibration of scalers for all instruments.
    """
    
    @staticmethod
    def auto_calibrate_if_needed(lookback_days: int = 60, force: bool = False, max_age_days: int = 7):
        """
        Check and calibrate scalers for all instruments if needed
        
        Args:
            lookback_days: Number of days to look back for calibration data
            force: Force recalibration regardless of age
            max_age_days: Maximum age of scalers in days
        """
        from data.database import get_connection
        
        # Get list of instruments
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT instrument_name 
                    FROM model_features_15m_tier1
                    ORDER BY instrument_name
                """)
                instruments = [row[0] for row in cur.fetchall()]
                
            logger.info(f"Checking {len(instruments)} instruments for scaler calibration")
            
            calibrated_count = 0
            for instrument in instruments:
                transformer = FeatureTransformer(instrument)
                
                if force or transformer.needs_recalibration(max_age_days):
                    logger.info(f"Calibrating scaler for {instrument}")
                    if transformer.calibrate_recent_lookback(lookback_days):
                        calibrated_count += 1
            
            logger.info(f"Calibrated {calibrated_count} out of {len(instruments)} instruments")
                
        except Exception as e:
            logger.error(f"Error in auto_calibrate_if_needed: {e}")
        
        finally:
            conn.close()


def calibrate_all_instrument_scalers(lookback_days: int = 60):
    """
    Calibrate scalers for all instruments based on recent data
    
    Args:
        lookback_days: Number of days of data to use for calibration
    """
    ScalerManager.auto_calibrate_if_needed(lookback_days, force=True)


def get_scaled_features(instrument_name: str, limit: int = 1000) -> pd.DataFrame:
    """
    Load and scale recent features for a specific instrument
    
    Args:
        instrument_name: The instrument to load data for
        limit: Maximum number of recent records to load
        
    Returns:
        DataFrame with scaled features
    """
    from data.database import get_connection
    
    # Get data
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM model_features_15m_tier1
                WHERE instrument_name = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (instrument_name, limit))
            
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchall()
            df = pd.DataFrame(data, columns=columns)
        
        # Sort by timestamp for time-series order
        df = df.sort_values('timestamp')
        
        if len(df) == 0:
            logger.warning(f"No data found for {instrument_name}")
            return pd.DataFrame()
        
        # Load transformer and scale features
        transformer = FeatureTransformer(instrument_name)
        scaled_df = transformer.transform(df)
        
        return scaled_df
    
    finally:
        conn.close()


def get_sequence_for_inference(instrument_name: str, seq_length: int = 64) -> np.ndarray:
    """
    Get the latest sequence of data for model inference
    
    Args:
        instrument_name: The instrument to load data for
        seq_length: Length of the sequence to create
        
    Returns:
        Array with shape [1, seq_length, num_features] for model input
    """
    # Get scaled features (load more than needed to ensure we have enough after filtering)
    scaled_df = get_scaled_features(instrument_name, limit=seq_length + 100)
    
    if len(scaled_df) < seq_length:
        logger.warning(f"Insufficient data for {instrument_name}: only {len(scaled_df)} samples available")
        return None
    
    # Create transformer for sequence creation
    transformer = FeatureTransformer(instrument_name)
    
    # Get the latest sequence
    sequences = transformer.preprocess_sequences(scaled_df, seq_length=seq_length, stride=1)
    
    if len(sequences) == 0:
        logger.warning(f"Failed to create sequences for {instrument_name}")
        return None
    
    # Return the last sequence as a batch of 1
    return sequences[-1:] 


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature transformer utilities")
    parser.add_argument("--calibrate-all", action="store_true", help="Recalibrate scalers for all instruments")
    parser.add_argument("--lookback", type=int, default=60, help="Days of history to use for calibration")
    parser.add_argument("--instrument", type=str, help="Process a specific instrument")
    parser.add_argument("--stats", action="store_true", help="Print scaler statistics")
    parser.add_argument("--sequence", action="store_true", help="Create a sample sequence for the specified instrument")
    parser.add_argument("--seq-length", type=int, default=64, help="Sequence length for sequence creation")

    args = parser.parse_args()

    if args.calibrate_all:
        calibrate_all_instrument_scalers(args.lookback)

    elif args.instrument:
        if args.stats:
            transformer = FeatureTransformer(args.instrument)
            stats = transformer.get_feature_stats()
            print(f"Feature statistics for {args.instrument}:")
            for feature, stat in stats['features'].items():
                print(f"  {feature}: {stat}")

        elif args.sequence:
            seq = get_sequence_for_inference(args.instrument, seq_length=args.seq_length)
            if seq is not None:
                print(f"Created sequence with shape: {seq.shape}")
                print(f"Sample values (first timestep, first 10 features):")
                print(seq[0, 0, :10])
            else:
                print(f"Failed to create sequence for {args.instrument}")

        else:
            scaled = get_scaled_features(args.instrument, limit=10)
            print(f"Sample scaled data for {args.instrument}:")
            print(scaled.head())

    else:
        print("No action specified. Use --calibrate-all or --instrument with additional flags.")
