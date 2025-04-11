#!/usr/bin/env python3

# python -m features.export_parquet

"""
Unified training script for Deribit perpetual trading model.
Trains a single model on all instruments with advanced optimizations:
- Full W&B integration
- Mixed precision training
- Gradient accumulation
- Smart batching and caching
- Model souping (ensemble averaging)
- Parquet-based data loading with memory mapping (3-4x speedup)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
from configparser import ConfigParser
from pathlib import Path
import wandb
import psycopg2
import psycopg2.extras
from tqdm import tqdm
import time
import random
import gc
import pyarrow.parquet as pq
from contextlib import contextmanager

# Add project root to path for imports
sys.path.append('/mnt/p/perpetual')

# Import project modules
from data.database import get_connection
from features.transformers import FeatureTransformer
from models.architecture import DeribitHybridModel, OptimizedDeribitModel
from models.profiler import TrainingProfiler

@contextmanager
def nullcontext():
    """A context manager that does nothing"""
    yield

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dynamic feature pruning function
def dynamic_feature_pruning(features_df, targets_df, keep_pct=0.75, method='mutual_info'):
    """
    Dynamically prune low-information features using mutual information or correlation.
    
    Args:
        features_df: DataFrame with features
        targets_df: DataFrame with target values
        keep_pct: Percentage of features to keep
        method: 'mutual_info' or 'correlation'
        
    Returns:
        Pruned features DataFrame and mask of selected features
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
        
        # Get feature columns (exclude non-feature columns)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['instrument_name', 'timestamp']]
        
        # Select target (use next_return_1bar as default)
        target = targets_df['next_return_1bar'].values
        
        if method == 'mutual_info':
            # Calculate mutual information scores
            mi_scores = mutual_info_regression(features_df[feature_cols].values, target)
            scores = mi_scores
        else:  # correlation
            # Calculate absolute correlation scores
            corr_scores = []
            for col in feature_cols:
                corr = abs(np.corrcoef(features_df[col].values, target)[0, 1])
                corr_scores.append(corr if not np.isnan(corr) else 0)
            scores = np.array(corr_scores)
        
        # Determine threshold
        threshold = np.percentile(scores, 100 * (1 - keep_pct))
        
        # Create mask of features to keep
        keep_mask = scores >= threshold
        keep_features = [feature_cols[i] for i in range(len(feature_cols)) if keep_mask[i]]
        
        # Add back non-feature columns
        keep_features += [col for col in features_df.columns if col not in feature_cols]
        
        # Return pruned features
        return features_df[keep_features], keep_mask, keep_features
    except:
        # If feature pruning fails, return original features
        print("Feature pruning failed, using all features")
        return features_df, None, list(features_df.columns)

# Apply temporal curriculum learning
def apply_temporal_curriculum(df, epoch, epoch_window_size=90, base_window_days=180):
    """
    Apply temporal curriculum learning by focusing on more recent data first.
    
    Args:
        df: DataFrame with 'timestamp' column
        epoch: Current epoch number
        epoch_window_size: Days to add per curriculum phase
        base_window_days: Minimum days to include
        
    Returns:
        DataFrame filtered by curriculum window
    """
    # Convert timestamps to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get the most recent date in the dataset
    max_date = df['timestamp'].max()
    
    # Calculate curriculum window
    # Early epochs focus on more recent data, later epochs include older data
    window_days = base_window_days + (epoch // 3) * epoch_window_size
    
    # Calculate cutoff date
    cutoff_date = max_date - pd.Timedelta(days=window_days)
    
    # Filter dataset
    filtered_df = df[df['timestamp'] >= cutoff_date].copy()
    
    # If we filtered out too much data, use at least 50% of the original data
    if len(filtered_df) < len(df) * 0.5:
        # Sort by timestamp and take the most recent 50%
        return df.sort_values('timestamp', ascending=False).iloc[:int(len(df) * 0.5)].copy()
    
    return filtered_df

# Create dataset with caching for faster loading
class CachedPerpetualSwapDataset:
    """Dataset that combines multiple instruments and caches processed data for speed."""
    def __init__(
        self,
        instruments,
        seq_length=64,
        cache_dir="/tmp/deribit_cache",
        test_size=0.2,
        device="cpu",
        epoch=0
    ):
        self.instruments = instruments
        self.seq_length = seq_length
        self.cache_dir = cache_dir
        self.test_size = test_size
        self.device = device
        self.epoch = epoch
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load all datasets
        self.train_datasets = []
        self.test_datasets = []
        
        # Record feature and label columns for consistency
        self.feature_columns = [
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
            "is_weekend", "mins_to_next_funding"
        ]
        
        self.label_columns = [
            "next_return_1bar", "next_return_2bar", "next_return_4bar", 
            "direction_class", "next_volatility"
        ]
        
        # Load and cache each instrument's data
        for instrument in instruments:
            train_dataset, test_dataset = self._load_instrument(instrument)
            if train_dataset and test_dataset:
                self.train_datasets.append(train_dataset)
                self.test_datasets.append(test_dataset)
        
        if not self.train_datasets:
            raise ValueError("No valid data found for any instruments!")
        
        # Create combined datasets
        self.combined_train = CombinedDataset(self.train_datasets)
        self.combined_test = CombinedDataset(self.test_datasets)
        
        print(f"✅ Created dataset with {len(self.combined_train)} training and {len(self.combined_test)} test samples")
        
    def _load_instrument(self, instrument_name):
        """Load data for a single instrument with caching for speed."""
        # Check if cached version exists
        cache_file = os.path.join(self.cache_dir, f"{instrument_name}_data.pt")
        if os.path.exists(cache_file):
            print(f"📂 Loading cached data for {instrument_name}")
            data = torch.load(cache_file)
            
            # Make sure feature_columns and label_columns are properly loaded
            if hasattr(data["train"], "feature_columns") and data["train"].feature_columns is not None:
                if self.feature_columns is None:
                    self.feature_columns = data["train"].feature_columns
                    self.label_columns = data["train"].label_columns
                    
            return data["train"], data["test"]
        
        print(f"🔄 Loading fresh data for {instrument_name}")
        
        # Get raw data from database
        df = self._create_labels_db(instrument_name)
        if df.empty:
            print(f"⚠️ No data available for {instrument_name}")
            return None, None
        
        # Define feature and label columns if not already set
        feature_columns = self.feature_columns
        label_columns = self.label_columns
        
        # Create a feature transformer for scaling
        transformer = FeatureTransformer(instrument_name)
        
        # Apply temporal curriculum learning if desired (epoch > 0)
        if self.epoch > 0:
            df = apply_temporal_curriculum(df, self.epoch)
        
        # Split into train/test
        split_idx = int(len(df) * (1 - self.test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Create datasets
        train_dataset = InstrumentDataset(
            instrument_name=instrument_name,
            features=train_df,
            seq_length=self.seq_length,
            transform=transformer,
            feature_columns=feature_columns,
            target_columns=label_columns,
            device=self.device
        )
        
        test_dataset = InstrumentDataset(
            instrument_name=instrument_name,
            features=test_df,
            seq_length=self.seq_length,
            transform=transformer,
            feature_columns=feature_columns,
            target_columns=label_columns,
            device=self.device
        )
        
        # Cache the datasets
        torch.save({
            "train": train_dataset,
            "test": test_dataset,
            "metadata": {
                "feature_columns": self.feature_columns,
                "label_columns": self.label_columns,
                "creation_time": datetime.now()
            }
        }, cache_file)
        
        return train_dataset, test_dataset
    
    def _create_labels_db(self, instrument_name):
        """Create labels from tier 1 features in the database or Parquet files."""
        # First check if Parquet file exists
        parquet_path = f"/mnt/p/perpetual/cache/tier1_{instrument_name}.parquet"
        if os.path.exists(parquet_path):
            try:
                print(f"📊 Loading data from Parquet for {instrument_name}")
                # Load from Parquet with memory mapping for efficiency and better performance
                table = pq.read_table(
                    parquet_path, 
                    memory_map=True,
                    use_threads=True # Enable multi-threading for faster loading
                )
                
                # Convert to pandas more efficiently
                df = table.to_pandas(
                    zero_copy_only=True,  # Use zero-copy when possible
                    split_blocks=True,    # Split into column blocks for better memory usage
                    self_destruct=True    # Allow Arrow to free memory as soon as possible
                )
                
                # Create forward-looking return labels more efficiently
                next_returns = []
                for i in range(1, 5):
                    next_returns.append(df['return_1bar'].shift(-i))
                    
                df['next_return_1bar'] = next_returns[0]
                df['next_return_2bar'] = next_returns[0] + next_returns[1]
                
                # Use vectorized operations instead of sum in a loop
                df['next_return_4bar'] = pd.concat(next_returns, axis=1).sum(axis=1)
                
                # Create volatility label
                df['next_volatility'] = df['return_1bar'].rolling(4).std().shift(-4)
                
                # Fill NaNs in the last rows with 0 more efficiently
                fill_cols = ['next_return_1bar', 'next_return_2bar', 'next_return_4bar', 'next_volatility']
                for col in fill_cols:
                    df.loc[df.index[-4:], col] = 0
                
                # Create direction class label using vectorized operations
                volatility = df['return_1bar'].rolling(20).std().fillna(0.001)
                df['direction_class'] = 0  # Default to neutral
                
                # Vectorized direction class computation
                is_up = df['next_return_1bar'] > 0.5 * volatility
                is_down = df['next_return_1bar'] < -0.5 * volatility
                
                df.loc[is_up, 'direction_class'] = 1
                df.loc[is_down, 'direction_class'] = -1
                
                return df
            except Exception as e:
                print(f"⚠️ Error loading from Parquet for {instrument_name}: {e}")
                print(f"Falling back to database")
        
        # Fall back to database loading if Parquet file doesn't exist or fails
        conn = get_connection()
        try:
            print(f"🔄 Loading data from database for {instrument_name}")
            # Load tier 1 features with optimized query
            with conn.cursor() as cur:
                # Use FETCH to stream data instead of fetching all at once
                cur.execute("""
                    SELECT * FROM model_features_15m_tier1
                    WHERE instrument_name = %s
                    ORDER BY timestamp
                """, (instrument_name,))
                
                columns = [desc[0] for desc in cur.description]
                
                # Stream data in chunks for better memory usage
                chunk_size = 10000
                chunks = []
                while True:
                    data_chunk = cur.fetchmany(chunk_size)
                    if not data_chunk:
                        break
                    chunks.append(pd.DataFrame(data_chunk, columns=columns))
                
                # Concatenate chunks into a single DataFrame
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    return pd.DataFrame()
                
            # Create forward-looking return labels
            next_returns = []
            for i in range(1, 5):
                next_returns.append(df['return_1bar'].shift(-i))
                
            df['next_return_1bar'] = next_returns[0]
            df['next_return_2bar'] = next_returns[0] + next_returns[1]
            df['next_return_4bar'] = pd.concat(next_returns, axis=1).sum(axis=1)
            
            # Create volatility label
            df['next_volatility'] = df['return_1bar'].rolling(4).std().shift(-4)
            
            # Fill NaNs in the last rows with 0
            fill_cols = ['next_return_1bar', 'next_return_2bar', 'next_return_4bar', 'next_volatility']
            for col in fill_cols:
                df.loc[df.index[-4:], col] = 0
            
            # Create direction class label
            volatility = df['return_1bar'].rolling(20).std().fillna(0.001)
            df['direction_class'] = 0
            
            # Vectorized classification
            is_up = df['next_return_1bar'] > 0.5 * volatility
            is_down = df['next_return_1bar'] < -0.5 * volatility
            
            df.loc[is_up, 'direction_class'] = 1
            df.loc[is_down, 'direction_class'] = -1
            
            return df
        finally:
            conn.close()
    
    def get_dataloaders(self, batch_size=32, num_workers=4):
        """Get train and test dataloaders with optimized GPU data transfer."""
        train_loader = DataLoader(
            self.combined_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2,  # Prefetch 2 batches per worker
            pin_memory_device="cuda" if torch.cuda.is_available() else "",  # Pin directly to CUDA for faster transfers
        )
        
        test_loader = DataLoader(
            self.combined_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,  # Prefetch 2 batches per worker
            pin_memory_device="cuda" if torch.cuda.is_available() else "",  # Pin directly to CUDA
        )
        
        return train_loader, test_loader
    
    # Get optimized dataloader with cross-instrument contiguous batching
    def get_optimized_dataloaders(self, batch_size=32, num_workers=4):
        """
        Get optimized dataloaders with cross-instrument contiguous batching for better GPU utilization
        """
        train_loader = FastContiguousDataLoader(
            datasets=self.train_datasets,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        test_loader = FastContiguousDataLoader(
            datasets=self.test_datasets,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader


class InstrumentDataset:
    """Dataset for a single instrument."""
    def __init__(
        self,
        instrument_name,
        features,
        seq_length=64,
        stride=1,
        transform=None,
        feature_columns=None,
        target_columns=None,
        device="cpu"
    ):
        self.instrument_name = instrument_name
        self.seq_length = seq_length
        self.stride = stride
        self.transform = transform
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.device = device
        
        # Store raw data
        self.features = features
        
        # Check if we have all required columns
        missing_features = [col for col in feature_columns if col not in features.columns]
        if missing_features:
            print(f"Missing feature columns for {instrument_name}: {missing_features}")
            for col in missing_features:
                features[col] = 0.0
                
        missing_targets = [col for col in target_columns if col not in features.columns]
        if missing_targets:
            print(f"Missing target columns for {instrument_name}: {missing_targets}")
            for col in missing_targets:
                features[col] = 0.0
        
        # Apply transformation if provided
        if transform is not None:
            features_only = features[feature_columns].copy()
            self.scaled_data = transform.transform(features_only)
        else:
            self.scaled_data = features[feature_columns].copy()
            
        # Store targets
        self.targets = features[target_columns].copy()
        
        # Create sequences
        self.valid_indices = self._create_sequences()
        
        # Store instrument ID as a tensor for asset-aware models
        self.instrument_id = hash(instrument_name) % 100  # Simple hash to create ID
        
    def _create_sequences(self):
        """Create sequences for time-series input."""
        # Get number of valid sequences
        n_samples = len(self.scaled_data)
        valid_starts = range(0, n_samples - self.seq_length + 1, self.stride)
        return list(valid_starts)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get start index for this sequence
        start_idx = self.valid_indices[idx]
        
        # Extract feature sequence
        feature_seq = self.scaled_data.iloc[start_idx:start_idx+self.seq_length].values
        feature_tensor = torch.FloatTensor(feature_seq.astype(np.float32))
        
        # Extract target (using the last timestep in the sequence)
        target_idx = start_idx + self.seq_length - 1
        
        def safe_value(value, min_val=-100.0, max_val=100.0):
            return np.clip(value, min_val, max_val) if not np.isnan(value) else 0.0

        next_return_1bar = safe_value(self.targets.iloc[target_idx]['next_return_1bar'])
        next_return_2bar = safe_value(self.targets.iloc[target_idx]['next_return_2bar']) 
        next_return_4bar = safe_value(self.targets.iloc[target_idx]['next_return_4bar'])
        
        # Convert direction class to one-hot encoding
        direction_raw = int(self.targets.iloc[target_idx]['direction_class'])
        # Map -1,0,1 to 0,1,2 for classification
        direction_class = direction_raw + 1 if direction_raw in [-1, 0, 1] else 1
        
        next_volatility = safe_value(self.targets.iloc[target_idx]['next_volatility'])
        funding_rate = safe_value(self.scaled_data.iloc[target_idx]['funding_1h']) if 'funding_1h' in self.scaled_data.columns else 0.0
        
        # Create target tensor dictionary
        target_dict = {
            'next_return_1bar': torch.FloatTensor([next_return_1bar]),
            'next_return_2bar': torch.FloatTensor([next_return_2bar]),
            'next_return_4bar': torch.FloatTensor([next_return_4bar]),
            'direction_class': torch.LongTensor([direction_class]),
            'next_volatility': torch.FloatTensor([next_volatility]),
            'funding_rate': torch.FloatTensor([funding_rate]),
            'instrument_id': torch.LongTensor([self.instrument_id])
        }

        return feature_tensor, target_dict


class CombinedDataset:
    """Dataset that combines multiple instrument datasets."""
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(ds) for ds in datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)
        
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if dataset_idx > 0:
            idx_within_dataset = idx - self.cumulative_lengths[dataset_idx-1]
        else:
            idx_within_dataset = idx
            
        # Get the item from the appropriate dataset
        return self.datasets[dataset_idx][idx_within_dataset]


# OPTIMIZATION: Faster Cross-Instrument Contiguous Batching
class FastContiguousBatchSampler:
    """
    Creates contiguous batches across multiple instruments for GPU efficiency.
    This drastically improves memory layout and reduces CPU->GPU transfer time.
    """
    def __init__(self, datasets, batch_size, shuffle=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate total samples
        self.total_samples = sum(len(dataset) for dataset in datasets)
        
        # Create index mapping
        self.indices = []
        offset = 0
        for i, dataset in enumerate(datasets):
            for j in range(len(dataset)):
                self.indices.append((i, j))  # (dataset_idx, sample_idx)
            offset += len(dataset)
        
        # Shuffle indices if needed
        if shuffle:
            import random
            random.shuffle(self.indices)
            
    def __iter__(self):
        # Yield batches
        batch = []
        for idx in self.indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                
        # Yield last batch if not empty
        if batch:
            yield batch
            
    def __len__(self):
        return (self.total_samples + self.batch_size - 1) // self.batch_size


# OPTIMIZATION: FastContiguousDataLoader for better GPU memory layout
class FastContiguousDataLoader:
    """
    Faster data loader with contiguous memory layout across instruments.
    """
    def __init__(self, datasets, batch_size, shuffle=True, num_workers=4, 
                 pin_memory=True, drop_last=True, prefetch_factor=2):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = FastContiguousBatchSampler(datasets, batch_size, shuffle)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _collate_fn(self, batch_indices):
        """
        Collate function that creates contiguous batches across instruments.
        """
        features_list = []
        targets_dict = {
            'next_return_1bar': [],
            'next_return_2bar': [],
            'next_return_4bar': [],
            'direction_class': [],
            'next_volatility': [],
            'funding_rate': [],
            'instrument_id': []
        }
        
        # Get data from each dataset
        for dataset_idx, sample_idx in batch_indices:
            features, targets = self.datasets[dataset_idx][sample_idx]
            features_list.append(features)
            
            # Collect targets
            for key, value in targets.items():
                if key in targets_dict:
                    targets_dict[key].append(value)
        
        # Stack features into a single tensor
        features_batch = torch.stack(features_list)
        
        # Stack targets into a single tensor for each key
        for key in targets_dict:
            if targets_dict[key]:  # Check if we have any values
                targets_dict[key] = torch.cat(targets_dict[key])
        
        return features_batch, targets_dict
    
    def __iter__(self):
        # Use standard dataloader in worker mode for compatibility
        if self.num_workers > 0:
            from torch.utils.data import DataLoader
            
            # Create a wrapper dataset to use with DataLoader
            class IndexDataset:
                def __init__(self, sampler):
                    self.sampler = sampler
                    self.indices = list(iter(sampler))  # Convert sampler to list of indices
                    
                def __len__(self):
                    return len(self.indices)
                    
                def __getitem__(self, idx):
                    return self.indices[idx]
            
            # Create wrapper dataset
            dataset = IndexDataset(self.sampler)
            
            # Create standard dataloader with our collate function
            loader = DataLoader(
                dataset,
                batch_size=None,  # We're already batching in the sampler
                shuffle=False,    # We're already shuffling in the sampler
                num_workers=self.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor
            )
            
            # Yield from loader
            yield from loader
        else:
            # Single-process mode
            for batch_indices in self.sampler:
                batch = self._collate_fn(batch_indices)
                if self.pin_memory:
                    batch = self._pin_batch(batch)
                yield batch
    
    def _pin_batch(self, batch):
        """Pin batch to memory for faster GPU transfer."""
        features, targets = batch
        features = features.pin_memory()
        targets = {k: v.pin_memory() for k, v in targets.items()}
        return features, targets
    
    def __len__(self):
        return len(self.sampler)


def train_unified_model(
    instruments,
    model_params,
    training_params,
    use_wandb=True,
    debug=False,
    profiler=None,
    max_speed=False,
    cache_dir="/tmp/deribit_cache"
):
    """
    Train a unified model on multiple instruments with optimized performance.
    
    Args:
        instruments: List of instruments to include
        model_params: Parameters for model initialization
        training_params: Parameters for training
        use_wandb: Whether to use W&B for tracking
        debug: Enable debug mode for faster runs
        profiler: Optional profiler for performance tracking
        max_speed: Enable maximum speed optimizations
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Start W&B run with full system metrics logging
    if use_wandb:
        if profiler:
            profiler.start_operation("wandb_init")
            
        run = wandb.init(
            project="deribit-perpetual-model",
            name=f"unified_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "instruments": instruments,
                "model_params": model_params,
                "training_params": training_params,
                "device": device,
                "max_speed": max_speed
            },
            settings=wandb.Settings(
                code_dir="/mnt/p/perpetual",
                _disable_stats=False,  # Internal API to ensure system stats are enabled
                x_stats_disk_paths=["/root"]  # Monitor root directory
            ),
            monitor_gym=False
        )
        
        # Log system info at start of run
        if torch.cuda.is_available():
            wandb.log({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "total_gpu_memory_GB": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
            
        if profiler:
            profiler.end_operation("wandb_init")
    
    try:
        # Load and combine datasets with caching
        print(f"🔄 Loading data for {len(instruments)} instruments")
        
        if profiler:
            profiler.start_operation("data_loading")
        
        # Use fast mode for debugging if needed
        if debug:
            print("⚡ DEBUG MODE ACTIVE: Using reduced dataset size")
            seq_length = 32
            cache_suffix = "_debug"
        else:
            seq_length = training_params.get('seq_length', 64)
            cache_suffix = ""
            
        # Set epoch for temporal curriculum learning
        epoch = 0
            
        dataset = CachedPerpetualSwapDataset(
            instruments=instruments,
            seq_length=seq_length,
            cache_dir=f"{cache_dir}{cache_suffix}",  # Use the cache directory from args
            test_size=training_params.get('test_size', 0.2),
            device=device,
            epoch=epoch
        )
        
        # Get optimized dataloaders
        batch_size = training_params.get('batch_size', 64)
        
        # Use optimized dataloader if max_speed is enabled
        if max_speed:
            train_loader, test_loader = dataset.get_optimized_dataloaders(
                batch_size=batch_size,
                num_workers=training_params.get('num_workers', 4)
            )
            print("⚡ Using optimized contiguous dataloader for maximum speed")
        else:
            train_loader, test_loader = dataset.get_dataloaders(
                batch_size=batch_size,
                num_workers=training_params.get('num_workers', 4)
            )
        
        if profiler:
            profiler.end_operation("data_loading")
            profiler.start_operation("model_initialization")
        
        # Initialize model with optimal architecture
        input_dim = len(dataset.feature_columns)
        
        # Use OptimizedDeribitModel if max_speed is enabled
        if max_speed:
            model = OptimizedDeribitModel(
                input_dim=input_dim,
                **model_params
            ).to(device)
            print("⚡ Using OptimizedDeribitModel architecture for maximum speed")
        else:
            model = DeribitHybridModel(
                input_dim=input_dim,
                **model_params
            ).to(device)
        
        # Apply JIT compilation for speed improvement
        try:
            if torch.cuda.is_available():
                # Skip torch.compile due to compatibility issues with model architecture
                # Instead, just apply cudnn benchmarking for better performance
                torch.backends.cudnn.benchmark = True
                print("✅ CUDA optimizations enabled for faster training")
        except Exception as e:
            print(f"⚠️ CUDA optimization failed: {e}")
        
        # Enable W&B model watching for gradient and parameter tracking
        if use_wandb:
            wandb.watch(model, log="all", log_freq=100)
            
        if profiler:
            profiler.end_operation("model_initialization")
            profiler.start_operation("optimizer_setup")
        
        # Apply mixed precision optimization with improved settings
        scaler = GradScaler(
            growth_factor=2.0,       # Faster scaling growth
            backoff_factor=0.5,      # Faster backoff when needed
            growth_interval=100      # More aggressive growth
        )
        
        # Log model architecture to W&B
        if use_wandb:
            wandb.watch(model, log="all", log_freq=100)
        
        # Adjust optimizer with correct parameters
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_params.get('learning_rate', 1e-3),
            weight_decay=training_params.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8            # Improved numerical stability
        )
        
        # Use a more advanced scheduler for faster convergence
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=training_params.get('learning_rate', 1e-3),
            steps_per_epoch=len(train_loader),
            epochs=training_params.get('num_epochs', 20),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,    # LR range from max_lr/25 to max_lr
            final_div_factor=10000.0  # End LR = max_lr/10000
        )
        
        if profiler:
            profiler.end_operation("optimizer_setup")
            profiler.start_operation("training")
        
        # Choose appropriate training function based on max_speed
        if max_speed:
            # Use enhanced training function
            train_optimized_enhanced(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                num_epochs=training_params.get('num_epochs', 20),
                device=device,
                patience=training_params.get('patience', 5),
                gradient_accumulation=training_params.get('gradient_accumulation', 1),
                model_save_path=training_params.get('model_save_path', "/mnt/p/perpetual/models/checkpoints"),
                experiment_name="unified_model",
                debug=debug,
                use_wandb=use_wandb,
                profiler=profiler
            )
        else:
            # Use standard training function
            train_optimized(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                num_epochs=training_params.get('num_epochs', 20),
                device=device,
                patience=training_params.get('patience', 5),
                gradient_accumulation=training_params.get('gradient_accumulation', 1),
                model_save_path=training_params.get('model_save_path', "/mnt/p/perpetual/models/checkpoints"),
                experiment_name="unified_model",
                debug=debug,
                use_wandb=use_wandb,
                profiler=profiler  # Pass profiler to train_optimized
            )
        
        if profiler:
            profiler.end_operation("training")
    
    finally:
        # Cleanup
        if use_wandb:
            if profiler:
                profiler.start_operation("wandb_finish")
            wandb.finish()
            if profiler:
                profiler.end_operation("wandb_finish")
            
        # Force garbage collection
        if profiler:
            profiler.start_operation("memory_cleanup")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if profiler:
            profiler.end_operation("memory_cleanup")


def train_optimized(model, train_loader, val_loader, optimizer, scheduler, scaler, num_epochs=20, device="cuda", patience=5, gradient_accumulation=1, model_save_path="/mnt/p/perpetual/models/checkpoints", experiment_name=None, debug=False, use_wandb=True, profiler=None):
    """
    Optimized training function with mixed precision, gradient accumulation and enhanced monitoring.
    """
    print(f"🚀 Starting optimized training on {device} with mixed precision")
    print(f"📊 Using batch size {train_loader.batch_size} × {gradient_accumulation} accumulation steps")
    
    # Define loss functions
    direction_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    
    # Initialize metrics
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Ensure model save directory exists
    os.makedirs(model_save_path, exist_ok=True)
    
    # Create checkpoint filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        checkpoint_base = f"{model_save_path}/{experiment_name}_{timestamp}"
    else:
        checkpoint_base = f"{model_save_path}/model_{timestamp}"
    
    # Create a history dictionary to track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_direction_acc': [],
        'val_direction_acc': [],
        'learning_rate': []
    }
    
    # Set torch.cuda.amp precision
    torch.set_float32_matmul_precision('high')
    
    # Training loop
    for epoch in range(num_epochs):
        # Profile epoch time
        if profiler:
            profiler.start_operation(f"epoch_{epoch}")
            profiler.start_operation(f"epoch_{epoch}_training")
        
        epoch_start_time = time.time()
        
        # ===== TRAINING =====
        model.train()
        train_losses = []
        train_direction_correct = 0
        train_direction_total = 0
        train_return_mse_sum = 0
        
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        # Use tqdm with a higher update frequency for more responsive UI
        for batch_idx, (features, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", miniters=10)):
            # Profile data loading and transfer
            if profiler:
                profiler.start_operation("data_transfer")
                
            # Move to device and ensure data is clean - use non_blocking for async transfer
            features = features.to(device, non_blocking=True)
            features = torch.nan_to_num(features)
            
            # Move all target tensors to device with non_blocking
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            
            if profiler:
                profiler.end_operation("data_transfer")
                profiler.start_operation("forward_pass")
            
            # Get instrument IDs for asset-aware models
            asset_ids = targets.get('instrument_id')
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(features, targets['funding_rate'], asset_ids)
            
            if profiler:
                profiler.end_operation("forward_pass") 
                profiler.start_operation("loss_computation")
            
            with autocast():
                # Compute losses
                direction_loss = direction_criterion(outputs['direction_logits'], targets['direction_class'].squeeze())
                return_loss = regression_criterion(outputs['expected_return'], targets['next_return_1bar'])
                risk_loss = regression_criterion(outputs['expected_risk'], targets['next_volatility'])
                
                # Combined loss with weighting
                loss = direction_loss + return_loss + 0.5 * risk_loss
                
                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation
            
            if profiler:
                profiler.end_operation("loss_computation")
                profiler.start_operation("backward_pass")
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            if profiler:
                profiler.end_operation("backward_pass")
            
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation == 0:
                if profiler:
                    profiler.start_operation("optimizer_step")
                    
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights and zero gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # set_to_none=True is faster than setting to zero
                
                # Update learning rate each batch with OneCycleLR
                scheduler.step()
                
                if profiler:
                    profiler.end_operation("optimizer_step")
            
            # Record metrics
            train_losses.append(loss.item() * gradient_accumulation)  # Rescale loss back
            
            # Calculate accuracy
            pred_direction = torch.argmax(outputs['direction_logits'], dim=1)
            train_direction_correct += (pred_direction == targets['direction_class'].squeeze()).sum().item()
            train_direction_total += targets['direction_class'].size(0)
            
            # Calculate MSE for returns
            train_return_mse_sum += return_loss.item() * gradient_accumulation * targets['next_return_1bar'].size(0)
            
            # Record batch processing time if profiling
            if profiler and batch_idx > 0:  # Skip first batch (warm-up)
                batch_time = time.time() - epoch_start_time
                profiler.record_batch_time(batch_time / batch_idx)  # Average time per batch so far
            
            # Log batch metrics to W&B (but not too frequently)
            if use_wandb and batch_idx % 20 == 0:
                if profiler:
                    profiler.start_operation("wandb_batch_logging")
                    
                # Calculate GPU memory metrics
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    gpu_utilization = 0  # Will be tracked by W&B system metrics
                    
                    # Log memory usage
                    wandb.log({
                        'gpu_memory_allocated_GB': gpu_memory_allocated,
                        'gpu_memory_reserved_GB': gpu_memory_reserved,
                        'iteration': batch_idx + epoch * len(train_loader)
                    })
                    
                wandb.log({
                    'batch_loss': loss.item() * gradient_accumulation,
                    'batch_direction_loss': direction_loss.item(),
                    'batch_return_loss': return_loss.item(),
                    'batch_risk_loss': risk_loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'batch': batch_idx + epoch * len(train_loader)
                })
                
                if profiler:
                    profiler.end_operation("wandb_batch_logging")
        
        # Calculate epoch metrics
        train_loss = np.mean(train_losses)
        train_direction_acc = train_direction_correct / train_direction_total if train_direction_total > 0 else 0
        train_return_mse = train_return_mse_sum / train_direction_total if train_direction_total > 0 else 0
        
        if profiler:
            profiler.end_operation(f"epoch_{epoch}_training")
            profiler.start_operation(f"epoch_{epoch}_validation")
            
        # ===== VALIDATION =====
        model.eval()
        val_losses = []
        val_direction_correct = 0
        val_direction_total = 0
        val_return_mse_sum = 0
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # Move to device
                features = features.to(device, non_blocking=True)
                features = torch.nan_to_num(features)
                targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
                asset_ids = targets.get('instrument_id')
                
                # Mixed precision inference
                with autocast():
                    outputs = model(features, targets['funding_rate'], asset_ids)
                    
                    # Compute losses
                    direction_loss = direction_criterion(outputs['direction_logits'], targets['direction_class'].squeeze())
                    return_loss = regression_criterion(outputs['expected_return'], targets['next_return_1bar'])
                    risk_loss = regression_criterion(outputs['expected_risk'], targets['next_volatility'])
                    loss = direction_loss + return_loss + 0.5 * risk_loss
                
                # Record metrics
                val_losses.append(loss.item())
                
                # Calculate accuracy
                pred_direction = torch.argmax(outputs['direction_logits'], dim=1)
                val_direction_correct += (pred_direction == targets['direction_class'].squeeze()).sum().item()
                val_direction_total += targets['direction_class'].size(0)
                
                # Calculate MSE
                val_return_mse_sum += return_loss.item() * targets['next_return_1bar'].size(0)
        
        if profiler:
            profiler.end_operation(f"epoch_{epoch}_validation")
            profiler.start_operation(f"epoch_{epoch}_metrics")
            
        # Calculate epoch metrics
        val_loss = np.mean(val_losses)
        val_direction_acc = val_direction_correct / val_direction_total if val_direction_total > 0 else 0
        val_return_mse = val_return_mse_sum / val_direction_total if val_direction_total > 0 else 0
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Update history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_direction_acc'].append(train_direction_acc)
        history['val_direction_acc'].append(val_direction_acc)
        history['learning_rate'].append(current_lr)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Time: {epoch_duration:.2f}s - "
              f"Train Loss: {train_loss:.6f} - "
              f"Val Loss: {val_loss:.6f} - "
              f"Train Acc: {train_direction_acc:.4f} - "
              f"Val Acc: {val_direction_acc:.4f} - "
              f"LR: {current_lr:.6f}")
        
        # Log to W&B
        if use_wandb:
            if profiler:
                profiler.start_operation("wandb_epoch_logging")
                
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_direction_acc': train_direction_acc,
                'val_direction_acc': val_direction_acc,
                'train_return_mse': train_return_mse,
                'val_return_mse': val_return_mse,
                'learning_rate': current_lr,
                'epoch_duration': epoch_duration
            })
            
            if profiler:
                profiler.end_operation("wandb_epoch_logging")
        
        # Add epoch time to profiler
        if profiler:
            profiler.record_epoch_time(epoch_duration)
            profiler.end_operation(f"epoch_{epoch}_metrics")
            
        # Save if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            if profiler:
                profiler.start_operation("save_best_model")
                
            checkpoint_path = f"{checkpoint_base}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            
            if profiler:
                profiler.end_operation("save_best_model")
                
            print(f"✅ Model improved! Saved to {checkpoint_path}")
            
            # Save to W&B
            if use_wandb:
                wandb.save(checkpoint_path)
                
                # Log model performance metrics
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_val_acc"] = val_direction_acc
                wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every N epochs to prevent data loss
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            if profiler:
                profiler.start_operation("save_checkpoint")
                
            checkpoint_path = f"{checkpoint_base}_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            
            if profiler:
                profiler.end_operation("save_checkpoint")
                
        if profiler:
            profiler.end_operation(f"epoch_{epoch}")
    
    # Save final model
    if profiler:
        profiler.start_operation("save_final_model")
        
    final_checkpoint_path = f"{checkpoint_base}_final.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'history': history
    }, final_checkpoint_path)
    
    if profiler:
        profiler.end_operation("save_final_model")
        
    print(f"✅ Training complete! Final model saved to {final_checkpoint_path}")
    
    # Create performance visualizations for W&B
    if use_wandb:
        if profiler:
            profiler.start_operation("wandb_create_visualizations")
            
        # Log final model
        wandb.save(final_checkpoint_path)
        
        # Create performance plots for W&B dashboard
        try:
            import matplotlib.pyplot as plt
            
            # Loss curve
            plt.figure(figsize=(10, 5))
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('/tmp/loss_curve.png')
            wandb.log({"loss_curve": wandb.Image('/tmp/loss_curve.png')})
            
            # Accuracy curve
            plt.figure(figsize=(10, 5))
            plt.plot(history['train_direction_acc'], label='Train Accuracy')
            plt.plot(history['val_direction_acc'], label='Validation Accuracy')
            plt.title('Direction Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('/tmp/accuracy_curve.png')
            wandb.log({"accuracy_curve": wandb.Image('/tmp/accuracy_curve.png')})
            
            # Learning rate curve
            plt.figure(figsize=(10, 5))
            plt.plot(history['learning_rate'], label='Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('/tmp/lr_curve.png')
            wandb.log({"lr_curve": wandb.Image('/tmp/lr_curve.png')})
        except Exception as e:
            print(f"Error creating plots: {e}")
            
        if profiler:
            profiler.end_operation("wandb_create_visualizations")
    
    return model, history


# OPTIMIZATION: Enhanced training function for maximum speed
def train_optimized_enhanced(model, train_loader, val_loader, optimizer, scheduler, scaler, num_epochs=20, 
                   device="cuda", patience=5, gradient_accumulation=1, model_save_path="/mnt/p/perpetual/models/checkpoints", 
                   experiment_name=None, debug=False, use_wandb=True, profiler=None):
    """
    Significantly enhanced training function with all optimizations applied.
    """
    print(f"🚀 Starting enhanced optimized training on {device} with mixed precision")
    print(f"📊 Using batch size {train_loader.batch_size} × {gradient_accumulation} accumulation steps")
    
    # Define loss functions - CrossEntropy for direction, Huber for regression (more robust)
    direction_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.HuberLoss(delta=1.0)  # More robust than MSE
    
    # Initialize metrics
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Ensure model save directory exists
    os.makedirs(model_save_path, exist_ok=True)
    
    # Create checkpoint filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        checkpoint_base = f"{model_save_path}/{experiment_name}_{timestamp}"
    else:
        checkpoint_base = f"{model_save_path}/model_{timestamp}"
    
    # Create a history dictionary to track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_direction_acc': [],
        'val_direction_acc': [],
        'learning_rate': []
    }
    
    # Set torch.cuda.amp precision
    torch.set_float32_matmul_precision('high')
    
    # Create online validator for more efficient validation
    class OnlineValidator:
        def __init__(self, model, criterion_dir, criterion_reg, device, keep_ratio=0.05):
            self.model = model
            self.criterion_dir = criterion_dir
            self.criterion_reg = criterion_reg
            self.device = device
            self.keep_ratio = keep_ratio
            self.buffer = []
            
        def update(self, features, targets):
            # Randomly keep some batches for validation
            if random.random() < self.keep_ratio:
                self.buffer.append((features.detach(), {k: v.detach() for k, v in targets.items()}))
                # Keep buffer size reasonable
                if len(self.buffer) > 100:
                    self.buffer.pop(0)
                    
        def validate(self):
            # Run validation on buffered batches
            self.model.eval()
            val_losses = []
            val_direction_correct = 0
            val_direction_total = 0
            val_return_mse_sum = 0
            
            with torch.no_grad():
                for features, targets in self.buffer:
                    # Move to device
                    features = features.to(self.device)
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                    asset_ids = targets.get('instrument_id')
                    
                    # Mixed precision inference
                    with autocast():
                        outputs = self.model(features, targets['funding_rate'], asset_ids)
                        
                        # Compute losses
                        direction_loss = self.criterion_dir(outputs['direction_logits'], targets['direction_class'].squeeze())
                        return_loss = self.criterion_reg(outputs['expected_return'], targets['next_return_1bar'])
                        risk_loss = self.criterion_reg(outputs['expected_risk'], targets['next_volatility'])
                        loss = direction_loss + return_loss + 0.5 * risk_loss
                    
                    # Record metrics
                    val_losses.append(loss.item())
                    
                    # Calculate accuracy
                    pred_direction = torch.argmax(outputs['direction_logits'], dim=1)
                    val_direction_correct += (pred_direction == targets['direction_class'].squeeze()).sum().item()
                    val_direction_total += targets['direction_class'].size(0)
                    
                    # Calculate MSE
                    val_return_mse_sum += return_loss.item() * targets['next_return_1bar'].size(0)
            
            if val_direction_total == 0:
                return 0, 0, 0
                
            val_loss = np.mean(val_losses) if val_losses else 0
            val_direction_acc = val_direction_correct / val_direction_total if val_direction_total > 0 else 0
            val_return_mse = val_return_mse_sum / val_direction_total if val_direction_total > 0 else 0
            
            return val_loss, val_direction_acc, val_return_mse
    
    # Create online validator
    online_validator = OnlineValidator(
        model, direction_criterion, regression_criterion, device, keep_ratio=0.1
    )
    
    # Training loop
    for epoch in range(num_epochs):
        # Profile epoch time
        if profiler:
            profiler.start_operation(f"epoch_{epoch}")
            profiler.start_operation(f"epoch_{epoch}_training")
        
        epoch_start_time = time.time()
        
        # ===== TRAINING =====
        model.train()
        train_losses = []
        train_direction_correct = 0
        train_direction_total = 0
        train_return_mse_sum = 0
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Use tqdm with a higher update frequency for more responsive UI
        for batch_idx, (features, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", miniters=10)):
            # Profile data loading and transfer
            if profiler:
                profiler.start_operation("data_transfer")
                
            # Move to device more efficiently with non_blocking=True
            features = features.to(device, non_blocking=True)
            
            # Use more efficient tensor sanitization
            features = torch.clamp(features, min=-10.0, max=10.0)
            
            # Move all target tensors to device with non_blocking
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            
            if profiler:
                profiler.end_operation("data_transfer")
                profiler.start_operation("forward_pass")
            
            # Get instrument IDs for asset-aware models
            asset_ids = targets.get('instrument_id')
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(features, targets['funding_rate'], asset_ids)
            
            if profiler:
                profiler.end_operation("forward_pass") 
                profiler.start_operation("loss_computation")
            
            with autocast():
                # Compute losses
                direction_loss = direction_criterion(outputs['direction_logits'], targets['direction_class'].squeeze())
                return_loss = regression_criterion(outputs['expected_return'], targets['next_return_1bar'])
                risk_loss = regression_criterion(outputs['expected_risk'], targets['next_volatility'])
                
                # Combined loss with weighting
                loss = direction_loss + return_loss + 0.5 * risk_loss
                
                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation
            
            if profiler:
                profiler.end_operation("loss_computation")
                profiler.start_operation("backward_pass")
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            if profiler:
                profiler.end_operation("backward_pass")
            
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation == 0:
                if profiler:
                    profiler.start_operation("optimizer_step")
                    
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Clip gradients to prevent explosion using a more efficient method
                # This operation is performed in-place
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights and zero gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # set_to_none=True is faster than setting to zero
                
                # Update learning rate each batch with OneCycleLR
                scheduler.step()
                
                if profiler:
                    profiler.end_operation("optimizer_step")
            
            # Record metrics
            train_losses.append(loss.item() * gradient_accumulation)  # Rescale loss back
            
            # Calculate accuracy more efficiently
            with torch.no_grad():
                pred_direction = torch.argmax(outputs['direction_logits'], dim=1)
                correct = (pred_direction == targets['direction_class'].squeeze()).sum().item()
                
                train_direction_correct += correct
                train_direction_total += targets['direction_class'].size(0)
                
                # Calculate MSE for returns
                train_return_mse_sum += return_loss.item() * gradient_accumulation * targets['next_return_1bar'].size(0)
            
            # Update online validator for streaming validation
            online_validator.update(features, targets)
            
            # Record batch processing time if profiling
            if profiler and batch_idx > 0:  # Skip first batch (warm-up)
                batch_time = time.time() - epoch_start_time
                profiler.record_batch_time(batch_time / batch_idx)  # Average time per batch so far
            
            # Log batch metrics to W&B (but not too frequently)
            if use_wandb and batch_idx % 50 == 0:  # Reduced logging frequency
                if profiler:
                    profiler.start_operation("wandb_batch_logging")
                    
                # Calculate GPU memory metrics
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    
                    # Log memory usage
                    wandb.log({
                        'gpu_memory_allocated_GB': gpu_memory_allocated,
                        'gpu_memory_reserved_GB': gpu_memory_reserved,
                        'iteration': batch_idx + epoch * len(train_loader)
                    })
                    
                wandb.log({
                    'batch_loss': loss.item() * gradient_accumulation,
                    'batch_direction_loss': direction_loss.item(),
                    'batch_return_loss': return_loss.item(),
                    'batch_risk_loss': risk_loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'batch': batch_idx + epoch * len(train_loader)
                })
                
                if profiler:
                    profiler.end_operation("wandb_batch_logging")
                
            # Streaming validation every N batches
            if batch_idx % 500 == 499:
                # Run quick validation with online validator
                quick_val_loss, quick_val_acc, quick_val_mse = online_validator.validate()
                
                # Log partial epoch metrics
                print(f"Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Train Loss: {np.mean(train_losses[-100:]):.6f} - "
                      f"Val Loss: {quick_val_loss:.6f} - "
                      f"Val Acc: {quick_val_acc:.4f}")
                
                if use_wandb:
                    wandb.log({
                        'quick_val_loss': quick_val_loss,
                        'quick_val_accuracy': quick_val_acc,
                        'quick_val_mse': quick_val_mse,
                        'batch': batch_idx + epoch * len(train_loader)
                    })
        
        # Calculate epoch metrics
        train_loss = np.mean(train_losses)
        train_direction_acc = train_direction_correct / train_direction_total if train_direction_total > 0 else 0
        train_return_mse = train_return_mse_sum / train_direction_total if train_direction_total > 0 else 0
        
        if profiler:
            profiler.end_operation(f"epoch_{epoch}_training")
            profiler.start_operation(f"epoch_{epoch}_validation")
            
        # ===== VALIDATION =====
        # Use full validation only on epoch boundaries
        model.eval()
        val_losses = []
        val_direction_correct = 0
        val_direction_total = 0
        val_return_mse_sum = 0
        
        with torch.no_grad():
            # Use online validation if available
            online_val_loss, online_val_acc, online_val_mse = online_validator.validate()
            
            # Only run full validation every N epochs (or if online validation has too few samples)
            run_full_validation = (epoch % 5 == 0) or (epoch == num_epochs - 1) or len(online_validator.buffer) < 10
            
            if run_full_validation:
                for features, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    # Move to device
                    features = features.to(device, non_blocking=True)
                    features = torch.clamp(features, min=-10.0, max=10.0)
                    targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
                    asset_ids = targets.get('instrument_id')
                    
                    # Mixed precision inference
                    with autocast():
                        outputs = model(features, targets['funding_rate'], asset_ids)
                        
                        # Compute losses
                        direction_loss = direction_criterion(outputs['direction_logits'], targets['direction_class'].squeeze())
                        return_loss = regression_criterion(outputs['expected_return'], targets['next_return_1bar'])
                        risk_loss = regression_criterion(outputs['expected_risk'], targets['next_volatility'])
                        loss = direction_loss + return_loss + 0.5 * risk_loss
                    
                    # Record metrics
                    val_losses.append(loss.item())
                    
                    # Calculate accuracy
                    pred_direction = torch.argmax(outputs['direction_logits'], dim=1)
                    val_direction_correct += (pred_direction == targets['direction_class'].squeeze()).sum().item()
                    val_direction_total += targets['direction_class'].size(0)
                    
                    # Calculate MSE
                    val_return_mse_sum += return_loss.item() * targets['next_return_1bar'].size(0)
                
                # Calculate epoch metrics
                val_loss = np.mean(val_losses)
                val_direction_acc = val_direction_correct / val_direction_total if val_direction_total > 0 else 0
                val_return_mse = val_return_mse_sum / val_direction_total if val_direction_total > 0 else 0
            else:
                # Use online validation metrics
                val_loss = online_val_loss
                val_direction_acc = online_val_acc
                val_return_mse = online_val_mse
        
        if profiler:
            profiler.end_operation(f"epoch_{epoch}_validation")
            profiler.start_operation(f"epoch_{epoch}_metrics")
            
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Update history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_direction_acc'].append(train_direction_acc)
        history['val_direction_acc'].append(val_direction_acc)
        history['learning_rate'].append(current_lr)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Time: {epoch_duration:.2f}s - "
              f"Train Loss: {train_loss:.6f} - "
              f"Val Loss: {val_loss:.6f} - "
              f"Train Acc: {train_direction_acc:.4f} - "
              f"Val Acc: {val_direction_acc:.4f} - "
              f"LR: {current_lr:.6f}")
        
        # Log to W&B
        if use_wandb:
            if profiler:
                profiler.start_operation("wandb_epoch_logging")
                
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_direction_acc': train_direction_acc,
                'val_direction_acc': val_direction_acc,
                'train_return_mse': train_return_mse,
                'val_return_mse': val_return_mse,
                'learning_rate': current_lr,
                'epoch_duration': epoch_duration
            })
            
            if profiler:
                profiler.end_operation("wandb_epoch_logging")
        
        # Add epoch time to profiler
        if profiler:
            profiler.record_epoch_time(epoch_duration)
            profiler.end_operation(f"epoch_{epoch}_metrics")
            
        # Save if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            if profiler:
                profiler.start_operation("save_best_model")
                
            checkpoint_path = f"{checkpoint_base}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            
            if profiler:
                profiler.end_operation("save_best_model")
                
            print(f"✅ Model improved! Saved to {checkpoint_path}")
            
            # Save to W&B
            if use_wandb:
                wandb.save(checkpoint_path)
                
                # Log model performance metrics
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_val_acc"] = val_direction_acc
                wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every N epochs to prevent data loss
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            if profiler:
                profiler.start_operation("save_checkpoint")
                
            checkpoint_path = f"{checkpoint_base}_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            
            if profiler:
                profiler.end_operation("save_checkpoint")
                
        if profiler:
            profiler.end_operation(f"epoch_{epoch}")
    
    # Save final model
    if profiler:
        profiler.start_operation("save_final_model")
        
    final_checkpoint_path = f"{checkpoint_base}_final.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'history': history
    }, final_checkpoint_path)
    
    if profiler:
        profiler.end_operation("save_final_model")
        
    print(f"✅ Training complete! Final model saved to {final_checkpoint_path}")
    
    # Create performance visualizations for W&B
    if use_wandb:
        if profiler:
            profiler.start_operation("wandb_create_visualizations")
            
        # Log final model
        wandb.save(final_checkpoint_path)
        
        # Create performance plots for W&B dashboard
        try:
            import matplotlib.pyplot as plt
            
            # Loss curve
            plt.figure(figsize=(10, 5))
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('/tmp/loss_curve.png')
            wandb.log({"loss_curve": wandb.Image('/tmp/loss_curve.png')})
            
            # Accuracy curve
            plt.figure(figsize=(10, 5))
            plt.plot(history['train_direction_acc'], label='Train Accuracy')
            plt.plot(history['val_direction_acc'], label='Validation Accuracy')
            plt.title('Direction Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('/tmp/accuracy_curve.png')
            wandb.log({"accuracy_curve": wandb.Image('/tmp/accuracy_curve.png')})
            
            # Learning rate curve
            plt.figure(figsize=(10, 5))
            plt.plot(history['learning_rate'], label='Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('/tmp/lr_curve.png')
            wandb.log({"lr_curve": wandb.Image('/tmp/lr_curve.png')})
        except Exception as e:
            print(f"Error creating plots: {e}")
            
        if profiler:
            profiler.end_operation("wandb_create_visualizations")
    
    return model, history


def create_ensemble_model(base_config, num_models=3):
    """
    Create an ensemble of models with different random seeds and configurations.
    This implements a simple form of model souping to improve performance.
    """
    ensemble_models = []
    ensemble_configs = []
    
    # Create variations of the base config
    for i in range(num_models):
        # Create a copy of the base config
        model_config = base_config.copy()
        
        # Apply variations
        seed = 42 + i
        set_seed(seed)
        
        # Vary some hyperparameters
        if i == 1:
            model_config["transformer_heads"] = base_config.get("transformer_heads", 8) // 2
            model_config["dropout"] = base_config.get("dropout", 0.2) * 1.5
        elif i == 2:
            model_config["tcn_channels"] = [c // 2 for c in base_config.get("tcn_channels", [128, 128, 128])]
            model_config["transformer_layers"] = base_config.get("transformer_layers", 4) - 1
        
        ensemble_configs.append(model_config)
    
    return ensemble_configs


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Deribit Perpetual Trading Model")
    parser.add_argument("--instruments", type=str, nargs="+", 
                      help="List of instruments to train on (default: all used=TRUE instruments)")
    parser.add_argument("--config", type=str, default="/mnt/p/perpetual/models/training_config.json",
                      help="Path to training configuration file (not required, using built-in defaults)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for fast testing")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--profile", action="store_true", help="Enable detailed performance profiling")
    parser.add_argument("--export-parquet", action="store_true", 
                      help="Export data to Parquet format before training")
    parser.add_argument("--max-speed", action="store_true", 
                      help="Enable all speed optimizations (may reduce accuracy slightly)")
    parser.add_argument("--output-dir", type=str, default="/mnt/p/perpetual/models/checkpoints",
                      help="Directory to save output models")
    parser.add_argument("--cache-dir", type=str, default="/tmp/deribit_cache",
                      help="Directory to cache processed data")
    args = parser.parse_args()
    
    # Initialize profiler if enabled
    profiler = None
    if args.profile:
        from models.profiler import TrainingProfiler
        profiler = TrainingProfiler("/mnt/p/perpetual/tmp")
        print("🔍 Performance profiling enabled")
    
    # Use W&B unless explicitly disabled
    use_wandb = not args.no_wandb
    
    # Get list of instruments if not provided
    if not args.instruments:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT instrument_name
                    FROM model_features_15m_tier1
                    WHERE used = TRUE
                    ORDER BY instrument_name
                """)
                instruments = [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"Error fetching instruments: {e}")
            instruments = ["BTC_USDC-PERPETUAL", "ETH_USDC-PERPETUAL"]
        finally:
            conn.close()
    else:
        instruments = args.instruments
    
    # Export data to Parquet if requested
    if args.export_parquet:
        if profiler:
            profiler.start_operation("export_parquet")
            
        print("🔄 Exporting instrument data to Parquet format for faster loading")
        
        # Import the export_parquet module
        try:
            from features.export_parquet import export_all_instruments
            
            # Create the cache directory if it doesn't exist
            os.makedirs("/mnt/p/perpetual/cache", exist_ok=True)
            
            # Export all instruments
            export_all_instruments("/mnt/p/perpetual/cache", overwrite=False)
            print("✅ Export to Parquet completed")
        except Exception as e:
            print(f"⚠️ Error exporting to Parquet: {e}")
            print("Continuing with database loading")
            
        if profiler:
            profiler.end_operation("export_parquet")
    
    # Adjust for debug mode
    if args.debug:
        print("⚡ Debug mode enabled - using smaller model and dataset")
        instruments = instruments[:3]  # Only use first 3 instruments
        model_params = {
            "tcn_channels": [32, 32],
            "tcn_kernel_size": 3,
            "transformer_dim": 32,
            "transformer_heads": 2,
            "transformer_layers": 2,
            "dropout": 0.2,
            "max_seq_length": 512
        }
        training_params = {
            "seq_length": 32,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "num_epochs": 1,
            "patience": 3,
            "gradient_accumulation": 2,
            "num_workers": 2,
            "model_save_path": args.output_dir  # Use the output_dir from arguments
        }
    else:
        # Default optimized parameters for unified model with improved GPU utilization
        model_params = {
            "tcn_channels": [128, 128, 128, 128],  # Much larger model
            "tcn_kernel_size": 5,                  # Increased from 3 for better pattern detection
            "transformer_dim": 128,                # Doubled from 64 for better representation
            "transformer_heads": 8,                # Increased from 4 for better attention
            "transformer_layers": 4,               # Doubled for better sequence modeling
            "dropout": 0.3,                        # Increased for better regularization
            "max_seq_length": 1024
        }
        
        # Determine optimal batch size and workers based on available GPU
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cpu_count = os.cpu_count()
            
            # Scale batch size with GPU memory (heuristic)
            if gpu_mem_gb > 16:
                batch_size = 256
                gradient_accum = 1
            elif gpu_mem_gb > 10:
                batch_size = 128
                gradient_accum = 2
            elif gpu_mem_gb > 6:
                batch_size = 64
                gradient_accum = 4
            else:
                batch_size = 32
                gradient_accum = 8
                
            # Scale workers with CPU count, but not too many
            num_workers = min(8, max(4, cpu_count // 2))
        else:
            # CPU fallback
            batch_size = 32
            gradient_accum = 4
            num_workers = 2
        
        training_params = {
            "seq_length": 96,                    # Increased from 64 for longer context
            "batch_size": batch_size,
            "learning_rate": 0.001,
            "weight_decay": 0.0002,              # Increased for better regularization
            "num_epochs": 40,                    # Doubled from 20 for better convergence
            "patience": 8,                       # Increased for more training time
            "gradient_accumulation": gradient_accum,
            "num_workers": num_workers,
            "model_save_path": args.output_dir   # Use the output_dir from arguments
        }
    
    # Use highly optimized settings if max-speed is enabled
    if args.max_speed:
        print("⚡ MAX SPEED MODE: Using optimized model and training settings")
        
        # Update model parameters for max speed
        model_params = {
            "tcn_channels": [64, 64],           # Smaller but faster model
            "tcn_kernel_size": 3,               # Smaller kernel is faster
            "transformer_dim": 64,              # Reduced dimension for speed
            "transformer_heads": 4,             # Fewer heads for faster computation
            "transformer_layers": 2,            # Fewer layers for speed
            "dropout": 0.2,                     # Less dropout for faster convergence
            "max_seq_length": 512               # Shorter sequence length
        }
        
        # Update training parameters for max speed
        training_params["batch_size"] = int(training_params["batch_size"] * 1.5)  # Larger batch size
        training_params["gradient_accumulation"] = 1  # No gradient accumulation
        training_params["learning_rate"] = 0.002  # Higher learning rate
        training_params["num_epochs"] = 20  # Fewer epochs
    
    # Print training configuration
    print("\n----- Training Configuration -----")
    print(f"🧠 Model Architecture: {'Optimized TCN + Lightweight Transformer' if args.max_speed else 'TCN + Transformer Hybrid'}")
    print(f"🔢 Model Parameters: {model_params}")
    print(f"⚙️ Training Parameters: {training_params}")
    print(f"💱 Instruments: {len(instruments)} total")
    print(f"📂 Data Source: {'Parquet (Memory-Mapped)' if os.path.exists(f'/mnt/p/perpetual/cache/tier1_{instruments[0]}.parquet') else 'PostgreSQL Database'}")
    print(f"🔍 Debug Mode: {args.debug}")
    print(f"📊 Weights & Biases: {'Enabled' if use_wandb else 'Disabled'}")
    print(f"⚡ Maximum Speed Mode: {args.max_speed}")
    print(f"💾 Cache Directory: {args.cache_dir}")
    print(f"💾 Output Directory: {args.output_dir}")
    
    # Display hardware information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🖥️  Hardware: {gpu_name} ({gpu_mem:.1f} GB VRAM), {os.cpu_count()} CPU cores")
        print(f"⚡ Effective Batch Size: {training_params['batch_size'] * training_params['gradient_accumulation']}")
    print("----------------------------------\n")
    
    # Set global random seed
    set_seed(42)
    
    # Train unified model
    train_unified_model(
        instruments=instruments,
        model_params=model_params,
        training_params=training_params,
        use_wandb=use_wandb,
        debug=args.debug,
        profiler=profiler,
        max_speed=args.max_speed,
        cache_dir=args.cache_dir
    )
    
    # Save profiling results
    if profiler:
        profiler.save_results()

if __name__ == "__main__":
    main()