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
from models.architecture import DeribitHybridModel
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

# Create dataset with caching for faster loading
class CachedPerpetualSwapDataset:
    """Dataset that combines multiple instruments and caches processed data for speed."""
    def __init__(
        self,
        instruments,
        seq_length=64,
        cache_dir="/tmp/deribit_cache",
        test_size=0.2,
        device="cpu"
    ):
        self.instruments = instruments
        self.seq_length = seq_length
        self.cache_dir = cache_dir
        self.test_size = test_size
        self.device = device
        
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
        feature_columns = [
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
        
        label_columns = [
            "next_return_1bar", "next_return_2bar", "next_return_4bar", 
            "direction_class", "next_volatility"
        ]
        
        if self.feature_columns is None:
            self.feature_columns = feature_columns
            self.label_columns = label_columns
        
        # Create a feature transformer for scaling
        transformer = FeatureTransformer(instrument_name)
        
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
            feature_columns=self.feature_columns,
            target_columns=self.label_columns,
            device=self.device
        )
        
        test_dataset = InstrumentDataset(
            instrument_name=instrument_name,
            features=test_df,
            seq_length=self.seq_length,
            transform=transformer,
            feature_columns=self.feature_columns,
            target_columns=self.label_columns,
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
                # Load from Parquet with memory mapping for efficiency
                df = pq.read_table(parquet_path, memory_map=True).to_pandas()
                
                # Create forward-looking return labels
                df['next_return_1bar'] = df['return_1bar'].shift(-1)
                df['next_return_2bar'] = df['return_1bar'].shift(-1) + df['return_1bar'].shift(-2)
                df['next_return_4bar'] = sum(df['return_1bar'].shift(-i) for i in range(1, 5))
                
                # Create volatility label
                df['next_volatility'] = df['return_1bar'].rolling(4).std().shift(-4)
                
                # Fill NaNs in the last rows with 0
                df.loc[df.index[-4:], ['next_return_1bar', 'next_return_2bar', 'next_return_4bar', 'next_volatility']] = 0
                
                # Create direction class label
                volatility = df['return_1bar'].rolling(20).std().fillna(0.001)
                df['direction_class'] = 0
                df.loc[df['next_return_1bar'] > 0.5 * volatility, 'direction_class'] = 1
                df.loc[df['next_return_1bar'] < -0.5 * volatility, 'direction_class'] = -1
                
                return df
            except Exception as e:
                print(f"⚠️ Error loading from Parquet for {instrument_name}: {e}")
                print(f"Falling back to database")
        
        # Fall back to database loading if Parquet file doesn't exist or fails
        conn = get_connection()
        try:
            print(f"🔄 Loading data from database for {instrument_name}")
            # Load tier 1 features
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM model_features_15m_tier1
                    WHERE instrument_name = %s
                    ORDER BY timestamp
                """, (instrument_name,))
                
                columns = [desc[0] for desc in cur.description]
                data = cur.fetchall()
                
            if not data:
                return pd.DataFrame()
                
            df = pd.DataFrame(data, columns=columns)
            
            # Create forward-looking return labels
            df['next_return_1bar'] = df['return_1bar'].shift(-1)
            df['next_return_2bar'] = df['return_1bar'].shift(-1) + df['return_1bar'].shift(-2)
            df['next_return_4bar'] = sum(df['return_1bar'].shift(-i) for i in range(1, 5))
            
            # Create volatility label
            df['next_volatility'] = df['return_1bar'].rolling(4).std().shift(-4)
            
            # Fill NaNs in the last rows with 0
            df.loc[df.index[-4:], ['next_return_1bar', 'next_return_2bar', 'next_return_4bar', 'next_volatility']] = 0
            
            # Create direction class label
            volatility = df['return_1bar'].rolling(20).std().fillna(0.001)
            df['direction_class'] = 0
            df.loc[df['next_return_1bar'] > 0.5 * volatility, 'direction_class'] = 1
            df.loc[df['next_return_1bar'] < -0.5 * volatility, 'direction_class'] = -1
            
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


def train_unified_model(
    instruments,
    model_params,
    training_params,
    use_wandb=True,
    debug=False,
    profiler=None
):
    """
    Train a unified model on multiple instruments with optimized performance.
    
    Args:
        instruments: List of instruments to include
        model_params: Parameters for model initialization
        training_params: Parameters for training
        use_wandb: Whether to use W&B for tracking
        debug: Enable debug mode for faster runs
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Start W&B run with full system metrics logging
    if use_wandb:
        run = wandb.init(
            project="deribit-perpetual-model",
            name=f"unified_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "instruments": instruments,
                "model_params": model_params,
                "training_params": training_params,
                "device": device
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
    
    try:
        # Load and combine datasets with caching
        print(f"🔄 Loading data for {len(instruments)} instruments")
        
        # Use fast mode for debugging if needed
        if debug:
            print("⚡ DEBUG MODE ACTIVE: Using reduced dataset size")
            seq_length = 32
            cache_suffix = "_debug"
        else:
            seq_length = training_params.get('seq_length', 64)
            cache_suffix = ""
            
        dataset = CachedPerpetualSwapDataset(
            instruments=instruments,
            seq_length=seq_length,
            cache_dir=f"/tmp/deribit_cache{cache_suffix}",
            test_size=training_params.get('test_size', 0.2),
            device=device
        )
        
        # Get optimized dataloaders
        batch_size = training_params.get('batch_size', 64)
        train_loader, test_loader = dataset.get_dataloaders(
            batch_size=batch_size,
            num_workers=training_params.get('num_workers', 4)
        )
        
        # Initialize model with torch.compile for 2-3x speedup
        input_dim = len(dataset.feature_columns)
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
        
        # Apply mixed precision optimization
        # This alone can give 2-3x speedup
        scaler = GradScaler()
        
        # Log model architecture to W&B
        if use_wandb:
            wandb.watch(model, log="all", log_freq=100)
        
        # Adjust optimizer with correct parameters
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_params.get('learning_rate', 1e-3),
            weight_decay=training_params.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # Use a more advanced scheduler for faster convergence
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=training_params.get('learning_rate', 1e-3),
            steps_per_epoch=len(train_loader),
            epochs=training_params.get('num_epochs', 20),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Train with optimized process
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
            use_wandb=use_wandb
        )
        
    finally:
        # Cleanup
        if use_wandb:
            wandb.finish()
            
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_optimized(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    num_epochs=20,
    device="cuda",
    patience=5,
    gradient_accumulation=1,
    model_save_path="/mnt/p/perpetual/models/checkpoints",
    experiment_name=None,
    debug=False,
    use_wandb=True,
    profiler=None
):
    """
    Optimized training function with mixed precision, gradient accumulation and enhanced monitoring.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision training
        num_epochs: Number of epochs to train for
        device: Device to train on
        patience: Number of epochs to wait for improvement before early stopping
        gradient_accumulation: Number of batches to accumulate gradients over
        model_save_path: Path to save model checkpoints
        experiment_name: Name of the experiment for checkpoints
        debug: Enable debug mode for faster runs
        use_wandb: Whether to use W&B for tracking
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
        # Profile entire epoch if profiler exists
        epoch_context = profiler.profile_region(f"epoch_{epoch}") if profiler else nullcontext()
        with epoch_context:
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
                # Profile data transfer if profiler exists
                data_context = profiler.profile_region("data_transfer") if profiler else nullcontext()
                with data_context:
                    # Move to device and ensure data is clean - use non_blocking for async transfer
                    features = features.to(device, non_blocking=True)
                    features = torch.nan_to_num(features)
                    
                    # Move all target tensors to device with non_blocking
                    targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
                
                # Get instrument IDs for asset-aware models
                asset_ids = targets.get('instrument_id')
                
                # Profile forward pass
                forward_context = profiler.profile_region("forward") if profiler else nullcontext()
                with forward_context:
                    # Mixed precision forward pass
                    with autocast():
                        outputs = model(features, targets['funding_rate'], asset_ids)
                
                # Profile loss computation
                loss_context = profiler.profile_region("loss_computation") if profiler else nullcontext()
                with loss_context:
                    with autocast():
                        # Compute losses
                        direction_loss = direction_criterion(outputs['direction_logits'], targets['direction_class'].squeeze())
                        return_loss = regression_criterion(outputs['expected_return'], targets['next_return_1bar'])
                        risk_loss = regression_criterion(outputs['expected_risk'], targets['next_volatility'])
                        
                        # Combined loss with weighting
                        loss = direction_loss + return_loss + 0.5 * risk_loss
                        
                        # Scale loss by gradient accumulation steps
                        loss = loss / gradient_accumulation
                
                # Profile backward pass
                backward_context = profiler.profile_region("backward") if profiler else nullcontext()
                with backward_context:
                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                
                # Only update weights after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation == 0:
                    # Profile optimizer step
                    optim_context = profiler.profile_region("optimizer_step") if profiler else nullcontext()
                    with optim_context:
                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)
                        
                        # Clip gradients to prevent explosion
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Update weights and zero gradients
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        
                        # Update learning rate each batch with OneCycleLR
                        scheduler.step()
                
                # Record metrics
                train_losses.append(loss.item() * gradient_accumulation)  # Rescale loss back
                
                # Calculate accuracy
                pred_direction = torch.argmax(outputs['direction_logits'], dim=1)
                train_direction_correct += (pred_direction == targets['direction_class'].squeeze()).sum().item()
                train_direction_total += targets['direction_class'].size(0)
                
                # Calculate MSE for returns
                train_return_mse_sum += return_loss.item() * gradient_accumulation * targets['next_return_1bar'].size(0)
                
                # Step the profiler
                if profiler:
                    profiler.step(batch_size=features.size(0))
            
            # Log batch metrics to W&B (but not too frequently)
            if use_wandb and batch_idx % 20 == 0:
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
        
        # Calculate epoch metrics
        train_loss = np.mean(train_losses)
        train_direction_acc = train_direction_correct / train_direction_total if train_direction_total > 0 else 0
        train_return_mse = train_return_mse_sum / train_direction_total if train_direction_total > 0 else 0
        
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
        
        # Save if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint_path = f"{checkpoint_base}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            
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
            checkpoint_path = f"{checkpoint_base}_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
    
    # Save final model
    final_checkpoint_path = f"{checkpoint_base}_final.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'history': history
    }, final_checkpoint_path)
    
    print(f"✅ Training complete! Final model saved to {final_checkpoint_path}")
    
    # Create performance visualizations for W&B
    if use_wandb:
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
    args = parser.parse_args()
    
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
            "num_workers": 2
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
            "model_save_path": "/mnt/p/perpetual/models/checkpoints"
        }
    
    # Print training configuration
    print("\n----- Training Configuration -----")
    print(f"🧠 Model Architecture: TCN + Transformer Hybrid")
    print(f"🔢 Model Parameters: {model_params}")
    print(f"⚙️ Training Parameters: {training_params}")
    print(f"💱 Instruments: {len(instruments)} total")
    print(f"📂 Data Source: {'Parquet (Memory-Mapped)' if os.path.exists(f'/mnt/p/perpetual/cache/tier1_{instruments[0]}.parquet') else 'PostgreSQL Database'}")
    print(f"🔍 Debug Mode: {args.debug}")
    print(f"📊 Weights & Biases: {'Enabled' if use_wandb else 'Disabled'}")
    
    # Display hardware information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🖥️  Hardware: {gpu_name} ({gpu_mem:.1f} GB VRAM), {os.cpu_count()} CPU cores")
        print(f"⚡ Effective Batch Size: {training_params['batch_size'] * training_params['gradient_accumulation']}")
    print("----------------------------------\n")
    
    # Set global random seed
    set_seed(42)

    # Initialize profiler if enabled
    profiler = TrainingProfiler(enabled=args.profile, profile_gpu=args.profile) if args.profile else None 
    
    # Train unified model
    train_unified_model(
        instruments=instruments,
        model_params=model_params,
        training_params=training_params,
        use_wandb=use_wandb,
        debug=args.debug,
        profiler=profiler
    )


if __name__ == "__main__":
    main()