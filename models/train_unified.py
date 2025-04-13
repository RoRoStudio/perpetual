#!/usr/bin/env python3
"""
Streamlined training script for Deribit perpetual trading model.
Focuses on maximum performance with minimal complexity.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import time
import random
import gc
import pyarrow.parquet as pq
import json
import tempfile

# Add project root to path for imports
sys.path.append('/mnt/p/perpetual')

# Import project modules
from data.database import get_connection
from features.transformers import FeatureTransformer
from models.architecture import DeribitHybridModel, OptimizedDeribitModel
from models.profiler import TrainingProfiler

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False  # Better performance but less reproducible
        torch.backends.cudnn.benchmark = True  # Optimize convolution for fixed input sizes

class FastDataset:
    """Simplified dataset with optimized loading from Parquet"""
    def __init__(self, instruments, seq_length=32, test_size=0.2):
        self.instruments = instruments
        self.seq_length = seq_length
        self.test_size = test_size
        
        self.feature_columns = None
        self.label_columns = [
            "future_return_1bar", "future_return_2bar", "future_return_4bar", 
            "direction_class", "direction_signal", "future_volatility", "signal_confidence"
        ]
        
        # Add this line to create asset ID mapping
        self.asset_ids = {instrument: i for i, instrument in enumerate(instruments)}
        
        # Load datasets
        self.train_data, self.test_data = self._load_data()
        
    def _load_data(self):
        """Load data from Parquet files with optimized settings"""
        all_train_features = []
        all_train_targets = []
        all_test_features = []
        all_test_targets = []
        # NEW: Add lists for asset IDs
        all_train_asset_ids = []
        all_test_asset_ids = []
        
        for instrument in self.instruments:
            # Try to load from Parquet (faster)
            parquet_path = f"/mnt/p/perpetual/cache/tier1_{instrument}.parquet"
            if os.path.exists(parquet_path):
                print(f"Loading data from Parquet for {instrument}")
                # Load using PyArrow with memory mapping for efficiency
                table = pq.read_table(parquet_path, memory_map=True)
                df = table.to_pandas()
                
                # Use the pre-computed forward-looking metrics directly
                # These are already in our database from compute_tier1.py
                df['next_volatility'] = df['future_volatility']
                # For backward compatibility, ensure these columns exist
                if 'future_return_1bar' not in df.columns:
                    print(f"Warning: 'future_return_1bar' not found in data, using fallback")
                    df['future_return_1bar'] = df['return_1bar'].shift(-1)
                    df['future_return_2bar'] = df['return_1bar'].shift(-1) + df['return_1bar'].shift(-2)
                    df['future_return_4bar'] = df['return_1bar'].shift(-1) + df['return_1bar'].shift(-2) + \
                                          df['return_1bar'].shift(-3) + df['return_1bar'].shift(-4)
                    df['future_volatility'] = df['return_1bar'].rolling(4).std().shift(-4)
                    df['signal_confidence'] = 1.0  # Default confidence
                
                # Fill NaNs
                for col in self.label_columns:
                    df.loc[df.index[-4:], col] = 0
                
                # Store feature columns if not already set
                if self.feature_columns is None:
                    # Select all columns except targets and metadata
                    exclude_cols = self.label_columns + ['instrument_name', 'timestamp']
                    self.feature_columns = [col for col in df.columns if col not in exclude_cols]
                
                # Apply transformation
                transformer = FeatureTransformer(instrument)
                features_df = transformer.transform(df[self.feature_columns])
                targets_df = df[self.label_columns]
                
                # Create sequences
                features, targets = self._create_sequences(features_df, targets_df)
                
                # Split into train/test
                split_idx = int(len(features) * (1 - self.test_size))
                train_features = features[:split_idx]
                train_targets = targets[:split_idx]
                test_features = features[split_idx:]
                test_targets = targets[split_idx:]
                
                # Add to list
                all_train_features.append(train_features)
                all_train_targets.append(train_targets)
                all_test_features.append(test_features)
                all_test_targets.append(test_targets)
                
                # NEW: Add these lines to create and store asset IDs
                instrument_id = self.asset_ids.get(instrument, 0)
                train_asset_ids = torch.full((len(train_features),), instrument_id, dtype=torch.long)
                test_asset_ids = torch.full((len(test_features),), instrument_id, dtype=torch.long)
                all_train_asset_ids.append(train_asset_ids)
                all_test_asset_ids.append(test_asset_ids)
        
        # Concatenate all data
        train_features = torch.cat(all_train_features) if all_train_features else torch.tensor([])
        train_targets = torch.cat(all_train_targets) if all_train_targets else torch.tensor([])
        test_features = torch.cat(all_test_features) if all_test_features else torch.tensor([])
        test_targets = torch.cat(all_test_targets) if all_test_targets else torch.tensor([])
        
        # NEW: Concatenate asset IDs
        train_asset_ids = torch.cat(all_train_asset_ids) if all_train_asset_ids else torch.tensor([])
        test_asset_ids = torch.cat(all_test_asset_ids) if all_test_asset_ids else torch.tensor([])
        
        print(f"✅ Loaded {len(train_features)} training and {len(test_features)} test samples")
        
        # NEW: Return asset IDs as part of the data tuples
        return (train_features, train_targets, train_asset_ids), (test_features, test_targets, test_asset_ids)
    
    def _create_sequences(self, features_df, targets_df):
        """Create sequences from DataFrames efficiently"""
        # Make sure all features are numeric
        for col in features_df.columns:
            if features_df[col].dtype == 'object':
                print(f"Warning: Converting object column {col} to float")
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        # Convert to numpy arrays, ensuring float32 dtype
        features_array = features_df.values.astype(np.float32)
        targets_array = targets_df.values.astype(np.float32)
        
        # Create sequences
        num_sequences = len(features_array) - self.seq_length + 1
        
        # Log dimensions for debugging
        print(f"Features shape: {features_array.shape}, Targets shape: {targets_array.shape}")
        print(f"Creating {num_sequences} sequences of length {self.seq_length}")
        
        # Pre-allocate tensors for better performance
        feature_sequences = torch.zeros((num_sequences, self.seq_length, features_array.shape[1]), 
                                    dtype=torch.float32)
        target_tensors = torch.zeros((num_sequences, targets_array.shape[1]), 
                                    dtype=torch.float32)
        
        # Fill sequences
        for i in range(num_sequences):
            # Get sequence slice
            seq_slice = features_array[i:i+self.seq_length]
            feature_sequences[i] = torch.from_numpy(seq_slice)
            target_tensors[i] = torch.from_numpy(targets_array[i+self.seq_length-1])
        
        # Ensure no NaNs
        feature_sequences = torch.nan_to_num(feature_sequences)
        target_tensors = torch.nan_to_num(target_tensors)
        
        return feature_sequences, target_tensors
    
    def get_dataloaders(self, batch_size=32, num_workers=4):
        """Get train and test dataloaders with optimized settings"""
        # Use TensorDataset for better performance
        from torch.utils.data import TensorDataset
        
        # NEW: Include asset_ids in the datasets
        train_dataset = TensorDataset(self.train_data[0], self.train_data[1], self.train_data[2])
        test_dataset = TensorDataset(self.test_data[0], self.test_data[1], self.test_data[2])
        
        # Configure DataLoader with optimal settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        
        return train_loader, test_loader

def train(model, train_loader, val_loader, optimizer, scheduler, scaler, device, 
          num_epochs=1, patience=3, gradient_accumulation=1, model_save_path="./models", 
          profiler=None, use_wandb=True,
          max_speed=False, instruments=None, model_params=None, training_params=None,
          feature_columns=None):
    """
    Simplified training function with essential optimizations
    """
    print(f"🚀 Starting training on {device} with mixed precision")
    print(f"📊 Using batch size {train_loader.dataset.tensors[0].shape[0] // len(train_loader)}")
    
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
    checkpoint_base = f"{model_save_path}/model_{timestamp}"
    
    # Set torch precision 
    torch.set_float32_matmul_precision('high')
    
    # Training loop
    for epoch in range(num_epochs):
        if profiler:
            profiler.start_operation(f"epoch_{epoch}")
        
        epoch_start_time = time.time()
        
        # ===== TRAINING =====
        model.train()
        train_losses = []
        train_direction_correct = 0
        train_direction_total = 0
        
        optimizer.zero_grad()
        
        for batch_idx, (features, targets, asset_ids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
            if profiler:
                profiler.start_operation("batch_processing")
                
            # Move to device with non_blocking for better performance
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # NEW: Move asset_ids to device
            asset_ids = asset_ids.to(device, non_blocking=True)
            
            # Extract both direction_class and direction_signal for dual training
            direction_class = targets[:, 3].long()  # Already 0,1,2 in database
            direction_signal = targets[:, 4].long()  # Also 0,1,2 in database
            future_return = targets[:, 0].unsqueeze(1)
            future_volatility = targets[:, 5].unsqueeze(1)
            signal_confidence = targets[:, 6].unsqueeze(1)
            
            # Mixed precision forward pass
            with autocast():
                # Create funding rate (but use real asset_ids now)
                funding_rate = torch.zeros(features.size(0), 1, device=device)
                
                # NEW: Use asset_ids instead of None
                outputs = model(features, funding_rate, asset_ids)
                
                # Compute losses with dual objectives
                direction_class_loss = direction_criterion(outputs['direction_logits'], direction_class)
                direction_signal_loss = direction_criterion(outputs['signal_logits'], direction_signal)
                return_loss = regression_criterion(outputs['expected_return'], future_return)
                risk_loss = regression_criterion(outputs['expected_risk'], future_volatility)
                confidence_loss = regression_criterion(outputs['predicted_confidence'], signal_confidence)

                # Combined loss with higher weight on direction_class (training target) but also learn direction_signal (trading target)
                loss = direction_class_loss + 0.5 * direction_signal_loss + return_loss + 0.5 * risk_loss + 0.3 * confidence_loss
                
                # Scale loss for gradient accumulation
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation == 0 or (batch_idx + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad
                if scheduler:
                    scheduler.step()
            
            # Record metrics
            train_losses.append(loss.item() * (gradient_accumulation if gradient_accumulation > 1 else 1))
            
            # Calculate accuracy
            pred_direction = torch.argmax(outputs['direction_logits'], dim=1)
            train_direction_correct += (pred_direction == direction_class).sum().item()
            train_direction_total += direction_class.size(0)
            
            if profiler:
                profiler.end_operation("batch_processing")
                
            # Log metrics to W&B (less frequently)
            if use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'batch_loss': loss.item() * (gradient_accumulation if gradient_accumulation > 1 else 1),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'iteration': batch_idx + epoch * len(train_loader)
                })
        
        # Calculate epoch metrics
        train_loss = np.mean(train_losses)
        train_direction_acc = train_direction_correct / train_direction_total if train_direction_total > 0 else 0
        
        # ===== VALIDATION =====
        if profiler:
            profiler.start_operation("validation")
            
        model.eval()
        val_losses = []
        val_direction_correct = 0
        val_direction_total = 0
        
        with torch.no_grad():
            for features, targets, asset_ids in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # Move to device
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                # NEW: Move asset_ids to device
                asset_ids = asset_ids.to(device, non_blocking=True)
                
                # Extract both direction_class and direction_signal for dual evaluation
                direction_class = targets[:, 3].long()  # Already 0,1,2 in database
                direction_signal = targets[:, 4].long()  # Also 0,1,2 in database
                future_return = targets[:, 0].unsqueeze(1)
                future_volatility = targets[:, 5].unsqueeze(1)
                signal_confidence = targets[:, 6].unsqueeze(1)
                
                # Create funding rate
                funding_rate = torch.zeros(features.size(0), 1, device=device)
                
                # NEW: Use asset_ids instead of None
                with autocast():
                    outputs = model(features, funding_rate, asset_ids)
                    
                    # Compute losses
                    direction_loss = direction_criterion(outputs['direction_logits'], direction_class)
                    return_loss = regression_criterion(outputs['expected_return'], future_return)
                    risk_loss = regression_criterion(outputs['expected_risk'], future_volatility)
                    loss = direction_loss + return_loss + 0.5 * risk_loss
                
                # Record metrics
                val_losses.append(loss.item())
                
                # Calculate accuracy
                pred_direction = torch.argmax(outputs['direction_logits'], dim=1)
                val_direction_correct += (pred_direction == direction_class).sum().item()
                val_direction_total += direction_class.size(0)
        
        if profiler:
            profiler.end_operation("validation")
            
        # Calculate epoch metrics
        val_loss = np.mean(val_losses)
        val_direction_acc = val_direction_correct / val_direction_total if val_direction_total > 0 else 0
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Time: {epoch_duration:.2f}s - "
              f"Train Loss: {train_loss:.6f} - "
              f"Val Loss: {val_loss:.6f} - "
              f"Train Acc: {train_direction_acc:.4f} - "
              f"Val Acc: {val_direction_acc:.4f}")
        
        # Log to W&B
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_direction_acc': train_direction_acc,
                'val_direction_acc': val_direction_acc,
                'epoch_duration': epoch_duration
            })
        
        # Record epoch time
        if profiler:
            profiler.record_epoch_time(epoch_duration)
            
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
                'val_loss': val_loss,
            }, checkpoint_path)

            # Save JSON sidecar
            with open(checkpoint_base + "_best.json", "w") as f:
                json.dump({
                    "model_type": "OptimizedDeribitModel" if max_speed else "DeribitHybridModel",
                    "instruments": instruments,
                    "model_params": model_params,
                    "training_params": training_params,
                    "feature_columns": list(feature_columns)
                }, f, indent=2)

            print(f"✅ Model improved! Saved to {checkpoint_path}")
            
            # Save to W&B
            if use_wandb:
                wandb.save(checkpoint_path)
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_val_acc"] = val_direction_acc
                wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
            break
        
        if profiler:
            profiler.end_operation(f"epoch_{epoch}")
    
    # Save final model
    final_checkpoint_path = f"{checkpoint_base}_final.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, final_checkpoint_path)
    
    print(f"✅ Training complete! Final model saved to {final_checkpoint_path}")
    
    # Save to W&B
    if use_wandb:
        wandb.save(final_checkpoint_path)
    
    return model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Deribit Perpetual Trading Model")
    parser.add_argument("--instruments", type=str, nargs="+", 
                      help="List of instruments to train on")
    parser.add_argument("--quick-test", action="store_true", 
                      help="Enable quick test mode for fast testing")
    parser.add_argument("--no-wandb", action="store_true", 
                      help="Disable Weights & Biases logging")
    parser.add_argument("--profile", action="store_true", 
                      help="Enable detailed performance profiling")
    parser.add_argument("--max-speed", action="store_true", 
                      help="Enable all speed optimizations")
    parser.add_argument("--output-dir", type=str, default="/mnt/p/perpetual/models/checkpoints",
                      help="Directory to save output models")
    parser.add_argument("--sweep-config", type=str, 
                      help="Path to sweep configuration file or JSON string")
    
    # Parse known args and capture unknown args (for dot notation parameters)
    args, unknown = parser.parse_known_args()
    
    # Process dot notation parameters from wandb sweep
    if unknown:
        # Create config structure for sweep parameters
        dot_config = {"model_params": {}, "training_params": {}}
        max_speed_from_args = False
        
        for arg in unknown:
            if "=" in arg and arg.startswith("--"):
                key, value = arg.split("=", 1)
                key = key[2:]  # Remove leading "--"
                
                # Convert underscores to hyphens for flags
                if key == "max_speed":
                    max_speed_from_args = value.lower() == "true"
                    args.max_speed = max_speed_from_args
                    continue
                
                # Handle model parameters
                elif key.startswith("model_params."):
                    param_name = key.split(".", 1)[1]
                    # Convert to appropriate types
                    if param_name in ["transformer_dim", "transformer_heads", 
                                     "transformer_layers", "max_seq_length", 
                                     "tcn_kernel_size"]:
                        dot_config["model_params"][param_name] = int(float(value))
                    else:
                        dot_config["model_params"][param_name] = float(value)
                # Handle training parameters
                elif key.startswith("training_params."):
                    param_name = key.split(".", 1)[1]
                    # Convert to appropriate types
                    if param_name in ["batch_size", "gradient_accumulation", 
                                     "num_epochs", "patience", "seq_length"]:
                        dot_config["training_params"][param_name] = int(float(value))
                    else:
                        dot_config["training_params"][param_name] = float(value)
        
        # Write config to temporary file if we collected parameters
        if dot_config["model_params"] or dot_config["training_params"]:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(dot_config, f, indent=2)
                args.sweep_config = f.name
                print(f"Created temporary sweep config from dot notation parameters: {args.sweep_config}")
    
    # Initialize profiler if enabled
    profiler = None
    if args.profile:
        profiler = TrainingProfiler("/mnt/p/perpetual/tmp")
        print("🔍 Performance profiling enabled")
    
    # Use W&B unless explicitly disabled
    use_wandb = not args.no_wandb
    
    # Default instruments if not provided - use instruments marked as 'used' in the database
    if not args.instruments:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT instrument_name FROM instruments 
                    WHERE used = TRUE ORDER BY instrument_name
                """)
                instruments = [row[0] for row in cur.fetchall()]
            
            # No fallback - require database to return instruments
            if not instruments:
                raise ValueError("No instruments found with used=TRUE in database. Check instruments table.")
                
            # For quick-test mode, just use BTC if it's in the list
            if args.quick_test:
                if "BTC_USDC-PERPETUAL" in instruments:
                    instruments = ["BTC_USDC-PERPETUAL"]
                else:
                    # Otherwise use the first instrument
                    instruments = [instruments[0]]
        except Exception as e:
            print(f"Error fetching instruments: {e}")
            raise  # Re-raise exception to halt execution
        finally:
            conn.close()
    else:
        instruments = args.instruments
    
    # Load parameters from sweep config if provided
    if args.sweep_config:
        try:
            # Check if it's a file path or a JSON string
            if os.path.isfile(args.sweep_config):
                with open(args.sweep_config, 'r') as f:
                    sweep_config = json.load(f)
            else:
                # Try to parse as JSON string
                sweep_config = json.loads(args.sweep_config)
            
            # Extract parameters from sweep config
            model_params = sweep_config.get('model_params', {})
            training_params = sweep_config.get('training_params', {})
            
            # Ensure transformer_dim is even (to avoid dimension mismatch error)
            if 'transformer_dim' in model_params:
                model_params['transformer_dim'] = model_params['transformer_dim'] + (model_params['transformer_dim'] % 2)
                
            max_speed = sweep_config.get('max_speed', args.max_speed)
            print(f"Loaded sweep config with {len(model_params)} model parameters and {len(training_params)} training parameters")
        except Exception as e:
            print(f"Error parsing sweep config: {e}, using defaults")
            max_speed = args.max_speed
            # Use defaults for model/training params (set later)
    else:
        max_speed = args.max_speed
        # Default parameters for quick-test mode
        if args.quick_test:
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
                "gradient_accumulation": 1,
                "num_workers": 2
            }
        else:
            # Default parameters for standard mode
            model_params = {
                "tcn_channels": [64, 64],  # Smaller but still effective
                "tcn_kernel_size": 3,
                "transformer_dim": 64,
                "transformer_heads": 4,
                "transformer_layers": 2,
                "dropout": 0.2,
                "max_seq_length": 512
            }
            
            training_params = {
                "seq_length": 64,
                "batch_size": 64,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "num_epochs": 10,
                "patience": 3,
                "gradient_accumulation": 1,
                "num_workers": 4
            }
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize W&B
    if use_wandb:
        wandb.init(
            project="deribit-perpetual-model",
            name=f"fast_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "instruments": instruments,
                "model_params": model_params,
                "training_params": training_params,
                "device": device,
                "max_speed": max_speed
            }
        )
    
    # Apply performance optimizations
    set_seed()
    
    # Load data
    data = FastDataset(
        instruments=instruments,
        seq_length=training_params.get('seq_length', 32),
        test_size=0.2
    )
    
    # Create dataloaders
    train_loader, val_loader = data.get_dataloaders(
        batch_size=training_params.get('batch_size', 32),
        num_workers=training_params.get('num_workers', 4)
    )
    
    # Initialize model
    input_dim = len(data.feature_columns)

    # Auto-select model class based on max_speed flag
    model_class = OptimizedDeribitModel if max_speed else DeribitHybridModel
    model_name = model_class.__name__

    model = model_class(
        input_dim=input_dim,
        **model_params
    ).to(device)

    print(f"🧠 Instantiated model: {model_name}")

    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_params.get('learning_rate', 0.001),
        weight_decay=training_params.get('weight_decay', 0.0001)
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=training_params.get('learning_rate', 0.001),
        steps_per_epoch=len(train_loader),
        epochs=training_params.get('num_epochs', 10),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Print configuration
    print("\n----- Training Configuration -----")
    print(f"🧠 Model: {'OptimizedDeribitModel' if max_speed else 'DeribitHybridModel'}")
    print(f"🔢 Model Parameters: {model_params}")
    print(f"⚙️ Training Parameters: {training_params}")
    print(f"💱 Instruments: {instruments}")
    print(f"📂 Output Directory: {args.output_dir}")
    print("----------------------------------\n")
    
    # Train model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        num_epochs=training_params.get('num_epochs', 10),
        patience=training_params.get('patience', 3),
        gradient_accumulation=training_params.get('gradient_accumulation', 1),
        model_save_path=args.output_dir,
        profiler=profiler,
        use_wandb=use_wandb,
        max_speed=max_speed,
        instruments=instruments,
        model_params=model_params,
        training_params=training_params,
        feature_columns=data.feature_columns
    )

    # Clean up temporary config file if we created one
    if unknown and 'tempfile' in locals() and os.path.exists(args.sweep_config):
        try:
            os.unlink(args.sweep_config)
        except:
            pass
    
    # Save profiling results
    if profiler:
        profiler.save_results()
    
    # Finish W&B run
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()