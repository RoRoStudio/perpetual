#!/usr/bin/env python3
"""
backtest.py
-----------------------------------------------------
Run backtests of trained models against historical
data to validate performance, accuracy, and ROI.
-----------------------------------------------------
"""

import argparse
import os
import json
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("backtest")

# Add project root to path for imports
import sys
sys.path.append('/mnt/p/perpetual')

from models.architecture import DeribitHybridModel, OptimizedDeribitModel
from features.transformers import FeatureTransformer
from data.database import get_connection, db_connection

# Define terms that indicate future-leaking features
LEAKY_TERMS = ('future_', 'next_', 'direction_', 'signal_', 'quantile')

def load_sidecar_config(checkpoint_path):
    """
    Load the model configuration from the sidecar JSON file.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pt file)
        
    Returns:
        Dictionary with model configuration
    """
    json_path = checkpoint_path.replace(".pt", ".json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing sidecar config: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)

def load_dataset(instruments, seq_length, feature_columns):
    """
    Load dataset from Parquet files with optimized memory mapping.
    
    Args:
        instruments: List of instrument names to load
        seq_length: Sequence length for model input
        feature_columns: List of feature column names to use
        
    Returns:
        Tuple of (features, targets, asset_ids)
    """
    all_features = []
    all_targets = []
    all_asset_ids = []
    asset_map = {inst: idx for idx, inst in enumerate(instruments)}
    label_cols = ["direction_class"]

    for instrument in instruments:
        # Try to load from separate Parquet files first (new structure)
        features_path = f"/mnt/p/perpetual/cache/tier1_features_{instrument}.parquet"
        labels_path = f"/mnt/p/perpetual/cache/tier1_labels_{instrument}.parquet"
        
        # Check if separate files exist
        if os.path.exists(features_path) and os.path.exists(labels_path):
            logger.info(f"Loading backtest data for {instrument} from separate files")
            
            # Load features and labels
            features_table = pq.read_table(features_path, memory_map=True)
            features_df = features_table.to_pandas()
            
            labels_table = pq.read_table(labels_path, memory_map=True)
            labels_df = labels_table.to_pandas()
            
            # Verify timestamps match
            common_timestamps = set(features_df['timestamp']).intersection(set(labels_df['timestamp']))
            if len(common_timestamps) < min(len(features_df), len(labels_df)) * 0.9:
                logger.warning(f"⚠️ Significant timestamp mismatch between features and labels for {instrument}")
                
            # Filter both dataframes to only include common timestamps
            features_df = features_df[features_df['timestamp'].isin(common_timestamps)]
            labels_df = labels_df[labels_df['timestamp'].isin(common_timestamps)]
            
            # Sort by timestamp to ensure alignment
            features_df = features_df.sort_values('timestamp')
            labels_df = labels_df.sort_values('timestamp')
                
            # Safety check: ensure no leaky terms in feature columns
            safe_feature_columns = []
            for col in feature_columns:
                if any(term in col for term in LEAKY_TERMS):
                    logger.warning(f"⚠️ Found potential leaky feature: {col}. Removing from feature list.")
                    continue
                if col in features_df.columns:
                    safe_feature_columns.append(col)
                    
            # Apply transformation
            transformer = FeatureTransformer(instrument)
            features_transformed = transformer.transform(features_df[safe_feature_columns])
                
        else:
            logger.error(f"No separate Parquet files found for {instrument}. Please run export_parquet.py first.")
            continue

        # Fill any NaN values in the labels
        for col in label_cols:
            if col in labels_df.columns:
                labels_df[col] = labels_df[col].fillna(1)  # Default to neutral class (1)

        # Convert to numpy arrays
        features_np = features_transformed.values.astype(np.float32)
        targets_np = labels_df[label_cols].values.astype(np.float32)

        # Create sequences
        num_seqs = len(features_np) - seq_length + 1
        if num_seqs <= 0:
            logger.warning(f"Not enough data for {instrument} after sequence creation")
            continue
            
        feat_seq = torch.zeros((num_seqs, seq_length, features_np.shape[1]))
        targ_seq = torch.zeros((num_seqs,), dtype=torch.long)
        ids_seq = torch.full((num_seqs,), asset_map[instrument], dtype=torch.long)

        for i in range(num_seqs):
            feat_seq[i] = torch.from_numpy(features_np[i:i+seq_length])
            # Get the target from the end of each sequence
            target_val = int(targets_np[i + seq_length - 1][0])
            # Ensure it's a valid class (0, 1, or 2)
            if target_val not in [0, 1, 2]:
                target_val = 1  # Default to neutral
            targ_seq[i] = target_val
            
        all_features.append(feat_seq)
        all_targets.append(targ_seq)
        all_asset_ids.append(ids_seq)

    # Ensure we have data to work with
    if not all_features:
        raise ValueError("No valid data found for any instruments!")
        
    return (
        torch.cat(all_features),
        torch.cat(all_targets),
        torch.cat(all_asset_ids)
    )

def log_confusion_matrix(y_true, y_pred):
    """
    Create and log confusion matrix to W&B.
    
    Args:
        y_true: True labels (0=down, 1=neutral, 2=up)
        y_pred: Predicted labels (0=down, 1=neutral, 2=up)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=["DOWN", "NEUTRAL", "UP"],
                yticklabels=["DOWN", "NEUTRAL", "UP"],
                cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

def log_prediction_distribution(y_true, y_pred):
    """
    Create and log prediction distribution histogram to W&B.
    
    Args:
        y_true: True labels (0=down, 1=neutral, 2=up)
        y_pred: Predicted labels (0=down, 1=neutral, 2=up)
    """
    fig, ax = plt.subplots()
    bins = np.arange(4) - 0.5
    ax.hist(y_true, bins=bins, alpha=0.5, label="Actual", edgecolor='black')
    ax.hist(y_pred, bins=bins, alpha=0.5, label="Predicted", edgecolor='black')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["DOWN", "NEUTRAL", "UP"])
    ax.set_title("Prediction Distribution")
    ax.legend()
    wandb.log({"prediction_distribution": wandb.Image(fig)})
    plt.close(fig)

def log_performance_metrics(instruments, y_true, y_pred, instrument_ids):
    """
    Log per-instrument performance metrics to W&B.
    
    Args:
        instruments: List of instrument names
        y_true: True labels (0=down, 1=neutral, 2=up)
        y_pred: Predicted labels (0=down, 1=neutral, 2=up)
        instrument_ids: Array of instrument IDs for each prediction
    """
    # Create a table with per-instrument metrics
    instrument_metrics = []
    
    # Calculate overall metrics
    overall_accuracy = (np.array(y_true) == np.array(y_pred)).mean()
    
    # Get per-instrument metrics
    instrument_map = {i: name for i, name in enumerate(instruments)}
    
    for i, instrument in enumerate(instruments):
        # Use array method for mask creation
        mask = (np.array(instrument_ids) == i)
        if not mask.any():  # Fixed boolean iteration issue
            continue
            
        inst_true = np.array(y_true)[mask]
        inst_pred = np.array(y_pred)[mask]
        
        # Calculate metrics
        accuracy = (inst_true == inst_pred).mean()
        
        # Count by direction
        down_count = np.sum(inst_true == 0)
        neutral_count = np.sum(inst_true == 1)
        up_count = np.sum(inst_true == 2)
        
        # Count by prediction
        down_pred = np.sum(inst_pred == 0)
        neutral_pred = np.sum(inst_pred == 1)
        up_pred = np.sum(inst_pred == 2)
        
        # Save metrics
        instrument_metrics.append({
            'instrument': instrument,
            'accuracy': accuracy,
            'down_actual': down_count / len(inst_true) if len(inst_true) > 0 else 0,
            'neutral_actual': neutral_count / len(inst_true) if len(inst_true) > 0 else 0,
            'up_actual': up_count / len(inst_true) if len(inst_true) > 0 else 0,
            'down_predicted': down_pred / len(inst_pred) if len(inst_pred) > 0 else 0,
            'neutral_predicted': neutral_pred / len(inst_pred) if len(inst_pred) > 0 else 0,
            'up_predicted': up_pred / len(inst_pred) if len(inst_pred) > 0 else 0,
            'sample_count': len(inst_true)
        })
    
    # Create a W&B Table
    columns = ['instrument', 'accuracy', 'down_actual', 'neutral_actual', 'up_actual', 
               'down_predicted', 'neutral_predicted', 'up_predicted', 'sample_count']
    instrument_table = wandb.Table(columns=columns)
    
    # Add rows to table
    for metrics in instrument_metrics:
        instrument_table.add_data(
            metrics['instrument'],
            metrics['accuracy'],
            metrics['down_actual'],
            metrics['neutral_actual'],
            metrics['up_actual'],
            metrics['down_predicted'],
            metrics['neutral_predicted'],
            metrics['up_predicted'],
            metrics['sample_count']
        )
    
    # Log to W&B
    wandb.log({
        "instrument_metrics": instrument_table,
        "overall_accuracy": overall_accuracy
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()
    
    # Use W&B unless explicitly disabled
    use_wandb = not args.no_wandb

    try:
        # Load checkpoint and sidecar config
        checkpoint = torch.load(args.model, map_location="cpu")
        metadata = load_sidecar_config(args.model)
        model_type = metadata["model_type"]
        instruments = metadata["instruments"]
        model_params = metadata["model_params"]
        training_params = metadata["training_params"]
        seq_length = training_params.get("seq_length", 64)

        # Read feature columns from training config
        feature_columns = metadata.get("feature_columns")
        if not feature_columns:
            raise ValueError("Missing 'feature_columns' in sidecar config.")

        logger.info(f"Loading data for {len(instruments)} instruments with sequence length {seq_length}")
        
        # Load dataset with correct feature column list
        features, targets, asset_ids = load_dataset(instruments, seq_length, feature_columns)
        dataset = TensorDataset(features, targets, asset_ids)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        input_dim = features.shape[2]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cls = OptimizedDeribitModel if model_type == "OptimizedDeribitModel" else DeribitHybridModel
        
        logger.info(f"Creating {model_type} with input dimension {input_dim}")
        model = model_cls(input_dim=input_dim, **model_params).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if use_wandb:
            wandb.init(
                project="deribit-perpetual-model",
                name=f"backtest_{os.path.basename(args.model)}",
                config=metadata
            )

        y_true, y_pred = [], []
        instrument_ids = []

        with torch.no_grad():
            for x, y, aid in tqdm(loader, desc="Running backtest"):
                x, y, aid = x.to(device), y.to(device), aid.to(device)
                funding_rate = torch.zeros((x.size(0), 1), device=device)
                out = model(x, funding_rate, aid)
                preds = torch.argmax(out["direction_logits"], dim=1)

                y_true += y.cpu().tolist()
                y_pred += preds.cpu().tolist()
                instrument_ids += aid.cpu().tolist()

        # Calculate and log metrics
        report = classification_report(y_true, y_pred, labels=[0, 1, 2], output_dict=True)
        metrics = {
            "backtest_accuracy": (np.array(y_true) == np.array(y_pred)).mean(),
            "direction/precision_0": report["0"]["precision"],
            "direction/recall_0": report["0"]["recall"],
            "direction/f1_0": report["0"]["f1-score"],
            "direction/precision_1": report["1"]["precision"],
            "direction/recall_1": report["1"]["recall"],
            "direction/f1_1": report["1"]["f1-score"],
            "direction/precision_2": report["2"]["precision"],
            "direction/recall_2": report["2"]["recall"],
            "direction/f1_2": report["2"]["f1-score"]
        }

        # Calculate per-instrument accuracy
        for inst_id, inst_name in enumerate(instruments):
            inst_mask = (np.array(instrument_ids) == inst_id)
            if np.sum(inst_mask) > 0:
                inst_acc = (np.array(y_true)[inst_mask] == np.array(y_pred)[inst_mask]).mean()
                metrics[f"{inst_name}_accuracy"] = inst_acc

        if use_wandb:
            wandb.log(metrics)
            log_confusion_matrix(y_true, y_pred)
            log_prediction_distribution(y_true, y_pred)
            log_performance_metrics(instruments, y_true, y_pred, instrument_ids)
            wandb.finish()
            
        logger.info(f"✅ Enhanced backtest complete — Overall accuracy: {metrics['backtest_accuracy']:.4f}")
        
        # Print summary even if not using W&B
        logger.info(f"\nDirection Class Metrics:")
        logger.info(f"DOWN (0): Precision={metrics['direction/precision_0']:.4f}, Recall={metrics['direction/recall_0']:.4f}, F1={metrics['direction/f1_0']:.4f}")
        logger.info(f"NEUTRAL (1): Precision={metrics['direction/precision_1']:.4f}, Recall={metrics['direction/recall_1']:.4f}, F1={metrics['direction/f1_1']:.4f}")
        logger.info(f"UP (2): Precision={metrics['direction/precision_2']:.4f}, Recall={metrics['direction/recall_2']:.4f}, F1={metrics['direction/f1_2']:.4f}")

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        logger.error(traceback.format_exc())
        if use_wandb and wandb.run is not None:
            wandb.finish(exit_code=1)
        sys.exit(1)

if __name__ == "__main__":
    main()