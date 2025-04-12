#!/usr/bin/env python3
"""
Backtest Deribit model on validation set and log metrics to W&B.
"""
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import wandb

sys.path.append('/mnt/p/perpetual')

from models.architecture import DeribitHybridModel, OptimizedDeribitModel
from features.transformers import FeatureTransformer
from torch.utils.data import DataLoader, TensorDataset
import pyarrow.parquet as pq

def load_data(instruments, seq_length=64, test_size=0.2):
    all_features, all_targets, all_asset_ids = [], [], []
    asset_ids = {name: idx for idx, name in enumerate(instruments)}

    for name in instruments:
        path = f"/mnt/p/perpetual/cache/tier1_{name}.parquet"
        if not os.path.exists(path):
            continue
        print(f"📥 Loading {name}")
        df = pq.read_table(path, memory_map=True).to_pandas()

        # Compute labels
        df['next_return_1bar'] = df['return_1bar'].shift(-1)
        df['next_return_2bar'] = df['return_1bar'].shift(-1) + df['return_1bar'].shift(-2)
        df['next_return_4bar'] = sum(df['return_1bar'].shift(-i) for i in range(1, 5))
        df['next_volatility'] = df['return_1bar'].rolling(4).std().shift(-4)
        volatility = df['return_1bar'].rolling(20).std().fillna(0.001)
        df['direction_class'] = 0
        df.loc[df['next_return_1bar'] > 0.5 * volatility, 'direction_class'] = 1
        df.loc[df['next_return_1bar'] < -0.5 * volatility, 'direction_class'] = -1
        for col in ['next_return_1bar', 'next_return_2bar', 'next_return_4bar', 'next_volatility']:
            df[col].fillna(0, inplace=True)

        # Features and targets
        label_cols = ['next_return_1bar', 'next_return_2bar', 'next_return_4bar', 'direction_class', 'next_volatility']
        exclude_cols = label_cols + ['timestamp', 'instrument_name']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        transformer = FeatureTransformer(name)
        X = transformer.transform(df[feature_cols]).values.astype(np.float32)
        y = df[label_cols].values.astype(np.float32)

        N = len(X) - seq_length + 1
        split = int(N * (1 - test_size))
        X_val = np.stack([X[i:i+seq_length] for i in range(split, N)])
        y_val = y[split + seq_length - 1:N + seq_length - 1]
        asset_id_val = np.full((len(X_val),), asset_ids[name], dtype=np.int64)

        all_features.append(torch.tensor(X_val))
        all_targets.append(torch.tensor(y_val))
        all_asset_ids.append(torch.tensor(asset_id_val))

    return (
        torch.cat(all_features),
        torch.cat(all_targets),
        torch.cat(all_asset_ids),
        asset_ids
    )

@torch.no_grad()
def evaluate(model, val_loader, device, asset_ids):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    all_correct, all_total = 0, 0
    instrument_acc = {name: [0, 0] for name in asset_ids}

    for X, y, aid in tqdm(val_loader, desc="📊 Running backtest"):
        X, y, aid = X.to(device), y.to(device), aid.to(device)
        direction_true = y[:, 3].long() + 1  # Shift -1,0,1 -> 0,1,2
        direction_pred = torch.argmax(model(X, torch.zeros(len(X), 1).to(device), aid)['direction_logits'], dim=1)

        correct = (direction_true == direction_pred).sum().item()
        total = direction_true.size(0)

        all_correct += correct
        all_total += total

        for idx, a in enumerate(aid.cpu().numpy()):
            name = list(asset_ids.keys())[list(asset_ids.values()).index(a)]
            instrument_acc[name][0] += int(direction_true[idx].item() == direction_pred[idx].item())
            instrument_acc[name][1] += 1

    val_acc = all_correct / all_total
    print(f"\n✅ Backtest Accuracy: {val_acc:.4f}")
    return val_acc, instrument_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt)")
    args = parser.parse_args()

    print(f"🔍 Loading checkpoint from {args.model}")
    checkpoint = torch.load(args.model, map_location="cpu")

    # Figure out model type
    is_fast = 'OptimizedDeribitModel' in args.model or 'max_speed' in args.model
    model_class = OptimizedDeribitModel if is_fast else DeribitHybridModel
    model_args = checkpoint.get('model_args') or {
        'input_dim': 43,
        'tcn_channels': [64, 64],
        'tcn_kernel_size': 3,
        'transformer_dim': 64,
        'transformer_heads': 4,
        'transformer_layers': 2,
        'dropout': 0.2,
        'max_seq_length': 512
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_class(**model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Use all cached instruments
    instruments = sorted([
        f[6:-8] for f in os.listdir("/mnt/p/perpetual/cache")
        if f.startswith("tier1_") and f.endswith(".parquet")
    ])

    print(f"📡 Instruments: {instruments}")
    print("🧪 Loading validation data...")
    X, y, asset_ids_tensor, asset_ids_map = load_data(instruments, seq_length=model_args.get('max_seq_length', 64))

    val_dataset = TensorDataset(X, y, asset_ids_tensor)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, pin_memory=True)

    run = wandb.init(
        project="deribit-perpetual-model",
        name=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={"model": args.model, "instruments": instruments}
    )

    acc, per_instrument = evaluate(model, val_loader, device, asset_ids_map)
    wandb.log({"val_direction_acc": acc})
    for name, (correct, total) in per_instrument.items():
        if total > 0:
            wandb.log({f"acc_{name}": correct / total})

    wandb.finish()

if __name__ == "__main__":
    main()
