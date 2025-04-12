import argparse
import os
import json
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from models.architecture import DeribitHybridModel, OptimizedDeribitModel
from features.transformers import FeatureTransformer
import pyarrow.parquet as pq

def load_sidecar_config(checkpoint_path):
    """Try to find a sidecar .json config with the same base name as the checkpoint"""
    config_path = checkpoint_path.replace(".pt", ".json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}

def load_dataset(instruments, seq_length):
    all_features = []
    all_targets = []
    all_asset_ids = []
    asset_map = {inst: idx for idx, inst in enumerate(instruments)}
    feature_columns = None

    for instrument in instruments:
        path = f"/mnt/p/perpetual/cache/tier1_{instrument}.parquet"
        table = pq.read_table(path, memory_map=True)
        df = table.to_pandas()

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

        label_cols = ["next_return_1bar", "next_return_2bar", "next_return_4bar", "direction_class", "next_volatility"]

        if feature_columns is None:
            exclude = label_cols + ['instrument_name', 'timestamp']
            feature_columns = [col for col in df.columns if col not in exclude]

        transformer = FeatureTransformer(instrument)
        features_df = transformer.transform(df[feature_columns])
        targets_df = df[label_cols]

        features_np = features_df.values.astype(np.float32)
        targets_np = targets_df.values.astype(np.float32)

        num_seqs = len(features_np) - seq_length + 1
        feat_seq = torch.zeros((num_seqs, seq_length, features_np.shape[1]))
        targ_seq = torch.zeros((num_seqs, targets_np.shape[1]))
        ids_seq = torch.full((num_seqs,), asset_map[instrument], dtype=torch.long)

        for i in range(num_seqs):
            feat_seq[i] = torch.tensor(features_np[i:i + seq_length])
            targ_seq[i] = torch.tensor(targets_np[i + seq_length - 1])

        all_features.append(feat_seq)
        all_targets.append(targ_seq)
        all_asset_ids.append(ids_seq)

    X = torch.cat(all_features)
    y = torch.cat(all_targets)
    ids = torch.cat(all_asset_ids)
    return TensorDataset(X, y, ids)

def run_backtest(checkpoint_path):
    print(f"🔍 Loading checkpoint from {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint and sidecar config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = load_sidecar_config(checkpoint_path)
    model_type = config.get("model_type", "OptimizedDeribitModel")
    instruments = config.get("instruments", [])
    model_params = config.get("model_params", {})
    training_params = config.get("training_params", {})
    seq_length = training_params.get("seq_length", 32)
    input_dim = model_params.get("input_dim", 43)  # default

    # Load test dataset
    dataset = load_dataset(instruments, seq_length)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Init model
    if model_type == "DeribitHybridModel":
        model = DeribitHybridModel(input_dim=input_dim, **model_params).to(device)
    else:
        model = OptimizedDeribitModel(input_dim=input_dim, **model_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Log W&B
    wandb.init(
        project="deribit-perpetual-model",
        name=f"backtest_{os.path.basename(checkpoint_path)}",
        config={
            "model_path": checkpoint_path,
            "model_type": model_type,
            "instruments": instruments,
            "seq_length": seq_length
        }
    )

    criterion = torch.nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    instrument_metrics = {inst: {"correct": 0, "total": 0} for inst in instruments}

    with torch.no_grad():
        for x, y, ids in tqdm(loader):
            x, y, ids = x.to(device), y.to(device), ids.to(device)
            funding_rate = torch.zeros((x.shape[0], 1), device=device)
            output = model(x, funding_rate, ids)
            logits = output["direction_logits"]
            preds = torch.argmax(logits, dim=1)
            labels = y[:, 3].long() + 1

            for i, inst_id in enumerate(ids):
                name = instruments[inst_id.item()]
                instrument_metrics[name]["total"] += 1
                if preds[i] == labels[i]:
                    instrument_metrics[name]["correct"] += 1

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    total_correct = sum(m["correct"] for m in instrument_metrics.values())
    total_samples = sum(m["total"] for m in instrument_metrics.values())
    overall_acc = total_correct / total_samples

    print(f"\n✅ Backtest complete - Overall Accuracy: {overall_acc:.4f}")
    wandb.log({"backtest_accuracy": overall_acc})

    for inst, stats in instrument_metrics.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"{inst}: {acc:.4f}")
        wandb.log({f"{inst}_accuracy": acc})

    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt model checkpoint")
    args = parser.parse_args()
    run_backtest(args.model)

if __name__ == "__main__":
    main()
