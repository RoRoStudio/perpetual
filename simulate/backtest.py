# simulate/backtest.py
import argparse
import os
import json
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq

from models.architecture import DeribitHybridModel, OptimizedDeribitModel
from features.transformers import FeatureTransformer

def load_sidecar_config(checkpoint_path):
    json_path = checkpoint_path.replace(".pt", ".json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing sidecar config: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)

def load_dataset(instruments, seq_length, feature_columns):
    all_features = []
    all_targets = []
    all_asset_ids = []
    asset_map = {inst: idx for idx, inst in enumerate(instruments)}
    label_cols = ["future_return_1bar", "future_return_2bar", "future_return_4bar",
                  "direction_class", "direction_signal", "future_volatility", "signal_confidence"]

    for instrument in instruments:
        path = f"/mnt/p/perpetual/cache/tier1_{instrument}.parquet"
        table = pq.read_table(path, memory_map=True)
        df = table.to_pandas()

        # Recompute targets from return_1bar
        df['future_return_1bar'] = df['return_1bar'].shift(-1)
        df['future_return_2bar'] = df['return_1bar'].shift(-1) + df['return_1bar'].shift(-2)
        df['future_return_4bar'] = sum(df['return_1bar'].shift(-i) for i in range(1, 5))
        df['future_volatility'] = df['return_1bar'].rolling(4).std().shift(-4)

        volatility = df['return_1bar'].rolling(20).std().fillna(0.001)
        df['direction_class'] = 0
        df.loc[df['future_return_1bar'] > 0.5 * volatility, 'direction_class'] = 1
        df.loc[df['future_return_1bar'] < -0.5 * volatility, 'direction_class'] = -1

        for col in label_cols:
            df[col].fillna(0, inplace=True)

        transformer = FeatureTransformer(instrument)
        features_df = transformer.transform(df[feature_columns])
        targets_df = df[label_cols]

        features_np = features_df.values.astype(np.float32)
        targets_np = targets_df.values.astype(np.float32)

        num_seqs = len(features_np) - seq_length + 1
        feat_seq = torch.zeros((num_seqs, seq_length, features_np.shape[1]))
        targ_seq = torch.zeros((num_seqs,), dtype=torch.long)
        ids_seq = torch.full((num_seqs,), asset_map[instrument], dtype=torch.long)

        for i in range(num_seqs):
            feat_seq[i] = torch.tensor(features_np[i:i + seq_length])
            targ_seq[i] = int(targets_np[i + seq_length - 1][3] + 1)  # shift to [0,1,2]

        all_features.append(feat_seq)
        all_targets.append(targ_seq)
        all_asset_ids.append(ids_seq)

    return (
        torch.cat(all_features),
        torch.cat(all_targets),
        torch.cat(all_asset_ids)
    )

def log_confusion_matrix(y_true, y_pred):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt model checkpoint")
    args = parser.parse_args()

    # Load checkpoint and sidecar config
    checkpoint = torch.load(args.model, map_location="cpu")
    metadata = load_sidecar_config(args.model)
    model_type = metadata["model_type"]
    instruments = metadata["instruments"]
    model_params = metadata["model_params"]
    training_params = metadata["training_params"]
    seq_length = training_params.get("seq_length", 64)

    # NEW: Read feature columns from training config
    feature_columns = metadata.get("feature_columns")
    if not feature_columns:
        raise ValueError("Missing 'feature_columns' in sidecar config.")

    # Load dataset with correct feature column list
    features, targets, asset_ids = load_dataset(instruments, seq_length, feature_columns)
    dataset = TensorDataset(features, targets, asset_ids)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    input_dim = features.shape[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cls = OptimizedDeribitModel if model_type == "OptimizedDeribitModel" else DeribitHybridModel
    model = model_cls(input_dim=input_dim, **model_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wandb.init(
        project="deribit-perpetual-model",
        name=f"backtest_{os.path.basename(args.model)}",
        config=metadata
    )

    y_true, y_pred = [], []
    per_asset = {inst: [] for inst in instruments}

    with torch.no_grad():
        for x, y, aid in tqdm(loader, desc="Backtesting"):
            x, y, aid = x.to(device), y.to(device), aid.to(device)
            funding_rate = torch.zeros((x.size(0), 1), device=device)
            out = model(x, funding_rate, aid)
            preds = torch.argmax(out["direction_logits"], dim=1)

            y_true += y.cpu().tolist()
            y_pred += preds.cpu().tolist()

            for a, pred, true in zip(aid.cpu().tolist(), preds.cpu().tolist(), y.cpu().tolist()):
                per_asset[instruments[a]].append(pred == true)

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

    for inst, correct_list in per_asset.items():
        if correct_list:
            metrics[f"{inst}_accuracy"] = np.mean(correct_list)

    wandb.log(metrics)
    log_confusion_matrix(y_true, y_pred)
    log_prediction_distribution(y_true, y_pred)
    wandb.finish()
    print("✅ Enhanced backtest complete — logged everything to Weights & Biases!")

if __name__ == "__main__":
    main()
