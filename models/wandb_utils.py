#!/usr/bin/env python3
"""
Weights & Biases utilities for the Deribit perpetual trading project.
Enhances model training with advanced tracking, visualization, and debugging.
"""

import os
import torch
import numpy as np
import pandas as pd
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

def init_wandb(project_name="deribit-perpetual-model", run_name=None, config=None):
    """
    Initialize W&B with proper configuration.
    
    Args:
        project_name: W&B project name
        run_name: Name for this specific run
        config: Configuration dictionary
        
    Returns:
        wandb run object
    """
    # Generate a run name if not provided
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize run
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        reinit=True,
        settings=wandb.Settings(start_method="thread")
    )
    
    # Log environment info
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda
        }
        wandb.config.update({"environment": gpu_info})
    
    return run

def log_model_graph(model, sample_input, funding_rate=None):
    """
    Log model architecture graph to W&B.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor
        funding_rate: Sample funding rate tensor
    """
    try:
        # Ensure model is in eval mode for graph capture
        model.eval()
        
        # Create sample inputs if not provided
        if sample_input is None:
            sample_input = torch.randn(1, 64, model.input_dim).to(next(model.parameters()).device)
        
        if funding_rate is None:
            funding_rate = torch.randn(1, 1).to(next(model.parameters()).device)
        
        # Register hooks for graph
        wandb.watch(model, log="all", log_freq=100)
    except Exception as e:
        print(f"Error logging model graph: {e}")

def log_feature_distributions(features_df, prefix="tier1"):
    """
    Log feature distributions to W&B.
    
    Args:
        features_df: DataFrame with features
        prefix: Prefix for the metric name
    """
    if features_df is None or features_df.empty:
        return
    
    try:
        # Log histograms for each feature
        for col in features_df.columns:
            if col in ['instrument_name', 'timestamp']:
                continue
                
            values = features_df[col].dropna().values
            if len(values) > 0:
                wandb.log({f"{prefix}_feature_{col}": wandb.Histogram(values)})
                
        # Log correlation matrix
        numeric_cols = features_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        if len(numeric_cols) > 1:
            corr_matrix = features_df[numeric_cols].corr()
            
            plt.figure(figsize=(12, 10))
            plt.matshow(corr_matrix, fignum=1)
            plt.title(f"{prefix.capitalize()} Feature Correlation Matrix")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"/tmp/{prefix}_correlation.png")
            
            wandb.log({f"{prefix}_correlation_matrix": wandb.Image(f"/tmp/{prefix}_correlation.png")})
    except Exception as e:
        print(f"Error logging feature distributions: {e}")

def log_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Log confusion matrix to W&B.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (e.g., ["Down", "Neutral", "Up"])
    """
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Default class names if not provided
        if class_names is None:
            class_names = ["Down", "Neutral", "Up"]
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create a pretty plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("/tmp/confusion_matrix.png")
        
        # Log to W&B
        wandb.log({"confusion_matrix": wandb.Image("/tmp/confusion_matrix.png")})
    except Exception as e:
        print(f"Error logging confusion matrix: {e}")

def log_performance_metrics(metrics, prefix="validation"):
    """
    Log performance metrics to W&B.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for the metric name
    """
    # Add prefix to metric names
    metrics_with_prefix = {f"{prefix}_{k}": v for k, v in metrics.items()}
    
    # Log to W&B
    wandb.log(metrics_with_prefix)

def log_model_predictions(inputs, true_values, predictions, num_samples=10):
    """
    Log model predictions to W&B for inspection.
    
    Args:
        inputs: Input features
        true_values: True target values
        predictions: Model predictions
        num_samples: Number of samples to log
    """
    try:
        # Create a table
        columns = ["index", "input_features", "true_direction", "pred_direction", 
                  "true_return", "pred_return", "confidence"]
        data = []
        
        # Sample random indices
        if len(true_values) > num_samples:
            indices = np.random.choice(len(true_values), num_samples, replace=False)
        else:
            indices = range(len(true_values))
            
        # Extract data
        for idx in indices:
            # Extract values based on the format of your data
            true_dir = true_values[idx].get('direction_class', 1).item()
            true_ret = true_values[idx].get('next_return_1bar', 0).item()
            
            pred_dir = predictions[idx].get('direction', 0).item()
            pred_ret = predictions[idx].get('expected_return', 0).item()
            confidence = predictions[idx].get('confidence', 0).item()
            
            # Map direction class (0,1,2) to labels (Down, Neutral, Up)
            dir_labels = ["Down", "Neutral", "Up"]
            true_dir_label = dir_labels[true_dir] if 0 <= true_dir < 3 else "Unknown"
            pred_dir_label = dir_labels[int(pred_dir+1)] if -1 <= pred_dir <= 1 else "Unknown"
            
            # Add to data
            data.append([
                idx, 
                np.array2string(inputs[idx][-1, :5].cpu().numpy(), precision=2),  # Show last timestep, first 5 features
                true_dir_label,
                pred_dir_label,
                f"{true_ret:.4f}",
                f"{pred_ret:.4f}",
                f"{confidence:.4f}"
            ])
        
        # Create W&B table
        predictions_table = wandb.Table(columns=columns, data=data)
        wandb.log({"prediction_samples": predictions_table})
    except Exception as e:
        print(f"Error logging model predictions: {e}")

def save_model_to_wandb(model, model_name="deribit_model", metadata=None):
    """
    Save model to W&B artifacts.
    
    Args:
        model: PyTorch model or state dict
        model_name: Name for the model artifact
        metadata: Additional metadata to log with the model
    """
    try:
        # Create a W&B artifact
        model_artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description=f"Deribit trading model saved on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        # Save the model to a file
        model_path = f"/tmp/{model_name}.pt"
        
        if isinstance(model, dict):
            # It's already a state dict
            torch.save(model, model_path)
        else:
            # It's a model object
            torch.save(model.state_dict(), model_path)
        
        # Add the file to the artifact
        model_artifact.add_file(model_path)
        
        # Add metadata if provided
        if metadata:
            for k, v in metadata.items():
                model_artifact.metadata[k] = v
        
        # Log the artifact
        wandb.log_artifact(model_artifact)
    except Exception as e:
        print(f"Error saving model to W&B: {e}")

def log_scaler_info(transformer, instrument_name):
    """
    Log feature scaler information to W&B.
    
    Args:
        transformer: FeatureTransformer object
        instrument_name: Instrument name
    """
    try:
        # Get feature statistics
        stats = transformer.get_feature_stats()
        
        # Create tables for different scaling types
        standard_data = []
        minmax_data = []
        
        for feature, info in stats['features'].items():
            if info['scaling'] == 'standard':
                standard_data.append([feature, info['mean'], info['std']])
            elif info['scaling'] == 'minmax':
                minmax_data.append([feature, info['min'], info['max']])
        
        # Create and log tables
        if standard_data:
            standard_table = wandb.Table(
                columns=["Feature", "Mean", "Std"],
                data=standard_data
            )
            wandb.log({f"{instrument_name}_standard_scaled_features": standard_table})
        
        if minmax_data:
            minmax_table = wandb.Table(
                columns=["Feature", "Min", "Max"],
                data=minmax_data
            )
            wandb.log({f"{instrument_name}_minmax_scaled_features": minmax_table})
            
        # Log summary info
        wandb.log({
            f"{instrument_name}_num_training_samples": stats['num_samples'],
            f"{instrument_name}_training_start": stats.get('training_start', 'N/A'),
            f"{instrument_name}_training_end": stats.get('training_end', 'N/A')
        })
    except Exception as e:
        print(f"Error logging scaler info: {e}")

def create_learning_curve_plot(history):
    """
    Create and log learning curve plots.
    
    Args:
        history: Dictionary containing training metrics
    """
    try:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['train_direction_acc'], label='Train Accuracy')
        ax2.plot(history['val_direction_acc'], label='Validation Accuracy')
        ax2.set_title('Direction Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Save and log
        plt.tight_layout()
        plt.savefig('/tmp/learning_curves.png')
        wandb.log({"learning_curves": wandb.Image('/tmp/learning_curves.png')})
    except Exception as e:
        print(f"Error creating learning curve plot: {e}")

def log_system_metrics():
    """
    Start logging system metrics (GPU, memory, CPU) to W&B.
    """
    try:
        # Enable system metrics monitoring
        wandb.init(monitor_gym=False, settings=wandb.Settings(system_metrics=True))
    except Exception as e:
        print(f"Error enabling system metrics: {e}")