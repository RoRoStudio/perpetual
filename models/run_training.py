#!/usr/bin/env python3
"""
Simple script to run model training with optimal settings.
This is a convenience wrapper around train_unified.py with optimal default settings.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Set environment variables for better performance
import os
os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 4))
os.environ["MKL_NUM_THREADS"] = str(min(8, os.cpu_count() or 4))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import subprocess
import argparse
import time
from datetime import datetime
import torch
import json
from models.profiler import TrainingProfiler

def apply_performance_optimizations():
    """Apply performance optimizations for better speed"""
    if torch.cuda.is_available():
        # Enable TF32 precision for faster math on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True
        
        # Faster but potentially less reproducible
        torch.backends.cudnn.deterministic = False
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        print(f"🖥️ CUDA optimizations applied for {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available, using CPU only")

def main():
    parser = argparse.ArgumentParser(description="Run Deribit model training with optimal settings")
    parser.add_argument("--fast", action="store_true", 
                      help="Run in fast mode (one instrument, few epochs)")
    parser.add_argument("--quick-test", action="store_true", 
                      help="Run in quick test mode (minimal training)")
    parser.add_argument("--profile", action="store_true", 
                      help="Enable detailed performance profiling")
    parser.add_argument("--max-speed", action="store_true", 
                      help="Enable all speed optimizations")
    parser.add_argument("--output-dir", type=str, 
                      default="/mnt/p/perpetual/models/checkpoints",
                      help="Directory to save model outputs")
    parser.add_argument("--sweep-id", type=str, 
                      help="WandB sweep ID to load parameters from")
    parser.add_argument("--sweep-config", type=str, help="Path to sweep configuration file or JSON string")
    args = parser.parse_args()
    
    # Initialize profiler
    profiler = None
    if args.profile:
        profiler = TrainingProfiler("/mnt/p/perpetual/tmp")
        print("🔍 Performance profiling enabled")
    
    # Determine instruments based on mode
    if args.quick_test:
        mode = "quick-test"
        instruments = ["BTC_USDC-PERPETUAL"]
    elif args.fast:
        mode = "fast"
        instruments = ["BTC_USDC-PERPETUAL", "ETH_USDC-PERPETUAL"]
    else:
        mode = "standard"
        instruments = ["BTC_USDC-PERPETUAL", "ETH_USDC-PERPETUAL", 
                     "SOL_USDC-PERPETUAL", "XRP_USDC-PERPETUAL", "DOGE_USDC-PERPETUAL"]
    
    # Build command
    cmd = ["python", "-m", "models.train_unified"]
    
    # Add instruments
    cmd.extend(["--instruments"] + instruments)
    
    # Add debug mode for quick test
    if args.quick_test:
        cmd.append("--debug")
    
    # Add profile flag
    if args.profile:
        cmd.append("--profile")
    
    # Add max-speed flag
    if args.max_speed:
        cmd.append("--max-speed")
    
    # Add output directory
    cmd.extend(["--output-dir", args.output_dir])
    
    # Add sweep config if provided
    if args.sweep_id:
        # Create temporary sweep config file
        sweep_config = create_sweep_config(args.sweep_id)
        if sweep_config:
            cmd.extend(["--sweep-config", sweep_config])
    
    # Print run information
    print("\n" + "="*50)
    print(f"STARTING DERIBIT MODEL TRAINING: {mode.upper()} MODE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Command: {' '.join(cmd)}")
    print("="*50 + "\n")
    
    # Apply performance optimizations
    apply_performance_optimizations()
    
    # Record start time
    start_time = time.time()
    
    # Run the command
    try:
        process = subprocess.run(cmd, check=True)
        success = process.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running training: {e}")
        success = False
    
    # Calculate duration
    duration = time.time() - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print completion message
    print("\n" + "="*50)
    print(f"TRAINING {'COMPLETED' if success else 'FAILED'}")
    print(f"Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Mode: {mode}")
    print("="*50 + "\n")
    
    if success:
        print("✅ Model training completed successfully!")
        print("View detailed results in Weights & Biases dashboard.")
    else:
        print("❌ Model training failed. Check logs for details.")

def create_sweep_config(sweep_id):
    """
    Create a temporary config file from wandb sweep parameters.
    Returns the path to the config file or None if failed.
    """
    try:
        import wandb
        api = wandb.Api()
        sweep = api.sweep(f"robertfreericks-roro-studio/deribit-perpetual-model/{sweep_id}")
        
        # Extract best configuration
        best_run = None
        best_metric = float('inf')
        
        for run in sweep.runs:
            if run.state == "finished":
                metric_value = run.summary.get("Relative Time (Process)", float('inf'))
                if metric_value < best_metric:
                    best_metric = metric_value
                    best_run = run
        
        if best_run:
            # Extract config from best run
            config = {
                "model_params": best_run.config.get("model_params", {}),
                "training_params": best_run.config.get("training_params", {}),
                "max_speed": best_run.config.get("max_speed", "true") == "true"
            }
            
            # Save to temporary file
            config_path = "/tmp/sweep_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            print(f"✅ Created sweep config from best run in sweep {sweep_id}")
            return config_path
        else:
            print(f"⚠️ No finished runs found in sweep {sweep_id}")
            return None
    except Exception as e:
        print(f"⚠️ Error creating sweep config: {e}")
        return None

if __name__ == "__main__":
    main()