#!/usr/bin/env python3
"""
Simple script to run model training with optimal settings.
This is a convenience wrapper around train_unified.py with optimal default settings.
"""

# OPTIMIZATION: Set PyTorch and NumPy thread settings BEFORE importing other modules
import os
os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 4))
os.environ["MKL_NUM_THREADS"] = str(min(8, os.cpu_count() or 4))
os.environ["NUMEXPR_NUM_THREADS"] = str(min(8, os.cpu_count() or 4))

# OPTIMIZATION: Set PyTorch environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Regular imports
import subprocess
import argparse
import time
from datetime import datetime
import torch
import numpy as np
import random
from models.profiler import TrainingProfiler

def main():
    # Configure argument parser with additional optimization options
    parser = argparse.ArgumentParser(description="Run Deribit model training with optimal settings")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode (one instrument, few epochs)")
    parser.add_argument("--quick-test", action="store_true", help="Run in quick test mode (minimal training to verify setup)")
    parser.add_argument("--full", action="store_true", help="Run full training on all instruments")
    parser.add_argument("--profile", action="store_true", help="Enable detailed performance profiling")
    parser.add_argument("--max-speed", action="store_true", help="Enable all speed optimizations (may reduce accuracy slightly)")
    parser.add_argument("--output-dir", type=str, default="/mnt/p/perpetual/models/checkpoints",
                        help="Directory to save output models")
    parser.add_argument("--cache-dir", type=str, default="/tmp/deribit_cache", 
                        help="Directory to cache processed data")
    args = parser.parse_args()
    
    # Initialize profiler if enabled
    profiler = None
    if args.profile:
        profiler = TrainingProfiler("/mnt/p/perpetual/tmp")
        print("🔍 Performance profiling enabled")
    
    # Determine run mode
    if args.quick_test:
        mode = "quick-test"
        cmd = ["python", "-m", "models.train_unified", "--debug", "--instruments", "BTC_USDC-PERPETUAL"]
    elif args.fast:
        mode = "fast"
        cmd = ["python", "-m", "models.train_unified", "--instruments", "BTC_USDC-PERPETUAL", "ETH_USDC-PERPETUAL"]
    elif args.full:
        mode = "full"
        cmd = ["python", "-m", "models.train_unified"]  # Train on all instruments
    else:
        # Default to standard mode with main instruments
        mode = "standard"
        cmd = ["python", "-m", "models.train_unified", "--instruments", 
               "BTC_USDC-PERPETUAL", "ETH_USDC-PERPETUAL", "SOL_USDC-PERPETUAL", 
               "XRP_USDC-PERPETUAL", "DOGE_USDC-PERPETUAL"]
    
    # Add flags based on arguments
    if args.profile:
        cmd.append("--profile")
        print("🔍 Performance profiling enabled")
        
    if args.max_speed:
        cmd.append("--max-speed")
        print("⚡ Maximum speed optimizations enabled")
    
    # Add output directory 
    cmd.extend(["--output-dir", args.output_dir])
    
    # Add cache directory
    cmd.extend(["--cache-dir", args.cache_dir])
    
    # Print run information
    print("\n" + "="*50)
    print(f"STARTING DERIBIT MODEL TRAINING: {mode.upper()} MODE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Command: {' '.join(cmd)}")
    print("="*50 + "\n")
    
    # Record start time
    start_time = time.time()
    
    # Apply performance optimizations
    apply_performance_optimizations()
    
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

def apply_performance_optimizations():
    """Apply various performance optimizations for better training speed."""
    # Configure NumPy for better performance
    np.set_printoptions(precision=4, suppress=True)
    
    # Set random seeds for deterministic results
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configure PyTorch for better performance
    if torch.cuda.is_available():
        # Enable TF32 precision for better performance on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocator for better performance
        try:
            torch.cuda.memory.set_per_process_memory_fraction(0.95)
        except:
            # Fall back if this specific API is not available
            pass
        
        # Set CUDA options
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()
        
        # Use cudnn benchmarking to find the best algorithm
        torch.backends.cudnn.benchmark = True
        
        # For maximum speed on consumer GPUs, disable deterministic algorithms
        # This can slightly affect result reproducibility but improves speed
        torch.backends.cudnn.deterministic = False
        
        # Print CUDA information
        print(f"🖥️ Using CUDA: {torch.version.cuda}")
        print(f"🖥️ Device: {torch.cuda.get_device_name(0)}")
        print(f"🖥️ Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print("⚠️ CUDA not available, using CPU only. This will be significantly slower.")
        # Optimize OpenMP operations for CPU training
        os.environ["OMP_SCHEDULE"] = "STATIC"
        os.environ["OMP_PROC_BIND"] = "CLOSE"

if __name__ == "__main__":
    main()