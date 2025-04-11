#!/usr/bin/env python3
"""
Simple script to run model training with optimal settings.
This is a convenience wrapper around train_unified.py with optimal default settings.
"""

import os
import subprocess
import argparse
import time
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run Deribit model training with optimal settings")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode (one instrument, few epochs)")
    parser.add_argument("--quick-test", action="store_true", help="Run in quick test mode (minimal training to verify setup)")
    parser.add_argument("--full", action="store_true", help="Run full training on all instruments")
    args = parser.parse_args()
    
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
    
    # Print run information
    print("\n" + "="*50)
    print(f"STARTING DERIBIT MODEL TRAINING: {mode.upper()} MODE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Command: {' '.join(cmd)}")
    print("="*50 + "\n")
    
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

if __name__ == "__main__":
    main()