
```
perpetual
├─ README.md
├─ commands.md
├─ config
│  ├─ config.ini
│  └─ credentials.env
├─ data
│  ├─ __pycache__
│  │  └─ database.cpython-39.pyc
│  ├─ database.py
│  └─ fetch
│     ├─ __pycache__
│     │  ├─ fetch_funding.cpython-39.pyc
│     │  ├─ fetch_instruments.cpython-39.pyc
│     │  ├─ fetch_ohlcv.cpython-39.pyc
│     │  ├─ fetch_volatility.cpython-39.pyc
│     │  └─ sync_live.cpython-39.pyc
│     ├─ fetch_funding.py
│     ├─ fetch_instruments.py
│     ├─ fetch_ohlcv.py
│     ├─ fetch_volatility.py
│     └─ sync_live.py
├─ deribit
│  └─ authentication
│     ├─ __init__.py
│     ├─ __pycache__
│     │  ├─ __init__.cpython-39.pyc
│     │  └─ authenticate.cpython-39.pyc
│     └─ authenticate.py
├─ features
│  ├─ __pycache__
│  │  ├─ compute_tier1.cpython-39.pyc
│  │  ├─ transformers.cpython-39.pyc
│  │  └─ utils.cpython-39.pyc
│  ├─ compute_tier1.py
│  ├─ compute_tier2.py
│  ├─ transformers.py
│  └─ utils.py
├─ ideas.md
├─ inference
│  ├─ executor.py
│  └─ run_inference.py
├─ logs
│  └─ training.log
├─ models
│  ├─ __pycache__
│  │  ├─ architecture.cpython-39.pyc
│  │  ├─ run_training.cpython-39.pyc
│  │  ├─ train_supervised.cpython-39.pyc
│  │  └─ train_unified.cpython-39.pyc
│  ├─ architecture.py
│  ├─ checkpoints
│  │  ├─ unified_model_20250410_204806_best.pt
│  │  ├─ unified_model_20250410_204806_epoch1.pt
│  │  ├─ unified_model_20250410_204806_epoch3.pt
│  │  ├─ unified_model_20250410_204806_final.pt
│  │  ├─ unified_model_20250411_102726_best.pt
│  │  └─ unified_model_20250411_102726_epoch1.pt
│  ├─ run_training.py
│  ├─ scalers
│  │  ├─ ADA_USDC_PERPETUAL_metadata.pkl
│  │  ├─ ADA_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ ADA_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ ALGO_USDC_PERPETUAL_metadata.pkl
│  │  ├─ ALGO_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ ALGO_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ AVAX_USDC_PERPETUAL_metadata.pkl
│  │  ├─ AVAX_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ AVAX_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ BCH_USDC_PERPETUAL_metadata.pkl
│  │  ├─ BCH_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ BCH_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ BTC_USDC_PERPETUAL_metadata.pkl
│  │  ├─ BTC_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ BTC_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ DOGE_USDC_PERPETUAL_metadata.pkl
│  │  ├─ DOGE_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ DOGE_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ DOT_USDC_PERPETUAL_metadata.pkl
│  │  ├─ DOT_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ DOT_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ ETH_USDC_PERPETUAL_metadata.pkl
│  │  ├─ ETH_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ ETH_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ LINK_USDC_PERPETUAL_metadata.pkl
│  │  ├─ LINK_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ LINK_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ LTC_USDC_PERPETUAL_metadata.pkl
│  │  ├─ LTC_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ LTC_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ NEAR_USDC_PERPETUAL_metadata.pkl
│  │  ├─ NEAR_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ NEAR_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ SOL_USDC_PERPETUAL_metadata.pkl
│  │  ├─ SOL_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ SOL_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ TRX_USDC_PERPETUAL_metadata.pkl
│  │  ├─ TRX_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ TRX_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ UNI_USDC_PERPETUAL_metadata.pkl
│  │  ├─ UNI_USDC_PERPETUAL_minmax_scaler.pkl
│  │  ├─ UNI_USDC_PERPETUAL_std_scaler.pkl
│  │  ├─ XRP_USDC_PERPETUAL_metadata.pkl
│  │  ├─ XRP_USDC_PERPETUAL_minmax_scaler.pkl
│  │  └─ XRP_USDC_PERPETUAL_std_scaler.pkl
│  ├─ train_rl.py
│  ├─ train_unified.py
│  └─ wandb_utils.py
├─ simulate
│  ├─ backtest.py
│  ├─ generate_labels.py
│  └─ trade_tracker.py
└─ wandb
   ├─ debug-internal.log
   ├─ debug.log
   ├─ latest-run
   │  ├─ files
   │  │  ├─ output.log
   │  │  ├─ requirements.txt
   │  │  └─ wandb-metadata.json
   │  ├─ logs
   │  │  ├─ debug-core.log
   │  │  ├─ debug-internal.log
   │  │  └─ debug.log
   │  ├─ run-6chk11ny.wandb
   │  └─ tmp
   │     └─ code
   └─ run-20250411_110528-6chk11ny
      ├─ files
      │  ├─ output.log
      │  ├─ requirements.txt
      │  └─ wandb-metadata.json
      ├─ logs
      │  ├─ debug-core.log
      │  ├─ debug-internal.log
      │  └─ debug.log
      ├─ run-6chk11ny.wandb
      └─ tmp
         └─ code

```