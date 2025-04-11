#!/usr/bin/env python3
# sync_live.py
# -------------------------------------------------------------------
# Real-time data orchestration loop for Deribit.
# - Triggers 15m polling, hourly metadata, and scheduled backfills.
# - Supports both full historical and ongoing live sync.
# -------------------------------------------------------------------
import subprocess
import time
import os
import sys
import logging
from datetime import datetime
import schedule
import threading
import argparse

# ---- Imports from project root (PYTHONPATH should be /mnt/p/perpetual)
from data.database import get_connection

# ---- Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("sync_live")

# ---- Constants
HISTORICAL_FETCH_SCRIPTS = [
    "data.fetch.fetch_instruments",
    "data.fetch.fetch_ohlcv",
    "data.fetch.fetch_funding",
    "data.fetch.fetch_volatility",
]

POLLING_INTERVAL_SECONDS = 5

polling_threads = {
    "ohlcv": None,
    "funding_vol": None,
}

# --- Script runner
def run_script(script_name, historical=False, debug=False):
    try:
        cmd = [sys.executable, "-m", script_name]
        if historical:
            cmd.append("--historical")
        if debug:
            cmd.append("--debug")
            
        logger.info(f"Running script: {script_name}{' (historical)' if historical else ''}{' (debug)' if debug else ''}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ Script {script_name} completed")
        
        # When in debug mode, also log the script's output
        if debug and result.stdout:
            for line in result.stdout.splitlines():
                if line.strip():  # Only log non-empty lines
                    logger.debug(f"📜 {script_name} output: {line}")
                    
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error running {script_name}: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        logger.error(f"Unexpected error running {script_name}: {e}")
        return False, str(e)

# --- Historical Backfill
def run_full_historical_backfill(debug=False):
    logger.info("🚀 Starting full historical backfill")
    run_script("data.fetch.fetch_instruments", debug=debug)
    run_script("data.fetch.fetch_ohlcv", historical=True, debug=debug)
    run_script("data.fetch.fetch_funding", historical=True, debug=debug)
    run_script("data.fetch.fetch_volatility", historical=True, debug=debug)
    logger.info("✅ Historical backfill complete")

def run_historical_fetches(debug=False):
    logger.info("🔁 Running historical sync (twice for safety)")
    for i in range(2):
        logger.info(f"📦 Historical sync round {i+1}/2")
        for script in HISTORICAL_FETCH_SCRIPTS:
            success, _ = run_script(script, debug=debug)
            if not success:
                logger.warning(f"⚠️ Non-critical failure in {script}")

# --- Polling Threads
def is_polling_active(polling_type):
    thread = polling_threads.get(polling_type)
    if thread and thread.is_alive():
        return True
    polling_threads[polling_type] = None
    return False

def run_ohlcv_intensive_polling(debug=False):
    if is_polling_active("ohlcv"):
        logger.warning("📉 OHLCV polling already running")
        return

    def worker():
        logger.info("⏱️ Starting OHLCV intensive polling")
        end_time = time.time() + 300
        while time.time() < end_time:
            run_script("data.fetch.fetch_ohlcv", debug=debug)
            time.sleep(POLLING_INTERVAL_SECONDS)
        polling_threads["ohlcv"] = None
        logger.info("✅ OHLCV polling complete")

    polling_threads["ohlcv"] = threading.Thread(target=worker, daemon=True)
    polling_threads["ohlcv"].start()

def run_funding_volatility_intensive_polling(debug=False):
    if is_polling_active("funding_vol"):
        logger.warning("💸 Funding/Vol polling already running")
        return

    def worker():
        logger.info("⏱️ Starting funding + volatility intensive polling")
        end_time = time.time() + 300
        while time.time() < end_time:
            run_script("data.fetch.fetch_funding", debug=debug)
            run_script("data.fetch.fetch_volatility", debug=debug)
            time.sleep(POLLING_INTERVAL_SECONDS)
        polling_threads["funding_vol"] = None
        logger.info("✅ Funding/Vol polling complete")

    polling_threads["funding_vol"] = threading.Thread(target=worker, daemon=True)
    polling_threads["funding_vol"].start()

# --- Schedule Management
def run_instrument_fetch(debug=False):
    run_script("data.fetch.fetch_instruments", debug=debug)

def check_intensive_polling_schedule(debug=False):
    now = datetime.utcnow()
    minute = now.minute
    second = now.second
    if second > 10:
        return
    if minute == 0:
        run_funding_volatility_intensive_polling(debug=debug)
    elif minute in [15, 30, 45]:
        run_ohlcv_intensive_polling(debug=debug)

def setup_schedules(debug=False):
    schedule.every().hour.at(":01").do(run_instrument_fetch, debug=debug)
    for i in range(60):
        schedule.every().hour.at(f":{i:02d}").do(check_intensive_polling_schedule, debug=debug)
    schedule.every().hour.at(":05").do(lambda: run_script("data.fetch.fetch_ohlcv", debug=debug))
    schedule.every().hour.at(":35").do(lambda: run_script("data.fetch.fetch_ohlcv", debug=debug))
    schedule.every().hour.at(":10").do(lambda: run_script("data.fetch.fetch_funding", debug=debug))
    schedule.every().hour.at(":40").do(lambda: run_script("data.fetch.fetch_funding", debug=debug))
    schedule.every().hour.at(":20").do(lambda: run_script("data.fetch.fetch_volatility", debug=debug))
    schedule.every().hour.at(":50").do(lambda: run_script("data.fetch.fetch_volatility", debug=debug))
    logger.info("✅ Schedule setup complete")

# --- Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--historical', action='store_true', help="Run historical backfill and exit")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    
    is_debug = args.debug
    if is_debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("🔍 Debug mode enabled")

    if args.historical:
        run_full_historical_backfill(debug=is_debug)
        return

    logger.info("🚀 Starting live data sync")
    run_historical_fetches(debug=is_debug)
    setup_schedules(debug=is_debug)
    run_instrument_fetch(debug=is_debug)
    run_script("data.fetch.fetch_ohlcv", debug=is_debug)
    run_script("data.fetch.fetch_funding", debug=is_debug)
    run_script("data.fetch.fetch_volatility", debug=is_debug)

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Graceful shutdown")
    except Exception as e:
        logger.error(f"❌ Unexpected crash: {e}")

if __name__ == "__main__":
    main()