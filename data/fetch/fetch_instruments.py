# Fetch Deribit instruments and store in DB
# /mnt/p/perpetual/data/fetching/fetch_instruments.py

import requests
from psycopg2.extras import execute_values
from datetime import datetime
import sys
import os

# Ensure package import works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data.database import get_connection


DERIBIT_API_URL = "https://www.deribit.com/api/v2/public/get_instruments?kind=future&expired=false"


def fetch_perpetual_instruments():
    response = requests.get(DERIBIT_API_URL)
    response.raise_for_status()
    all_instruments = response.json().get("result", [])

    # Filter perpetuals and reshape
    perpetuals = []
    for i in all_instruments:
        if i.get("settlement_period") != "perpetual":
            continue

        perpetuals.append({
            "instrument_name": i["instrument_name"],
            "base_currency": i["base_currency"],
            "quote_currency": i["quote_currency"],
            "counter_currency": i.get("counter_currency"),
            "settlement_currency": i.get("settlement_currency"),
            "kind": i["kind"],
            "settlement_period": i["settlement_period"],
            "instrument_type": i["instrument_type"],
            "is_active": i["is_active"],
            "instrument_id": i["instrument_id"],
            "price_index": i["price_index"],
            "creation_timestamp": datetime.utcfromtimestamp(i["creation_timestamp"] / 1000),
            "expiration_timestamp": datetime.utcfromtimestamp(i["expiration_timestamp"] / 1000),
            "contract_size": float(i["contract_size"]),
            "tick_size": float(i["tick_size"]),
            "taker_commission": float(i["taker_commission"]),
            "maker_commission": float(i["maker_commission"]),
            "max_leverage": float(i["max_leverage"]),
            "max_liquidation_commission": float(i["max_liquidation_commission"]),
            "min_trade_amount": float(i["min_trade_amount"]),
            "block_trade_tick_size": float(i["block_trade_tick_size"]),
            "block_trade_min_trade_amount": float(i["block_trade_min_trade_amount"]),
            "block_trade_commission": float(i["block_trade_commission"]),
            "rfq": i["rfq"],
            "used": False  # default; toggle manually for selected pairs
        })

    return perpetuals


def upsert_instruments(conn, instruments):
    if not instruments:
        print("⚠️ No instruments found.")
        return

    rows = [tuple(i.values()) for i in instruments]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO instruments (
                instrument_name, base_currency, quote_currency, counter_currency, settlement_currency,
                kind, settlement_period, instrument_type, is_active, instrument_id, price_index,
                creation_timestamp, expiration_timestamp, contract_size, tick_size,
                taker_commission, maker_commission, max_leverage, max_liquidation_commission,
                min_trade_amount, block_trade_tick_size, block_trade_min_trade_amount,
                block_trade_commission, rfq, used
            ) VALUES %s
            ON CONFLICT (instrument_name) DO UPDATE SET
                base_currency = EXCLUDED.base_currency,
                quote_currency = EXCLUDED.quote_currency,
                counter_currency = EXCLUDED.counter_currency,
                settlement_currency = EXCLUDED.settlement_currency,
                kind = EXCLUDED.kind,
                settlement_period = EXCLUDED.settlement_period,
                instrument_type = EXCLUDED.instrument_type,
                is_active = EXCLUDED.is_active,
                instrument_id = EXCLUDED.instrument_id,
                price_index = EXCLUDED.price_index,
                creation_timestamp = EXCLUDED.creation_timestamp,
                expiration_timestamp = EXCLUDED.expiration_timestamp,
                contract_size = EXCLUDED.contract_size,
                tick_size = EXCLUDED.tick_size,
                taker_commission = EXCLUDED.taker_commission,
                maker_commission = EXCLUDED.maker_commission,
                max_leverage = EXCLUDED.max_leverage,
                max_liquidation_commission = EXCLUDED.max_liquidation_commission,
                min_trade_amount = EXCLUDED.min_trade_amount,
                block_trade_tick_size = EXCLUDED.block_trade_tick_size,
                block_trade_min_trade_amount = EXCLUDED.block_trade_min_trade_amount,
                block_trade_commission = EXCLUDED.block_trade_commission,
                rfq = EXCLUDED.rfq;
        """, rows)
        conn.commit()
        print(f"✅ Upserted {len(rows)} instruments.")


def main():
    print("📡 Fetching perpetual swap instruments from Deribit...")
    instruments = fetch_perpetual_instruments()
    print(f"🔢 Found {len(instruments)} perpetual instruments.")

    conn = get_connection()
    try:
        upsert_instruments(conn, instruments)
    finally:
        conn.close()
        print("🔌 Database connection closed.")


if __name__ == "__main__":
    main()
