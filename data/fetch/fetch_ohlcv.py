# Fetch 15m OHLCV candles from Deribit and store in DB
# /mnt/p/perpetual/data/fetching/fetch_ohlcv.py
import requests
import psycopg2
from psycopg2.extras import execute_values
import time
from datetime import datetime, timedelta
import sys
import os
import logging

# Set timezone to UTC explicitly
os.environ['TZ'] = 'UTC'
if hasattr(time, 'tzset'):
    time.tzset()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("fetch_ohlcv")

# Ensure parent dir is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data.database import get_connection
from deribit.authentication import authenticate_http, add_auth_to_params

HTTP_URL = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
RESOLUTION = "15"  # 15-minute candles
MILLISECONDS_PER_BAR = 15 * 60 * 1000
MAX_CANDLES_PER_CALL = 1000  # Deribit usually limits to 1000 candles per request

# How far back to fetch data (in days)
MAX_HISTORY_DAYS = 1825  # 5 years


def get_usdc_perps(conn):
    """Get all USDC-based perpetual contracts marked as 'used'"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT instrument_name FROM instruments
            WHERE used = TRUE
              AND quote_currency = 'USDC'
              AND kind = 'future'
              AND instrument_name ILIKE '%-PERPETUAL';
        """)
        return [row[0] for row in cur.fetchall()]


def get_instrument_creation_date(conn, instrument_name):
    """Get the creation date of an instrument from the database"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT creation_timestamp FROM instruments
            WHERE instrument_name = %s;
        """, (instrument_name,))
        row = cur.fetchone()
        if row and row[0]:
            return row[0]
    
    # If no creation date found, default to 5 years ago
    return datetime.utcnow() - timedelta(days=MAX_HISTORY_DAYS)


def insert_ohlcv_data(conn, instrument_name, candles):
    """Insert OHLCV data into the database"""
    if not candles.get("ticks") or not candles.get("open"):
        logger.debug(f"⚠️ INSERT: No valid candles data to insert for {instrument_name}")
        return 0

    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM historical_ohlcv_data 
            WHERE instrument_name = %s
        """, (instrument_name,))
        count_before = cur.fetchone()[0]
        logger.debug(f"🔢 DB BEFORE: {count_before} existing rows for {instrument_name}")

    # Create rows to insert as before
    rows = [
        (
            instrument_name,
            datetime.utcfromtimestamp(t / 1000.0),
            o, h, l, c, v
        )
        for t, o, h, l, c, v in zip(
            candles["ticks"],
            candles["open"],
            candles["high"],
            candles["low"],
            candles["close"],
            candles["volume"]
        )
    ]
    
    # Log the latest candle we're trying to insert
    if rows and logger.level <= logging.DEBUG:
        latest_row = sorted(rows, key=lambda x: x[1], reverse=True)[0]
        logger.debug(f"📅 LATEST CANDLE TO INSERT: {instrument_name} @ {latest_row[1]} - O:{latest_row[2]} H:{latest_row[3]} L:{latest_row[4]} C:{latest_row[5]} V:{latest_row[6]}")

    with conn.cursor() as cur:
        logger.debug(f"💾 INSERTING: Attempting to insert {len(rows)} rows for {instrument_name}")
        execute_values(cur, """
            INSERT INTO historical_ohlcv_data (
                instrument_name, timestamp, open, high, low, close, volume
            ) VALUES %s
            ON CONFLICT (instrument_name, timestamp) DO NOTHING;
        """, rows)
        conn.commit()

        cur.execute("""
            SELECT COUNT(*) FROM historical_ohlcv_data 
            WHERE instrument_name = %s
        """, (instrument_name,))
        count_after = cur.fetchone()[0]
        logger.debug(f"🔢 DB AFTER: {count_after} total rows for {instrument_name}")

    inserted = count_after - count_before
    if inserted == 0:
        logger.debug(f"⚠️ INSERT RESULT: No new rows inserted for {instrument_name} (all rows already existed)")
    else:
        logger.debug(f"✅ INSERT RESULT: Successfully inserted {inserted} new rows for {instrument_name}")
    return inserted


def fetch_ohlcv_chunk(instrument_name, start_ts, end_ts, access_token=None):
    """Fetch OHLCV data for a specific time range using HTTP API"""
    # Print the actual timestamps we're requesting (for debugging)
    start_dt = datetime.utcfromtimestamp(start_ts / 1000.0)
    end_dt = datetime.utcfromtimestamp(end_ts / 1000.0)
    logger.debug(f"🔍 REQUEST: {instrument_name} from {start_dt} to {end_dt}")
    logger.debug(f"🔢 TIMESTAMPS: start={start_ts}, end={end_ts}")
    
    params = {
        "instrument_name": instrument_name,
        "start_timestamp": int(start_ts),
        "end_timestamp": int(end_ts),
        "resolution": RESOLUTION
    }
    
    # Add authentication if provided
    if access_token:
        params = add_auth_to_params(params, access_token)
    
    try:
        response = requests.get(HTTP_URL, params=params)
        
        if response.status_code != 200:
            logger.error(f"API request failed: {response.text}")
            return {}
            
        result = response.json()
        
        # Print the first and last few ticks if available (for debugging)
        if "result" in result and "ticks" in result["result"] and result["result"]["ticks"]:
            ticks = result["result"]["ticks"]
            first_tick = datetime.utcfromtimestamp(ticks[0] / 1000.0)
            last_tick = datetime.utcfromtimestamp(ticks[-1] / 1000.0)
            logger.debug(f"📊 RESPONSE: {len(ticks)} candles from {first_tick} to {last_tick}")
            if logger.level <= logging.DEBUG:
                # Print the first 2 and last 2 candles
                if len(ticks) > 4:
                    sample_start = zip(ticks[:2], result["result"]["open"][:2], result["result"]["close"][:2])
                    sample_end = zip(ticks[-2:], result["result"]["open"][-2:], result["result"]["close"][-2:])
                    logger.debug(f"📈 FIRST CANDLES: {[{'time': datetime.utcfromtimestamp(t/1000), 'open': o, 'close': c} for t, o, c in sample_start]}")
                    logger.debug(f"📉 LAST CANDLES: {[{'time': datetime.utcfromtimestamp(t/1000), 'open': o, 'close': c} for t, o, c in sample_end]}")
                else:
                    sample = zip(ticks, result["result"]["open"], result["result"]["close"])
                    logger.debug(f"📊 ALL CANDLES: {[{'time': datetime.utcfromtimestamp(t/1000), 'open': o, 'close': c} for t, o, c in sample]}")
        else:
            logger.debug(f"⚠️ RESPONSE: No candles returned")

        return result.get("result", {})
    except Exception as e:
        logger.error(f"Exception during request: {e}")
        return {}


def fetch_all_ohlcv_for_pair(conn, instrument_name, is_historical=False):
    """Fetch all OHLCV data for a specific instrument with pagination"""
    logger.info(f"🔄 Fetching OHLCV data for: {instrument_name}")

    # Public endpoints don't require authentication
    access_token = None
    if logger.level <= logging.DEBUG:
        logger.debug("🔓 Bypassing authentication for public endpoint")

    # Determine start and end timestamps
    now_dt = datetime.utcnow()
    end_time_dt = now_dt + timedelta(hours=1)  # Work with datetime objects
    now = int(now_dt.timestamp() * 1000)  # Convert to milliseconds for API
    end_ts = int(end_time_dt.timestamp() * 1000)  # Convert to milliseconds for API

    logger.debug(f"⏱️ CURRENT TIME: {now_dt} UTC")
    logger.debug(f"⏱️ END TIME: {end_time_dt} UTC")
    logger.debug(f"⏱️ TIMESTAMPS: now={now}, end={end_ts}")
    
    # If fetching historical data, start from the earliest point
    if is_historical:
        # Get the creation date of the instrument
        creation_date = get_instrument_creation_date(conn, instrument_name)
        start_ts = int(creation_date.timestamp() * 1000)
        logger.info(f"📊 Historical backfill from {creation_date} to {now}")
    else:
        # For regular updates, just get recent data from the last timestamp we have
        with conn.cursor() as cur:
            cur.execute("""
                SELECT MAX(timestamp) FROM historical_ohlcv_data
                WHERE instrument_name = %s;
            """, (instrument_name,))
            latest_ts = cur.fetchone()[0]
        
        if latest_ts:
            start_ts = int(latest_ts.timestamp() * 1000)
            logger.debug(f"📈 LATEST DB TIMESTAMP: {latest_ts} UTC / {start_ts} ms")
        else:
            # If no data yet, get the creation date
            creation_date = get_instrument_creation_date(conn, instrument_name)
            start_ts = int(creation_date.timestamp() * 1000)
            logger.debug(f"📦 NO DATA YET: Starting from creation date {creation_date} UTC / {start_ts} ms")
    
    total_inserted = 0
    candles_per_chunk = MAX_CANDLES_PER_CALL * MILLISECONDS_PER_BAR
    
    # For historical mode, work backwards from now to beginning
    if is_historical:
        current_end = end_ts
        
        # Continue fetching until we reach the start timestamp or get no more data
        while current_end > start_ts:
            current_start = max(start_ts, current_end - candles_per_chunk)
            
            logger.info(f"Fetching chunk: {datetime.utcfromtimestamp(current_start/1000)} to {datetime.utcfromtimestamp(current_end/1000)}")
            
            candles = fetch_ohlcv_chunk(instrument_name, current_start, current_end, access_token)
            
            if not candles.get("ticks"):
                logger.warning(f"No data in chunk, moving to next chunk")
                current_end = current_start
                continue
            
            inserted = insert_ohlcv_data(conn, instrument_name, candles)
            total_inserted += inserted
            
            if inserted > 0:
                logger.info(f"➕ Inserted {inserted} new candles")
            
            # Move to previous chunk, with small overlap to ensure continuity
            current_end = current_start - 1000  # 1 second before current start
            
            # Rate limiting to avoid API issues
            time.sleep(0.5)
    else:
        # For normal mode, just get the latest chunk
        candles = fetch_ohlcv_chunk(instrument_name, start_ts, end_ts, access_token)
        
        if candles.get("ticks"):
            inserted = insert_ohlcv_data(conn, instrument_name, candles)
            total_inserted += inserted
            
            if inserted > 0:
                logger.info(f"➕ Inserted {inserted} new candles")
            else:
                logger.info(f"⏭️ No new data")
        else:
            logger.warning(f"🟡 No candles returned")
    
    logger.info(f"✅ Total inserted for {instrument_name}: {total_inserted} candles")
    return total_inserted


def main():
    # Check for history mode flag
    is_historical = "--historical" in sys.argv
    
    # Enable debug logging if requested
    if "--debug" in sys.argv:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Add comprehensive timestamp debugging
    now_utc = datetime.utcnow()
    now_local = datetime.now()
    now_ts = now_utc.timestamp()
    
    logger.debug(f"🕒 SYSTEM TIME CHECK:")
    logger.debug(f"  - Local datetime.now(): {now_local}")
    logger.debug(f"  - UTC datetime.utcnow(): {now_utc}")
    logger.debug(f"  - Timestamp now_utc.timestamp(): {now_ts}")
    logger.debug(f"  - Converting back: datetime.fromtimestamp(now_ts): {datetime.fromtimestamp(now_ts)}")
    logger.debug(f"  - Converting back as UTC: datetime.utcfromtimestamp(now_ts): {datetime.utcfromtimestamp(now_ts)}")
    
    # Check TZ environment variable
    tz = os.environ.get('TZ')
    logger.debug(f"  - TZ environment variable: {tz}")
    
    # Check if tzset function was called
    tzset_available = hasattr(time, 'tzset')
    logger.debug(f"  - time.tzset available: {tzset_available}")
    
    logger.info(f"Mode: {'Historical backfill' if is_historical else 'Regular update'}")
    logger.info(f"Current time (UTC): {now_utc}")
    
    conn = get_connection()
    try:
        instruments = get_usdc_perps(conn)
        logger.info(f"Found {len(instruments)} USDC perps with used = TRUE.")

        total_candles = 0
        for inst in instruments:
            candles = fetch_all_ohlcv_for_pair(conn, inst, is_historical)
            total_candles += candles
            
        logger.info(f"✅ Total: {total_candles} new candles inserted into DB.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()