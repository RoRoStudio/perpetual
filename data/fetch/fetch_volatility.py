# Fetch historical BTC/ETH volatility index
# /mnt/p/perpetual/data/fetching/fetch_volatility.py
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import time
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
logger = logging.getLogger("fetch_volatility")

# Ensure parent dir is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data.database import get_connection
from deribit.authentication import authenticate_http, add_auth_to_params


API_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
MILLISECONDS_PER_HOUR = 3600 * 1000
MAX_HOURS_PER_CALL = 720  # 30 days per request
RESOLUTION = "3600"  # 1-hour resolution (in seconds)

CURRENCIES = ["BTC", "ETH"]  # Supported by Deribit

# How far back to fetch data (in days)
MAX_HISTORY_DAYS = 1825  # 5 years

# --- DB Functions ---

def get_latest_timestamp(conn, currency):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(timestamp) FROM historical_volatility_index
            WHERE currency = %s;
        """, (currency,))
        return cur.fetchone()[0]


def insert_volatility_data(conn, currency, records):
    if not records:
        logger.debug("⚠️ INSERT: No records to insert")
        return 0

    # Get count before insertion
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM historical_volatility_index 
            WHERE currency = %s
        """, (currency,))
        count_before = cur.fetchone()[0]
        logger.debug(f"🔢 DB BEFORE: {count_before} existing rows for {currency}")

    # Show sample of raw records
    if records and logger.level <= logging.DEBUG:
        if len(records) > 2:
            sample_entries = records[:1] + records[-1:]
            logger.debug(f"📋 SAMPLE RECORDS: {sample_entries}")
        else:
            logger.debug(f"📋 ALL RECORDS: {records}")

    rows = [
        (
            currency,
            datetime.utcfromtimestamp(entry[0] / 1000),
            entry[1],
            entry[2],
            entry[3],
            entry[4]
        )
        for entry in records
    ]

    # Log the latest volatility index we're trying to insert
    if rows and logger.level <= logging.DEBUG:
        latest_row = sorted(rows, key=lambda x: x[1], reverse=True)[0]
        logger.debug(f"📅 LATEST VOL INDEX TO INSERT: {currency} @ {latest_row[1]} - O:{latest_row[2]} H:{latest_row[3]} L:{latest_row[4]} C:{latest_row[5]}")

    try:
        with conn.cursor() as cur:
            logger.debug(f"💾 INSERTING: Attempting to insert {len(rows)} rows for {currency}")
            execute_values(cur, """
                INSERT INTO historical_volatility_index (
                    currency, timestamp, open, high, low, close
                ) VALUES %s
                ON CONFLICT (currency, timestamp) DO NOTHING;
            """, rows)
            conn.commit()
            
            # Get count after insertion
            cur.execute("""
                SELECT COUNT(*) FROM historical_volatility_index 
                WHERE currency = %s
            """, (currency,))
            count_after = cur.fetchone()[0]
            logger.debug(f"🔢 DB AFTER: {count_after} total rows for {currency}")

        # Return actual number of new rows
        inserted = count_after - count_before
        if inserted == 0:
            logger.debug(f"⚠️ INSERT RESULT: No new rows inserted for {currency} (all rows already existed)")
        else:
            logger.debug(f"✅ INSERT RESULT: Successfully inserted {inserted} new rows for {currency}")
        return inserted
    except Exception as e:
        logger.error(f"Database error during insertion: {e}")
        conn.rollback()
        return 0

# --- HTTP Fetch ---

def fetch_vol_chunk(currency, start_ts, end_ts, access_token=None):
    """Make HTTP request to fetch volatility index data"""
    params = {
        "currency": currency,
        "start_timestamp": int(start_ts),
        "end_timestamp": int(end_ts),
        "resolution": RESOLUTION
    }
    
    # Add authentication if provided
    if access_token:
        params = add_auth_to_params(params, access_token)
    
    # Enhanced request logging
    start_date = datetime.utcfromtimestamp(start_ts/1000)
    end_date = datetime.utcfromtimestamp(end_ts/1000)
    logger.debug(f"🔍 REQUEST: Volatility index for {currency} from {start_date} to {end_date}")
    logger.debug(f"🔢 TIMESTAMPS: start={start_ts}, end={end_ts}")
    
    try:
        response = requests.get(API_URL, params=params)
        
        if response.status_code != 200:
            logger.error(f"API Request failed: {response.status_code} - {response.text}")
            raise Exception(f"API Request failed: {response.text}")
        
        result = response.json().get("result", {}).get("data", [])
        
        # Enhanced response logging
        if result:
            first_entry = result[0]
            last_entry = result[-1]
            first_time = datetime.utcfromtimestamp(first_entry[0] / 1000) if first_entry else None
            last_time = datetime.utcfromtimestamp(last_entry[0] / 1000) if last_entry else None
            logger.debug(f"📊 RESPONSE: {len(result)} volatility records from {first_time} to {last_time}")
            
            # Log sample data
            if logger.level <= logging.DEBUG:
                if len(result) > 4:
                    # Show first and last 2 entries
                    sample_entries = result[:2] + result[-2:]
                    samples_formatted = [{
                        'time': datetime.utcfromtimestamp(entry[0]/1000),
                        'open': entry[1],
                        'high': entry[2],
                        'low': entry[3],
                        'close': entry[4]
                    } for entry in sample_entries]
                    logger.debug(f"📈 SAMPLE RECORDS: {samples_formatted}")
                else:
                    samples_formatted = [{
                        'time': datetime.utcfromtimestamp(entry[0]/1000),
                        'open': entry[1],
                        'high': entry[2],
                        'low': entry[3],
                        'close': entry[4]
                    } for entry in result]
                    logger.debug(f"📈 ALL RECORDS: {samples_formatted}")
        else:
            logger.debug(f"⚠️ RESPONSE: No volatility index data returned")
        
        return result
    except Exception as e:
        logger.error(f"Exception during API request: {e}")
        raise e


# --- Fetch All for One Currency ---

def fetch_all_vol_for_currency(conn, currency, is_historical=False):
    """Fetch all volatility index data for a currency with pagination"""
    logger.info(f"🔄 Fetching volatility index for: {currency}")
    
    # Public endpoints don't require authentication
    access_token = None
    if logger.level <= logging.DEBUG:
        logger.debug("🔓 Bypassing authentication for public endpoint")

    # Current time as end timestamp
    now_dt = datetime.utcnow()
    now = int(now_dt.timestamp() * 1000)
    logger.debug(f"⏱️ CURRENT TIME: {now_dt} UTC / {now} ms")
    
    # For historical backfill, start from 5 years ago or earliest available
    if is_historical:
        start_ts = int((datetime.utcnow() - timedelta(days=MAX_HISTORY_DAYS)).timestamp() * 1000)
        logger.info(f"📊 Historical backfill from {datetime.utcfromtimestamp(start_ts/1000)} to {datetime.utcfromtimestamp(now/1000)}")
    else:
        # For regular updates, just get recent data from the last timestamp we have
        latest_ts = get_latest_timestamp(conn, currency)
        if latest_ts:
            start_ts = int((latest_ts + timedelta(minutes=5)).timestamp() * 1000)  # Smaller increment to ensure overlap
            logger.debug(f"📈 LATEST DB TIMESTAMP: {latest_ts} UTC / {start_ts} ms")
        else:
            # If no data yet, start 5 years back
            start_ts = int((datetime.utcnow() - timedelta(days=MAX_HISTORY_DAYS)).timestamp() * 1000)
            logger.debug(f"📦 NO DATA YET: Starting from {datetime.utcfromtimestamp(start_ts/1000)} UTC / {start_ts} ms")

    total_inserted = 0
    
    # For historical mode, work backwards from now
    if is_historical:
        current_end = now
        
        # Continue fetching until we reach the start timestamp or get no more data
        while current_end > start_ts:
            # Calculate chunk start time (max 30 days per call)
            current_start = max(start_ts, current_end - (MAX_HOURS_PER_CALL * MILLISECONDS_PER_HOUR))
            
            logger.info(f"Fetching chunk: {datetime.utcfromtimestamp(current_start/1000)} to {datetime.utcfromtimestamp(current_end/1000)}")
            
            try:
                records = fetch_vol_chunk(currency, current_start, current_end, access_token)
            except Exception as e:
                logger.error(f"❌ Error fetching chunk: {e}")
                break
            
            if not records:
                logger.warning(f"No data in chunk, moving to next chunk")
                current_end = current_start
                continue
            
            inserted = insert_volatility_data(conn, currency, records)
            total_inserted += inserted
            
            if inserted > 0:
                logger.info(f"➕ Inserted {inserted} new rows")
            
            # Move to previous chunk
            current_end = current_start
            
            # Rate limiting to avoid API issues
            time.sleep(0.5)
    else:
        # For regular mode, just get data from the latest timestamp to now
        while start_ts < now:
            end_ts = min(start_ts + (MAX_HOURS_PER_CALL * MILLISECONDS_PER_HOUR), now)
            
            try:
                records = fetch_vol_chunk(currency, start_ts, end_ts, access_token)
            except Exception as e:
                logger.error(f"❌ Error fetching chunk: {e}")
                break
            
            if not records:
                logger.warning(f"🟡 No data in range: {datetime.utcfromtimestamp(start_ts/1000)}")
                start_ts = end_ts  # Move forward to next time period
                continue
            
            inserted = insert_volatility_data(conn, currency, records)
            total_inserted += inserted
            
            if inserted > 0:
                logger.info(f"➕ Inserted {inserted} new rows ({datetime.utcfromtimestamp(start_ts/1000)} → {datetime.utcfromtimestamp(end_ts/1000)})")
            else:
                logger.info(f"⏭️ No new data ({datetime.utcfromtimestamp(start_ts/1000)} → {datetime.utcfromtimestamp(end_ts/1000)})")
            
            start_ts = end_ts
            time.sleep(0.25)

    logger.info(f"✅ Done: {currency} — Total rows inserted: {total_inserted}")
    return total_inserted


# --- Main Entrypoint ---

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
        for currency in CURRENCIES:
            fetch_all_vol_for_currency(conn, currency, is_historical)
    finally:
        conn.close()


if __name__ == "__main__":
    main()