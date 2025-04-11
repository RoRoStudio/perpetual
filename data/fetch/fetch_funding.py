# Fetch hourly/8h funding rates and index prices
# /mnt/p/perpetual/data/fetching/fetch_funding.py
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import time
import sys
import os
import logging
import traceback

# Set timezone to UTC explicitly
os.environ['TZ'] = 'UTC'
if hasattr(time, 'tzset'):
    time.tzset()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("fetch_funding")

# Ensure parent dir is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data.database import get_connection
from deribit.authentication import authenticate_http, add_auth_to_params


API_URL = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
MILLISECONDS_PER_HOUR = 60 * 60 * 1000
MAX_HOURS_PER_CALL = 720  # Fetch 30 days at a time (720 hours)

# How far back to fetch data (in days)
MAX_HISTORY_DAYS = 1825  # 5 years

# --- SQL Helper Functions ---

def get_used_instruments(conn):
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


def get_latest_timestamp(conn, instrument_name):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(timestamp) FROM historical_funding_rates
            WHERE instrument_name = %s;
        """, (instrument_name,))
        ts = cur.fetchone()[0]
        return ts


def insert_funding_rates(conn, instrument_name, records):
    if not records:
        logger.debug("⚠️ INSERT: No records to insert")
        return 0

    # Log raw records for debugging
    logger.debug(f"🔄 PROCESSING: {len(records)} records for {instrument_name}")
    if len(records) > 0 and logger.level <= logging.DEBUG:
        logger.debug(f"📋 SAMPLE RECORD: {records[0]}")

    # Get count before insertion
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM historical_funding_rates 
            WHERE instrument_name = %s
        """, (instrument_name,))
        count_before = cur.fetchone()[0]
        logger.debug(f"🔢 DB BEFORE: {count_before} existing rows for {instrument_name}")

    # Prepare rows
    rows = []
    for entry in records:
        try:
            timestamp = entry.get("timestamp")
            if not timestamp:
                logger.warning(f"Skipping record with missing timestamp: {entry}")
                continue
                
            rows.append((
                instrument_name,
                datetime.utcfromtimestamp(timestamp / 1000),
                entry.get("interest_1h"),
                entry.get("interest_8h"),
                entry.get("index_price"),
                entry.get("prev_index_price")
            ))
        except Exception as e:
            logger.error(f"Error processing record {entry}: {e}")
            continue

    if not rows:
        logger.warning("No valid rows to insert")
        return 0

    # Log the latest funding rate we're trying to insert
    if rows and logger.level <= logging.DEBUG:
        latest_row = sorted(rows, key=lambda x: x[1], reverse=True)[0]
        logger.debug(f"📅 LATEST FUNDING RATE TO INSERT: {instrument_name} @ {latest_row[1]} - 1h:{latest_row[2]} 8h:{latest_row[3]} price:{latest_row[4]}")

    try:
        with conn.cursor() as cur:
            logger.debug(f"💾 INSERTING: Attempting to insert {len(rows)} rows for {instrument_name}")
            execute_values(cur, """
                INSERT INTO historical_funding_rates (
                    instrument_name, timestamp,
                    interest_1h, interest_8h,
                    index_price, prev_index_price
                ) VALUES %s
                ON CONFLICT (instrument_name, timestamp) DO NOTHING;
            """, rows)
            conn.commit()
            
            # Get count after insertion
            cur.execute("""
                SELECT COUNT(*) FROM historical_funding_rates 
                WHERE instrument_name = %s
            """, (instrument_name,))
            count_after = cur.fetchone()[0]
            logger.debug(f"🔢 DB AFTER: {count_after} total rows for {instrument_name}")

        # Return actual number of new rows
        inserted = count_after - count_before
        if inserted == 0:
            logger.debug(f"⚠️ INSERT RESULT: No new rows inserted for {instrument_name} (all rows already existed)")
        else:
            logger.debug(f"✅ INSERT RESULT: Successfully inserted {inserted} new rows for {instrument_name}")
        return inserted
    except Exception as e:
        logger.error(f"Database error during insertion: {e}")
        conn.rollback()
        return 0

# --- HTTP Request Logic ---

def fetch_funding_chunk(instrument_name, start_ts, end_ts, access_token=None):
    """Make HTTP request to fetch funding rate data"""
    params = {
        "instrument_name": instrument_name,
        "start_timestamp": int(start_ts),
        "end_timestamp": int(end_ts)
    }
    
    # Add authentication if provided
    if access_token:
        params = add_auth_to_params(params, access_token)
    
    # Enhanced request logging
    start_date = datetime.utcfromtimestamp(start_ts/1000)
    end_date = datetime.utcfromtimestamp(end_ts/1000)
    logger.debug(f"🔍 REQUEST: Funding rates for {instrument_name} from {start_date} to {end_date}")
    logger.debug(f"🔢 TIMESTAMPS: start={start_ts}, end={end_ts}")
    
    try:
        response = requests.get(API_URL, params=params)
        
        if response.status_code != 200:
            logger.error(f"API Request failed: {response.status_code} - {response.text}")
            return []
        
        result = response.json()
        
        if "error" in result:
            logger.error(f"API error: {result['error']}")
            return []
            
        data = result.get("result", [])
        
        # Enhanced response logging
        if data:
            first_entry = data[0]
            last_entry = data[-1]
            first_time = datetime.utcfromtimestamp(first_entry.get("timestamp", 0) / 1000)
            last_time = datetime.utcfromtimestamp(last_entry.get("timestamp", 0) / 1000)
            logger.debug(f"📊 RESPONSE: {len(data)} funding rate records from {first_time} to {last_time}")
            
            # Log sample data
            if logger.level <= logging.DEBUG:
                if len(data) > 4:
                    # Show first and last 2 entries
                    sample_entries = data[:2] + data[-2:]
                    logger.debug(f"📈 SAMPLE RECORDS: {sample_entries}")
                else:
                    logger.debug(f"📈 ALL RECORDS: {data}")
        else:
            logger.debug(f"⚠️ RESPONSE: No funding rate data returned")
            
        return data
    except Exception as e:
        logger.error(f"Exception during API request: {e}")
        return []


def fetch_all_for_instrument(conn, instrument_name, is_historical=False):
    """Fetch all funding rate data for a specific instrument with pagination"""
    logger.info(f"🔄 Fetching funding rates for: {instrument_name}")

    # Public endpoints don't require authentication
    access_token = None
    if logger.level <= logging.DEBUG:
        logger.debug("🔓 Bypassing authentication for public endpoint")

    # Current time as end timestamp
    now_dt = datetime.utcnow()
    now = int(now_dt.timestamp() * 1000)
    logger.debug(f"⏱️ CURRENT TIME: {now_dt} UTC / {now} ms")
    
    # For historical backfill, start from the earliest point
    if is_historical:
        # Get the creation date of the instrument
        creation_date = get_instrument_creation_date(conn, instrument_name)
        start_ts = int(creation_date.timestamp() * 1000)
        logger.info(f"📊 Historical backfill from {creation_date} to {datetime.utcfromtimestamp(now/1000)}")
    else:
        # For regular updates, just get recent data from the last timestamp we have
        latest_ts = get_latest_timestamp(conn, instrument_name)
        if latest_ts:
            start_ts = int((latest_ts + timedelta(minutes=5)).timestamp() * 1000)  # Smaller increment to ensure overlap
            logger.debug(f"📈 LATEST DB TIMESTAMP: {latest_ts} UTC / {start_ts} ms")
        else:
            # If no data yet, get the creation date
            creation_date = get_instrument_creation_date(conn, instrument_name)
            start_ts = int(creation_date.timestamp() * 1000)
            logger.debug(f"📦 NO DATA YET: Starting from creation date {creation_date} UTC / {start_ts} ms")


    total_inserted = 0
    
    # For historical mode, work forwards from creation date
    if is_historical:
        current_start = start_ts
        
        # Continue fetching until we reach now
        while current_start < now:
            # Calculate chunk end time (max 30 days per call)
            current_end = min(current_start + (MAX_HOURS_PER_CALL * MILLISECONDS_PER_HOUR), now)
            
            logger.info(f"Fetching chunk: {datetime.utcfromtimestamp(current_start/1000)} to {datetime.utcfromtimestamp(current_end/1000)}")
            
            try:
                records = fetch_funding_chunk(instrument_name, current_start, current_end, access_token)
            except Exception as e:
                logger.error(f"❌ Error fetching chunk: {e}")
                break
            
            if not records:
                logger.warning(f"No data in chunk, moving to next chunk")
                current_start = current_end
                continue
            
            logger.info(f"Received {len(records)} funding rate records")
            
            inserted = insert_funding_rates(conn, instrument_name, records)
            total_inserted += inserted
            
            if inserted > 0:
                logger.info(f"➕ Inserted {inserted} new rows")
            
            # Move to next chunk
            current_start = current_end
            
            # Rate limiting to avoid API issues
            time.sleep(0.5)
    else:
        # For regular mode, just get data from the latest timestamp to now
        while start_ts < now:
            end_ts = min(start_ts + (MAX_HOURS_PER_CALL * MILLISECONDS_PER_HOUR), now)
            
            try:
                records = fetch_funding_chunk(instrument_name, start_ts, end_ts, access_token)
            except Exception as e:
                logger.error(f"❌ Error fetching chunk: {e}")
                break
            
            if not records:
                logger.warning(f"🟡 No more records returned for this time period.")
                # Move forward to next time period
                start_ts = end_ts
                continue
            
            inserted = insert_funding_rates(conn, instrument_name, records)
            total_inserted += inserted
            
            if inserted > 0:
                logger.info(f"➕ Inserted {inserted} new rows ({datetime.utcfromtimestamp(start_ts / 1000)} → {datetime.utcfromtimestamp(end_ts / 1000)})")
            else:
                logger.info(f"⏭️ No new data ({datetime.utcfromtimestamp(start_ts / 1000)} → {datetime.utcfromtimestamp(end_ts / 1000)})")
            
            start_ts = end_ts
            time.sleep(0.3)

    logger.info(f"✅ Done: {instrument_name} — Total rows inserted: {total_inserted}")
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
        instruments = get_used_instruments(conn)
        logger.info(f"Found {len(instruments)} instruments marked used = TRUE.")

        total_inserted = 0
        for instrument in instruments:
            inserted = fetch_all_for_instrument(conn, instrument, is_historical)
            total_inserted += inserted
            
        logger.info(f"✅ Grand total: {total_inserted} funding rate records inserted")
    finally:
        conn.close()


if __name__ == "__main__":
    main()