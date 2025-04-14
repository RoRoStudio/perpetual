#!/usr/bin/env python3
"""
export_parquet.py
-----------------------------------------------------
Exports tier1 features and labels from PostgreSQL to Parquet format
for faster data loading during model training. This
optimization provides 3-4x speedup for data loading.
-----------------------------------------------------
"""

import os
import sys
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import traceback

# Add project root to path for imports
sys.path.append('/mnt/p/perpetual')

from data.database import get_connection, db_connection

# Define terms that indicate future-leaking features
LEAKY_TERMS = ('future_', 'next_', 'direction_', 'signal_', 'quantile')

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("export_parquet")

def export_separate_tables(instrument, output_dir, overwrite=False):
    """
    Export tier1 features and labels for a specific instrument to separate Parquet files.
    
    Args:
        instrument: Name of the instrument
        output_dir: Directory to save Parquet files
        overwrite: Whether to overwrite existing files
    
    Returns:
        Tuple of (features_path, labels_path) or (None, None) if failed
    """
    features_path = os.path.join(output_dir, f"tier1_features_{instrument}.parquet")
    labels_path = os.path.join(output_dir, f"tier1_labels_{instrument}.parquet")
    
    # Check if files already exist and skip if not overwriting
    if os.path.exists(features_path) and os.path.exists(labels_path) and not overwrite:
        logger.info(f"Skipping {instrument} - files already exist")
        return features_path, labels_path
    
    try:
        logger.info(f"Exporting feature and label data for {instrument}")
        with db_connection() as conn:
            # Get feature data from PostgreSQL
            features_df = pd.read_sql(
                "SELECT * FROM tier1_features_15m WHERE instrument_name = %s ORDER BY timestamp",
                conn, params=(instrument,)
            )
            
            # Get label data
            labels_df = pd.read_sql(
                "SELECT * FROM tier1_labels_15m WHERE instrument_name = %s ORDER BY timestamp",
                conn, params=(instrument,)
            )
            
            # Check if we got any data
            if len(features_df) == 0 or len(labels_df) == 0:
                logger.warning(f"No data found for {instrument} in separate tables")
                return None, None
                
            logger.info(f"Retrieved {len(features_df)} rows for {instrument}")
            
            # Convert to PyArrow table and write to Parquet
            features_table = pa.Table.from_pandas(features_df)
            labels_table = pa.Table.from_pandas(labels_df)
            
            pq.write_table(features_table, features_path, compression="ZSTD")
            pq.write_table(labels_table, labels_path, compression="ZSTD")
            
            logger.info(f"Exported {instrument} to {features_path} and {labels_path}")
            return features_path, labels_path
            
    except Exception as e:
        logger.error(f"Error exporting {instrument}: {e}")
        logger.error(traceback.format_exc())
        return None, None

def get_feature_whitelist(conn, instrument_name):
    """
    Get only safe feature columns from tier1_features_15m.
    
    Args:
        conn: Database connection
        instrument_name: Name of the instrument
        
    Returns:
        List of column names that are safe to use as features
    """
    # Query tier1_features_15m to get column names
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'tier1_features_15m'
        """)
        all_columns = [row[0] for row in cur.fetchall()]
    
    # Apply whitelist filter
    feature_columns = [
        col for col in all_columns
        if not any(term in col for term in LEAKY_TERMS)
        and col not in ('instrument_name', 'timestamp')
    ]
    
    return feature_columns

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export tier1 features and labels from PostgreSQL to Parquet format")
    parser.add_argument("--instrument", type=str, help="Export data for a specific instrument")
    parser.add_argument("--output-dir", type=str, default="/mnt/p/perpetual/cache", 
                      help="Directory to save Parquet files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel export workers")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = datetime.now()
    
    if args.instrument:
        # Export a single instrument
        export_separate_tables(args.instrument, args.output_dir, args.overwrite)
    else:
        # Export all instruments in parallel
        with db_connection() as conn:
            # Get instrument list
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT instrument_name 
                    FROM tier1_features_15m 
                    ORDER BY instrument_name
                """)
                instruments = [row[0] for row in cur.fetchall()]
                
        # Export each instrument
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(export_separate_tables, instrument, args.output_dir, args.overwrite): instrument
                for instrument in instruments
            }
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                            total=len(instruments), 
                            desc="Exporting instruments"):
                instrument = futures[future]
                try:
                    feature_path, label_path = future.result()
                    if feature_path and label_path:
                        results.append((feature_path, label_path))
                except Exception as e:
                    logger.error(f"Error exporting {instrument}: {e}")
                    logger.error(traceback.format_exc())
                    
        logger.info(f"Successfully exported {len(results)} out of {len(instruments)} instruments")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"Export completed in {duration:.2f} seconds")

if __name__ == "__main__":
    main()