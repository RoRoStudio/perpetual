import psycopg2
import psycopg2.extras
import os
from configparser import ConfigParser
from dotenv import load_dotenv
from contextlib import contextmanager
from psycopg2 import pool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("database")

# Create a global connection pool
_connection_pool = None

def initialize_connection_pool(min_connections=2, max_connections=30):
    """
    Initialize the global connection pool.
    Should be called at application startup.
    """
    global _connection_pool
    
    # Load DB credentials from .env
    load_dotenv('/mnt/p/perpetual/config/credentials.env')
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    
    # Load host/port/db from config.ini
    config = ConfigParser()
    config.read("/mnt/p/perpetual/config/config.ini")
    
    db_host = config.get("DATABASE", "DB_HOST")
    db_port = config.get("DATABASE", "DB_PORT")
    db_name = config.get("DATABASE", "DB_NAME")
    
    # Create pool if it doesn't exist
    if _connection_pool is None:
        logger.info(f"Creating connection pool: {min_connections}-{max_connections} connections")
        _connection_pool = pool.ThreadedConnectionPool(
            min_connections,
            max_connections,
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )

def get_connection():
    """
    Get a connection from the pool.
    If the pool doesn't exist, initialize it first.
    
    Returns:
        A database connection from the pool
    """
    global _connection_pool
    if _connection_pool is None:
        initialize_connection_pool()
    
    try:
        return _connection_pool.getconn()
    except Exception as e:
        logger.error(f"Error getting connection from pool: {e}")
        
        # Fallback to direct connection if pool fails
        load_dotenv('/mnt/p/perpetual/config/credentials.env')
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        
        config = ConfigParser()
        config.read("/mnt/p/perpetual/config/config.ini")
        
        db_host = config.get("DATABASE", "DB_HOST")
        db_port = config.get("DATABASE", "DB_PORT")
        db_name = config.get("DATABASE", "DB_NAME")
        
        return psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )

def release_connection(conn):
    """
    Release a connection back to the pool.
    
    Args:
        conn: The connection to release
    """
    global _connection_pool
    if _connection_pool is not None:
        try:
            _connection_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error releasing connection to pool: {e}")
            try:
                conn.close()
            except:
                pass

@contextmanager
def db_connection():
    """
    Context manager for database connections.
    Automatically releases the connection back to the pool.
    
    Example:
        with db_connection() as conn:
            # Use conn here
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        release_connection(conn)

def close_all_connections():
    """Close all pool connections and clear the pool."""
    global _connection_pool
    if _connection_pool is not None:
        try:
            _connection_pool.closeall()
            _connection_pool = None
            logger.info("All database connections closed")
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")