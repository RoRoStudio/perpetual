import psycopg2
import psycopg2.extras
import os
from configparser import ConfigParser
from dotenv import load_dotenv

def get_connection():
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

    return psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password
    )