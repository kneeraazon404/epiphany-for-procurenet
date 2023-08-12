import psycopg2

from logger_setup import logger

DATABASE_CONFIG = {
    "database": "apppgsqldb",
    "user": "apppgsqluser",
    "password": "p0$tGrr3$qladM!n18072023",
    "host": "43.207.8.208",
    "port": "5432",
}

#  Create a new connection
try:
    logger.info("Connecting to the PostgreSQL database...")
    conn = psycopg2.connect(**DATABASE_CONFIG)
    logger.info("Connection established.")
    c = conn.cursor()
except Exception as e:
    logger.error(f"Error connecting to the PostgreSQL database: {e}")
    raise
conn = psycopg2.connect(**DATABASE_CONFIG)


def create_updated_products_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS updated_products (
            id INTEGER PRIMARY KEY,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            product_name TEXT,
            product_url TEXT,
            retry_failed BOOLEAN DEFAULT FALSE
        )
        """
    )
    logger.info("Table 'updated_products' created or already exists.")


def create_failed_products_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS failed_products (
            id INTEGER PRIMARY KEY,
            product_name TEXT,
            product_url TEXT
        )
        """
    )
    logger.info("Table 'failed_products' created or already exists.")


create_updated_products_table(c)
create_failed_products_table(c)
conn.commit()
