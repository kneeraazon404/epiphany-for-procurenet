import psycopg2

from logger_setup import logger

# Database connection and table creation

conn = psycopg2.connect(
    database="apppgsqldb",
    user="apppgsqluser",
    password="p0$tGrr3$qladM!n18072023",
    host="43.207.8.208",
    port="5432",
)
c = conn.cursor()


def create_tables():
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS updated_products (
            id INTEGER PRIMARY KEY,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            product_name TEXT,
            product_url TEXT,
            retry_failed BOOLEAN DEFAULT FALSE
        );

        CREATE TABLE IF NOT EXISTS failed_products (
            id INTEGER PRIMARY KEY,
            product_name TEXT,
            product_url TEXT
        )
    """
    )
    conn.commit()


create_tables()
logger.info(
    "PostgreSQL database connected and table updated_products created if not exist."
)
