import logging
import os
import sys


def setup_logger(name, log_directory="logs", log_level=logging.INFO):
    """
    Configure logger for given name.

    Args:
    - name (str): Name of the logger.
    - log_directory (str, optional): Directory where logs will be stored. Defaults to "logs".
    - log_level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
    - logger: Configured logger.
    """
    # Check and create the logging directory
    os.makedirs(log_directory, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # File handler for logging
    log_file = os.path.join(log_directory, "logfile.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Console handler for logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Formatters
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(message)s")

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Use the function to set up the logger
logger = setup_logger(__name__)
