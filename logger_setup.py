# Logging setup

import logging
import os
import sys

# Logging setup
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = os.path.join(log_directory, "logfile.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(file_handler)
logger.addHandler(console_handler)
