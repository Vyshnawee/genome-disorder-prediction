import logging
from datetime import datetime

from src.config import LOG_DIR


# ---------------------------------------------------------
# Create a separate folder for every execution
# ---------------------------------------------------------

RUN_TIMESTAMP = datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)

RUN_LOG_DIR = LOG_DIR / RUN_TIMESTAMP

RUN_LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Create log file for this execution
# ---------------------------------------------------------

LOG_FILE = RUN_LOG_DIR / "agdpm.log"


# ---------------------------------------------------------
# Configure logger
# ---------------------------------------------------------

logger = logging.getLogger("AGDPM")

logger.setLevel(logging.INFO)

logger.propagate = False


# Remove existing handlers
if logger.handlers:
    logger.handlers.clear()


# File handler ONLY
file_handler = logging.FileHandler(
    LOG_FILE,
    encoding="utf-8"
)

file_handler.setLevel(logging.INFO)


# Log format
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

file_handler.setFormatter(formatter)


# Attach ONLY the file handler
logger.addHandler(file_handler)
