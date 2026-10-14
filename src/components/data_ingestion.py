import shutil
from dataclasses import dataclass

import pandas as pd

from src.config import DATA_FILE, RAW_DATA_FILE, WORKING_DATA_DIR
from src.exception import AGDPMException
from src.logger import logger


@dataclass
class DataIngestionArtifact:
    data: pd.DataFrame


class DataIngestion:
    """
    Copies the original Kaggle file from data/raw into data/working
    (only if the copy does not exist yet) and reads the copy.
    The file in data/raw is never modified.
    """

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        logger.info("========== DATA INGESTION STARTED ==========")

        try:
            WORKING_DATA_DIR.mkdir(parents=True, exist_ok=True)

            if DATA_FILE.exists():
                logger.info(f"Working copy already exists: {DATA_FILE}")
            else:
                shutil.copy2(RAW_DATA_FILE, DATA_FILE)
                logger.info(f"Copied {RAW_DATA_FILE} -> {DATA_FILE}")

            df = pd.read_csv(DATA_FILE)
            logger.info(f"Data shape: {df.shape}")

            logger.info("========== DATA INGESTION COMPLETED ==========")
            return DataIngestionArtifact(data=df)

        except Exception as e:
            logger.exception("Error occurred during data ingestion.")
            raise AGDPMException(e, None) from e
