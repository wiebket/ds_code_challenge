import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


class Config:
    """Application configuration from environment variables."""

    # Paths
    PROJ_ROOT = Path(__file__).resolve().parents[1]
    logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

    DATA_DIR = PROJ_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    INTERIM_DATA_DIR = DATA_DIR / "interim"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"

    MODELS_DIR = PROJ_ROOT / "models"

    REPORTS_DIR = PROJ_ROOT / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"

    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "af-south-1")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

    @classmethod
    def validate(cls):
        """Validate required environment variables."""
        required = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET_NAME"]
        missing = [var for var in required if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist."""
        for dir_path in [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.INTERIM_DATA_DIR,
            cls.EXTERNAL_DATA_DIR,
            cls.MODELS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Validate on import
Config.validate()
