import os
from pathlib import Path

import boto3
import geopandas as gpd
import pandas as pd

from ds_code_challenge.config import Config


def download_from_s3(prefix, destination="raw"):
    """
    Download data from S3 to local data directory.

    Args:
        prefix: S3 object name
        destination: Subfolder in data/ ('raw', 'processed', 'interim', 'external')
    """

    # Create s3 connection
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
        region_name=Config.AWS_DEFAULT_REGION,
    )

    # Get items with common prefix
    response = s3_client.list_objects_v2(Bucket=Config.S3_BUCKET_NAME, Prefix=prefix)

    # Download each file in prefix
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".bak"):
            print(f"Skipping {key}")
            continue

        # Save to data directory
        local_path = Config.DATA_DIR / destination / key
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if os.path.isfile(local_path):
            print(f"Already downloaded {key}")
        else:
            s3_client.download_file(Config.S3_BUCKET_NAME, key, str(local_path))
            print(f"Downloaded to {local_path}")


def load_data(filename, data_dir="raw"):
    filepath = Config.DATA_DIR / data_dir / filename

    # check if filename is geojson or csv / csv.gz, then load accordingly
    if Path(filepath).suffix.lower() == ".geojson":
        try:
            gdf = gpd.read_file(filepath)
            return gdf
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    else:
        try:
            df = pd.read_csv(filepath, index_col=0)
            return df
        except Exception as e:
            print(f"Error reading {filename}: {e}")
