import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from ds_code_challenge.utils import Timer

logging.basicConfig(level=logging.INFO)


def spatial_join(sr_df, hex_df):
    """
    Transform services requests data with the following operations:
    1. rename hex_df index column to h3_level8_index
    2. join hex_df to sr_df
    3. if lat, long in joined df are nan, set h3_level8_index to '0' (str type)
    4. select columns to keep so that joined df has same columns as sr_df + h3_level8_index

    The function logs time taken for the join operation and how many records failed to join.
    The join error threshold is set to 22.56%, which is just above the percentage of missing
    latitude/longitude records in sr_df.

    Parameters
    ----------
    sr_df : pandas dataframe with service requests data
    hex_df : geopandas dataframe with hex-polygons data

    Returns
    -------
    pandas dataframe with joined service requests and hex data
    """

    logger = logging.getLogger(__name__)

    # Convert pandas df to GeoDataFrame
    sr_df["geometry"] = sr_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    sr_geo_df = gpd.GeoDataFrame(sr_df, geometry="geometry", crs="EPSG:4326")

    # Rename index column in hex8
    hex_df.rename(columns={"index": "h3_level8_index"}, inplace=True)

    # Spatial join to find intersection (i.e. any overlap) between service request geometry & hexagon geometry
    with Timer() as t:
        sr_hex_df = gpd.sjoin(
            sr_geo_df, hex_df, how="left", predicate="intersects", rsuffix="right"
        )

    # Check for failed joins (NaN in joined columns)
    failed_joins = sr_hex_df[sr_hex_df["index_right"].isna()]
    join_error = len(failed_joins) / len(sr_geo_df) * 100

    # Log results
    logger.info(f" Join completed in {t.duration:.2f} seconds.")

    if len(failed_joins) > 0:
        logger.warning(
            f" Failed to join {len(failed_joins)} out of {len(sr_geo_df)} rows ({join_error:.2f}%)."
        )
    elif join_error > 22.56:
        raise ValueError(
            "Aborting. Join failure rate exceeds missing latitude/longitude values. Check join syntax."
        )
    else:
        logger.info(f"Successfully joined all {len(sr_geo_df)} rows")

    sr_hex_df["h3_level8_index"].fillna("0", inplace=True)
    keep_columns = sr_df.columns.to_list()[:-1] + ["h3_level8_index"]

    return sr_hex_df[keep_columns].set_index("notification_number")


def validate_join(df, val_df):
    """
    Validate dataframe join.

    Parameters
    ----------
    df : pandas dataframe to validate
    val_df : validation dataframe to compare against

    Returns
    -------
    difference between the dataframes
    """

    diff = df.compare(val_df)

    if diff.empty:
        print("DataFrames are identical")
    else:
        print(f"Found {len(diff)} differences:\n{diff}")

    return diff


def requests_by_department_per_day(df):
    """
    Aggregate service requests to daily counts per department.

    Parameters
    ----------
    df : Service request dataframe with 'creation_timestamp' and 'department' columns

    Returns
    -------
    dataframe with columns: [date, department, request_count]
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Aggregating {len(df)} service requests by department and day")

    # Convert timestamp to date
    df = df.copy()
    df["creation_timestamp"] = pd.to_datetime(df["creation_timestamp"])
    df["date"] = df["creation_timestamp"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate by date and department
    daily_counts = df.groupby(["date", "department"]).size().reset_index(name="request_count")

    logger.info(f"Aggregated to {len(daily_counts)} date-department combinations")
    logger.info(f"Date range: {daily_counts['date'].min()} to {daily_counts['date'].max()}")
    logger.info(f"Departments: {daily_counts['department'].nunique()}")

    return daily_counts


def filter_date_range(daily_counts, start_date, end_date):
    """
    Filter daily counts to specific date range.

    Parameters
    ----------
    daily_counts : pandas dataframe with [date, department, request_count]
    start_date : start date for date filter (string 'year-month-day')
    end_date : end date for date filter (string 'year-month-day')

    Returns
    -------
    dataframe filtered to date range
    """

    logger = logging.getLogger(__name__)

    logger.info(f"Filtering to date range: {start_date} to {end_date}")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter to date range
    filtered = daily_counts[
        (daily_counts["date"] >= start_date) & (daily_counts["date"] <= end_date)
    ].copy()

    logger.info(f"Filtered to {len(filtered)} records")

    return filtered
