import logging

logger = logging.getLogger(__name__)


def detect_anomalies_zscore(daily_counts, threshold=3):
    """
    Detect anomalies using Z-score.

    Parameters
    ----------
    daily_counts : pandas dataframe with [date, department, request_count]
    threshold : Z-score threshold (typically 2-3)

    Returns
    -------
    dataframe with anomalies
    """
    logger.info(f"Detecting anomalies with Z-score threshold: {threshold}")

    # Calculate statistics per department
    dept_stats = (
        daily_counts.groupby("department")["request_count"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )

    # Merge statistics back
    daily_with_stats = daily_counts.merge(dept_stats, on="department")

    # Calculate Z-score
    daily_with_stats["z_score"] = (
        daily_with_stats["request_count"] - daily_with_stats["mean"]
    ) / daily_with_stats["std"]

    # Flag anomalies
    daily_with_stats["is_anomaly"] = daily_with_stats["z_score"].abs() > threshold

    # Get anomalies
    anomalies = daily_with_stats[daily_with_stats["is_anomaly"]].copy()

    logger.info(f"Found {len(anomalies)} anomalous days")
    logger.info(f"Departments affected: {anomalies['department'].nunique()}")

    return anomalies.sort_values(["department", "date"], ascending=True)
