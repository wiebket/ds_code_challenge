import logging

import geopandas as gpd
from shapely.geometry import Point


def spatial_join(sr_df, hex_df):
    # rename hex_df index column to h3_level8_index, then join to sr_df;
    # log how many records failed to join; add threshold for join error
    # log time taken for join
    # if lat, long in joined df is nan, set index to 0

    logger = logging.getLogger(__name__)

    # Convert pandas df to GeoDataFrame
    sr_df['geometry'] = sr_df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    sr_geo_df = gpd.GeoDataFrame(sr_df, geometry='geometry', crs='EPSG:4326')

    # Rename index column in hex8
    hex_df.rename(columns={'index':'h3_level8_index'}, inplace=True)

    # Spatial join to find intersection (i.e. any overlap) between service request geometry & hexagon geometry
    sr_hex_df = gpd.sjoin(sr_geo_df, hex_df, how='left', predicate='intersects', rsuffix='right')

    return sr_hex_df


def validate_join(joined_df):
    # validate against sr_hex.csv.gz
    return