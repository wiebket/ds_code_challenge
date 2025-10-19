def basic_overview(df):
    """
    Print basic overview of data frame.

    Parameters
    ----------
    df : pandas dataframe
    """
    print("Basic dataset overview:")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}\n")
    print(f"Column names: {df.columns.tolist()}\n")
    print(df.info())  # Data types, non-null counts
    print(df.describe())  # Summary statistics
    print(df.head())  # First few rows


def check_duplicates(df, column_names):
    """
    Check if column in dataframe contains duplicate entries, excluding nan.

    Parameters
    ----------
    df : pandas dataframe
    column_names : list of strings
    """
    duplicate = len(df.dropna(subset=column_names)[df.duplicated(subset=column_names, keep=False)])
    print(f"Duplicate values in {column_names}: {duplicate}")
