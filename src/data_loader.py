"""
Tasks 1 & 3.
Implement every function below. Do NOT change function signatures.
"""
import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Task 1a  [3 pts]

    Reads the raw dataset from disk and prepares the date information
    that all downstream tasks rely on. It must load the CSV file,
    ensure that the 'date' column is stored as a proper datetime type,
    and create a new integer column called 'yyyymm' that encodes each
    date as year * 100 + month (e.g. March 2005 becomes 200503).

    Input:  A string filepath pointing to the .csv file.
    Output: A pandas.DataFrame containing all original columns, with
            'date' converted to datetime and a new 'yyyymm' column
            appended.
    """
    raise NotImplementedError


def filter_data(df: pd.DataFrame,
                start_date: str = "2000-01-31") -> pd.DataFrame:
    """
    Task 1b  [2 pts]

    Restricts the dataset to a desired time window. Keeps only the
    rows whose 'date' is on or after the given start_date and returns
    a DataFrame with a cleanly reset index.

    Input:  A DataFrame (output of load_data) and a date string.
    Output: A filtered DataFrame with the index reset.
    """
    raise NotImplementedError


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Task 1c  [3 pts]

    Discovers which columns in the DataFrame are usable features for
    regression. A column qualifies as a feature if it is numeric and
    is not one of the non-feature columns listed in config.py under
    NON_FEATURE_COLS (which includes 'id', 'date', 'size_grp',
    'yyyymm', and the target 'r_1'). The returned list must be sorted
    alphabetically so that column ordering is deterministic across all
    students.

    Input:  A DataFrame.
    Output: A sorted Python list of column-name strings.
    """
    raise NotImplementedError


def train_test_split(df: pd.DataFrame,
                     features: list,
                     target: str,
                     n_train: int) -> tuple:
    """
    Task 3  [5 pts]

    Splits the data chronologically into a training set and a test
    set. The sorted unique months are divided so that the first
    n_train months form the training period and all remaining months
    form the test period. From each period the function extracts the
    feature matrix, the target vector, and the corresponding yyyymm
    dates.

    Input:  The filtered DataFrame, the feature-name list, the target
            column name, and the integer n_train.
    Output: A six-element tuple
            (S_train, y_train, S_test, y_test, dates_train, dates_test)
            where S arrays have dtype float64 and shape (n_samples, P),
            y arrays have dtype float64 and shape (n_samples,), and
            dates arrays contain yyyymm integers with shape (n_samples,).
    """
    raise NotImplementedError
