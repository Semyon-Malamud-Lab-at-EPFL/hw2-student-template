"""
Task 2 — Exploratory Data Analysis.
Implement every function below. Do NOT change function signatures.
"""
import numpy as np
import pandas as pd


def count_stocks_per_month(df: pd.DataFrame) -> np.ndarray:
    """
    Task 2a  [3 pts]

    Counts how many distinct stocks appear in each month. This gives
    a picture of how the breadth of the cross-section evolves over
    time.

    Input:  The filtered DataFrame.
    Output: A numpy array of shape (M, 2). Column 0 holds the yyyymm
            integer; column 1 holds the count of unique id values.
            Rows are sorted ascending by date.
    """
    raise NotImplementedError


def mean_return_by_month(df: pd.DataFrame) -> np.ndarray:
    """
    Task 2b  [3 pts]

    Computes the average excess return across all stocks in each
    month, providing a snapshot of overall market conditions over
    time.

    Input:  The filtered DataFrame.
    Output: A numpy array of shape (M, 2). Column 0 is yyyymm;
            column 1 is the mean of 'r_1'. Sorted ascending by date.
    """
    raise NotImplementedError


def return_std_by_month(df: pd.DataFrame) -> np.ndarray:
    """
    Task 2c  [3 pts]

    Computes the cross-sectional standard deviation of excess returns
    in each month. This measures how dispersed stock returns are
    within a given month — a high value means stocks had very
    different outcomes that month.

    Input:  The filtered DataFrame.
    Output: A numpy array of shape (M, 2). Column 0 is yyyymm;
            column 1 is the standard deviation (ddof=0) of 'r_1'
            across all stocks in that month. Sorted ascending by date.
    """
    raise NotImplementedError


def feature_target_correlation(df: pd.DataFrame,
                               features: list,
                               target: str = "r_1") -> np.ndarray:
    """
    Task 2d  [3 pts]

    Computes how linearly related each feature is to the target
    across the entire dataset, giving a first indication of which
    signals might be useful predictors.

    Input:  The filtered DataFrame, the list of feature names, and
            the target column name.
    Output: A numpy array of shape (P,) holding the Pearson
            correlation of each feature with 'r_1'. The ordering must
            match the input features list.
    """
    raise NotImplementedError
