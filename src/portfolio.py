"""
Task 7 — Managed Portfolio Construction and Evaluation.
Implement every function below. Do NOT change function signatures.
"""
import numpy as np
import pandas as pd


def compute_managed_returns(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    """
    Task 7a  [11 pts]

    Converts test-set predictions into a monthly portfolio return
    series. For each month the managed return is the cross-sectional
    average of the element-wise product of predicted and actual
    returns (see Background section). The predictions come from Ridge
    regression fitted with the best shrinkage selected in Task 6a.

    Input:  Predicted returns (N,), actual returns (N,), and yyyymm
            integers (N,) — all 1-D arrays over the test set.
    Output: A numpy array of shape (M, 2). Column 0 is the yyyymm
            integer; column 1 is the managed return for that month.
            Sorted ascending by date.
    """
    raise NotImplementedError


def performance_metrics(monthly_returns: np.ndarray) -> dict:
    """
    Task 7b  [9 pts]

    Summarises a monthly return series with three standard statistics.

    Input:  The (M, 2) array from compute_managed_returns.
    Output: A Python dict with three keys:
              'mean'   — average monthly return, annualised (* 12)
              'vol'    — std of monthly returns, annualised (* sqrt(12))
              'sharpe' — Sharpe Ratio (annualised mean / annualised vol)
    """
    raise NotImplementedError
