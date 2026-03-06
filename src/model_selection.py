"""
Task 6 — Model Selection via Walk-Forward Cross-Validation.
Implement every function below. Do NOT change function signatures.
This is the ONLY file where for/while loops are allowed (for iterating
over cross-validation folds).
"""
import numpy as np
from src.regression import ridge_regression, lasso_regression


def oos_r_squared(y_actual: np.ndarray,
                  y_predicted: np.ndarray) -> float:
    """
    Task 6 helper  [2 pts]

    Computes the out-of-sample R-squared as defined in the Background
    section. This helper is used inside the cross-validation routines
    below and is also tested independently by the autograder.

    Input:  Two 1-D arrays of the same length.
    Output: A single float.
    """
    raise NotImplementedError


def select_ridge_shrinkage(
    S_train: np.ndarray,
    y_train: np.ndarray,
    dates_train: np.ndarray,
    shrinkage_list: list,
) -> tuple:
    """
    Task 6a  [15 pts]

    Selects the best Ridge shrinkage value using walk-forward
    (expanding-window) cross-validation with K=5 folds. In each fold
    the training window starts at the first month and grows; the
    validation window immediately follows. Concretely, let
    m_1, ..., m_N be the sorted unique months. For fold k (k=1...5):
        CV-train:      m_1  through  m_{floor(N*k/6)}
        CV-validation: m_{floor(N*k/6)+1}  through  m_{floor(N*(k+1)/6)}
    Inside each fold, call ridge_regression with the full shrinkage
    grid so that all candidates are evaluated from a single
    eigendecomposition. After all five folds, average the R-squared
    scores and pick the shrinkage with the highest average.

    Input:  Training features (T, P), training target (T,), yyyymm
            dates (T,), and the shrinkage grid (list of K floats,
            defined in config.py).
    Output: A tuple (best_shrinkage, cv_scores) where best_shrinkage
            is a float and cv_scores is a numpy array of shape (K,).
    """
    raise NotImplementedError


def select_lasso_alpha(
    S_train: np.ndarray,
    y_train: np.ndarray,
    dates_train: np.ndarray,
    alpha_list: list,
) -> tuple:
    """
    Task 6b  [8 pts]

    Identical fold structure to select_ridge_shrinkage, but for
    Lasso. Inside each fold, iterate over alpha_list and call
    lasso_regression for each candidate alpha.

    Input:  Same structure as select_ridge_shrinkage, but with an
            alpha grid (defined in config.py).
    Output: A tuple (best_alpha, cv_scores) with the same format as
            select_ridge_shrinkage.
    """
    raise NotImplementedError
