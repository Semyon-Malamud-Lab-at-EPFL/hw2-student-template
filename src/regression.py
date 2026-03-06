"""
Tasks 4 & 5 — Ridge and Lasso Regression.
Implement every function below. Do NOT change function signatures.
"""
import numpy as np


def ridge_regression(
    S_train: np.ndarray,
    y_train: np.ndarray,
    S_test: np.ndarray,
    shrinkage_list: list,
) -> tuple:
    """
    Task 4  [20 pts]

    Solves the Ridge regression problem for an entire grid of
    shrinkage values at once. Rather than inverting a matrix
    separately for every shrinkage value, this function performs the
    eigendecomposition once and obtains all solutions cheaply through
    diagonal scaling. Refer to the lecture notebook (Experiments with
    Simulated High-Dimensional Regressions) for the derivation and
    reference implementation.

    Input:  S_train (T, P), y_train (T,), S_test (T2, P), and a
            list of K shrinkage values.
    Output: A tuple (betas, predictions) where betas has shape
            (P, K) and predictions has shape (T2, K). Column k
            corresponds to shrinkage value z_k.
    """
    raise NotImplementedError


def lasso_regression(
    S_train: np.ndarray,
    y_train: np.ndarray,
    S_test: np.ndarray,
    alpha: float,
) -> tuple:
    """
    Task 5  [10 pts]

    Fits a Lasso model using scikit-learn and returns both the
    coefficient vector and the test-set predictions. To guarantee
    that every student obtains identical, deterministic results, the
    Lasso object must be created with exactly three keyword
    arguments: fit_intercept=False, max_iter=10000, tol=1e-6. Any
    deviation will produce different coefficients and the autograder
    will flag them as incorrect.

    Input:  Training features (T, P), training target (T,), test
            features (T2, P), and a scalar regularisation strength
            alpha.
    Output: A tuple (beta, y_pred) where beta has shape (P,) and
            y_pred has shape (T2,).
    """
    raise NotImplementedError
