"""
Local sanity-check tests for Assignment 2.
Run with:   python -m pytest tests/test_local.py -v
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
from config import (N_TRAIN, SHRINKAGE_LIST, ALPHA_LASSO,
                    DATA_PATH, TARGET_COL, START_DATE)


@pytest.fixture(scope="module")
def raw_df():
    from src.data_loader import load_data
    return load_data(DATA_PATH)

@pytest.fixture(scope="module")
def filtered_df(raw_df):
    from src.data_loader import filter_data
    return filter_data(raw_df, START_DATE)

@pytest.fixture(scope="module")
def features(filtered_df):
    from src.data_loader import get_feature_columns
    return get_feature_columns(filtered_df)

@pytest.fixture(scope="module")
def split_data(filtered_df, features):
    from src.data_loader import train_test_split
    return train_test_split(filtered_df, features, TARGET_COL, N_TRAIN)


class TestTask1:
    def test_load_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_date_is_datetime(self, raw_df):
        assert np.issubdtype(raw_df["date"].dtype, np.datetime64)

    def test_yyyymm_exists_and_format(self, raw_df):
        assert "yyyymm" in raw_df.columns
        assert 190000 < raw_df["yyyymm"].iloc[0] < 210000

    def test_filter_reduces_rows(self, raw_df, filtered_df):
        assert len(filtered_df) < len(raw_df)

    def test_filter_index_reset(self, filtered_df):
        assert filtered_df.index[0] == 0

    def test_features_is_sorted_list(self, features):
        assert isinstance(features, list)
        assert features == sorted(features)

    def test_target_not_in_features(self, features):
        assert TARGET_COL not in features

    def test_non_feature_excluded(self, features):
        for col in ["date", "size_grp", "id", "yyyymm", "r_1"]:
            assert col not in features, f"'{col}' should not be in features"


class TestTask2:
    def test_count_stocks_shape(self, filtered_df):
        from src.eda import count_stocks_per_month
        r = count_stocks_per_month(filtered_df)
        assert isinstance(r, np.ndarray) and r.ndim == 2 and r.shape[1] == 2

    def test_count_stocks_sorted(self, filtered_df):
        from src.eda import count_stocks_per_month
        r = count_stocks_per_month(filtered_df)
        assert np.all(r[:-1, 0] <= r[1:, 0])

    def test_mean_return_shape(self, filtered_df):
        from src.eda import mean_return_by_month
        r = mean_return_by_month(filtered_df)
        assert r.ndim == 2 and r.shape[1] == 2

    def test_return_std_shape(self, filtered_df):
        from src.eda import return_std_by_month
        r = return_std_by_month(filtered_df)
        assert isinstance(r, np.ndarray) and r.ndim == 2 and r.shape[1] == 2

    def test_return_std_sorted(self, filtered_df):
        from src.eda import return_std_by_month
        r = return_std_by_month(filtered_df)
        assert np.all(r[:-1, 0] <= r[1:, 0])

    def test_return_std_positive(self, filtered_df):
        from src.eda import return_std_by_month
        r = return_std_by_month(filtered_df)
        assert np.all(r[:, 1] >= 0)

    def test_feature_corr_shape(self, filtered_df, features):
        from src.eda import feature_target_correlation
        r = feature_target_correlation(filtered_df, features)
        assert isinstance(r, np.ndarray)
        assert r.shape == (len(features),)

    def test_feature_corr_range(self, filtered_df, features):
        from src.eda import feature_target_correlation
        r = feature_target_correlation(filtered_df, features)
        assert np.all(np.abs(r) <= 1.0 + 1e-10)


class TestTask3:
    def test_returns_six_elements(self, split_data):
        assert len(split_data) == 6

    def test_S_train_dtype(self, split_data):
        assert split_data[0].dtype == np.float64

    def test_S_test_dtype(self, split_data):
        assert split_data[2].dtype == np.float64

    def test_y_train_1d(self, split_data):
        assert split_data[1].ndim == 1

    def test_feature_dim_matches(self, split_data, features):
        assert split_data[0].shape[1] == len(features)
        assert split_data[2].shape[1] == len(features)

    def test_dates_match_rows(self, split_data):
        assert len(split_data[4]) == split_data[0].shape[0]
        assert len(split_data[5]) == split_data[2].shape[0]


class TestTask4:
    def test_ridge_shapes(self, split_data):
        from src.regression import ridge_regression
        zs = [0.1, 1.0, 10.0]
        betas, preds = ridge_regression(split_data[0], split_data[1], split_data[2], zs)
        assert betas.shape == (split_data[0].shape[1], len(zs))
        assert preds.shape == (split_data[2].shape[0], len(zs))

    def test_ridge_no_nan(self, split_data):
        from src.regression import ridge_regression
        b, p = ridge_regression(split_data[0], split_data[1], split_data[2], [1.0])
        assert not np.any(np.isnan(b)) and not np.any(np.isnan(p))

    def test_ridge_p_ge_t(self):
        from src.regression import ridge_regression
        np.random.seed(42)
        b, p = ridge_regression(np.random.randn(50, 80), np.random.randn(50),
                                np.random.randn(10, 80), [1.0, 10.0])
        assert b.shape == (80, 2) and p.shape == (10, 2)


class TestTask5:
    def test_lasso_shapes(self, split_data):
        from src.regression import lasso_regression
        beta, pred = lasso_regression(split_data[0], split_data[1], split_data[2], 0.001)
        assert beta.shape == (split_data[0].shape[1],)
        assert pred.shape == (split_data[2].shape[0],)

    def test_lasso_deterministic(self, split_data):
        from src.regression import lasso_regression
        _, p1 = lasso_regression(split_data[0], split_data[1], split_data[2], 0.001)
        _, p2 = lasso_regression(split_data[0], split_data[1], split_data[2], 0.001)
        assert np.allclose(p1, p2)


class TestTask6:
    def test_oos_r2_perfect(self):
        from src.model_selection import oos_r_squared
        y = np.array([1.0, 2.0, 3.0])
        assert np.isclose(oos_r_squared(y, y), 1.0)

    def test_oos_r2_zero(self):
        from src.model_selection import oos_r_squared
        y = np.array([1.0, 2.0, 3.0])
        assert np.isclose(oos_r_squared(y, np.zeros(3)), 0.0)

    def test_ridge_cv_shapes(self, split_data):
        from src.model_selection import select_ridge_shrinkage
        best, scores = select_ridge_shrinkage(
            split_data[0], split_data[1], split_data[4], SHRINKAGE_LIST[:5])
        assert isinstance(best, float)
        assert scores.shape == (5,)

    def test_lasso_cv_shapes(self, split_data):
        from src.model_selection import select_lasso_alpha
        best, scores = select_lasso_alpha(
            split_data[0], split_data[1], split_data[4], ALPHA_LASSO[:3])
        assert isinstance(best, float)
        assert scores.shape == (3,)


class TestTask7:
    def test_managed_returns_shape(self, split_data):
        from src.portfolio import compute_managed_returns
        y_fake = np.random.randn(split_data[2].shape[0])
        result = compute_managed_returns(y_fake, y_fake, split_data[5])
        M = len(np.unique(split_data[5]))
        assert result.shape == (M, 2)

    def test_managed_returns_sorted(self, split_data):
        from src.portfolio import compute_managed_returns
        y_fake = np.random.randn(split_data[2].shape[0])
        result = compute_managed_returns(y_fake, y_fake, split_data[5])
        assert np.all(result[:-1, 0] <= result[1:, 0])

    def test_performance_keys(self, split_data):
        from src.portfolio import compute_managed_returns, performance_metrics
        y_fake = np.random.randn(split_data[2].shape[0])
        mret = compute_managed_returns(y_fake, y_fake, split_data[5])
        perf = performance_metrics(mret)
        for key in ("mean", "vol", "sharpe"):
            assert key in perf, f"Missing key '{key}'"
            assert isinstance(perf[key], float)
