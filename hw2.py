"""
Assignment 2 Pipeline Orchestrator — DO NOT MODIFY THIS FILE.
"""
import numpy as np, json, os
from config import (N_TRAIN, SHRINKAGE_LIST, ALPHA_LASSO, DATA_PATH,
                    TARGET_COL, START_DATE)
from src.data_loader import load_data, filter_data, get_feature_columns, train_test_split
from src.eda import (count_stocks_per_month, mean_return_by_month,
                     return_std_by_month, feature_target_correlation)
from src.regression import ridge_regression, lasso_regression
from src.model_selection import (oos_r_squared, select_ridge_shrinkage,
                                 select_lasso_alpha)
from src.portfolio import compute_managed_returns, performance_metrics


def main():
    print(f"N_TRAIN = {N_TRAIN}")
    res = {}

    print("\n[Task 1] Loading data ...")
    df = load_data(DATA_PATH)
    df = filter_data(df, START_DATE)
    features = get_feature_columns(df)
    print(f"  Shape: {df.shape},  #features: {len(features)}")
    res["n_features"] = len(features)

    print("\n[Task 2] EDA ...")
    spm = count_stocks_per_month(df)
    mrm = mean_return_by_month(df)
    rsm = return_std_by_month(df)
    ftc = feature_target_correlation(df, features)
    print(f"  Months: {len(spm)}")
    res["stocks_per_month"] = spm.tolist()
    res["returns_by_month"] = mrm.tolist()
    res["return_std_by_month"] = rsm.tolist()
    res["feat_corr"] = ftc.tolist()

    print("\n[Task 3] Train / test split ...")
    S_tr, y_tr, S_te, y_te, d_tr, d_te = train_test_split(
        df, features, TARGET_COL, N_TRAIN)
    print(f"  S_train {S_tr.shape}   S_test {S_te.shape}")
    res["train_shape"] = list(S_tr.shape)
    res["test_shape"]  = list(S_te.shape)

    print("\n[Task 4] Ridge regression (full grid) ...")
    betas_ridge, preds_ridge = ridge_regression(S_tr, y_tr, S_te, SHRINKAGE_LIST)
    r2_ridge = np.array([oos_r_squared(y_te, preds_ridge[:, k])
                         for k in range(len(SHRINKAGE_LIST))])
    print(f"  R2 range: [{r2_ridge.min():.6f}, {r2_ridge.max():.6f}]")
    res["r2_ridge_grid"] = r2_ridge.tolist()

    print("\n[Task 5] Lasso regression (sample alpha) ...")
    mid = len(ALPHA_LASSO) // 2
    beta_l, pred_l = lasso_regression(S_tr, y_tr, S_te, ALPHA_LASSO[mid])
    r2_l = oos_r_squared(y_te, pred_l)
    print(f"  Alpha={ALPHA_LASSO[mid]:.6g}  R2={r2_l:.6f}")
    res["lasso_sample_r2"] = r2_l

    print("\n[Task 6a] CV — Ridge ...")
    best_z, cv_ridge = select_ridge_shrinkage(S_tr, y_tr, d_tr, SHRINKAGE_LIST)
    print(f"  Best z = {best_z:.6g},  CV R2 = {cv_ridge.max():.6f}")
    res["best_shrinkage"] = best_z
    res["cv_ridge"] = cv_ridge.tolist()

    print("[Task 6b] CV — Lasso ...")
    best_a, cv_lasso = select_lasso_alpha(S_tr, y_tr, d_tr, ALPHA_LASSO)
    print(f"  Best alpha = {best_a:.6g},  CV R2 = {cv_lasso.max():.6f}")
    res["best_alpha"] = best_a
    res["cv_lasso"] = cv_lasso.tolist()

    best_idx = SHRINKAGE_LIST.index(best_z)
    y_pred_best = preds_ridge[:, best_idx]
    oos_r2 = oos_r_squared(y_te, y_pred_best)
    print(f"\n  Final Ridge OOS R2 = {oos_r2:.6f}")
    res["oos_r2"] = oos_r2

    print("\n[Task 7] Managed portfolio ...")
    mret = compute_managed_returns(y_pred_best, y_te, d_te)
    perf = performance_metrics(mret)
    print(f"  Sharpe = {perf['sharpe']:.4f}")
    res["managed_returns"] = mret.tolist()
    res["managed_perf"]    = perf

    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
