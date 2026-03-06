# Assignment 2

## Overview

In this assignment you will build a **return
prediction** pipeline from scratch. Using a panel of monthly stock
characteristics (signals) and returns, you will implement Ridge
regression via eigendecomposition, fit Lasso regression with
scikit-learn, select regularisation parameters through walk-forward
cross-validation, and evaluate predictions by constructing a managed
portfolio.

You are provided with a pickle file containing monthly observations for
U.S. stocks. Each row is one stock–month
pair with preprocessed signal columns (features) and a target column
(`r_1`) representing the next-month excess return.

> **Important:** All implementations must be **vectorised** (using
> pandas/numpy operations). Explicit Python `for`/`while` loops are
> permitted **only** in `model_selection.py` for iterating over
> cross-validation folds. Loops anywhere else will result in zero
> credit for the affected function.

## Repository Structure
```
assignment2-<username>/
├── .github/
│   └── workflows/
│       └── autograder.yml          # CI/CD (do not modify)
├── data/
│   └── data.pkl                    # Monthly stock data (provided)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # read, filter, features, split
│   ├── eda.py                      # exploratory data analysis
│   ├── regression.py               # Ridge & Lasso regression
│   ├── model_selection.py          # walk-forward cross-validation
│   └── portfolio.py                # managed returns & performance
├── tests/
│   └── test_local.py               # Local shape/dtype checks
├── config.py                       # Configuration (do not modify)
├── hw2.py                          # Main entry point (do not modify)
├── requirements.txt                # Dependencies (do not modify)
└── README.md                       # This file
```

## Getting Started
```bash
# Clone your repository
git clone <your-repo-url>
cd assignment2-<your-username>

# Install dependencies
pip install -r requirements.txt

# Run the pipeline (once you have implemented the functions)
python hw2.py
```

## Local Testing

Public tests are provided in the `tests/` directory so you can check
your work **before** pushing. These tests verify types, shapes, and
basic properties — they do **not** check numerical correctness (that
is done by the autograder on push).
```bash
# Run all public tests
pytest tests/test_local.py -v

# Run only Task 1 tests
pytest tests/test_local.py::TestTask1 -v

# Run only Task 4 (Ridge) tests
pytest tests/test_local.py::TestTask4 -v
```

If a test fails with `NotImplementedError`, it means you haven't
implemented that function yet — this is expected.

## What to Implement

Open each file under `src/` and implement the functions marked with
`raise NotImplementedError`. Every function has a detailed docstring
that specifies inputs, outputs, and the exact computation required.

| File | Function(s) | Task | Description |
|------|-------------|------|-------------|
| `src/data_loader.py` | `load_data` | 1a | Load pickle, parse dates, create `yyyymm` |
| `src/data_loader.py` | `filter_data` | 1b | Keep rows from 2000 onward, reset index |
| `src/data_loader.py` | `get_feature_columns` | 1c | Identify numeric feature columns |
| `src/data_loader.py` | `train_test_split` | 3 | Chronological train/test split by month |
| `src/eda.py` | `count_stocks_per_month` | 2a | Distinct stock count per month |
| `src/eda.py` | `mean_return_by_month` | 2b | Cross-sectional mean return per month |
| `src/eda.py` | `return_std_by_month` | 2c | Cross-sectional return dispersion per month |
| `src/eda.py` | `feature_target_correlation` | 2d | Pearson correlation of each feature with target |
| `src/regression.py` | `ridge_regression` | 4 | Eigendecomposition-based Ridge for a shrinkage grid |
| `src/regression.py` | `lasso_regression` | 5 | Lasso via scikit-learn |
| `src/model_selection.py` | `oos_r_squared` | 6 | Out-of-sample R² (helper) |
| `src/model_selection.py` | `select_ridge_shrinkage` | 6a | Walk-forward CV for Ridge |
| `src/model_selection.py` | `select_lasso_alpha` | 6b | Walk-forward CV for Lasso |
| `src/portfolio.py` | `compute_managed_returns` | 7a | Monthly managed portfolio returns |
| `src/portfolio.py` | `performance_metrics` | 7b | Annualised mean, vol, Sharpe |

## Grading

Your submission is graded **automatically**. The autograder runs via
GitHub Actions each time you push, testing each function independently
and awarding **partial credit** on a 100-point scale.

### Grading Breakdown

| Task | Component | Points |
|------|-----------|--------|
| 1a | `load_data` | 3 |
| 1b | `filter_data` | 2 |
| 1c | `get_feature_columns` | 3 |
| 2a | `count_stocks_per_month` | 3 |
| 2b | `mean_return_by_month` | 3 |
| 2c | `return_std_by_month` | 3 |
| 2d | `feature_target_correlation` | 3 |
| 3 | `train_test_split` | 5 |
| 4 | `ridge_regression` | 20 |
| 5 | `lasso_regression` | 10 |
| 6 | `oos_r_squared` + `select_ridge_shrinkage` + `select_lasso_alpha` | 25 |
| 7a | `compute_managed_returns` | 11 |
| 7b | `performance_metrics` | 9 |
| | **Total** | **100** |

> **Note:** Each student is assigned a unique `N_TRAIN` parameter
> (number of training months) derived from their GitHub repository
> name. The autograder evaluates your code using your assigned
> parameter. Make sure your functions work correctly for **any** valid
> `N_TRAIN` value in [120, 180], not just a hard-coded one.

## Rules

- Do **not** modify `config.py`, `hw2.py`, or `requirements.txt`.
- Do **not** modify or remove `data/data.pkl`.
- Do **not** modify or remove the `.github/` directory.
- Do **not** add any additional dependencies beyond what is listed in
  `requirements.txt`.
- **Allowed libraries:** `numpy`, `pandas`, `scikit-learn` (Lasso only).
- Ridge regression must be implemented **from scratch** via
  eigendecomposition (refer to the lecture notebook).
- `for`/`while` loops are permitted **only** in `model_selection.py`
  for iterating over CV folds. All other code must be fully
  vectorised.