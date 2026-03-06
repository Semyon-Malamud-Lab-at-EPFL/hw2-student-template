"""
Configuration for Assignment 2.
DO NOT MODIFY THIS FILE.
"""
import hashlib, os
import numpy as np

def _get_repo_name():
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if repo:
        return repo.split("/")[-1]
    return os.path.basename(os.path.abspath(os.path.dirname(__file__)))

def _derive_n_train(repo_name, low=120, high=180):
    h = int(hashlib.sha256(repo_name.encode()).hexdigest(), 16)
    return low + (h % (high - low + 1))

REPO_NAME      = _get_repo_name()
N_TRAIN        = _derive_n_train(REPO_NAME)

SHRINKAGE_LIST = np.logspace(-4, 4, 30).tolist()
ALPHA_LASSO    = np.logspace(-6, -1, 20).tolist()

DATA_PATH      = os.path.join(os.path.dirname(__file__), "data", "data.csv")
TARGET_COL     = "r_1"
DATE_COL       = "date"
ID_COL         = "id"
SIZE_COL       = "size_grp"
YYYYMM_COL     = "yyyymm"
START_DATE     = "2000-01-31"
N_FOLDS        = 5
NON_FEATURE_COLS = {"id", "date", "size_grp", "yyyymm", "r_1"}
