"""
Nuisance learners for Double ML.

DML fits two auxiliary ("nuisance") models, then regresses residuals on residuals:
    model_y : E[Y | X]  -- predicts stress_level from covariates X
    model_t : E[T | X]  -- predicts the treatment T from covariates X

Each factory below returns a fresh, unfitted scikit-learn-compatible regressor,
so the same function can serve as model_y or model_t (EconML clones it).

CPU only. All randomness is seeded.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "StressLevelDataset.csv"


def make_mlp(seed: int = SEED):
    """
    Neural-net nuisance learner: StandardScaler -> MLPRegressor.

    The scaler is part of the returned estimator because MLPs train badly on
    unscaled inputs (our columns range from 0-5 up to 0-30). Wrapping it in a
    Pipeline keeps it a single fit/predict object, so EconML needs no extras.
    """
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),  # keyword: not positional in recent sklearn
            alpha=1e-2,                   # L2 penalty; n is only 1,100
            max_iter=1000,
            early_stopping=True,
            random_state=seed,
        ),
    )


def make_lgbm(seed: int = SEED):
    """
    Gradient-boosting nuisance learner. Small, shallow trees: with 1,100 rows a
    big model just memorises noise, and DML needs *out-of-fold* predictions
    that generalise.
    """
    return LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=8,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=1,        # single thread -> deterministic results
        verbose=-1,
    )


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    from A4.python.camille.dag import OUTCOME, covariates_for  # run as: python src/nuisance.py

    treatment = "study_load"
    df = load_data()
    X = df[covariates_for(treatment)]
    print(f"Treatment: {treatment}   n={len(df)}   X = {list(X.columns)}")

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    targets = {"model_y": df[OUTCOME], "model_t": df[treatment]}

    for role, y in targets.items():
        print(f"\n{role}: predicting {y.name}")
        for name, factory in [("MLP", make_mlp), ("LightGBM", make_lgbm)]:
            r2 = cross_val_score(factory(), X, y, cv=cv, scoring="r2")
            print(f"  {name:<9} 5-fold CV R2 = {r2.mean():+.3f}  (sd {r2.std():.3f})")
