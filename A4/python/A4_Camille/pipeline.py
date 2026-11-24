"""
Q5 pipeline: ATE of each treatment on stress_level via Double ML.

For every treatment T we fit, with discrete_treatment=False:
    LinearDML       -- CATE is a linear function of X
    CausalForestDML -- CATE is a forest (nonparametric) function of X
using each of the two nuisance learners (MLP, LightGBM) as BOTH model_y and
model_t. X = the DAG4-derived covariates for that treatment (no descendants of T).

Notation: Y = stress_level, T = treatment, X = covariates, ATE = average
treatment effect = mean of the CATE over the sample.
"""

import warnings

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from sklearn.model_selection import KFold

from A4.python.camille.dag import OUTCOME, TREATMENTS, covariates_for
from A4.python.camille.nuisance import make_lgbm, make_mlp

NUISANCE = {"MLP": make_mlp, "LightGBM": make_lgbm}
ALPHA = 0.05  # -> 95% confidence intervals


def _make_estimator(kind: str, factory, seed: int):
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    common = dict(
        model_y=factory(seed),
        model_t=factory(seed),
        discrete_treatment=False,
        cv=cv,
        random_state=seed,
    )
    if kind == "LinearDML":
        return LinearDML(**common)
    if kind == "CausalForestDML":
        return CausalForestDML(
            n_estimators=200, min_samples_leaf=10, n_jobs=1, **common
        )
    raise ValueError(kind)


def fit_one(df: pd.DataFrame, treatment: str, kind: str, nuisance: str, seed: int):
    """Fit one (treatment, estimator, nuisance) combination; return (est, X)."""
    X = df[covariates_for(treatment)].to_numpy(dtype=float)
    est = _make_estimator(kind, NUISANCE[nuisance], seed)
    est.fit(df[OUTCOME].to_numpy(dtype=float), df[treatment].to_numpy(dtype=float), X=X)
    return est, X


def run_pipeline(df: pd.DataFrame, threshold: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    Return a summary table: one row per treatment x estimator x nuisance learner,
    with ATE, standard error and 95% CI.

    `threshold` is the cut-off for the binarised study_load_high used by the
    meta-learners (Q5 part 2). The DML fits here use the raw ordinal treatments
    (discrete_treatment=False), so it is accepted for a stable signature but unused.
    """
    np.random.seed(seed)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # econml/sklearn FutureWarnings
        for t in TREATMENTS:
            for kind in ("LinearDML", "CausalForestDML"):
                for nuis in NUISANCE:
                    est, X = fit_one(df, t, kind, nuis, seed)
                    inf = est.ate_inference(X)
                    lo, hi = inf.conf_int_mean(alpha=ALPHA)
                    rows.append(
                        {
                            "treatment": t,
                            "estimator": kind,
                            "nuisance": nuis,
                            "ate": float(np.squeeze(inf.mean_point)),
                            "se": float(np.squeeze(inf.stderr_mean)),
                            "ci_low": float(np.squeeze(lo)),
                            "ci_high": float(np.squeeze(hi)),
                        }
                    )
    out = pd.DataFrame(rows)
    out["significant"] = (out.ci_low > 0) | (out.ci_high < 0)
    return out


def plot_ates(res: pd.DataFrame):
    """Forest plot: ATE and 95% CI per treatment, one panel per estimator."""
    import matplotlib.pyplot as plt

    kinds = list(res.estimator.unique())
    fig, axes = plt.subplots(1, len(kinds), figsize=(11, 4), sharey=True, sharex=True)
    colors = {"MLP": "#1f77b4", "LightGBM": "#ff7f0e"}
    treatments = list(res.treatment.unique())
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        for j, nuis in enumerate(NUISANCE):
            d = res[(res.estimator == kind) & (res.nuisance == nuis)].set_index("treatment")
            y = np.arange(len(treatments)) + (j - 0.5) * 0.25
            d = d.loc[treatments]
            ax.errorbar(d.ate, y, xerr=[d.ate - d.ci_low, d.ci_high - d.ate],
                        fmt="o", color=colors[nuis], capsize=3, label=nuis)
        ax.axvline(0, color="grey", lw=1, ls="--")
        ax.set_yticks(range(len(treatments)), treatments)
        ax.set_title(kind)
        ax.set_xlabel("ATE on stress_level (95% CI)")
    np.atleast_1d(axes)[0].invert_yaxis()
    np.atleast_1d(axes)[-1].legend(title="nuisance learner")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import time

    from A4.python.camille.nuisance import load_data

    t0 = time.time()
    res = run_pipeline(load_data())
    pd.set_option("display.width", 160)
    print(res.round(3).to_string(index=False))
    print(f"\n{time.time() - t0:.1f}s")
