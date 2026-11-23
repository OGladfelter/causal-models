"""
Q6 - Non-causal ML models used to "estimate" causal impacts.

A purely predictive model has no notion of confounding, adjustment sets,
mediators or colliders. We fit strong predictors on ALL covariates and read off
a quantity analogous to an ATE: the average predicted change in the outcome when
a treatment is raised one unit, everything else held at observed values
(a partial-dependence contrast). It LOOKS like an ATE and is not one.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import lightgbm as lgb

SEED = 42
OUTCOME = "stress_level"
TREATMENTS = ["study_load", "noise_level", "living_conditions",
              "academic_performance", "future_career_concerns"]


def make_model(name, seed=SEED):
    if name == "LightGBM":
        return lgb.LGBMRegressor(random_state=seed, verbose=-1, n_jobs=1)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    if name == "MLP":
        return make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(64, 32),
                                          max_iter=600, random_state=seed))
    raise ValueError(name)


def pd_effect(model, X, treatment, delta=1.0):
    """Average predicted change in outcome from a +1 unit shift in `treatment`."""
    Xb = X.copy()
    Xb[treatment] = Xb[treatment] + delta
    return float(np.mean(model.predict(Xb) - model.predict(X)))


def run_q6(df, models=("LightGBM", "RandomForest", "MLP"), n_boot=200, seed=SEED):
    rng = np.random.default_rng(seed)
    preds = [c for c in df.columns if c != OUTCOME]
    X, y = df[preds], df[OUTCOME]
    rows = []

    for name in models:
        # 1. predictive quality - what a non-causal model is actually good at
        r2 = cross_val_score(make_model(name), X, y,
                             cv=KFold(5, shuffle=True, random_state=seed),
                             scoring="r2").mean()

        # 2. point estimates: ONE fit, all treatments
        fitted = make_model(name).fit(X, y)
        point = {t: pd_effect(fitted, X, t) for t in TREATMENTS}

        # 3. bootstrap: ONE refit per resample, all treatments from it
        boots = {t: [] for t in TREATMENTS}
        for _ in range(n_boot):
            idx = rng.integers(0, len(df), len(df))
            Xb, yb = X.iloc[idx], y.iloc[idx]
            m = make_model(name).fit(Xb, yb)
            for t in TREATMENTS:
                boots[t].append(pd_effect(m, Xb, t))

        # 4. placebo: permute each treatment, refit, re-measure
        placebo = {}
        for t in TREATMENTS:
            Xp = X.copy()
            Xp[t] = rng.permutation(Xp[t].values)
            placebo[t] = pd_effect(make_model(name).fit(Xp, y), Xp, t)

        for t in TREATMENTS:
            lo, hi = np.percentile(boots[t], [2.5, 97.5])
            rows.append({"model": name, "cause": t,
                         "effect": round(point[t], 5),
                         "CI_low": round(lo, 5), "CI_high": round(hi, 5),
                         "sig": "yes" if lo * hi > 0 else "no",
                         "placebo_abs": round(abs(placebo[t]), 5),
                         "r2_outcome": round(r2, 5)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv("data/StressLevelDataset.csv")
    out = run_q6(df, n_boot=50)   # 50 for a fast draft (~2 min); use 200 for the final
    print(out.to_string(index=False))
    out.to_csv("outputs/q6_noncausal.csv", index=False)

