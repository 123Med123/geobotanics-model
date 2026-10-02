"""Zone-vs-control comparison, baseline classifiers and the cross-belt transferability test.

Read this before trusting any number that comes out of here:

* The unit of independent evidence is the BELT. Phase 1 has two. Patch-level sample
  sizes (thousands) are not independent observations of "mineralised vs not": every
  patch in a zone shares one label, one geology and one land-use history.
* Within-belt classification mostly learns "which of these two places is this", which
  is why the cross-belt transfer test is the result that matters.
* Accuracy is reported next to the majority-class rate, balanced accuracy and the
  fraction of test samples predicted positive (a classifier that labels everything
  "mineralised" can look fine on accuracy alone).
"""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- descriptive comparison ------------------------------------------------------------


def pair_differences(stats: pd.DataFrame) -> pd.DataFrame:
    """Mineralised minus control, per belt / date / index.

    ``stats`` is the long zone-date table from extract.process_scene (one row per
    zone, date, index). Returns median/mean differences and a standardised mean
    difference (difference / pooled SD - a within-date effect size).
    """
    cols = ["mean", "median", "var"]
    m = stats[stats.role == "mineralised"].set_index(["belt", "date", "index"])[cols]
    c = stats[stats.role == "control"].set_index(["belt", "date", "index"])[cols]
    j = m.join(c, lsuffix="_min", rsuffix="_ctl", how="inner").reset_index()
    j["diff_median"] = j.median_min - j.median_ctl
    j["diff_mean"] = j.mean_min - j.mean_ctl
    pooled = np.sqrt((j.var_min + j.var_ctl) / 2.0)
    j["std_diff"] = np.where(pooled > 0, j.diff_mean / pooled, np.nan)
    return j


def sign_test(diffs) -> dict:
    """Two-sided exact binomial sign test on the sign of per-date differences.

    With 3-6 dates this has very little power: even 6/6 same-sign dates gives
    p = 0.031, and 5/6 gives p = 0.22. It only asks "is the difference consistently in
    one direction", not "is it caused by mineralisation".
    """
    d = np.asarray([x for x in diffs if np.isfinite(x) and x != 0])
    n = int(d.size)
    k = int((d < 0).sum())
    if n == 0:
        return {"n": 0, "n_negative": 0, "p_two_sided": np.nan}
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return {"n": n, "n_negative": k, "p_two_sided": float(min(1.0, 2 * tail))}


def summarise_differences(diffs: pd.DataFrame) -> pd.DataFrame:
    """Per belt and index: how often mineralised < control, and the moisture link.

    ``corr_with_ndmi_diff`` is the across-date Pearson correlation between this
    index's difference and the NDMI difference. A strong positive value means the
    mineralised zone is lower exactly when it is also drier than its control - a water
    explanation fits at least as well as a metal one.
    """
    out = []
    for belt, g in diffs.groupby("belt"):
        nd = g[g["index"] == "ndmi"].set_index("date")["diff_median"]
        for index, gi in g.groupby("index"):
            st = sign_test(gi.diff_median)
            s = gi.set_index("date")["diff_median"]
            both = pd.concat([s, nd], axis=1, keys=["a", "b"]).dropna()
            corr = float(both.a.corr(both.b)) if len(both) >= 3 and index != "ndmi" else np.nan
            out.append({
                "belt": belt, "index": index, "n_dates": st["n"],
                "n_mineralised_lower": st["n_negative"], "sign_test_p": st["p_two_sided"],
                "median_of_diffs": float(gi.diff_median.median()),
                "median_std_diff": float(gi.std_diff.median()),
                "corr_with_ndmi_diff": corr,
            })
    return pd.DataFrame(out)


# --- classifiers -----------------------------------------------------------------------


def _models(random_state: int) -> dict:
    return {
        "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(max_iter=2000)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                              random_state=random_state, n_jobs=-1)),
    }


def _metrics(y_true, y_pred, y_score) -> dict:
    y_true = np.asarray(y_true)
    res = {
        "n_test": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "majority_rate": float(max(y_true.mean(), 1 - y_true.mean())) if y_true.size else np.nan,
        "predicted_positive_fraction": float(np.mean(y_pred)) if y_true.size else np.nan,
    }
    try:
        res["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        res["roc_auc"] = np.nan
    return res


def zone_date_features(stats: pd.DataFrame, feature_indices) -> pd.DataFrame:
    """One row per (zone, date): mean and spatial variance of each index (the brief's features)."""
    s = stats[stats["index"].isin(feature_indices)]
    wide = s.pivot_table(index=["belt", "zone_id", "label", "date"], columns="index", values=["mean", "var"])
    wide.columns = [f"{idx}_{stat}" for stat, idx in wide.columns]
    return wide.reset_index()


def zone_date_classifier(feat: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """The literal classifier from the brief, on per-zone-per-date rows.

    Within belt: leave-one-date-out (train on the other dates, test on both zones of the
    held-out date). Across belts: train on one belt, test on the other.
    There are only (2 zones x n_dates) rows per belt, so these numbers are anecdotes;
    they are reported because the brief asks for them and to contrast with the patch
    level, not as evidence.
    """
    fcols = [c for c in feat.columns if c not in ("belt", "zone_id", "label", "date")]
    rows = []
    for name, model in _models(random_state).items():
        for belt, g in feat.groupby("belt"):
            yt, yp, ys = [], [], []
            for d in sorted(g.date.unique()):
                tr, te = g[g.date != d], g[g.date == d]
                if tr.label.nunique() < 2 or te.empty:
                    continue
                m = clone(model).fit(tr[fcols], tr.label)
                yt += list(te.label); yp += list(m.predict(te[fcols])); ys += list(m.predict_proba(te[fcols])[:, 1])
            if yt:
                rows.append({"level": "zone_date", "model": name, "test": f"within {belt} (leave-one-date-out)",
                             "train_belt": belt, "test_belt": belt} | _metrics(yt, yp, ys))
        for a in feat.belt.unique():
            for b in feat.belt.unique():
                if a == b:
                    continue
                tr, te = feat[feat.belt == a], feat[feat.belt == b]
                if tr.label.nunique() < 2 or te.label.nunique() < 2:
                    continue
                m = clone(model).fit(tr[fcols], tr.label)
                rows.append({"level": "zone_date", "model": name, "test": f"transfer {a} -> {b}",
                             "train_belt": a, "test_belt": b}
                            | _metrics(te.label, m.predict(te[fcols]), m.predict_proba(te[fcols])[:, 1]))
    return pd.DataFrame(rows)


def patch_features(patches: pd.DataFrame, seasons_by_date: dict, feature_indices) -> pd.DataFrame:
    """One row per patch: median of each index per season (pooled over years).

    Seasons, not calendar dates, are the features so that two belts imaged on different
    days still share a feature space.
    """
    p = patches.copy()
    p["season"] = p["date"].map(seasons_by_date)
    p = p.dropna(subset=["season"])
    long = p.melt(id_vars=["belt", "zone_id", "label", "x", "y", "season"], value_vars=list(feature_indices),
                  var_name="index", value_name="value")
    wide = long.pivot_table(index=["belt", "zone_id", "label", "x", "y"], columns=["index", "season"],
                            values="value", aggfunc="median")
    wide.columns = [f"{i}_{s}" for i, s in wide.columns]
    return wide.reset_index()


def spatial_blocks(df: pd.DataFrame, block_m: float) -> pd.Series:
    return (df.zone_id.astype(str) + ":" + (df.x // block_m).astype(int).astype(str) + ":"
            + (df.y // block_m).astype(int).astype(str))


def _standardise_per_belt(df: pd.DataFrame, fcols) -> pd.DataFrame:
    """Z-score each feature within each belt using BOTH zones pooled (label-free).

    This removes the belt's overall climate/biome level (forest vs scrub) so the model
    can only use relative differences. It never looks at labels, so it doesn't leak.
    """
    out = df.copy()
    for belt, idx in df.groupby("belt").groups.items():
        sub = df.loc[idx, fcols]
        out.loc[idx, fcols] = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
    return out


def patch_classifier(feat: pd.DataFrame, block_m: float, cv_folds: int, random_state: int = 42) -> pd.DataFrame:
    """Within-belt spatial CV and cross-belt transfer, on patch features.

    Transfer uses only feature columns that exist (non-empty) in both belts, and is run
    on raw features and on per-belt standardised features.
    """
    base_cols = [c for c in feat.columns if c not in ("belt", "zone_id", "label", "x", "y")]
    rows = []
    models = _models(random_state)

    for belt, g in feat.groupby("belt"):
        fcols = [c for c in base_cols if g[c].notna().any()]
        groups = spatial_blocks(g, block_m)
        n_groups_min = g.groupby("label").apply(lambda x: spatial_blocks(x, block_m).nunique()).min()
        k = int(min(cv_folds, n_groups_min))
        if k < 2 or g.label.nunique() < 2:
            continue
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state)
        for name, model in models.items():
            yt, yp, ys = [], [], []
            for tr, te in cv.split(g[fcols], g.label, groups):
                m = clone(model).fit(g.iloc[tr][fcols], g.iloc[tr].label)
                yt += list(g.iloc[te].label)
                yp += list(m.predict(g.iloc[te][fcols]))
                ys += list(m.predict_proba(g.iloc[te][fcols])[:, 1])
            rows.append({"level": "patch", "model": name, "features": "raw",
                         "test": f"within {belt} (spatial-block CV, k={k})", "train_belt": belt, "test_belt": belt,
                         "n_train": int(len(g) * (k - 1) / k)} | _metrics(yt, yp, ys))

    belts = list(feat.belt.unique())
    for variant in ("raw", "per_belt_standardised"):
        data = feat if variant == "raw" else _standardise_per_belt(feat, base_cols)
        for a in belts:
            for b in belts:
                if a == b:
                    continue
                tr, te = data[data.belt == a], data[data.belt == b]
                fcols = [c for c in base_cols if tr[c].notna().any() and te[c].notna().any()]
                if not fcols or tr.label.nunique() < 2 or te.label.nunique() < 2:
                    continue
                for name, model in models.items():
                    m = clone(model).fit(tr[fcols], tr.label)
                    rows.append({"level": "patch", "model": name, "features": variant,
                                 "test": f"transfer {a} -> {b}", "train_belt": a, "test_belt": b,
                                 "n_train": int(len(tr)), "n_features": len(fcols)}
                                | _metrics(te.label, m.predict(te[fcols]), m.predict_proba(te[fcols])[:, 1]))
    return pd.DataFrame(rows)


def direction_agreement(feat: pd.DataFrame) -> pd.DataFrame:
    """For each patch feature: sign of (mineralised median - control median) in each belt.

    The simplest transferability check: if a feature is lower over the deposit in one
    belt and higher in the other, no classifier can transfer on it.
    """
    base_cols = [c for c in feat.columns if c not in ("belt", "zone_id", "label", "x", "y")]
    rows = []
    for c in base_cols:
        r = {"feature": c}
        signs = []
        for belt, g in feat.groupby("belt"):
            d = g[g.label == 1][c].median() - g[g.label == 0][c].median()
            r[f"diff_{belt}"] = float(d) if pd.notna(d) else np.nan
            if pd.notna(d) and d != 0:
                signs.append(np.sign(d))
        r["same_direction_all_belts"] = bool(len(signs) >= 2 and len(set(signs)) == 1)
        rows.append(r)
    return pd.DataFrame(rows)
