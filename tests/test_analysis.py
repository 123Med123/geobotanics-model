"""Synthetic-data checks: the analysis must find a planted effect, and must NOT claim
transfer when the effect reverses between belts."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import analysis
from src.extract import extract_patches
from rasterio.transform import from_origin

SEASONS = {date(2024, 4, 1): "pre_monsoon", date(2024, 8, 1): "monsoon", date(2024, 11, 1): "post_monsoon"}


def test_sign_test_exact_values():
    assert analysis.sign_test([-1] * 6)["p_two_sided"] == pytest.approx(2 / 64)
    assert analysis.sign_test([-1] * 5 + [1])["p_two_sided"] == pytest.approx(2 * 7 / 64)
    assert analysis.sign_test([])["n"] == 0


def _synthetic_patches(effect_by_belt, n=150, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for b_i, (belt, effect) in enumerate(effect_by_belt.items()):
        base = 0.4 + 0.3 * b_i  # belts differ in overall level (e.g. scrub vs forest)
        for role, label in (("mineralised", 1), ("control", 0)):
            for k in range(n):
                x, y = (k % 15) * 200 + (50000 if label == 0 else 0), (k // 15) * 200
                for d, s in SEASONS.items():
                    v = base + (effect if label else 0.0) + rng.normal(0, 0.03)
                    rows.append({"belt": belt, "zone_id": belt + ("" if label else "_control"), "role": role,
                                 "label": label, "date": d, "x": x, "y": y, "ndvi": v, "ndre": v / 2})
    return pd.DataFrame(rows)


def test_transfer_succeeds_when_effect_is_shared():
    p = _synthetic_patches({"a": -0.1, "b": -0.1})
    feat = analysis.patch_features(p, SEASONS, ["ndvi", "ndre"])
    res = analysis.patch_classifier(feat, block_m=1000, cv_folds=3)
    t = res[(res.test == "transfer a -> b") & (res.features == "per_belt_standardised")]
    assert (t.balanced_accuracy > 0.9).all()
    d = analysis.direction_agreement(feat)
    assert d.same_direction_all_belts.all()


def test_transfer_fails_when_effect_reverses():
    p = _synthetic_patches({"a": -0.1, "b": +0.1})
    feat = analysis.patch_features(p, SEASONS, ["ndvi", "ndre"])
    res = analysis.patch_classifier(feat, block_m=1000, cv_folds=3)
    within = res[res.test.str.startswith("within")]
    transfer = res[res.test.str.startswith("transfer") & (res.features == "per_belt_standardised")]
    assert (within.balanced_accuracy > 0.9).all()      # each belt separates on its own...
    assert (transfer.balanced_accuracy < 0.2).all()     # ...but the rule inverts across belts
    assert not analysis.direction_agreement(feat).same_direction_all_belts.any()


def test_pair_differences_and_summary():
    rows = []
    for d, diff in zip(SEASONS, (-0.1, -0.05, -0.2)):
        for role, off in (("mineralised", diff), ("control", 0.0)):
            for index in ("ndvi", "ndmi"):
                rows.append({"belt": "a", "role": role, "date": d, "index": index,
                             "mean": 0.5 + off, "median": 0.5 + off, "var": 0.01})
    diffs = analysis.pair_differences(pd.DataFrame(rows))
    np.testing.assert_allclose(sorted(diffs[diffs["index"] == "ndvi"].diff_median), [-0.2, -0.1, -0.05])
    s = analysis.summarise_differences(diffs).set_index("index")
    assert s.loc["ndvi", "n_mineralised_lower"] == 3
    # NDMI moved identically -> the NDVI difference is perfectly explained by moisture.
    assert s.loc["ndvi", "corr_with_ndmi_diff"] == pytest.approx(1.0)


def test_patch_grid_anchored_to_map_coordinates():
    # Two rasters of the same area whose origins differ by 3 pixels must give the same patch ids.
    idx = {"ndvi": np.ones((50, 50))}
    valid = np.ones((50, 50), dtype=bool)
    a = extract_patches(idx, valid, from_origin(1000, 2000, 10, 10), size=10, min_valid=0.5)
    b = extract_patches(idx, valid, from_origin(970, 2030, 10, 10), size=10, min_valid=0.5)
    common = set(zip(a.x, a.y)) & set(zip(b.x, b.y))
    assert len(common) >= 16
    assert all((x - 100) % 100 == 0 or (x - 50) % 100 == 0 for x, _ in common)
