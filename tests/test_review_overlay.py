import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("review_overlay_script", ROOT / "scripts/make_review_overlay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scene():
    n = 20
    bands = {b: np.full((n, n), 0.1) for b in ("B02", "B04", "B05", "B06", "B07", "B08", "B11")}
    bands["B04"][:] = 0.05
    bands["B08"][:] = 0.40                       # NDVI = 0.78 everywhere
    scl = np.full((n, n), 4, np.int16)
    scl[0:3, :] = 9                              # cloud rows
    scl[3:5, :] = 5                              # SCL not vegetation
    scl[5, :] = 4; bands["B08"][5, :] = 0.06     # SCL vegetation but NDVI below the floor
    scl[19, :] = -32768                          # off-swath row, all bands NaN
    for b in bands:
        bands[b][19, :] = np.nan
    return SimpleNamespace(bands=bands, scl=scl, transform=from_origin(0, 200, 10, 10))


def test_overlay_categories_agree_with_the_pipeline_mask():
    mod = _load()
    zone = SimpleNamespace(geometry=box(0, 0, 200, 200))
    mc = {"scl_keep": [4], "ndvi_floor": 0.2, "mine_buffer_m": 20}
    footprints = [box(0, 100, 40, 140)]          # 40 x 40 m, buffer 20 m -> 80 x 80 m (8 x 8 px)
    cat, ref = mod.categorise(zone, _scene(), footprints, mc, 0.01)
    counts = {c: int((cat == c).sum()) for c in range(7)}
    assert counts[6] == ref.n_valid                      # valid agrees with build_analysis_mask
    assert counts[2] == 3 * 20 and counts[1] == 20       # cloud rows, off-swath row
    assert counts[3] == 2 * 20 and counts[4] == 20       # SCL non-veg rows, NDVI-floor row
    assert counts[5] == ref.n_footprint and counts[5] > 0
    assert counts[0] == 0                                # zone covers the whole scene
    assert sum(counts[c] for c in range(1, 7)) == ref.n_zone
