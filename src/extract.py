"""Turn downloaded scenes into numbers: per-zone-per-date statistics and per-patch samples."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import indices as ix
from .masking import build_analysis_mask

REFLECTANCE_BANDS = ("B02", "B04", "B05", "B06", "B07", "B08", "B11")


class ReflectanceScaleError(RuntimeError):
    pass


@dataclass
class Scene:
    bands: dict[str, np.ndarray]   # reflectance, NaN = nodata
    scl: np.ndarray
    transform: object
    epsg: int


def read_scene(path: Path, band_order: Sequence[str], scale: float, offset: float, nodata_dn) -> Scene:
    import rasterio

    with rasterio.open(path) as src:
        if src.count != len(band_order):
            raise ValueError(f"{path}: {src.count} bands in file, manifest lists {len(band_order)}")
        # Prefer band descriptions written by the backend; fall back to manifest order.
        names = [d if d else band_order[i] for i, d in enumerate(src.descriptions)]
        if set(names) != set(band_order):
            names = list(band_order)
        data = {name: src.read(i + 1) for i, name in enumerate(names)}
        epsg = src.crs.to_epsg() if src.crs else None
        transform = src.transform
    scl = data.pop("SCL").astype(np.int16)
    refl = {b: ix.dn_to_reflectance(data[b], scale=scale, offset=offset, nodata=nodata_dn) for b in REFLECTANCE_BANDS}
    return Scene(bands=refl, scl=scl, transform=transform, epsg=epsg)


def check_reflectance(scene: Scene, where: np.ndarray, label: str) -> None:
    """Stop on reflectance that is implausible for vegetation (usually a wrong L2A offset).

    A 1000-DN offset error shifts every band by 0.1, which silently corrupts every
    ratio index. The two failure directions look different:
      * offset NOT applied when it should be -> nothing in the scene is dark: the 1st
        percentile of blue (B02) over all pixels stays around 0.1 or higher, where real
        dark targets (shadow, water, dense canopy) sit well below 0.05.
      * offset applied TWICE -> many negative reflectances.
    Plus a loose plausibility check on vegetated NIR (B08 roughly 0.1-0.7).
    """
    blue_all = scene.bands["B02"][np.isfinite(scene.bands["B02"])]
    if blue_all.size < 100 or where.sum() < 50:
        return
    blue_p01 = float(np.percentile(blue_all, 1))
    nir = float(np.nanmedian(scene.bands["B08"][where]))
    neg = float(np.mean([(scene.bands[b][where] < 0).mean() for b in REFLECTANCE_BANDS]))
    if blue_p01 > 0.08 or neg > 0.01 or not (0.1 <= nir <= 0.7):
        raise ReflectanceScaleError(
            f"{label}: implausible reflectance (B02 1st percentile={blue_p01:.3f}, vegetated median "
            f"B08={nir:.3f}, negative fraction={neg:.3f}). Check reflectance.offset in "
            "config/pipeline.yaml (L2A baseline >= 04.00 uses BOA_ADD_OFFSET=-1000)."
        )


def _summ(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {k: np.nan for k in ("mean", "median", "var", "std", "p25", "p75")} | {"n": 0}
    return {
        "mean": float(v.mean()), "median": float(np.median(v)), "var": float(v.var(ddof=1)) if v.size > 1 else np.nan,
        "std": float(v.std(ddof=1)) if v.size > 1 else np.nan,
        "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)), "n": int(v.size),
    }


def process_scene(zone, d, scene: Scene, footprints, masking_cfg: dict, hmssi_eps: float, patch_size: int,
                  min_patch_valid: float) -> tuple[list[dict], pd.DataFrame, dict]:
    """Return (zone-date stats rows, patch table, mask summary) for one zone on one date."""
    idx = ix.compute_all(scene.bands, hmssi_psri_eps=hmssi_eps)
    mask = build_analysis_mask(
        zone.geometry, scene.scl, idx["ndvi"], scene.bands, footprints, scene.transform,
        scl_keep=masking_cfg["scl_keep"], ndvi_floor=masking_cfg["ndvi_floor"],
        mine_buffer_m=masking_cfg["mine_buffer_m"],
    )
    check_reflectance(scene, mask.valid, f"{zone.zone_id} {d}")

    base = {"belt": zone.belt, "zone_id": zone.zone_id, "role": zone.role, "label": zone.label, "date": d}
    rows = []
    for name, arr in idx.items():
        s = _summ(arr[mask.valid])
        nan_frac = float(np.mean(~np.isfinite(arr[mask.valid]))) if mask.n_valid else np.nan
        rows.append(base | {"index": name, "nan_fraction": nan_frac} | s)

    patches = extract_patches(idx, mask.valid, scene.transform, patch_size, min_patch_valid)
    for k, v in base.items():
        patches[k] = v
    return rows, patches, base | mask.as_dict()


def extract_patches(idx: dict[str, np.ndarray], valid: np.ndarray, transform, size: int,
                    min_valid: float) -> pd.DataFrame:
    """Median of each index over non-overlapping size x size pixel patches.

    Patch edges are anchored to absolute map coordinates (multiples of size x pixel
    size), not to the array's corner. The patch id is the centre's map coordinate, so
    the same patch gets the same id on every date even if a download's extent shifts
    by a few pixels.
    """
    h, w = valid.shape
    res_x, res_y = abs(transform.a), abs(transform.e)
    x0, y0 = transform.c, transform.f  # upper-left corner (north-up raster)
    px, py = size * res_x, size * res_y
    c_start = int(round(((-x0) % px) / res_x)) % size
    r_start = int(round((y0 % py) / res_y)) % size
    out = []
    for r0 in range(r_start, h - size + 1, size):
        for c0 in range(c_start, w - size + 1, size):
            m = valid[r0:r0 + size, c0:c0 + size]
            frac = m.mean()
            if frac < min_valid:
                continue
            x, y = transform * (c0 + size / 2.0, r0 + size / 2.0)
            row = {"x": round(float(x)), "y": round(float(y)), "patch_valid_fraction": float(frac)}
            for name, arr in idx.items():
                vals = arr[r0:r0 + size, c0:c0 + size][m]
                vals = vals[np.isfinite(vals)]
                row[name] = float(np.median(vals)) if vals.size else np.nan
            out.append(row)
    return pd.DataFrame(out)
