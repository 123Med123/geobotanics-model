"""Vegetation index band math for Sentinel-2 L2A surface reflectance.

All functions take reflectance arrays (floats, typically 0..1) and return float64
arrays of the same shape. A pixel whose denominator is zero or near zero, or whose
input is non-finite, comes back as NaN rather than inf or a huge number, so
downstream means don't silently blow up.

Sentinel-2 band centres used here (S2A, nm): B02 492, B04 665, B05 704, B06 740,
B07 783, B08 833, B11 1614.

There are TWO red-edge chlorophyll indices here and they are NOT interchangeable:

* ``ci_rededge_b8``  = B08/B05 - 1   the common Sentinel-2 variant (NIR 833 nm)
* ``ci_rededge_b7``  = B07/B05 - 1   Zhang et al. 2018, Eq. 1: (R783/R705) - 1

They differ in the numerator band (B08 vs B07). HMSSI is defined on the B07 version.
Do not "simplify" one into the other.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

# Denominators with absolute value at or below this are treated as zero.
DEFAULT_EPS = 1e-6


def _as_float(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def safe_ratio(num, den, eps: float = DEFAULT_EPS) -> np.ndarray:
    """num / den, NaN where |den| <= eps or either input is non-finite."""
    num = _as_float(num)
    den = _as_float(den)
    num, den = np.broadcast_arrays(num, den)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > eps)
    out[ok] = num[ok] / den[ok]
    return out


def normalized_difference(a, b, eps: float = DEFAULT_EPS) -> np.ndarray:
    """(a - b) / (a + b)."""
    a = _as_float(a)
    b = _as_float(b)
    return safe_ratio(a - b, a + b, eps)


def dn_to_reflectance(dn, scale: float = 10000.0, offset: float = 0.0, nodata=0) -> np.ndarray:
    """Convert L2A digital numbers to reflectance: (DN + offset) / scale.

    ``nodata`` DN values become NaN. For processing baseline >= 04.00 products the
    raw offset is -1000 (see config/pipeline.yaml for why the default here is 0).
    """
    dn = _as_float(dn)
    out = (dn + offset) / scale
    if nodata is not None:
        out = np.where(dn == nodata, np.nan, out)
    return out


# --- indices ---------------------------------------------------------------------------

def ndvi(b08, b04) -> np.ndarray:
    """NDVI = (B8 - B4) / (B8 + B4)."""
    return normalized_difference(b08, b04)


def ndre(b08, b05) -> np.ndarray:
    """Red-edge NDVI = (B8 - B5) / (B8 + B5)."""
    return normalized_difference(b08, b05)


def ci_rededge_b8(b08, b05) -> np.ndarray:
    """Chlorophyll index red-edge, B8 variant: B8 / B5 - 1.

    NOT the Zhang et al. 2018 definition - that uses B7 (see ``ci_rededge_b7``).
    """
    return safe_ratio(b08, b05) - 1.0


def ci_rededge_b7(b07, b05) -> np.ndarray:
    """Chlorophyll index red-edge as in Zhang et al. 2018 (Sensors 18(7):2172), Eq. 1.

    CI_red-edge = (R783 / R705) - 1  ->  Sentinel-2: B7 / B5 - 1.
    Uses B7, not B8 - keep this distinct from ``ci_rededge_b8``.
    """
    return safe_ratio(b07, b05) - 1.0


def psri(b04, b02, b06) -> np.ndarray:
    """Plant senescence reflectance index, Zhang et al. 2018 Eq. 2.

    PSRI = (R680 - R500) / R750  ->  Sentinel-2: (B4 - B2) / B6.
    """
    b04 = _as_float(b04)
    b02 = _as_float(b02)
    return safe_ratio(b04 - b02, b06)


def hmssi(b07, b05, b04, b02, b06, psri_eps: float = 0.01) -> np.ndarray:
    """Heavy metal stress sensitive index, Zhang et al. 2018 (Sensors 18(7):2172), Eq. 3.

    HMSSI = CI_red-edge / PSRI, with CI_red-edge = B7/B5 - 1 (Eq. 1, the B7 variant)
    and PSRI = (B4 - B2)/B6 (Eq. 2).

    Caveat: for healthy green vegetation B4 and B2 are both small and similar, so
    PSRI sits near zero and can change sign. The ratio then explodes or flips sign
    for reasons unrelated to plant stress. Pixels with |PSRI| <= ``psri_eps`` are
    returned as NaN; summarise HMSSI with medians, not means, and report the fraction
    of NaN pixels (the pipeline does both). The paper developed HMSSI on rice paddies;
    its behaviour on semi-arid scrub and forest canopy is exactly what Phase 1 tests.
    """
    ci = ci_rededge_b7(b07, b05)
    p = psri(b04, b02, b06)
    return safe_ratio(ci, p, eps=psri_eps)


def ndmi(b08, b11) -> np.ndarray:
    """Normalised difference moisture index = (B8 - B11) / (B8 + B11).

    Not a metal-stress index. It is a water-stress diagnostic: drought depresses NDVI
    and red-edge indices the same way metal stress does, so a zone-vs-control
    difference that tracks NDMI is more plausibly moisture than mineralisation.
    """
    return normalized_difference(b08, b11)


# Names used throughout the pipeline, results and plots.
INDEX_NAMES = ("ndvi", "ndre", "ci_rededge_b8", "ci_rededge_b7", "psri", "hmssi", "ndmi")


def compute_all(bands: Mapping[str, np.ndarray], hmssi_psri_eps: float = 0.01) -> dict[str, np.ndarray]:
    """Compute every index from a dict of reflectance arrays keyed B02, B04, ... B11."""
    b = bands
    return {
        "ndvi": ndvi(b["B08"], b["B04"]),
        "ndre": ndre(b["B08"], b["B05"]),
        "ci_rededge_b8": ci_rededge_b8(b["B08"], b["B05"]),
        "ci_rededge_b7": ci_rededge_b7(b["B07"], b["B05"]),
        "psri": psri(b["B04"], b["B02"], b["B06"]),
        "hmssi": hmssi(b["B07"], b["B05"], b["B04"], b["B02"], b["B06"], psri_eps=hmssi_psri_eps),
        "ndmi": ndmi(b["B08"], b["B11"]),
    }
