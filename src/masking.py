"""Pixel masks: which pixels are allowed into any statistic.

A pixel is analysed only if it is
  * inside the zone polygon,
  * classified as vegetation by the Sentinel-2 Scene Classification Layer (SCL),
  * above the NDVI floor (catches sparse/bare pixels SCL lets through),
  * OUTSIDE the buffered mine footprint (pits, dumps, tailings, plant, smelter),
  * finite in every band.

The mine-footprint exclusion is the point of the whole exercise: without it the
"signal" is just "there is a mine here", not stressed vegetation over ore.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from rasterio.features import geometry_mask
from shapely.geometry.base import BaseGeometry

# Sentinel-2 L2A SCL classes, for reference.
SCL_CLASSES = {
    0: "no_data", 1: "saturated_or_defective", 2: "dark_area", 3: "cloud_shadow",
    4: "vegetation", 5: "not_vegetated", 6: "water", 7: "unclassified",
    8: "cloud_medium_prob", 9: "cloud_high_prob", 10: "thin_cirrus", 11: "snow_ice",
}
SCL_CLOUDY = (3, 8, 9, 10)


class FootprintError(RuntimeError):
    pass


def geometry_to_mask(geoms: Iterable[BaseGeometry], transform, shape) -> np.ndarray:
    """Boolean raster, True for pixels whose centre falls inside any geometry."""
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, dtype=bool)
    return geometry_mask(geoms, out_shape=shape, transform=transform, invert=True)


def scl_mask(scl: np.ndarray, keep: Iterable[int] = (4,)) -> np.ndarray:
    return np.isin(np.asarray(scl), list(keep))


def footprint_mask(footprints: Iterable[BaseGeometry], buffer_m: float, transform, shape) -> np.ndarray:
    """True inside the (buffered) mine footprint. Geometries must be in the raster CRS."""
    buffered = [g.buffer(buffer_m) for g in footprints if g is not None and not g.is_empty]
    return geometry_to_mask(buffered, transform, shape)


@dataclass
class MaskResult:
    valid: np.ndarray           # final analysis mask
    n_zone: int                 # pixels inside the zone polygon
    n_cloudy: int               # of those: cloud / shadow / cirrus
    n_nonveg: int               # of those: clear but not SCL vegetation or below NDVI floor
    n_footprint: int            # of those: vegetated but inside the mine buffer
    n_valid: int

    @property
    def valid_fraction(self) -> float:
        return self.n_valid / self.n_zone if self.n_zone else 0.0

    @property
    def clear_fraction(self) -> float:
        return 1.0 - (self.n_cloudy / self.n_zone) if self.n_zone else 0.0

    def as_dict(self) -> dict:
        return {
            "n_zone_px": self.n_zone, "n_cloudy_px": self.n_cloudy, "n_nonveg_px": self.n_nonveg,
            "n_footprint_px": self.n_footprint, "n_valid_px": self.n_valid,
            "valid_fraction": self.valid_fraction, "clear_fraction": self.clear_fraction,
        }


def build_analysis_mask(
    zone_geom: BaseGeometry,
    scl: np.ndarray,
    ndvi: np.ndarray,
    bands: Mapping[str, np.ndarray],
    footprints: Iterable[BaseGeometry],
    transform,
    scl_keep: Iterable[int] = (4,),
    ndvi_floor: float = 0.2,
    mine_buffer_m: float = 500.0,
) -> MaskResult:
    shape = scl.shape
    in_zone = geometry_to_mask([zone_geom], transform, shape)
    cloudy = np.isin(scl, SCL_CLOUDY)
    finite = np.ones(shape, dtype=bool)
    for arr in bands.values():
        finite &= np.isfinite(arr)
    veg = scl_mask(scl, scl_keep) & (np.nan_to_num(ndvi, nan=-np.inf) >= ndvi_floor) & finite
    fp = footprint_mask(footprints, mine_buffer_m, transform, shape)

    valid = in_zone & veg & ~fp
    return MaskResult(
        valid=valid,
        n_zone=int(in_zone.sum()),
        n_cloudy=int((in_zone & cloudy).sum()),
        n_nonveg=int((in_zone & ~cloudy & ~veg).sum()),
        n_footprint=int((in_zone & veg & fp).sum()),
        n_valid=int(valid.sum()),
    )


def load_footprints(path: str | Path, epsg: int, required: bool) -> list[BaseGeometry]:
    """Read a footprint GeoJSON and reproject to ``epsg``.

    For a mineralised zone (``required=True``) a missing or empty file is an error
    unless the file says ``"reviewed_empty": true`` - an empty mask must be a decision,
    never an accident.
    """
    import geopandas as gpd

    path = Path(path)
    if not path.exists():
        if required:
            raise FootprintError(
                f"Mine footprint file missing: {path}. Run scripts/fetch_osm_footprints.py and "
                "review it against imagery (see config/mine_footprints/README.md)."
            )
        return []
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    features = raw.get("features") or []
    if not features:
        if required and not raw.get("reviewed_empty", False):
            raise FootprintError(
                f"{path} has no polygons. If this zone truly has no mining disturbance, set "
                '"reviewed_empty": true in the file; otherwise add the pit/dump polygons.'
            )
        return []
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(epsg)
    return [g for g in gdf.geometry if g is not None and not g.is_empty]
