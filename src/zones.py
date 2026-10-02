"""Load zone definitions (config/zones.yaml) and build their geometries.

Geometries are built in the local UTM zone (metres) so buffers and offsets are exact.
A control zone is the mineralised zone's shape translated by ``offset_km``, which keeps
area and shape identical between the two.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pyproj import Transformer
from shapely import affinity
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

CONTROL_MIN_KM = 20.0
CONTROL_MAX_KM = 50.0


class ZoneConfigError(ValueError):
    pass


@dataclass
class Zone:
    zone_id: str            # "<belt>" or "<belt>_control"
    belt: str
    role: str               # "mineralised" | "control"
    name: str
    epsg: int               # UTM EPSG code the geometry is expressed in
    geometry: BaseGeometry  # UTM metres
    verified: bool
    verification_notes: str = ""
    sources: list[str] = field(default_factory=list)

    @property
    def label(self) -> int:
        return 1 if self.role == "mineralised" else 0

    def bbox_lonlat(self, margin_m: float = 0.0) -> tuple[float, float, float, float]:
        """(west, south, east, north) in EPSG:4326, after expanding by ``margin_m``."""
        g = self.geometry.buffer(margin_m) if margin_m else self.geometry
        to_ll = Transformer.from_crs(self.epsg, 4326, always_xy=True).transform
        w, s, e, n = shp_transform(to_ll, g.envelope).bounds
        return (w, s, e, n)

    def bbox_utm(self, margin_m: float = 0.0) -> tuple[float, float, float, float]:
        g = self.geometry.buffer(margin_m) if margin_m else self.geometry
        return g.bounds


@dataclass
class BeltPair:
    belt: str
    climate: str
    commodity: str
    mineralised: Zone
    control: Zone

    @property
    def verified(self) -> bool:
        return self.mineralised.verified and self.control.verified

    @property
    def zones(self) -> tuple[Zone, Zone]:
        return (self.mineralised, self.control)


def utm_epsg(lon: float, lat: float) -> int:
    """WGS84 / UTM EPSG code for a point (326xx north, 327xx south)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _is_verified(block: dict | None) -> bool:
    return bool(block) and block.get("status") == "verified"


def build_pair(belt: str, spec: dict) -> BeltPair:
    deposits = spec.get("deposits") or []
    if not deposits:
        raise ZoneConfigError(f"{belt}: no deposits listed")
    radius = float(spec.get("radius_m", 2500))
    if radius <= 0:
        raise ZoneConfigError(f"{belt}: radius_m must be > 0")

    lat0 = sum(d["lat"] for d in deposits) / len(deposits)
    lon0 = sum(d["lon"] for d in deposits) / len(deposits)
    epsg = utm_epsg(lon0, lat0)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True).transform

    circles = [Point(*to_utm(d["lon"], d["lat"])).buffer(radius) for d in deposits]
    mineral_geom = unary_union(circles)

    ctrl = spec.get("control") or {}
    offset = ctrl.get("offset_km")
    if not offset or len(offset) != 2:
        raise ZoneConfigError(f"{belt}: control.offset_km must be [east_km, north_km]")
    dx, dy = float(offset[0]) * 1000.0, float(offset[1]) * 1000.0
    control_geom = affinity.translate(mineral_geom, xoff=dx, yoff=dy)

    dist_km = (dx**2 + dy**2) ** 0.5 / 1000.0
    if not (CONTROL_MIN_KM <= dist_km <= CONTROL_MAX_KM):
        raise ZoneConfigError(
            f"{belt}: control offset is {dist_km:.1f} km; must be {CONTROL_MIN_KM:.0f}-{CONTROL_MAX_KM:.0f} km"
        )
    if control_geom.intersects(mineral_geom):
        raise ZoneConfigError(f"{belt}: control zone overlaps the mineralised zone")

    ver = spec.get("verification") or {}
    cver = ctrl.get("verification") or {}
    mineralised = Zone(
        zone_id=belt,
        belt=belt,
        role="mineralised",
        name=", ".join(d["name"] for d in deposits),
        epsg=epsg,
        geometry=mineral_geom,
        verified=_is_verified(ver),
        verification_notes=(ver.get("notes") or "").strip(),
        sources=list(spec.get("sources") or []),
    )
    control = Zone(
        zone_id=f"{belt}_control",
        belt=belt,
        role="control",
        name=ctrl.get("name", f"{belt} control"),
        epsg=epsg,
        geometry=control_geom,
        verified=_is_verified(cver),
        verification_notes=(cver.get("notes") or "").strip(),
        sources=list(ctrl.get("sources") or []),
    )
    return BeltPair(
        belt=belt,
        climate=spec.get("climate", ""),
        commodity=spec.get("commodity", ""),
        mineralised=mineralised,
        control=control,
    )


def load_pairs(path: str | Path = "config/zones.yaml", only_enabled: bool = True) -> dict[str, BeltPair]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    pairs = {}
    for belt, spec in (cfg.get("belts") or {}).items():
        if only_enabled and not spec.get("enabled", False):
            continue
        pairs[belt] = build_pair(belt, spec)
    return pairs
