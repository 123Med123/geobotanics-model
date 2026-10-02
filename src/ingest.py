"""Copernicus Data Space Ecosystem (CDSE) search and download.

Two CDSE services, no other data source:
  * STAC catalogue (https://stac.dataspace.copernicus.eu/v1) - open, no auth. Used only
    to list candidate acquisition dates and their tile-level cloud cover.
  * openEO (https://openeo.dataspace.copernicus.eu) - authenticated with OAuth client
    credentials. Used to download each zone's bands clipped to its bounding box and
    resampled to 10 m, so we never pull whole 100 km tiles.

Credentials come from the environment only (never from files in the repo):
  CDSE_CLIENT_ID, CDSE_CLIENT_SECRET   OAuth client created in the CDSE dashboard
  CDSE_OIDC_PROVIDER                   optional, defaults to "CDSE"
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

REQUIRED_ENV = ("CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET")


class CredentialsMissing(RuntimeError):
    pass


def require_credentials() -> tuple[str, str, str]:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise CredentialsMissing(
            "Missing environment variable(s): " + ", ".join(missing) + ". Create an OAuth client "
            "in the CDSE dashboard and set them as environment secrets (never commit them)."
        )
    return (
        os.environ["CDSE_CLIENT_ID"],
        os.environ["CDSE_CLIENT_SECRET"],
        os.environ.get("CDSE_OIDC_PROVIDER", "CDSE"),
    )


# --- date selection (STAC) ---------------------------------------------------------------

def search_scenes(
    bbox_lonlat: Sequence[float], start: str, end: str, stac_url: str, collection: str
) -> pd.DataFrame:
    """All L2A items intersecting ``bbox_lonlat`` between start and end (inclusive).

    Cloud filtering is done client-side on ``eo:cloud_cover`` so we don't depend on
    which query/filter extensions the catalogue supports.
    Returns columns: date, item_id, cloud_pct.
    """
    from pystac_client import Client
    from pystac_client.exceptions import APIError

    client = Client.open(stac_url)
    rows: dict[str, dict] = {}
    # The CDSE STAC gateway answers slowly (tens of seconds per page for a multi-year
    # query) and sometimes with 504, so ask quarter by quarter and retry each chunk.
    # The union of the chunks is the same item set as one query over start..end.
    for lo, hi in _date_chunks(date.fromisoformat(start), date.fromisoformat(end)):
        for attempt in range(1, _STAC_RETRIES + 1):
            try:
                search = client.search(collections=[collection], bbox=list(bbox_lonlat),
                                       datetime=f"{lo.isoformat()}/{hi.isoformat()}", limit=_STAC_PAGE_SIZE)
                items = list(search.items())
                break
            except (APIError, OSError):
                if attempt == _STAC_RETRIES:
                    raise
                time.sleep(5 * attempt)
        for item in items:
            dt = item.datetime or datetime.fromisoformat(item.properties["start_datetime"].replace("Z", "+00:00"))
            rows[item.id] = {
                "date": dt.date(),
                "item_id": item.id,
                "cloud_pct": float(item.properties.get("eo:cloud_cover", 100.0)),
            }
    return pd.DataFrame(list(rows.values()), columns=["date", "item_id", "cloud_pct"])


_STAC_RETRIES = 4
_STAC_PAGE_SIZE = 50


def _date_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split start..end (inclusive) into calendar-quarter chunks."""
    chunks, cur = [], start
    while cur <= end:
        q_end_month = ((cur.month - 1) // 3 + 1) * 3
        nxt = date(cur.year + (q_end_month == 12), q_end_month % 12 + 1, 1)
        hi = min(nxt - timedelta(days=1), end)
        chunks.append((cur, hi))
        cur = hi + timedelta(days=1)
    return chunks


def season_of(d: date, seasons: dict[str, Sequence[str]]) -> str | None:
    md = d.strftime("%m-%d")
    for name, (lo, hi) in seasons.items():
        if lo <= md <= hi:
            return name
    return None


def select_pair_dates(
    zone_scenes: pd.DataFrame,
    control_scenes: pd.DataFrame,
    years: Iterable[int],
    seasons: dict[str, Sequence[str]],
    max_cloud_pct: float,
    per_season: int = 1,
) -> pd.DataFrame:
    """Pick acquisition dates usable for BOTH the mineralised zone and its control.

    Using the same date for both is what lets the control absorb that day's weather and
    phenology. A date qualifies when every tile covering each zone that day is at or
    below ``max_cloud_pct``. Per (year, season) we keep the ``per_season`` dates with the
    lowest worst-case cloud. Returns columns: year, season, date, zone_cloud, control_cloud.
    """
    def per_day(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        return df.groupby("date")["cloud_pct"].max()

    z = per_day(zone_scenes)
    c = per_day(control_scenes)
    common = sorted(set(z.index) & set(c.index))
    years = set(years)
    rows = []
    for d in common:
        s = season_of(d, seasons)
        if s is None or d.year not in years:
            continue
        zc, cc = float(z[d]), float(c[d])
        if zc <= max_cloud_pct and cc <= max_cloud_pct:
            rows.append({"year": d.year, "season": s, "date": d, "zone_cloud": zc, "control_cloud": cc})
    cols = ["year", "season", "date", "zone_cloud", "control_cloud"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["worst"] = df[["zone_cloud", "control_cloud"]].max(axis=1)
    df = (
        df.sort_values(["year", "season", "worst", "date"])
        .groupby(["year", "season"], as_index=False)
        .head(per_season)
        .sort_values("date")
    )
    return df[cols].reset_index(drop=True)


# --- download (openEO) -------------------------------------------------------------------

def connect(openeo_url: str):
    import openeo

    client_id, client_secret, provider = require_credentials()
    conn = openeo.connect(openeo_url)
    conn.authenticate_oidc_client_credentials(
        client_id=client_id, client_secret=client_secret, provider_id=provider
    )
    return conn


@dataclass
class SceneFile:
    zone_id: str
    date: date
    path: Path
    manifest: dict


def scene_path(data_dir: Path, zone_id: str, d: date) -> Path:
    return Path(data_dir) / "raw" / zone_id / f"{zone_id}_{d.isoformat()}.tif"


def download_zone_date(
    conn,
    zone,
    d: date,
    data_dir: Path,
    collection: str,
    bands: Sequence[str],
    resolution_m: float = 10,
    margin_m: float = 200,
    retries: int = 3,
) -> SceneFile:
    """Download one zone on one date as a multi-band GeoTIFF (band order = ``bands``).

    Idempotent: an existing file + manifest is reused. Same-day granules from adjacent
    tiles are merged by taking the first valid observation.
    """
    out = scene_path(data_dir, zone.zone_id, d)
    man_path = out.with_suffix(".json")
    if out.exists() and man_path.exists():
        return SceneFile(zone.zone_id, d, out, json.loads(man_path.read_text()))

    out.parent.mkdir(parents=True, exist_ok=True)
    # Snap the extent outward to whole kilometres so every date of a zone requests the
    # identical grid (pixel edges on multiples of the resolution).
    w, s, e, n = zone.bbox_utm(margin_m)
    w, s = math.floor(w / 1000) * 1000.0, math.floor(s / 1000) * 1000.0
    e, n = math.ceil(e / 1000) * 1000.0, math.ceil(n / 1000) * 1000.0
    extent = {"west": w, "south": s, "east": e, "north": n, "crs": zone.epsg}
    temporal = [d.isoformat(), (d + timedelta(days=1)).isoformat()]

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            cube = conn.load_collection(collection, spatial_extent=extent, temporal_extent=temporal, bands=list(bands))
            cube = cube.resample_spatial(resolution=resolution_m, projection=zone.epsg, method="near")
            cube = cube.reduce_dimension(dimension="t", reducer="first")
            cube.download(str(out), format="GTiff")
            break
        except Exception as err:  # network / backend hiccups: retry with backoff
            last_err = err
            if out.exists():
                out.unlink()
            if attempt == retries:
                raise RuntimeError(f"openEO download failed for {zone.zone_id} {d}: {err}") from err
            time.sleep(2 ** attempt)

    manifest = {
        "zone_id": zone.zone_id,
        "date": d.isoformat(),
        "epsg": zone.epsg,
        "bands": list(bands),
        "collection": collection,
        "resolution_m": resolution_m,
        "spatial_extent": extent,
        "temporal_extent": temporal,
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "last_error_before_success": str(last_err) if last_err else None,
    }
    man_path.write_text(json.dumps(manifest, indent=2))
    return SceneFile(zone.zone_id, d, out, manifest)
