from datetime import date

import pandas as pd
import pytest

from src.ingest import select_pair_dates
from src.zones import ZoneConfigError, build_pair, load_pairs, utm_epsg

SPEC = {
    "radius_m": 2000,
    "deposits": [{"name": "A", "lat": 28.0, "lon": 75.8}],
    "verification": {"status": "verified"},
    "control": {"offset_km": [-30, 0], "verification": {"status": "unverified"}},
}


def test_utm_epsg():
    assert utm_epsg(75.8, 28.0) == 32643   # Khetri
    assert utm_epsg(80.7, 22.0) == 32644   # Malanjkhand


def test_control_is_same_shape_translated():
    p = build_pair("t", SPEC)
    assert p.control.geometry.area == pytest.approx(p.mineralised.geometry.area)
    dx = p.control.geometry.centroid.x - p.mineralised.geometry.centroid.x
    assert dx == pytest.approx(-30000, abs=1)
    assert p.mineralised.verified and not p.control.verified and not p.verified


@pytest.mark.parametrize("offset", [[-10, 0], [0, 60]])
def test_control_distance_limits(offset):
    spec = dict(SPEC, control={"offset_km": offset})
    with pytest.raises(ZoneConfigError):
        build_pair("t", spec)


def test_repo_zones_file_loads_and_nothing_is_marked_verified_by_accident():
    pairs = load_pairs("config/zones.yaml", only_enabled=False)
    assert {"khetri", "malanjkhand"} <= set(pairs)
    # Phase 1 guard: no U/Th/REE targets may appear in the config.
    text = open("config/zones.yaml", encoding="utf-8").read().lower()
    for word in ("tummalapalle", "seshachalam"):
        assert f"name: {word}" not in text and f"  {word}:" not in text


SEASONS = {"pre_monsoon": ["03-01", "05-31"], "monsoon": ["07-01", "09-30"], "post_monsoon": ["10-15", "12-15"]}


def _scenes(rows):
    return pd.DataFrame(rows, columns=["date", "item_id", "cloud_pct"])


def test_select_pair_dates_requires_both_zones_clear_same_day():
    z = _scenes([(date(2024, 4, 1), "a", 5), (date(2024, 4, 6), "b", 2), (date(2024, 8, 1), "c", 10)])
    c = _scenes([(date(2024, 4, 1), "d", 50), (date(2024, 4, 6), "e", 1), (date(2024, 8, 1), "f", 20)])
    out = select_pair_dates(z, c, [2024], SEASONS, max_cloud_pct=30, per_season=1)
    assert list(out.date) == [date(2024, 4, 6), date(2024, 8, 1)]
    assert list(out.season) == ["pre_monsoon", "monsoon"]


def test_select_pair_dates_worst_tile_counts_and_out_of_season_dropped():
    # Two tiles on 2024-04-10 for the zone: one cloudy -> that day fails.
    z = _scenes([(date(2024, 4, 10), "a", 5), (date(2024, 4, 10), "b", 80), (date(2024, 6, 15), "c", 0)])
    c = _scenes([(date(2024, 4, 10), "d", 5), (date(2024, 6, 15), "e", 0)])
    out = select_pair_dates(z, c, [2024], SEASONS, max_cloud_pct=30)
    assert out.empty   # April day too cloudy on one tile; June is outside every season window
