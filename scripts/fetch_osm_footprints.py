#!/usr/bin/env python
"""Pull candidate mine-footprint polygons from OpenStreetMap (Overpass API).

    python scripts/fetch_osm_footprints.py --belt khetri            # -> config/mine_footprints/khetri.geojson
    python scripts/fetch_osm_footprints.py --belt khetri --control  # -> khetri_control.geojson

The output is a STARTING POINT. OSM coverage of Indian mines is patchy, so check the
polygons against imagery and add any pit, dump, tailings pond or plant that is missing
(see config/mine_footprints/README.md). Needs network access to overpass-api.de.
Only closed ways are converted; multipolygon relations become their outer rings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.zones import load_pairs  # noqa: E402

OVERPASS = "https://overpass-api.de/api/interpreter"
TAGS = [
    ('landuse', 'quarry'), ('landuse', 'industrial'), ('landuse', 'landfill'), ('landuse', 'brownfield'),
    ('man_made', 'spoil_heap'), ('man_made', 'mineshaft'), ('man_made', 'adit'), ('man_made', 'works'),
    ('industrial', 'mine'), ('industrial', 'smelter'),
]


def build_query(s, w, n, e) -> str:
    parts = []
    for k, v in TAGS:
        parts.append(f'way["{k}"="{v}"]({s},{w},{n},{e});')
        parts.append(f'relation["{k}"="{v}"]({s},{w},{n},{e});')
    return "[out:json][timeout:120];(" + "".join(parts) + ");out geom;"


def ring(geom) -> Polygon | None:
    pts = [(p["lon"], p["lat"]) for p in geom or []]
    if len(pts) >= 4 and pts[0] == pts[-1]:
        poly = Polygon(pts)
        return poly if poly.is_valid and poly.area > 0 else poly.buffer(0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--belt", required=True)
    ap.add_argument("--control", action="store_true", help="fetch for the control zone instead")
    ap.add_argument("--margin-m", type=float, default=1000.0)
    args = ap.parse_args()

    pairs = load_pairs(ROOT / "config/zones.yaml", only_enabled=False)
    pair = pairs[args.belt]
    zone = pair.control if args.control else pair.mineralised
    w, s, e, n = zone.bbox_lonlat(args.margin_m)

    r = requests.post(OVERPASS, data={"data": build_query(s, w, n, e)}, timeout=180)
    r.raise_for_status()
    feats = []
    for el in r.json().get("elements", []):
        polys = []
        if el["type"] == "way":
            p = ring(el.get("geometry"))
            if p is not None:
                polys.append(p)
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") == "outer":
                    p = ring(m.get("geometry"))
                    if p is not None:
                        polys.append(p)
        if polys:
            feats.append({"type": "Feature", "geometry": mapping(unary_union(polys)),
                          "properties": {"osm_type": el["type"], "osm_id": el["id"], **(el.get("tags") or {}),
                                         "source": "OpenStreetMap contributors (ODbL)"}})

    out = ROOT / "config/mine_footprints" / (f"{args.belt}_control.geojson" if args.control else f"{args.belt}.geojson")
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=1))
    print(f"{len(feats)} polygon feature(s) -> {out}")
    if not feats:
        print("No OSM footprints found. Digitise the pit/dumps by hand - an empty file will be rejected "
              "for a mineralised zone unless you mark it reviewed_empty.")
    print("REVIEW AGAINST IMAGERY before running the pipeline.")


if __name__ == "__main__":
    main()
