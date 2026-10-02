#!/usr/bin/env python
"""Check CDSE access before a real run: network, STAC search, openEO login.

    python scripts/check_access.py

Prints what works and what doesn't, and never prints secret values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import ingest  # noqa: E402

HOSTS = [
    "https://stac.dataspace.copernicus.eu/v1",
    "https://openeo.dataspace.copernicus.eu/openeo/1.2/",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/.well-known/openid-configuration",
]


def main():
    cfg = yaml.safe_load(open(ROOT / "config/pipeline.yaml", encoding="utf-8"))
    ok = True
    for url in HOSTS:
        try:
            r = requests.get(url, timeout=20)
            print(f"[{'ok' if r.status_code < 400 else 'FAIL'}] {r.status_code} {url}")
            ok &= r.status_code < 400
        except Exception as e:
            print(f"[FAIL] {url}: {type(e).__name__}: {e}")
            ok = False

    try:
        df = ingest.search_scenes((75.75, 28.0, 75.8, 28.05), "2024-03-01", "2024-03-31",
                                  cfg["cdse"]["stac_url"], cfg["cdse"]["stac_collection"])
        print(f"[ok] STAC search returned {len(df)} Sentinel-2 L2A item(s) for a test box, March 2024")
    except Exception as e:
        print(f"[FAIL] STAC search: {type(e).__name__}: {e}")
        ok = False

    try:
        ingest.require_credentials()
        conn = ingest.connect(cfg["cdse"]["openeo_url"])
        print(f"[ok] openEO authenticated; collection {cfg['cdse']['openeo_collection']} present: "
              f"{cfg['cdse']['openeo_collection'] in conn.list_collection_ids()}")
    except Exception as e:
        print(f"[FAIL] openEO: {type(e).__name__}: {e}")
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
