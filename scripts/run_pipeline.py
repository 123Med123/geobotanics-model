#!/usr/bin/env python
"""End-to-end Phase 1 pipeline.

    python scripts/run_pipeline.py                       # all enabled belts
    python scripts/run_pipeline.py --belts khetri        # one belt (no transfer test)
    python scripts/run_pipeline.py --skip-download       # re-analyse files already in data/raw

Steps: verification gate -> STAC date selection (same dates for zone and control) ->
openEO download -> masking + indices -> zone-date stats and patches -> comparisons,
classifiers, transfer test -> plots and results/report.md.

The pipeline stops instead of substituting when something is wrong: unverified zones
(unless --allow-unverified), a missing mine-footprint file, too few usable dates, or
implausible reflectance. It never switches to another belt or data source on its own.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import analysis, ingest, plots  # noqa: E402
from src.extract import process_scene, read_scene  # noqa: E402
from src.indices import INDEX_NAMES  # noqa: E402
from src.masking import load_footprints  # noqa: E402
from src.zones import load_pairs  # noqa: E402

UNVERIFIED_STAMP = "UNVERIFIED COORDINATES - not a result"
# NDMI is a confound diagnostic, not a classifier feature.
FEATURE_INDICES = [i for i in INDEX_NAMES if i != "ndmi"]


class PipelineStop(RuntimeError):
    pass


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--belts", nargs="*", help="belt ids from config/zones.yaml (default: all enabled)")
    p.add_argument("--zones", default=str(ROOT / "config/zones.yaml"))
    p.add_argument("--config", default=str(ROOT / "config/pipeline.yaml"))
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--skip-download", action="store_true", help="use scenes already in data/raw")
    p.add_argument("--allow-unverified", action="store_true",
                   help="run on unverified coordinates; every output is stamped as such")
    return p.parse_args()


def choose_dates(pair, cfg, data_dir: Path, skip_download: bool) -> pd.DataFrame:
    out = data_dir / "dates" / f"{pair.belt}_dates.csv"
    if skip_download and out.exists():
        df = pd.read_csv(out, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df
    dc, cd = cfg["dates"], cfg["cdse"]
    start, end = f"{min(dc['years'])}-01-01", f"{max(dc['years'])}-12-31"
    scenes = {z.zone_id: ingest.search_scenes(z.bbox_lonlat(), start, end, cd["stac_url"], cd["stac_collection"])
              for z in pair.zones}
    df = ingest.select_pair_dates(scenes[pair.mineralised.zone_id], scenes[pair.control.zone_id],
                                  dc["years"], dc["seasons"], dc["max_scene_cloud_pct"], dc["dates_per_season"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def run_belt(pair, cfg, data_dir: Path, conn, skip_download: bool):
    dates = choose_dates(pair, cfg, data_dir, skip_download)
    if dates.empty:
        raise PipelineStop(f"{pair.belt}: STAC found no date with both zones under "
                           f"{cfg['dates']['max_scene_cloud_pct']}% cloud.")
    fp_dir = ROOT / cfg["masking"]["footprints_dir"]
    footprints = {
        pair.mineralised.zone_id: load_footprints(fp_dir / f"{pair.belt}.geojson", pair.mineralised.epsg, required=True),
        pair.control.zone_id: load_footprints(fp_dir / f"{pair.belt}_control.geojson", pair.control.epsg, required=False),
    }
    rc, ac, mc = cfg["reflectance"], cfg["analysis"], cfg["masking"]
    stats_rows, patch_frames, mask_rows = [], [], []
    for _, row in dates.iterrows():
        d: date = row["date"]
        per_zone = {}
        for z in pair.zones:
            if skip_download:
                path = ingest.scene_path(data_dir, z.zone_id, d)
                man = path.with_suffix(".json")
                if not (path.exists() and man.exists()):
                    raise PipelineStop(f"--skip-download but {path} is missing")
                bands = json.loads(man.read_text())["bands"]
            else:
                sf = ingest.download_zone_date(conn, z, d, data_dir, cfg["cdse"]["openeo_collection"],
                                               cfg["cdse"]["bands"], cfg["cdse"]["resolution_m"])
                path, bands = sf.path, sf.manifest["bands"]
            scene = read_scene(path, bands, rc["scale"], rc["offset"], rc["nodata_dn"])
            per_zone[z.zone_id] = process_scene(z, d, scene, footprints[z.zone_id], mc, ac["hmssi_psri_eps"],
                                                ac["patch_size_px"], ac["min_patch_valid_fraction"])
        # A date counts only if BOTH zones are usable that day (the control must see the same weather).
        fracs = {zid: r[2]["valid_fraction"] for zid, r in per_zone.items()}
        usable = all(f >= cfg["dates"]["min_valid_fraction"] for f in fracs.values())
        for zid, (rows, patches, mask_info) in per_zone.items():
            mask_rows.append(mask_info | {"season": row["season"], "used": usable})
            if usable:
                stats_rows += [r | {"season": row["season"], "year": row["year"]} for r in rows]
                patch_frames.append(patches)

    masks = pd.DataFrame(mask_rows)
    n_used = int(masks[masks.used].date.nunique()) if not masks.empty else 0
    if n_used < cfg["dates"]["min_dates_per_belt"]:
        raise PipelineStop(
            f"{pair.belt}: only {n_used} date(s) usable after masking (need "
            f"{cfg['dates']['min_dates_per_belt']}). See data/dates/{pair.belt}_dates.csv and the mask table. "
            "Options: relax max_scene_cloud_pct, add years, or enable a reserve belt in zones.yaml - "
            "this is a decision for the project owner, so the pipeline does not switch belts itself."
        )
    stats = pd.DataFrame(stats_rows)
    patches = pd.concat(patch_frames, ignore_index=True) if patch_frames else pd.DataFrame()
    return stats, patches, masks


def fmt(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df.empty:
        return "_(none)_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    header = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False))
    return f"{header}\n{sep}\n{body}\n"


def write_report(path: Path, pairs, masks, summary, zd, pc, direction, plots_made, stamp, stopped):
    L = []
    L.append("# Phase 1 results (auto-generated)\n")
    if stamp:
        L.append(f"> **{stamp}.** One or more zones are not verified in config/zones.yaml. "
                 "Nothing below may be cited as a finding.\n")
    L.append("> Read the README section **\"What this does and doesn't show\"** before reading any number here. "
             "This report states what was measured; it deliberately makes no claim about what it means.\n")
    L.append("## Belts\n")
    for b, p in pairs.items():
        L.append(f"- **{b}** ({p.commodity}, {p.climate}); mineralised verified: {p.mineralised.verified}, "
                 f"control verified: {p.control.verified}")
    if stopped:
        L.append("\n## Belts that could not be analysed\n")
        L += [f"- {m}" for m in stopped]
    L.append("\n## Masking (per zone and date)\n")
    L.append("Pixels removed for cloud, for not being vegetation, and for falling inside the buffered mine footprint. "
             "If the footprint removes most of a mineralised zone, the zone is mostly mine, not vegetation over ore.\n")
    L.append(fmt(masks))
    L.append("\n## Mineralised minus control, per index\n")
    L.append("`n_mineralised_lower` counts dates on which the mineralised zone's median was lower. "
             "`corr_with_ndmi_diff` near +1 means the difference tracks moisture (a drought explanation fits as well).\n")
    L.append(fmt(summary))
    L.append("\n## Classifier: per-zone-per-date features (brief's baseline; tiny n)\n")
    L.append(fmt(zd))
    L.append("\n## Classifier: patch level, within-belt spatial CV and cross-belt transfer\n")
    L.append("Compare `accuracy` with `majority_rate`, and check `predicted_positive_fraction` for one-class collapse. "
             "**The transfer rows are the transferability test.** A transfer balanced accuracy near 0.5 means "
             "what separated zone from control in one belt does not carry over to the other.\n")
    L.append(fmt(pc))
    L.append("\n## Direction agreement across belts (patch features)\n")
    L.append(fmt(direction))
    L.append("\n## Plots\n")
    L += [f"- `{p}`" for p in plots_made]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    pairs = load_pairs(args.zones)
    if args.belts:
        unknown = set(args.belts) - set(pairs)
        if unknown:
            sys.exit(f"Unknown or disabled belt(s): {', '.join(sorted(unknown))}")
        pairs = {b: pairs[b] for b in args.belts}
    if not pairs:
        sys.exit("No enabled belts in config/zones.yaml")

    unverified = [z.zone_id for p in pairs.values() for z in p.zones if not z.verified]
    stamp = None
    if unverified:
        if not args.allow_unverified:
            sys.exit("Refusing to run on unverified zones: " + ", ".join(unverified) +
                     "\nVerify them in config/zones.yaml (see the header there), or pass --allow-unverified "
                     "for a plumbing test whose outputs will be stamped as unverified.")
        stamp = UNVERIFIED_STAMP
        print(f"WARNING: {stamp}: {', '.join(unverified)}")

    data_dir, results_dir = Path(args.data_dir), Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    conn = None if args.skip_download else ingest.connect(cfg["cdse"]["openeo_url"])

    all_stats, all_patches, all_masks, stopped = [], [], [], []
    for belt, pair in pairs.items():
        print(f"== {belt}")
        try:
            s, p, m = run_belt(pair, cfg, data_dir, conn, args.skip_download)
        except PipelineStop as e:
            print(f"STOP: {e}")
            stopped.append(str(e))
            continue
        all_stats.append(s); all_patches.append(p); all_masks.append(m)

    if not all_stats:
        sys.exit("No belt could be analysed:\n" + "\n".join(stopped))

    stats = pd.concat(all_stats, ignore_index=True)
    patches = pd.concat(all_patches, ignore_index=True)
    masks = pd.concat(all_masks, ignore_index=True)
    stats.to_csv(results_dir / "zone_date_stats.csv", index=False)
    patches.to_csv(results_dir / "patches.csv", index=False)
    masks.to_csv(results_dir / "masking.csv", index=False)

    ac = cfg["analysis"]
    diffs = analysis.pair_differences(stats)
    summary = analysis.summarise_differences(diffs)
    diffs.to_csv(results_dir / "pair_differences.csv", index=False)
    summary.to_csv(results_dir / "difference_summary.csv", index=False)

    zd = analysis.zone_date_classifier(analysis.zone_date_features(stats, FEATURE_INDICES), ac["random_state"])
    seasons_by_date = dict(zip(stats["date"], stats["season"]))
    pfeat = analysis.patch_features(patches, seasons_by_date, FEATURE_INDICES)
    pc = analysis.patch_classifier(pfeat, ac["spatial_block_m"], ac["cv_folds"], ac["random_state"])
    direction = analysis.direction_agreement(pfeat) if pfeat.belt.nunique() > 1 else pd.DataFrame()
    zd.to_csv(results_dir / "classifier_zone_date.csv", index=False)
    pc.to_csv(results_dir / "classifier_patch.csv", index=False)
    direction.to_csv(results_dir / "direction_agreement.csv", index=False)
    if stats.belt.nunique() < 2:
        print("NOTE: only one belt analysed - no transferability test possible.")

    plots_made = []
    for belt in stats.belt.unique():
        out = results_dir / "plots" / f"{belt}_timeseries.png"
        plots.plot_belt_timeseries(stats, diffs, belt, out, list(INDEX_NAMES), stamp)
        plots_made.append(out.relative_to(results_dir))

    write_report(results_dir / "report.md", pairs, masks, summary, zd, pc, direction, plots_made, stamp, stopped)
    print(f"Done. See {results_dir / 'report.md'}")


if __name__ == "__main__":
    main()
