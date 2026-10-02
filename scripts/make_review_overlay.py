#!/usr/bin/env python
"""Build a self-contained HTML review map for one zone (a REVIEW AID, not a result).

    python scripts/make_review_overlay.py --belt malanjkhand            # mineralised zone
    python scripts/make_review_overlay.py --belt malanjkhand --control  # its control
    -> results/review/<belt>[_control]_overlay.html   (open in a browser; needs internet for tiles)

It draws, over satellite imagery: the zone outline, the deposit points, the mine-footprint polygons
from config/mine_footprints/ with their buffer (masking.mine_buffer_m), and - for each date whose
scene is already cached in data/raw/ - the pixel categories the pipeline's own masking assigns
(cloud, no-data, SCL non-vegetation, NDVI below the floor, inside the footprint buffer, valid).
It also has a polygon draw tool that exports GeoJSON, for adding pits/dumps/tailings that OSM lacks.

It only reads the repo. Re-run it after editing a footprint file to see what changed. Nothing here
changes a mask, a threshold or a zone, and it never marks anything reviewed.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.warp import Resampling, calculate_default_transform, reproject  # noqa: E402
from shapely.geometry import mapping, shape  # noqa: E402
from shapely.ops import transform as shp_transform, unary_union  # noqa: E402

from src import ingest  # noqa: E402
from src.extract import read_scene  # noqa: E402
from src.indices import compute_all  # noqa: E402
from src.masking import SCL_CLOUDY, build_analysis_mask, footprint_mask, geometry_to_mask, load_footprints  # noqa: E402
from src.zones import load_pairs  # noqa: E402

# category code -> (key, label, RGBA)
CATS = {
    1: ("nodata", "No data / off-swath", (40, 40, 40, 190)),
    2: ("cloud", "Cloud / shadow / cirrus", (150, 190, 235, 170)),
    3: ("scl_nonveg", "SCL not vegetation", (240, 140, 30, 150)),
    4: ("ndvi_floor", "SCL vegetation but NDVI below floor", (245, 215, 60, 150)),
    5: ("footprint", "Vegetated but inside footprint buffer", (225, 40, 40, 170)),
    6: ("valid", "Valid (enters the analysis)", (40, 170, 70, 140)),
}


def categorise(zone, scene, footprints, mc, eps):
    """Per-pixel category image (uint8, 0 = outside the zone) plus the pipeline's own MaskResult."""
    idx = compute_all(scene.bands, hmssi_psri_eps=eps)
    shp = scene.scl.shape
    in_zone = geometry_to_mask([zone.geometry], scene.transform, shp)
    finite = np.ones(shp, dtype=bool)
    for arr in scene.bands.values():
        finite &= np.isfinite(arr)
    nodata = ~finite | (scene.scl == -32768)
    cloudy = np.isin(scene.scl, SCL_CLOUDY)
    scl_veg = np.isin(scene.scl, list(mc["scl_keep"]))
    ndvi_ok = np.nan_to_num(idx["ndvi"], nan=-np.inf) >= mc["ndvi_floor"]
    fp = footprint_mask(footprints, mc["mine_buffer_m"], scene.transform, shp)
    cat = np.zeros(shp, np.uint8)
    cat[in_zone] = 3
    cat[in_zone & scl_veg & ~ndvi_ok] = 4
    cat[in_zone & scl_veg & ndvi_ok & fp] = 5
    cat[in_zone & scl_veg & ndvi_ok & ~fp] = 6
    cat[in_zone & cloudy] = 2
    cat[in_zone & nodata] = 1
    ref = build_analysis_mask(zone.geometry, scene.scl, idx["ndvi"], scene.bands, footprints, scene.transform,
                              scl_keep=mc["scl_keep"], ndvi_floor=mc["ndvi_floor"], mine_buffer_m=mc["mine_buffer_m"])
    if int((cat == 6).sum()) != ref.n_valid:  # the overlay must agree with the real mask
        raise RuntimeError(f"overlay valid count {(cat == 6).sum()} != pipeline n_valid {ref.n_valid}")
    return cat, ref


def to_png_overlay(cat: np.ndarray, transform, epsg: int):
    """Warp the category image to EPSG:4326 (nearest) and return (png data-URI, [[s, w], [n, e]])."""
    h, w = cat.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    dst_t, dw, dh = calculate_default_transform(f"EPSG:{epsg}", "EPSG:4326", w, h,
                                                left=left, bottom=bottom, right=right, top=top)
    dst = np.zeros((dh, dw), np.uint8)
    reproject(cat, dst, src_transform=transform, src_crs=f"EPSG:{epsg}", dst_transform=dst_t,
              dst_crs="EPSG:4326", resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
    lut = np.zeros((7, 4), np.uint8)
    for code, (_, _, rgba) in CATS.items():
        lut[code] = rgba
    buf = io.BytesIO()
    mpimg.imsave(buf, lut[dst], format="png")
    west, north = dst_t.c, dst_t.f
    east, south = west + dw * dst_t.a, north + dh * dst_t.e
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return uri, [[south, west], [north, east]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--belt", required=True)
    ap.add_argument("--control", action="store_true", help="show the control zone instead of the mineralised one")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "review"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config/pipeline.yaml", encoding="utf-8"))
    zones_cfg = yaml.safe_load(open(ROOT / "config/zones.yaml", encoding="utf-8"))["belts"][args.belt]
    pair = load_pairs(ROOT / "config/zones.yaml", only_enabled=False)[args.belt]
    zone = pair.control if args.control else pair.mineralised
    rc, ac, mc = cfg["reflectance"], cfg["analysis"], cfg["masking"]
    to_ll = Transformer.from_crs(zone.epsg, 4326, always_xy=True).transform
    to_utm = Transformer.from_crs(4326, zone.epsg, always_xy=True).transform

    fp_path = ROOT / mc["footprints_dir"] / (f"{args.belt}_control.geojson" if args.control else f"{args.belt}.geojson")
    fp_raw = json.loads(fp_path.read_text(encoding="utf-8")) if fp_path.exists() else {"type": "FeatureCollection", "features": []}
    fps = load_footprints(fp_path, zone.epsg, required=False) if fp_path.exists() else []
    union = unary_union(fps) if fps else None
    buffered = union.buffer(mc["mine_buffer_m"]) if union is not None else None

    # footprint features for display, with area in ha (computed in UTM) and an OSM link
    feats = []
    for ft in fp_raw.get("features", []):
        g_utm = shp_transform(to_utm, shape(ft["geometry"]))
        props = dict(ft.get("properties") or {})
        props["area_ha"] = round(g_utm.area / 1e4, 1)
        props["in_zone_ha"] = round(g_utm.intersection(zone.geometry).area / 1e4, 1)
        if props.get("osm_type") and props.get("osm_id"):
            props["osm_url"] = f"https://www.openstreetmap.org/{props['osm_type']}/{props['osm_id']}"
        feats.append({"type": "Feature", "geometry": ft["geometry"], "properties": props})

    zone_area = zone.geometry.area
    cover = {
        "zone_km2": round(zone_area / 1e6, 2),
        "footprint_pct_of_zone": round(100 * zone.geometry.intersection(union).area / zone_area, 1) if union is not None else 0.0,
        "buffered_pct_of_zone": round(100 * zone.geometry.intersection(buffered).area / zone_area, 1) if buffered is not None else 0.0,
        "n_polygons": len(feats), "buffer_m": mc["mine_buffer_m"],
        "reviewed_empty": bool(fp_raw.get("reviewed_empty", False)),
        "file": str(fp_path.relative_to(ROOT)).replace("\\", "/"),
        "file_modified": datetime.fromtimestamp(fp_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if fp_path.exists() else "missing",
    }

    deposits = [] if args.control else [{"name": d["name"], "lat": d["lat"], "lon": d["lon"]} for d in zones_cfg["deposits"]]

    # per-date category overlays from cached scenes
    dates_csv = ROOT / "data" / "dates" / f"{args.belt}_dates.csv"
    masks, skipped = [], []
    if dates_csv.exists():
        for _, r in pd.read_csv(dates_csv, parse_dates=["date"]).iterrows():
            d = r["date"].date()
            path = ingest.scene_path(ROOT / "data", zone.zone_id, d)
            man = path.with_suffix(".json")
            if not (path.exists() and path.stat().st_size > 0 and man.exists()):
                skipped.append(str(d))
                continue
            bands = json.loads(man.read_text())["bands"]
            scene = read_scene(path, bands, rc["scale"], rc["offset"], rc["nodata_dn"])
            cat, ref = categorise(zone, scene, fps, mc, ac["hmssi_psri_eps"])
            uri, bounds = to_png_overlay(cat, scene.transform, scene.epsg or zone.epsg)
            n = max(int((cat > 0).sum()), 1)
            pct = {CATS[c][0]: round(100 * int((cat == c).sum()) / n, 1) for c in CATS}
            masks.append({"date": str(d), "season": r["season"], "uri": uri, "bounds": bounds, "pct": pct,
                          "valid_fraction_pipeline": round(ref.valid_fraction, 3)})

    data = {
        "title": f"{args.belt}{' control' if args.control else ''}: footprint review",
        "role": zone.role, "zone_name": zone.name, "verified": zone.verified,
        "zone": {"type": "Feature", "geometry": mapping(shp_transform(to_ll, zone.geometry)), "properties": {}},
        "deposits": deposits, "footprints": {"type": "FeatureCollection", "features": feats},
        "buffered": mapping(shp_transform(to_ll, buffered)) if buffered is not None else None,
        "cover": cover, "masks": masks, "skipped": skipped,
        "cats": {str(k): {"key": v[0], "label": v[1], "rgba": list(v[2])} for k, v in CATS.items()},
        "min_valid_fraction": cfg["dates"]["min_valid_fraction"],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.belt}{'_control' if args.control else ''}_overlay.html"
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print(f"{out}  ({len(feats)} footprint polygon(s), {len(masks)} date overlay(s); "
          f"no scene cached for: {', '.join(skipped) or 'none'})")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Footprint review overlay</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css">
<style>
  :root { --bg:#fff; --fg:#1b1f24; --muted:#59636e; --line:#d0d7de; --warn-bg:#fff4d6; --warn-line:#e0b400; --panel:#f6f8fa; }
  @media (prefers-color-scheme: dark) { :root { --bg:#0f1318; --fg:#e6edf3; --muted:#9aa4af; --line:#30363d; --warn-bg:#3a2f08; --warn-line:#a67c00; --panel:#161b22; } }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--fg); }
  .wrap { display:flex; min-height:100vh; }
  #map { flex:1 1 60%; min-height:100vh; }
  aside { flex:0 0 430px; max-width:100%; padding:16px; overflow:auto; max-height:100vh; border-left:1px solid var(--line); background:var(--panel); }
  h1 { font-size:17px; margin:0 0 8px; } h2 { font-size:13px; margin:18px 0 6px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
  .warn { background:var(--warn-bg); border:1px solid var(--warn-line); border-radius:6px; padding:8px 10px; font-size:13px; }
  table { border-collapse:collapse; width:100%; font-size:12.5px; } th,td { padding:3px 6px; border-bottom:1px solid var(--line); text-align:right; }
  th:first-child, td:first-child { text-align:left; } tr.sel td { background:rgba(80,140,255,.15); }
  .sw { display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-1px; margin-right:6px; border:1px solid rgba(0,0,0,.3); }
  select, button { font:inherit; padding:5px 8px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--fg); }
  button { cursor:pointer; } label { display:block; margin:6px 0 2px; color:var(--muted); font-size:12.5px; }
  .muted { color:var(--muted); font-size:12.5px; } code { font-size:12px; }
  @media (max-width: 900px) { .wrap { flex-direction:column; } #map { min-height:60vh; flex:none; height:60vh; } aside { flex:none; max-height:none; border-left:0; border-top:1px solid var(--line); } }
</style></head>
<body><div class="wrap"><div id="map"></div><aside>
  <h1 id="title"></h1>
  <div class="warn" id="warn"></div>
  <h2>Zone and footprint file</h2><div id="cover"></div>
  <h2>Mask overlay (the pipeline's own categories)</h2>
  <label for="datesel">Date</label><select id="datesel"></select>
  <label for="op">Overlay opacity</label><input id="op" type="range" min="0" max="1" step="0.05" value="0.8" style="width:100%">
  <div id="legend" style="margin-top:8px"></div>
  <h2>Per date, % of zone pixels</h2><div id="tbl"></div><div class="muted" id="skipped"></div>
  <h2>Add a missed pit, dump or tailings pond</h2>
  <div class="muted">Use the polygon tool (top-left of the map), trace what you can see in the imagery, then download. The download is a GeoJSON file of your polygons only. It does not change any mask and is not applied anywhere; merge it into the footprint file yourself and fill in <code>source</code> with the imagery and its date.</div>
  <p><button id="dl">Download drawn polygons</button> <span class="muted" id="ndrawn">0 drawn</span></p>
</aside></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>
<script>
const D = __DATA__;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('title').textContent = D.title;
document.title = D.title;
document.getElementById('warn').innerHTML = '<b>Review aid, not a result.</b> ' +
  (D.verified ? '' : 'This zone is <b>UNVERIFIED</b> in config/zones.yaml. ') +
  'Footprints from OpenStreetMap are an <b>unreviewed</b> starting point and OSM coverage in rural India is patchy. Nothing here has been marked reviewed. See the README, "What this does and doesn\'t show".';

const map = L.map('map');
const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom:19, attribution:'Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community'}).addTo(map);
const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom:19, attribution:'&copy; OpenStreetMap contributors'});
const layers = {};
layers['Zone outline'] = L.geoJSON(D.zone, {style:{color:'#00e5ff', weight:2.5, fill:false}}).addTo(map);
layers['Deposit point(s)'] = L.layerGroup(D.deposits.map(p => L.circleMarker([p.lat,p.lon], {radius:6, color:'#fff', weight:2, fillColor:'#7c3aed', fillOpacity:1}).bindTooltip(esc(p.name)))).addTo(map);
layers['Footprint polygons (file)'] = L.geoJSON(D.footprints, {
  style:{color:'#ff2d2d', weight:2, fillColor:'#ff2d2d', fillOpacity:0.12},
  onEachFeature:(f,l) => { const p=f.properties; let h='<b>'+esc(p.name||p.landuse||p.man_made||p.industrial||'polygon')+'</b><br>';
    for (const k of ['landuse','man_made','industrial','operator','resource']) if (p[k]) h+=k+': '+esc(p[k])+'<br>';
    h+='area '+p.area_ha+' ha ('+p.in_zone_ha+' ha inside the zone)'; if (p.osm_url) h+='<br><a href="'+p.osm_url+'" target="_blank" rel="noopener">OSM '+esc(p.osm_type)+'/'+esc(p.osm_id)+'</a>'; l.bindPopup(h); }
}).addTo(map);
if (D.buffered) layers['Footprint + '+D.cover.buffer_m+' m buffer'] = L.geoJSON(D.buffered, {style:{color:'#ff9800', weight:2, dashArray:'6 5', fill:false}}).addTo(map);
const overlays = Object.assign({}, layers);
const drawn = new L.FeatureGroup().addTo(map);
overlays['Drawn polygons (not in any file)'] = drawn;
L.control.layers({'Satellite (Esri)':esri, 'OpenStreetMap':osm}, overlays, {collapsed: window.innerWidth < 900}).addTo(map);
map.addControl(new L.Control.Draw({edit:{featureGroup:drawn}, draw:{polygon:{allowIntersection:false, shapeOptions:{color:'#00ff6a'}}, polyline:false, rectangle:true, circle:false, marker:false, circlemarker:false}}));
const updN = () => document.getElementById('ndrawn').textContent = drawn.getLayers().length + ' drawn';
map.on(L.Draw.Event.CREATED, e => { drawn.addLayer(e.layer); updN(); });
map.on(L.Draw.Event.DELETED, updN); map.on(L.Draw.Event.EDITED, updN);
document.getElementById('dl').onclick = () => {
  const gj = drawn.toGeoJSON(); gj.features.forEach(f => f.properties = {source:'hand-digitised on '+D.title+' (REPLACE with imagery name and date)', kind:'(pit | waste_dump | tailings | plant | road | other)', reviewed_by:null});
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(gj,null,1)], {type:'application/geo+json'})); a.download = 'drawn_polygons.geojson'; a.click(); };
const fit = () => { map.invalidateSize(); map.fitBounds(layers['Zone outline'].getBounds().pad(0.15)); };
map.setView(layers['Zone outline'].getBounds().getCenter(), 13);
// The container's size can settle after first paint; refit on every resize until the user touches the map.
let touched = false;
['mousedown','wheel','touchstart','keydown'].forEach(ev => document.getElementById('map').addEventListener(ev, () => { touched = true; }, {passive:true}));
new ResizeObserver(() => { map.invalidateSize(); if (!touched) fit(); }).observe(document.getElementById('map'));
window.addEventListener('load', fit);

const c = D.cover;
document.getElementById('cover').innerHTML =
  '<table><tr><td>Zone</td><td>'+esc(D.zone_name)+' ('+c.zone_km2+' km&sup2;)</td></tr>' +
  '<tr><td>Footprint file</td><td><code>'+esc(c.file)+'</code></td></tr>' +
  '<tr><td>Last modified</td><td>'+esc(c.file_modified)+'</td></tr>' +
  '<tr><td>Polygons</td><td>'+c.n_polygons+(c.reviewed_empty?' (file says reviewed_empty)':'')+'</td></tr>' +
  '<tr><td>Footprint covers</td><td>'+c.footprint_pct_of_zone+'% of the zone</td></tr>' +
  '<tr><td>With '+c.buffer_m+' m buffer</td><td><b>'+c.buffered_pct_of_zone+'%</b> of the zone</td></tr></table>';
document.getElementById('legend').innerHTML = Object.values(D.cats).map(k => '<div><span class="sw" style="background:rgba('+k.rgba[0]+','+k.rgba[1]+','+k.rgba[2]+','+Math.max(k.rgba[3]/255,0.7)+')"></span>'+esc(k.label)+'</div>').join('');
const keys = Object.values(D.cats).map(k => k.key);
let cur = null;
const sel = document.getElementById('datesel');
sel.innerHTML = '<option value="-1">(none)</option>' + D.masks.map((m,i) => '<option value="'+i+'">'+m.date+' ('+esc(m.season)+')</option>').join('');
const ovl = D.masks.map(m => L.imageOverlay(m.uri, m.bounds, {opacity:0.8, interactive:false}));
function show(i) { if (cur !== null) map.removeLayer(ovl[cur]); cur = i >= 0 ? i : null; if (cur !== null) { ovl[cur].setOpacity(+document.getElementById('op').value); ovl[cur].addTo(map); }
  document.querySelectorAll('#tbl tr[data-i]').forEach(r => r.classList.toggle('sel', +r.dataset.i === i)); }
sel.onchange = () => show(+sel.value);
document.getElementById('op').oninput = e => { if (cur !== null) ovl[cur].setOpacity(+e.target.value); };
const short = {nodata:'no data', cloud:'cloud', scl_nonveg:'SCL non-veg', ndvi_floor:'NDVI&lt;floor', footprint:'footprint', valid:'valid'};
document.getElementById('tbl').innerHTML = D.masks.length ?
  '<table><tr><th>date</th>'+keys.map(k => '<th>'+short[k]+'</th>').join('')+'</tr>' +
  D.masks.map((m,i) => '<tr data-i="'+i+'" style="cursor:pointer"><td>'+m.date+'</td>'+keys.map(k => '<td'+(k==='valid'&&m.valid_fraction_pipeline<D.min_valid_fraction?' style="color:#d1242f;font-weight:600"':'')+'>'+m.pct[k]+'</td>').join('')+'</tr>').join('') + '</table>' +
  '<div class="muted">Red valid % is below the pipeline gate ('+Math.round(D.min_valid_fraction*100)+'% per zone per date). Valid % here equals the pipeline\'s valid_fraction.</div>' :
  '<div class="muted">No cached scenes for this zone, so no mask overlay.</div>';
document.querySelectorAll('#tbl tr[data-i]').forEach(r => r.onclick = () => { sel.value = r.dataset.i; show(+r.dataset.i); });
document.getElementById('skipped').textContent = D.skipped.length ? 'No cached scene for: ' + D.skipped.join(', ') : '';
</script></body></html>
"""

if __name__ == "__main__":
    main()
