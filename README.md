# Geobotanical prospecting, Phase 1: method proof

Does a vegetation-index anomaly computed from free Sentinel-2 imagery separate **known**
base-metal mineralisation from **matched clean control** zones? And does whatever
separates them in one belt still work in a different belt?

Phase 1 is a test of a published idea on two Indian copper belts with different
climates: Khetri (semi-arid, Rajasthan) and Malanjkhand (forested, Madhya Pradesh).
It is not a detector.

## What this does and doesn't show

**Read this before any result.** It was written before any data was pulled, on purpose.

- **It tests a correlation on 2 belts with a small number of dates.** Each belt gives
  one mineralised zone and one control, seen on roughly 3-6 dates. The independent
  unit of evidence is the belt, so n = 2. Thousands of 200 m patches don't change
  that: every patch in a zone shares one label, one geology and one land-use history.
- **It is not a validated detector.** A positive result would mean "in these two places,
  on these dates, vegetation over known ore looked different from vegetation 20-50 km
  away". It would not mean the method finds unknown deposits.
- **Mines disturb vegetation for reasons that have nothing to do with ore underground.**
  Pits, waste dumps, tailings, haul roads, dust and (at Khetri) smelter emissions all
  stress or remove vegetation. We mask OSM mine/industrial footprints plus a 500 m
  buffer, and SCL non-vegetation pixels, *before* computing anything. Otherwise the
  test would only detect "there is a mine here". The mask can't remove everything:
  airborne contamination travels further than 500 m, and OSM misses features. A
  mineralised-zone anomaly is therefore still compatible with **anthropogenic
  contamination from mining**, not natural geogenic stress, and Phase 1 can't tell the
  two apart.
- **Metal stress looks like drought and salinity stress.** The 2024 Remote Sensing
  review this project follows says so explicitly. We track NDMI (a moisture index) for
  every zone and date and report whether each index's zone-vs-control difference moves
  with the moisture difference. A difference that tracks NDMI has at least as good a
  water explanation as a metal one. Salinity is not measured at all.
- **The control zones are imperfect by construction.** A control is the same shape moved
  20-50 km, chosen to have no *known* mineralisation. "No known mineralisation" is not
  "none", and no two places 30 km apart have identical soil, slope, land use and
  management. Any difference found may be one of these.
- **Transferability is expected to be poor, and a poor result is a finding.** The
  literature reports that these indices transfer badly across regions and elements.
  Khetri scrub and Malanjkhand sal forest differ in biome, rainfall and canopy. The
  cross-belt test is reported whatever it shows.
- **HMSSI was designed for rice paddies** (Zhang et al. 2018, Hunan). Its denominator
  (PSRI) sits near zero over healthy green vegetation, so the ratio is unstable. We set
  it to NaN where |PSRI| < 0.01 and report how many pixels that removes. Its behaviour
  on scrub and forest is part of what's being tested, not assumed.
- **It says nothing about uranium, thorium or rare earths.** Phase 1 uses copper belts
  only. Different elements have different uptake, toxicity and host-rock settings. Even
  a clean Phase 1 success would not carry over to those targets without its own test.
- **Any positive result needs field validation before it means anything operationally:**
  soil and leaf geochemistry, species surveys and ground spectra on the anomalous
  patches and the controls.

## How it works

```
config/zones.yaml        deposit points (with sources + verification status) and control offsets
config/pipeline.yaml     every parameter that affects a result
config/mine_footprints/  <belt>.geojson mine-disturbance polygons (required for mineralised zones)
src/zones.py             zone geometry in UTM; control = same shape translated 20-50 km
src/ingest.py            CDSE STAC search (date choice) + openEO download (clipped, 10 m)
src/indices.py           band math (unit-tested)
src/masking.py           zone ∩ SCL vegetation ∩ NDVI floor, minus buffered mine footprint
src/extract.py           per-zone-per-date stats, 200 m patch samples, reflectance sanity check
src/analysis.py          zone-vs-control differences, classifiers, cross-belt transfer test
src/plots.py             time-series plots
scripts/run_pipeline.py  end-to-end run -> results/
scripts/check_access.py  verify network + CDSE credentials before a run
scripts/fetch_osm_footprints.py  starting footprints from OpenStreetMap (review by hand!)
tests/                   index math, masking, date selection, analysis on synthetic data
```

1. **Dates.** For each belt, STAC lists every Sentinel-2 L2A acquisition over both zones
   in 2023-2024. A date is kept only if **both** the mineralised zone and its control
   are under the cloud limit **on the same day**, so the control sees the same weather.
   The pipeline takes one date per season (pre-monsoon Mar-May, monsoon Jul-Sep,
   post-monsoon mid-Oct-mid-Dec) per year.
2. **Download.** openEO clips bands B02, B04, B05, B06, B07, B08, B11 and SCL to each
   zone and resamples them to 10 m in the local UTM zone.
3. **Mask.** A pixel is used only if it is inside the zone, SCL class 4 (vegetation),
   NDVI ≥ 0.2, and outside the buffered mine footprint. A date is dropped for the whole
   belt unless both zones keep at least 30% of their pixels.
4. **Indices.**

   | Index | Formula | Sentinel-2 |
   |---|---|---|
   | NDVI | (NIR − Red)/(NIR + Red) | (B8 − B4)/(B8 + B4) |
   | NDRE | (NIR − RE1)/(NIR + RE1) | (B8 − B5)/(B8 + B5) |
   | CI red-edge (B8 variant) | NIR/RE1 − 1 | B8/B5 − 1 |
   | CI red-edge (Zhang 2018 Eq. 1) | R783/R705 − 1 | **B7**/B5 − 1 |
   | PSRI (Zhang 2018 Eq. 2) | (R680 − R500)/R750 | (B4 − B2)/B6 |
   | HMSSI (Zhang 2018 Eq. 3) | CI red-edge / PSRI | (B7/B5 − 1) / ((B4 − B2)/B6) |
   | NDMI (moisture diagnostic only) | (NIR − SWIR1)/(NIR + SWIR1) | (B8 − B11)/(B8 + B11) |

   The two CI red-edge variants use different numerator bands (B8 vs B7) and are kept
   as separate functions on purpose.
5. **Compare.** Per belt, date and index: median, mean, spatial variance and IQR per
   zone. Mineralised minus control per date, a sign test across dates, and correlation
   with the NDMI difference.
6. **Classify.** (a) The brief's baseline: per-zone-per-date mean and variance features,
   logistic regression and random forest, leave-one-date-out within a belt. (b) 200 m
   patches with per-season index medians as features, spatial-block cross-validation
   within a belt (nearby patches never split between train and test).
7. **Transfer test.** Train on one belt, test on the other, in both directions. It runs
   on raw features and on features standardised within each belt (label-free, which
   removes the forest-vs-scrub level shift). The report also shows, per feature, whether
   the mineralised-minus-control difference points the same way in both belts. If it
   doesn't, nothing can transfer on that feature.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # index math etc. - must pass first

export CDSE_CLIENT_ID=...                # OAuth client from the CDSE dashboard
export CDSE_CLIENT_SECRET=...            # (set as environment secrets; never commit)
python scripts/check_access.py           # network + STAC + openEO login

python scripts/fetch_osm_footprints.py --belt khetri
python scripts/fetch_osm_footprints.py --belt malanjkhand
#   -> review/extend config/mine_footprints/*.geojson against imagery

python scripts/run_pipeline.py           # refuses unverified zones
python scripts/run_pipeline.py --allow-unverified   # plumbing test; every output stamped UNVERIFIED
```

Outputs land in `results/`: `report.md`, `plots/<belt>_timeseries.png`, and CSVs
(`zone_date_stats`, `pair_differences`, `difference_summary`, `classifier_zone_date`,
`classifier_patch`, `direction_agreement`, `masking`). Downloads are cached in
`data/raw/` and reused; `--skip-download` re-runs the analysis offline.

Network access needed: `*.dataspace.copernicus.eu` (STAC, openEO, and
`identity.dataspace.copernicus.eu` for OAuth tokens), plus `overpass-api.de` for the OSM
footprint script.

### Stops instead of guessing

The pipeline stops, and does not substitute anything, when:
- a zone isn't marked verified in `config/zones.yaml` (unless `--allow-unverified`);
- a mineralised zone has no reviewed mine-footprint file;
- a belt has fewer than 3 dates usable for both zones (it names the belt; switching to
  a reserve belt is the project owner's decision);
- reflectance looks like the L2A offset is wrong (see `config/pipeline.yaml`).

## Status (2026-10-02)

- [ ] Zone coordinates verified against primary sources. None are yet: see the notes in `config/zones.yaml`.
- [ ] Control zones checked against GSI Bhukosh for mapped mineralisation and land-cover match.
- [x] Mine footprints fetched from OSM (khetri 7 polygons, malanjkhand 4, khetri_control 1, malanjkhand_control 0).
- [ ] Mine footprints reviewed against imagery. **Not done.** Nothing is marked `reviewed_empty`. No OSM pit lies within 3 km of the Kolihan or Chandmari points, so Khetri is certainly incomplete.
- [x] L2A offset on CDSE openEO confirmed on real scenes: `reflectance.offset: 0` is correct (openEO already applies BOA_ADD_OFFSET). The `extract.py` check passed on all 22 downloaded scenes; blue-band 1st percentile was -0.0003 to 0.055 (the check fails above 0.08), with no excess of negative reflectance. It is a coarse heuristic, not a calibration.
- [ ] First run. **Not done.** The `--allow-unverified` plumbing test downloaded the scenes and then stopped at the date gate for both belts (details below). No `results/` were produced.

### Plumbing test, 2026-10-02 (UNVERIFIED coordinates; nothing here is evidence)

The pipeline stopped, as designed, and nothing was substituted: no reserve belt, no threshold change.

- **khetri: 2 of 6 candidate dates usable (need 3).** Failures were low vegetation fractions after the SCL and NDVI >= 0.2 mask, not clouds and not the mine mask (footprints plus the 500 m buffer cover about 10% of the zone). The control zone was 1.8%, 8% and 27% valid on three dates; the mineralised zone was 12-19% valid on the two March dates. This is the land-cover mismatch flagged for the Khetri control in `config/zones.yaml`, in a semi-arid belt. The table does not separate the SCL-vegetation effect from the NDVI floor, so which of the two removes the pixels is not established.
- **malanjkhand: 0 of 5 candidate dates usable (need 3).** The control was 41-99% valid. The mineralised zone was 1-28% valid: SCL non-vegetation removed 20-96% of it and the footprint mask a further 3-32%. The OSM "Malanjkhand Copper Mine" polygon is about 976 ha, and with the 500 m buffer the footprint covers about 66% of the zone, so the zone as drawn is mostly mine. That is the case the masking section above warns about.
- Both belts have only 5-6 candidate dates under the 30% tile-cloud limit for 2023-2024.
- Off-swath area (raster nodata) was 0-16% of the Khetri mineralised zone on some dates and 0% elsewhere. It did not decide any date.
- A run that stops at the date gate now still writes `results/masking.csv` (per zone and date, with a `used` flag), so the numbers behind the stop are kept. It writes nothing else in that case: no report, no statistics. A stop before masking (no dates, missing footprint file) has no mask table to write. A failed reflectance check (`ReflectanceScaleError`) is now a stop as well, and it ends the whole run, because `reflectance.*` is shared by every belt: no other belt is processed and no results are written, even for a belt that finished earlier. Only the mask table for the dates finished so far is kept.

### Code fixes made during the plumbing test (no `config/pipeline.yaml` change)

- `src/ingest.py`: the STAC date search now queries per quarter with retries. The CDSE gateway returned 504s part-way through one 2-year paged query. The set of items searched is unchanged.
- `src/extract.py`: pixels equal to the raster's own nodata tag (int16 -32768, from off-swath areas) are now NaN. Before, they became reflectance -3.2768 and blinded the blue-band percentile check on scenes with >= 1% fill. Valid-pixel masks are unchanged. Covered by `tests/test_extract.py`.

### Environment note

From the author's Windows machine, `requests` could not resolve `overpass-api.de` with the default dual-stack lookup and Overpass rejected the default `python-requests` User-Agent (406). Forcing IPv4 and sending a descriptive User-Agent worked; `scripts/fetch_osm_footprints.py` itself is unchanged.

## Data and licences

- Sentinel-2 L2A: Copernicus Data Space Ecosystem, free and open data licence (commercial use allowed). Google Earth Engine is deliberately **not** used (its free tier excludes work for a company).
- OpenStreetMap footprints: © OpenStreetMap contributors, ODbL.
- GSI Bhukosh (bhukosh.gsi.gov.in): consulted manually for geological context. It is not scraped.

## References

- Liu, M. et al. (2018). Heavy metal-induced stress in rice crops detected using multi-temporal Sentinel-2 satellite images. *Sci. Total Environ.* 637-638, 18-29.
- Zhang, Z., Liu, M., Liu, X., Zhou, G. (2018). A new vegetation index based on multitemporal Sentinel-2 images for discriminating heavy metal stress levels in rice. *Sensors* 18(7), 2172.
- Monitoring heavy metals and metalloids in soils and vegetation by remote sensing: a review. *Remote Sensing* 16(17), 3221 (2024).
