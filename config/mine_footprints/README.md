# Mine footprints

One GeoJSON per mineralised zone, named `<belt>.geojson` (e.g. `khetri.geojson`), with
polygons for open pits, waste dumps, tailings, plants and smelters (any CRS; it is
reprojected on load).

Why: without this mask the analysis only detects "there is a mine here". Every one of
these polygons is buffered by `masking.mine_buffer_m` and removed before any statistic
is computed. See the README's "What this does and doesn't show" section.

How to produce them:

1. `python scripts/fetch_osm_footprints.py --belt khetri` pulls OSM `landuse=quarry|industrial|landfill`
   and `man_made=spoil_heap|mineshaft|adit` inside the zone's bounding box plus a margin, via
   the Overpass API (`overpass-api.de`, which must be reachable).
2. **Always review the result against imagery.** OSM coverage in rural India is patchy.
   If a pit, dump or tailings pond is missing, draw it (QGIS or geojson.io) and add it
   to the file.
3. The pipeline refuses to run a mineralised zone whose footprint file is missing or
   empty. If a zone really has no disturbance, write a file with an empty
   FeatureCollection and `"reviewed_empty": true` at the top level.

Control zones use the same mechanism, as `<belt>_control.geojson`. That file is optional,
but any quarry or industry inside a control must be masked too.
