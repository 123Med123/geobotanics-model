import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.extract import REFLECTANCE_BANDS, read_scene

BANDS = list(REFLECTANCE_BANDS) + ["SCL"]


def _write_scene(path, fill_value=-32768):
    data = np.full((len(BANDS), 4, 4), 1500, dtype=np.int16)
    data[:, 0, 0] = fill_value          # off-swath pixel in every band, incl. SCL
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=len(BANDS), dtype="int16",
                       crs="EPSG:32643", transform=from_origin(0, 40, 10, 10), nodata=fill_value) as dst:
        dst.write(data)


def test_read_scene_masks_the_files_own_nodata_value(tmp_path):
    p = tmp_path / "s.tif"
    _write_scene(p)
    scene = read_scene(p, BANDS, scale=10000, offset=0, nodata_dn=0)
    for b in REFLECTANCE_BANDS:
        assert np.isnan(scene.bands[b][0, 0])
        assert np.isclose(scene.bands[b][1, 1], 0.15)
    assert np.isfinite(np.concatenate([scene.bands[b].ravel() for b in REFLECTANCE_BANDS])).sum() == 15 * len(REFLECTANCE_BANDS)


def test_read_scene_still_masks_configured_nodata_dn(tmp_path):
    p = tmp_path / "s.tif"
    _write_scene(p, fill_value=0)
    scene = read_scene(p, BANDS, scale=10000, offset=0, nodata_dn=0)
    assert np.isnan(scene.bands["B04"][0, 0])
