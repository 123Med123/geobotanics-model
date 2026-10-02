import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

from src.masking import build_analysis_mask, footprint_mask, scl_mask

# 10 x 10 raster, 10 m pixels, upper-left at (0, 100): pixel (r, c) covers x c*10..c*10+10, y 100-r*10..
T = from_origin(0, 100, 10, 10)
SHAPE = (10, 10)


def test_scl_keeps_only_vegetation():
    scl = np.array([[4, 5, 8], [9, 3, 4]])
    np.testing.assert_array_equal(scl_mask(scl), [[True, False, False], [False, False, True]])


def test_footprint_mask_with_buffer():
    fp = [box(40, 40, 60, 60)]  # 2 x 2 pixels
    assert footprint_mask(fp, 0, T, SHAPE).sum() == 4
    # 10 m buffer grows to 40 x 40 m minus rounded corners -> the 4 x 4 block, corners excluded by centre test
    m = footprint_mask(fp, 10, T, SHAPE)
    assert 12 <= m.sum() <= 16
    assert footprint_mask([], 500, T, SHAPE).sum() == 0


def test_analysis_mask_excludes_footprint_cloud_and_bare():
    zone = box(0, 0, 100, 100)                # whole raster
    scl = np.full(SHAPE, 4)
    scl[0, :] = 9                              # a row of cloud
    scl[1, :] = 5                              # a row of bare soil
    ndvi = np.full(SHAPE, 0.6)
    ndvi[2, :] = 0.1                           # below the NDVI floor
    bands = {"B04": np.full(SHAPE, 0.05)}
    fp = [box(0, 0, 30, 30)]                   # bottom-left 3 x 3 pixels
    res = build_analysis_mask(zone, scl, ndvi, bands, fp, T, ndvi_floor=0.2, mine_buffer_m=0)
    assert res.n_zone == 100
    assert res.n_cloudy == 10
    assert res.n_nonveg == 20                  # bare row + low-NDVI row
    assert res.n_footprint == 9
    assert res.n_valid == 100 - 10 - 20 - 9
    assert not res.valid[9, 0] and res.valid[9, 5]


def test_analysis_mask_respects_zone_polygon():
    zone = box(0, 50, 50, 100)                 # upper-left quarter
    scl = np.full(SHAPE, 4)
    ndvi = np.full(SHAPE, 0.6)
    res = build_analysis_mask(zone, scl, ndvi, {"B04": np.full(SHAPE, 0.05)}, [], T)
    assert res.n_zone == 25 and res.n_valid == 25
    assert res.valid[:5, :5].all() and not res.valid[5:, :].any()
