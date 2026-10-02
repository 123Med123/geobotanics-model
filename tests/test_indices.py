"""Known inputs -> known outputs for every index. Hand-computed values in comments."""
import numpy as np
import pytest

from src import indices as ix


def test_ndvi_known_value():
    # (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6
    assert ix.ndvi(np.array([0.5]), np.array([0.1]))[0] == pytest.approx(2 / 3)


def test_ndvi_array_and_range():
    b8 = np.array([0.5, 0.3, 0.1, 0.2])
    b4 = np.array([0.1, 0.3, 0.3, 0.0])
    np.testing.assert_allclose(ix.ndvi(b8, b4), [2 / 3, 0.0, -0.5, 1.0])


def test_ndre_known_value():
    # (0.5 - 0.2) / (0.5 + 0.2) = 0.3 / 0.7
    assert ix.ndre(np.array([0.5]), np.array([0.2]))[0] == pytest.approx(0.3 / 0.7)


def test_ci_rededge_b8_known_value():
    # 0.5 / 0.2 - 1 = 1.5
    assert ix.ci_rededge_b8(np.array([0.5]), np.array([0.2]))[0] == pytest.approx(1.5)


def test_ci_rededge_b7_zhang_eq1_known_value():
    # Zhang et al. 2018 Eq.1: R783/R705 - 1 -> B7/B5 - 1 = 0.45 / 0.2 - 1 = 1.25
    assert ix.ci_rededge_b7(np.array([0.45]), np.array([0.2]))[0] == pytest.approx(1.25)


def test_psri_zhang_eq2_known_value():
    # (R680 - R500) / R750 -> (B4 - B2) / B6 = (0.05 - 0.03) / 0.4 = 0.05
    assert ix.psri(np.array([0.05]), np.array([0.03]), np.array([0.4]))[0] == pytest.approx(0.05)


def test_psri_argument_order_matters():
    # Swapping B4 and B2 flips the sign - guards against a transposed call.
    assert ix.psri(np.array([0.03]), np.array([0.05]), np.array([0.4]))[0] == pytest.approx(-0.05)


def test_hmssi_zhang_eq3_known_value():
    # CI_red-edge(B7=0.45, B5=0.2) = 1.25 ; PSRI(B4=0.05, B2=0.03, B6=0.4) = 0.05 ; HMSSI = 25
    out = ix.hmssi(b07=np.array([0.45]), b05=np.array([0.2]), b04=np.array([0.05]),
                   b02=np.array([0.03]), b06=np.array([0.4]))
    assert out[0] == pytest.approx(25.0)


def test_hmssi_uses_b7_not_b8():
    # compute_all must feed B7 (not B8) into HMSSI. Make B7 and B8 very different and
    # check HMSSI matches the B7 version only.
    bands = {k: np.array([v]) for k, v in
             dict(B02=0.03, B04=0.05, B05=0.2, B06=0.4, B07=0.45, B08=0.60, B11=0.25).items()}
    out = ix.compute_all(bands)
    assert out["hmssi"][0] == pytest.approx(25.0)             # uses B7: 1.25 / 0.05
    assert out["ci_rededge_b7"][0] == pytest.approx(1.25)
    assert out["ci_rededge_b8"][0] == pytest.approx(2.0)      # 0.6/0.2 - 1: a different number
    assert out["ci_rededge_b7"][0] != out["ci_rededge_b8"][0]


def test_hmssi_nan_when_psri_near_zero():
    # B4 == B2 -> PSRI = 0 -> HMSSI undefined, must be NaN (not inf / huge).
    # Second pixel: PSRI = (0.0439 - 0.04) / 0.4 = 0.00975, below eps -> NaN.
    # Third pixel:  PSRI = (0.0441 - 0.04) / 0.4 = 0.01025, above eps -> 1.25 / 0.01025.
    out = ix.hmssi(np.full(3, 0.45), np.full(3, 0.2), np.array([0.04, 0.0439, 0.0441]),
                   np.full(3, 0.04), np.full(3, 0.4), psri_eps=0.01)
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(1.25 / 0.01025)


def test_ndmi_known_value():
    # (0.4 - 0.2) / (0.4 + 0.2) = 1/3
    assert ix.ndmi(np.array([0.4]), np.array([0.2]))[0] == pytest.approx(1 / 3)


def test_zero_denominators_give_nan_not_inf():
    z = np.array([0.0])
    assert np.isnan(ix.ndvi(z, z)[0])
    assert np.isnan(ix.ci_rededge_b8(np.array([0.5]), z)[0])
    assert np.isnan(ix.ci_rededge_b7(np.array([0.5]), z)[0])
    assert np.isnan(ix.psri(np.array([0.1]), np.array([0.05]), z)[0])


def test_nan_inputs_propagate_as_nan():
    out = ix.ndvi(np.array([np.nan, 0.5]), np.array([0.1, 0.1]))
    assert np.isnan(out[0]) and out[1] == pytest.approx(2 / 3)


def test_dn_to_reflectance_offset_and_nodata():
    dn = np.array([1500, 0, 11000])
    out = ix.dn_to_reflectance(dn, scale=10000, offset=-1000, nodata=0)
    assert out[0] == pytest.approx(0.05)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(1.0)


def test_shapes_preserved_2d():
    b8 = np.full((3, 4), 0.5)
    b4 = np.full((3, 4), 0.1)
    out = ix.ndvi(b8, b4)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out, 2 / 3)
