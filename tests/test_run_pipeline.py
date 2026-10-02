import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("run_pipeline_script", ROOT / "scripts/run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_pair(belt):
    zones = [SimpleNamespace(zone_id=f"{belt}", verified=True), SimpleNamespace(zone_id=f"{belt}_control", verified=True)]
    return SimpleNamespace(belt=belt, zones=zones)


def _masks(belt):
    return pd.DataFrame([
        {"belt": belt, "zone_id": belt, "date": "2023-01-01", "valid_fraction": 0.1, "season": "pre_monsoon", "used": False},
        {"belt": belt, "zone_id": f"{belt}_control", "date": "2023-01-01", "valid_fraction": 0.9, "season": "pre_monsoon", "used": False},
    ])


def test_masking_csv_is_written_when_every_belt_stops(tmp_path, monkeypatch):
    rp = _load_script()
    monkeypatch.setattr(rp, "load_pairs", lambda *_a, **_k: {"b1": _fake_pair("b1")})

    def stop(pair, *_a, **_k):
        raise rp.PipelineStop("b1: only 0 date(s) usable after masking (need 3).", masks=_masks("b1"))

    monkeypatch.setattr(rp, "run_belt", stop)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--skip-download", "--results-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        rp.main()
    assert "No belt could be analysed" in str(exc.value)
    out = pd.read_csv(tmp_path / "masking.csv")
    assert len(out) == 2 and not out["used"].any()
    assert not (tmp_path / "report.md").exists()  # no analysis output was produced


def test_reflectance_error_in_process_scene_becomes_a_stop_and_ends_the_run(tmp_path, monkeypatch):
    """A ReflectanceScaleError on the 2nd belt must stop the whole run, keep the masks seen so far
    (including the finished 1st belt) and write no results."""
    rp = _load_script()
    pairs = {"b1": _fake_pair("b1"), "b2": _fake_pair("b2")}
    monkeypatch.setattr(rp, "load_pairs", lambda *_a, **_k: pairs)
    calls = []

    def fake_run_belt(pair, *_a, **_k):
        calls.append(pair.belt)
        if pair.belt == "b1":
            return pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [1]}), _masks("b1").assign(used=True)
        raise rp.ReflectanceStop("b2: implausible reflectance (B02 1st percentile=0.121).", masks=_masks("b2"))

    monkeypatch.setattr(rp, "run_belt", fake_run_belt)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--skip-download", "--results-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        rp.main()
    assert "reflectance check" in str(exc.value)
    assert calls == ["b1", "b2"]
    assert sorted(pd.read_csv(tmp_path / "masking.csv")["belt"].unique()) == ["b1", "b2"]
    assert not (tmp_path / "report.md").exists() and not (tmp_path / "zone_date_stats.csv").exists()


def test_reflectance_stop_skips_later_belts(tmp_path, monkeypatch):
    rp = _load_script()
    pairs = {"b1": _fake_pair("b1"), "b2": _fake_pair("b2")}
    monkeypatch.setattr(rp, "load_pairs", lambda *_a, **_k: pairs)
    calls = []

    def fake_run_belt(pair, *_a, **_k):
        calls.append(pair.belt)
        raise rp.ReflectanceStop("b1: implausible reflectance")

    monkeypatch.setattr(rp, "run_belt", fake_run_belt)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--skip-download", "--results-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        rp.main()
    assert calls == ["b1"]  # b2 was never processed


def test_run_belt_converts_reflectance_scale_error_to_reflectance_stop(tmp_path, monkeypatch):
    """Exercise the real run_belt: process_scene raising ReflectanceScaleError must surface as ReflectanceStop."""
    rp = _load_script()
    from src.extract import ReflectanceScaleError

    z = SimpleNamespace(zone_id="b1", role="mineralised", epsg=32643)
    c = SimpleNamespace(zone_id="b1_control", role="control", epsg=32643)
    pair = SimpleNamespace(belt="b1", zones=[z, c], mineralised=z, control=c)
    cfg = {"dates": {"max_scene_cloud_pct": 30, "min_valid_fraction": 0.3, "min_dates_per_belt": 3},
           "masking": {"footprints_dir": "config/mine_footprints"},
           "reflectance": {"scale": 10000, "offset": 0, "nodata_dn": 0},
           "analysis": {"hmssi_psri_eps": 0.01, "patch_size_px": 20, "min_patch_valid_fraction": 0.5},
           "cdse": {}}
    dates = pd.DataFrame([{"year": 2023, "season": "pre_monsoon", "date": pd.Timestamp("2023-03-05").date()}])
    scene_file = tmp_path / "s.tif"
    scene_file.write_bytes(b"x")
    scene_file.with_suffix(".json").write_text('{"bands": []}')
    monkeypatch.setattr(rp, "choose_dates", lambda *a, **k: dates)
    monkeypatch.setattr(rp, "load_footprints", lambda *a, **k: [])
    monkeypatch.setattr(rp.ingest, "scene_path", lambda *a, **k: scene_file)
    monkeypatch.setattr(rp, "read_scene", lambda *a, **k: object())

    def boom(*a, **k):
        raise ReflectanceScaleError("b1_control 2023-03-05: implausible reflectance (B02 1st percentile=0.121)")

    monkeypatch.setattr(rp, "process_scene", boom)
    with pytest.raises(rp.ReflectanceStop) as exc:
        rp.run_belt(pair, cfg, tmp_path, None, skip_download=True)
    assert "implausible reflectance" in str(exc.value)
    assert "decision for the project owner" in str(exc.value)
    assert exc.value.masks is None  # failed on the very first scene: nothing finished yet


def test_stop_without_masks_writes_no_masking_csv(tmp_path, monkeypatch):
    rp = _load_script()
    monkeypatch.setattr(rp, "load_pairs", lambda *_a, **_k: {"b1": _fake_pair("b1")})

    def stop(pair, *_a, **_k):
        raise rp.PipelineStop("b1: no footprint file")  # stopped before any masking happened

    monkeypatch.setattr(rp, "run_belt", stop)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--skip-download", "--results-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        rp.main()
    assert not (tmp_path / "masking.csv").exists()
