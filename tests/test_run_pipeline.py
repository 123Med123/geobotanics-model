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
