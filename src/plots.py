"""Time-series plots: mineralised vs control per belt, plus their difference."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

MINERALISED = "#2a78d6"
CONTROL = "#eb6834"
DIFF = "#52514e"
GRID = "#e4e3df"

LABELS = {
    "ndvi": "NDVI", "ndre": "NDRE", "ci_rededge_b8": "CI red-edge (B8/B5-1)",
    "ci_rededge_b7": "CI red-edge, Zhang Eq.1 (B7/B5-1)", "psri": "PSRI (B4-B2)/B6",
    "hmssi": "HMSSI (Zhang 2018)", "ndmi": "NDMI (moisture diagnostic)",
}


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_belt_timeseries(stats: pd.DataFrame, diffs: pd.DataFrame, belt: str, out_path: Path,
                         indices, stamp: str | None = None) -> Path:
    s = stats[stats.belt == belt].copy()
    d = diffs[diffs.belt == belt].copy()
    s["date"] = pd.to_datetime(s["date"])
    d["date"] = pd.to_datetime(d["date"])

    fig, axes = plt.subplots(len(indices), 2, figsize=(11, 2.3 * len(indices)), sharex=True,
                             gridspec_kw={"width_ratios": [3, 2]})
    for i, index in enumerate(indices):
        ax, axd = axes[i, 0], axes[i, 1]
        for role, color, label in (("mineralised", MINERALISED, "Mineralised zone"),
                                   ("control", CONTROL, "Matched control")):
            g = s[(s["index"] == index) & (s.role == role)].sort_values("date")
            if g.empty:
                continue
            yerr = [g["median"] - g["p25"], g["p75"] - g["median"]]
            ax.errorbar(g["date"], g["median"], yerr=yerr, color=color, marker="o", markersize=6,
                        linewidth=2, capsize=3, elinewidth=1, label=label)
        ax.set_ylabel(LABELS.get(index, index), fontsize=8)
        _style(ax)

        gd = d[d["index"] == index].sort_values("date")
        axd.axhline(0, color="#8a8984", linewidth=1)
        axd.plot(gd["date"], gd["diff_median"], color=DIFF, marker="o", markersize=6, linewidth=2)
        axd.set_ylabel("mineralised - control", fontsize=8)
        _style(axd)

    axes[0, 0].legend(fontsize=8, frameon=False, loc="best")
    axes[0, 0].set_title("Zone medians (bars: interquartile range of masked pixels)", fontsize=9, loc="left")
    axes[0, 1].set_title("Difference of medians", fontsize=9, loc="left")
    title = f"{belt}: vegetated pixels outside the buffered mine footprint"
    if stamp:
        title += f"\n{stamp}"
    fig.suptitle(title, fontsize=11, x=0.01, ha="left", color="#b00020" if stamp else "#0b0b0b")
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
