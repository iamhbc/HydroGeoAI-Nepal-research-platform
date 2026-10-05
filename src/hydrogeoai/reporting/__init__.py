"""Paper-ready tables and figures (Section 32). Matplotlib only, headless backend."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a6f97", "#e07a1f", "#3a8a5c", "#b5403a", "#7b5ea7", "#8c6d46", "#c25b9a", "#6b7280"]
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10})


def write_table(df: pd.DataFrame, path: Path, floatfmt: str = ".4f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    try:
        md = df.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:  # tabulate optional
        md = df.to_string(index=False)
    path.with_suffix(".md").write_text(md)


def write_json(obj, path: Path) -> None:
    from ..experiments.registry import _j
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(obj), indent=2))


def ablation_bars(df: pd.DataFrame, path: Path, metric="auprc"):
    modes = list(dict.fromkeys(df["mode"]))
    variants = list(dict.fromkeys(df["variant"]))
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(variants), 3.2))
    w = 0.8 / len(modes)
    for i, m in enumerate(modes):
        d = df[df["mode"] == m].set_index("variant").reindex(variants)
        x = np.arange(len(variants)) + i * w
        err = np.vstack([d[metric] - d[f"{metric}_ci_low"], d[f"{metric}_ci_high"] - d[metric]]).clip(0)
        ax.bar(x, d[metric], w, yerr=err, label=m, color=PALETTE[i % len(PALETTE)], capsize=2)
    if "base_rate" in df:
        ax.axhline(df.base_rate.mean(), color="k", lw=0.8, ls="--", label="base rate")
    ax.set_xticks(np.arange(len(variants)) + w * (len(modes) - 1) / 2, variants, rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title("Ablation study (95% station-block bootstrap CI)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def reliability_plot(curves: dict[str, dict], path: Path):
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    mx = 0.0
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["mean_predicted"], c["observed_frequency"], "o-", ms=3, label=name, color=PALETTE[i % 8])
        mx = max(mx, max(c["mean_predicted"] + c["observed_frequency"]))
    ax.set_xlim(0, min(1, mx * 1.1 + 0.01))
    ax.set_ylim(0, min(1, mx * 1.1 + 0.01))
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability (test)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def line_plot(df: pd.DataFrame, x: str, y: str, hue: str, path: Path, title: str, ylabel: str | None = None):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for i, (k, g) in enumerate(df.groupby(hue, sort=False)):
        ax.plot(g[x], g[y], "o-", ms=3, label=str(k), color=PALETTE[i % 8])
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def station_map(stations: pd.DataFrame, values: pd.Series, basins: dict | None, path: Path, title: str,
                cmap="viridis", label=""):
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    if basins:
        for f in basins["features"]:
            polys = [f["geometry"]["coordinates"]] if f["geometry"]["type"] == "Polygon" else f["geometry"]["coordinates"]
            for poly in polys:
                xy = np.asarray(poly[0])
                ax.plot(xy[:, 0], xy[:, 1], color="#9ca3af", lw=0.6)
    st = stations.set_index("station_id")
    v = values.reindex(st.index)
    sc = ax.scatter(st.lon, st.lat, c=v, s=36, cmap=cmap, edgecolor="k", lw=0.4)
    ax.scatter(st.lon[v.isna()], st.lat[v.isna()], s=20, facecolor="white", edgecolor="#6b7280", lw=0.6)
    fig.colorbar(sc, ax=ax, shrink=0.8, label=label)
    ax.set_xlabel("Longitude (EPSG:4326)")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect(1 / np.cos(np.deg2rad(28)))
    fig.savefig(path)
    plt.close(fig)


def bar(series: pd.Series, path: Path, title: str, xlabel: str, note: str | None = None):
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(4.4, 0.25 * len(s) + 1.0))
    ax.barh(s.index.astype(str), s.values, color=PALETTE[0])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if note:
        fig.subplots_adjust(bottom=0.22)
        fig.text(0.01, 0.01, note, fontsize=6, style="italic", wrap=True)
    fig.savefig(path)
    plt.close(fig)
