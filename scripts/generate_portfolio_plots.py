#!/usr/bin/env python3
"""Regenerate every README figure from the frozen values under docs/figures/data/.

Six PNGs are produced with one style: training_convergence, efficiency,
encoder_connector_interaction, structural_augmentation, confirmatory_results and
surface_composition. Every plotted number comes from a values JSON that names its
source row; nothing is recomputed here. The encoder colours are a colour-vision-
deficiency-validated trio and the same mapping is used in every figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ENCODERS = ("clip", "dinov2", "ijepa")
COLOURS = {"clip": "#2a78d6", "dinov2": "#e07020", "ijepa": "#1a9c6b"}
MARKERS = {"clip": "o", "dinov2": "s", "ijepa": "^"}
LABELS = {"clip": "CLIP", "dinov2": "DINOv2", "ijepa": "I-JEPA"}
CONNECTORS = ("mlp256", "qformer32", "pool32")
CONNECTOR_LABELS = {"mlp256": "MLP256", "qformer32": "Q-Former32", "pool32": "Pool32"}
RAMP3 = ("#86b6ef", "#5598e7", "#256abf")
RAMP4 = ("#86b6ef", "#5598e7", "#256abf", "#104281")
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DPI = 170
FIGURES = (
    "training_convergence.png", "efficiency.png", "encoder_connector_interaction.png",
    "structural_augmentation.png", "confirmatory_results.png", "surface_composition.png",
)


def apply_style() -> None:
    """One sans-serif, recessive-chrome style for every public figure."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.labelsize": 11, "axes.labelcolor": INK2,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.fontsize": 10, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def grid(ax, axis: str = "y") -> None:
    """Hairline solid gridlines drawn beneath the data."""
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def subtitle(ax, *lines: str) -> None:
    """Secondary-ink lines under the title naming units and evidence tier."""
    ax.text(0.0, 1.015, "\n".join(lines), transform=ax.transAxes, fontsize=9.5, color=INK2,
            va="bottom", linespacing=1.35)


def title(ax, text: str, lines: int = 1) -> None:
    """Left-aligned bold title with room for the subtitle lines beneath it."""
    ax.set_title(text, loc="left", pad=22 + 14 * (lines - 1))


def spread_labels(values: list[float], min_gap: float) -> list[float]:
    """Push label positions apart so neighbouring end labels never overlap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [values[i] for i in order]
    for k in range(1, len(placed)):
        if placed[k] - placed[k - 1] < min_gap:
            placed[k] = placed[k - 1] + min_gap
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = placed[k]
    return out


def save(fig, out: Path) -> None:
    """Write a PNG without a software-version stamp so regeneration is stable."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


def plot_convergence(payload: dict, out: Path) -> None:
    """Matched corrected-protocol MLP256 tune-500 accuracy by epoch."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for encoder in ENCODERS:
        rows = payload["encoders"][encoder]
        epochs = [row["epoch"] for row in rows]
        accuracy = [row["accuracy"] for row in rows]
        ax.plot(epochs, accuracy, marker=MARKERS[encoder], linewidth=2.0, markersize=7,
                color=COLOURS[encoder], label=LABELS[encoder], markeredgecolor="white",
                markeredgewidth=1.0)
        ax.scatter([epochs[-1]], [accuracy[-1]], s=120, facecolors="white",
                   edgecolors=COLOURS[encoder], linewidths=2.0, zorder=4)
        ax.text(epochs[-1] + 0.08, accuracy[-1], f"{accuracy[-1]:.1f}", va="center",
                fontsize=9.5, color=INK2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.8, 5.45)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Tune-500 full-string exact match (%)")
    title(ax, "Corrected-protocol MLP256 development trajectory", lines=2)
    subtitle(ax, "500 development questions, seed 42; y-axis truncated",
             "Open marker: fixed reported endpoint (epoch 5); no per-encoder epoch selection")
    grid(ax)
    ax.legend(loc="lower right", ncol=3)
    fig.tight_layout()
    save(fig, out)


def plot_efficiency(payload: dict, out: Path) -> None:
    """Controlled per-stage inference latency benchmark."""
    rows = payload["configurations"]
    stage_keys = ("vision_ms", "connector_ms", "prefill_ms", "decode_ms")
    stage_labels = ("Vision encoder", "Connector", "Text prefill", "Decode")
    labels = []
    for row in rows:
        encoder, connector = row["configuration"].split(":")
        labels.append(f"{LABELS[encoder]} / {CONNECTOR_LABELS[connector]}")
    y_positions = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    for y, row in zip(y_positions, rows):
        if row["status"] != "measured":
            ax.plot(0, y, marker="x", markersize=8, color=INK2, markeredgewidth=1.5)
            ax.text(1.7, y, "not constructed", va="center", fontsize=9.5, color=INK2)
            continue
        left = 0.0
        for key, colour, label in zip(stage_keys, RAMP4, stage_labels):
            ax.barh(y, row[key], left=left, height=0.6, color=colour, edgecolor="white",
                    linewidth=1.2, label=label if y == y_positions[0] else None)
            left += row[key]
        ax.text(left + 0.6, y, f"{left:.1f} ms", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Median GPU model-inference latency (ms)")
    ax.set_xlim(0, 58)
    title(ax, "Controlled inference benchmark")
    subtitle(ax, "One RTX PRO 6000, bf16, batch 1, greedy decoding; medians over 600 "
                 "measurements per configuration")
    grid(ax, axis="x")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, out)


def plot_interaction(payload: list[dict], out: Path) -> None:
    """Six Test-Dev cells as a slope chart with image-clustered intervals."""
    by_name = {row["name"]: row for row in payload}
    key = {"clip": "CLIP", "dinov2": "DINOv2", "ijepa": "I-JEPA"}
    offsets = {"clip": -0.035, "dinov2": 0.0, "ijepa": 0.035}
    cost_shift = {"clip": 0.55, "dinov2": -0.65, "ijepa": 0.5}
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ends = {}
    for encoder in ENCODERS:
        cells = [by_name[f"testdev_full_string_exact_match_{key[encoder]}_{conn}"]
                 for conn in ("MLP256", "QFormer32")]
        xs = [0 + offsets[encoder], 1 + offsets[encoder]]
        ys = [cell["value"] for cell in cells]
        err = [[cell["value"] - cell["ci95"][0] for cell in cells],
               [cell["ci95"][1] - cell["value"] for cell in cells]]
        ax.errorbar(xs, ys, yerr=err, color=COLOURS[encoder], linewidth=2.0, elinewidth=1.3,
                    capsize=4, marker=MARKERS[encoder], markersize=8, markeredgecolor="white",
                    markeredgewidth=1.0, label=LABELS[encoder], zorder=3)
        ends[encoder] = ys[1]
        cost = by_name[f"testdev_contrast_Connector_{key[encoder]}: MLP256 minus Q-Former32"]
        ax.text(0.5, (ys[0] + ys[1]) / 2 + cost_shift[encoder], f"−{cost['value']:.2f}",
                ha="center", va="center", fontsize=9.5, color=INK2,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5}, zorder=4)
    label_y = spread_labels([ends[e] for e in ENCODERS], min_gap=0.75)
    for encoder, y in zip(ENCODERS, label_y):
        ax.text(1.1, y, LABELS[encoder], va="center", fontsize=10.5, color=INK)
    ax.set_xticks([0, 1], ["MLP256", "Q-Former32"])
    ax.set_xlim(-0.35, 1.45)
    ax.set_ylim(35, 48.5)
    ax.set_ylabel("Full-string exact match (%)")
    ax.set_xlabel("Connector configuration")
    title(ax, "GQA Test-Dev: three encoders × two connectors", lines=2)
    subtitle(ax, "12,578 questions over 398 images; image-clustered 95% intervals; y-axis truncated",
             "Line labels give the MLP256 − Q-Former32 change for each encoder")
    grid(ax)
    ax.legend(loc="lower left", ncol=3)
    fig.tight_layout()
    save(fig, out)


def plot_structural(payload: list[dict], out: Path) -> None:
    """Forest plot of the structural-text contrasts on Test-Dev."""
    rows = [row for row in payload if "ci95" in row]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows):
        primary = row.get("role") == "primary"
        colour = COLOURS["clip"] if primary else MUTED
        lo, hi = row["ci95"]
        ax.plot([lo, hi], [y, y], color=colour, linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.plot(row["value"], y, marker="D" if primary else "o", markersize=9 if primary else 8,
                color=colour, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        ax.text(hi + 0.03, y, f"{row['value']:+.2f} [{lo:+.2f}, {hi:+.2f}]", va="center",
                fontsize=9.5, color=INK2)
        if primary:
            ax.text(lo - 0.03, y, "non-detection", va="center", ha="right", fontsize=9.5,
                    color=INK2)
    ax.axvline(0, color=AXIS, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(ys, [row["name"] + (" (primary)" if row.get("role") == "primary" else "")
                       for row in rows])
    ax.set_xlim(-1.35, 1.35)
    ax.set_xlabel("Difference in official-rule accuracy (percentage points)")
    title(ax, "Structural-text study on GQA Test-Dev", lines=2)
    subtitle(ax, "12,578 questions over 398 images; official-rule accuracy",
             "Paired, image-clustered 95% intervals; the primary contrast is highlighted")
    grid(ax, axis="x")
    fig.tight_layout()
    save(fig, out)


def plot_confirmatory(payload: dict, out: Path) -> None:
    """Grouped bars of the confirmatory-surface cells with the text-only floor."""
    cells = {(cell["encoder"], cell["connector"]): cell["value"] for cell in payload["cells"]}
    unavailable = {(cell["encoder"], cell["connector"]) for cell in payload["unavailable"]}
    width, gap = 0.24, 0.02
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    for i, encoder in enumerate(ENCODERS):
        xs = [g + (i - 1) * (width + gap) for g in range(len(CONNECTORS))]
        for x, connector in zip(xs, CONNECTORS):
            if (encoder, connector) in unavailable:
                ax.plot(x, 0.9, marker="x", markersize=8, color=INK2, markeredgewidth=1.5)
                ax.text(x, 3.0, "not\ntrained", ha="center", va="bottom", fontsize=8.5, color=INK2)
                continue
            value = cells[(encoder, connector)]
            ax.bar(x, value, width, color=COLOURS[encoder], edgecolor="white", linewidth=1.2,
                   label=LABELS[encoder] if connector == "mlp256" else None, zorder=3)
            ax.text(x, value + 0.8, f"{value:.2f}", ha="center", va="bottom", fontsize=9.5,
                    color=INK2)
    floor = payload["floor"]["value"]
    ax.axhline(floor, color=INK2, linewidth=1.2, linestyle="--", zorder=2,
               label=f"{payload['floor']['label']} ({floor:.2f})")
    ax.set_xticks(range(len(CONNECTORS)), [CONNECTOR_LABELS[c] for c in CONNECTORS])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 60)
    ax.set_ylabel("Full-string exact match (%)")
    ax.set_xlabel("Connector configuration")
    title(ax, "Confirmatory surface: three encoders × three connectors")
    subtitle(ax, f"{payload['n_questions']:,} frozen questions, never used for selection; "
                 "seed 42, epoch 5; point estimates (no interval stored per cell)")
    grid(ax)
    ax.legend(loc="upper right", ncol=4)
    fig.tight_layout()
    save(fig, out)


def plot_composition(payload: dict, out: Path) -> None:
    """Share of each reasoning category on the three corrected-protocol surfaces."""
    categories = payload["categories"]
    surfaces = payload["surfaces"]
    width, gap = 0.26, 0.02
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for i, (surface, colour) in enumerate(zip(surfaces, RAMP3)):
        xs = [c + (i - 1) * (width + gap) for c in range(len(categories))]
        shares = [100 * surface["counts"][cat] / surface["n"] for cat in categories]
        ax.bar(xs, shares, width, color=colour, edgecolor="white", linewidth=1.2,
               label=surface["label"], zorder=3)
        for x, share in zip(xs, shares):
            ax.text(x, share + 0.6, f"{share:.0f}", ha="center", va="bottom", fontsize=8.5,
                    color=INK2)
    ax.set_xticks(range(len(categories)), categories)
    ax.set_ylim(0, 48)
    ax.set_ylabel("Share of questions (%)")
    ax.set_xlabel("Reasoning category")
    title(ax, "Reasoning-category composition of the frozen surfaces")
    subtitle(ax, "Same ordered category rule on every surface; count is zero everywhere, "
                 "so no counting claim is made")
    grid(ax)
    ax.legend(loc="upper center", ncol=3)
    fig.tight_layout()
    save(fig, out)


def main() -> None:
    """Load frozen values and write all six public figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "docs/figures/data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    apply_style()
    load = lambda name: json.loads((args.data_dir / name).read_text())  # noqa: E731
    plot_convergence(load("training_convergence.json"), args.out_dir / "training_convergence.png")
    plot_efficiency(load("efficiency.json"), args.out_dir / "efficiency.png")
    plot_interaction(load("encoder_connector_interaction.json"),
                     args.out_dir / "encoder_connector_interaction.png")
    plot_structural(load("structural_augmentation.json"),
                    args.out_dir / "structural_augmentation.png")
    plot_confirmatory(load("confirmatory_results.json"), args.out_dir / "confirmatory_results.png")
    plot_composition(load("surface_composition.json"), args.out_dir / "surface_composition.png")
    for name in FIGURES:
        digest = hashlib.sha256((args.out_dir / name).read_bytes()).hexdigest()
        print(f"{digest}  {args.out_dir / name}")


if __name__ == "__main__":
    main()
