#!/usr/bin/env python3
"""
Abliteration experiment results dashboard.

Per-model scatter plots: refusals vs KL divergence. Each experiment tagged
by color, baselines shown, shared axes labels, legend outside plots.

Usage:
    python scripts/dashboard.py
    python scripts/dashboard.py --output assets/dashboard.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_ORDER = [
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
]

MODEL_LABELS = {
    "google/gemma-4-E2B-it": "E2B (2.3B)",
    "google/gemma-4-E4B-it": "E4B (4.5B)",
    "google/gemma-4-26B-A4B-it": "26B MoE",
    "google/gemma-4-31B-it": "31B",
}

BASELINES = {
    "google/gemma-4-E2B-it": 98,
    "google/gemma-4-E4B-it": 99,
    "google/gemma-4-26B-A4B-it": 98,
    "google/gemma-4-31B-it": 100,
}

# Distinct colors per experiment type
TAG_PALETTE = {
    # Biprojection variants
    "bp-100pct": "#2196F3",
    "bp-cosmic-100pct": "#1565C0",
    "bp-stripped-markers": "#1976D2",
    "bp-70pct": "#42A5F5",
    "bp-cosmic-70pct": "#64B5F6",
    "bp-v2": "#90CAF9",
    "bp-scale1.3": "#7E57C2",
    "bp-scale1.5-all": "#9575CD",
    "bp-no-winsorize": "#FF7043",
    "bp-position-aware-50w": "#78909C",
    "bp-allayers": "#26A69A",
    "bp-debug": "#BDBDBD",
    "bp-default": "#E0E0E0",
    # EGA
    "ega-full": "#4CAF50",
    "moe-bp-100pct": "#FF9800",
    # E4B
    "e4b-bp-100pct": "#2196F3",
    "e4b-bp-70pct": "#42A5F5",
    # 31B
    "31b-bp-100pct": "#2196F3",
    # Task vector
    "tv": "#EF5350",
}

MARKERS = {
    "biprojection": "o",
    "ega": "s",
    "task_vector_negation": "^",
}


def tag_color(tag: str) -> str:
    """Get color for a tag, with fallback."""
    if tag in TAG_PALETTE:
        return TAG_PALETTE[tag]
    if tag.startswith("tv"):
        return TAG_PALETTE["tv"]
    return "#888888"


def load_experiments(experiments_dir: Path) -> dict[str, list[dict]]:
    """Load experiments grouped by model."""
    by_model: dict[str, list[dict]] = {}

    for f in sorted(experiments_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        if "refusals" not in data and "results" not in data:
            continue

        # Task vector multi-result
        if "results" in data and isinstance(data["results"], list) and data["results"] and "alpha" in data["results"][0]:
            model = data.get("instruct_model", data.get("model", "unknown"))
            for r in data["results"]:
                by_model.setdefault(model, []).append({
                    "tag": f"tv α={r['alpha']}",
                    "method": "task_vector_negation",
                    "refusals": r["refusals"],
                    "n_prompts": r["n_prompts"],
                    "kl_divergence": r["kl_divergence"],
                })
            continue

        if "refusals" not in data:
            continue

        tag = data.get("tag", f.stem)
        model = data.get("model", "unknown")

        # Skip duplicates
        if any(s in tag for s in ("reeval", "audit", "combined", "save", "export")):
            continue
        if "refusal-audit" in f.name or "sanity-check" in f.name:
            continue

        by_model.setdefault(model, []).append({
            "tag": tag,
            "method": data.get("method", "unknown"),
            "refusals": data["refusals"],
            "n_prompts": data.get("n_prompts", 100),
            "kl_divergence": data.get("kl_divergence", 0),
        })

    return by_model


def generate_dashboard(by_model: dict[str, list[dict]], output_path: Path) -> None:
    """Generate per-model scatter plots with seaborn styling."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)

    models_with_data = [m for m in MODEL_ORDER if m in by_model]
    n = len(models_with_data)
    if n == 0:
        return

    cols = 2
    rows = (n + 1) // 2

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(14, 6 * rows),
        gridspec_kw={"hspace": 0.25, "wspace": 0.25},
        squeeze=False,
    )

    # Collect all legend entries across all plots (deduplicated, ordered)
    legend_entries: dict[str, tuple[str, str, str]] = {}  # tag -> (color, marker, label)

    for idx, model_id in enumerate(models_with_data):
        row, col_idx = divmod(idx, cols)
        ax = axes[row][col_idx]
        results = by_model[model_id]
        label = MODEL_LABELS.get(model_id, model_id)

        best = min(results, key=lambda r: (r["refusals"], r["kl_divergence"]))

        # Baseline
        bl = BASELINES.get(model_id, 98)
        ax.scatter(0, bl, color="#999", marker="X", s=100, zorder=5, linewidths=1.5)
        legend_entries["baseline"] = ("#999", "X", f"Baseline (unmodified)")

        for r in results:
            color = tag_color(r["tag"])
            marker = MARKERS.get(r["method"], "o")
            is_best = r is best

            ax.scatter(
                r["kl_divergence"], r["refusals"],
                c=color, marker=marker,
                s=140 if is_best else 70,
                edgecolors="gold" if is_best else "white",
                linewidths=2.5 if is_best else 0.8,
                zorder=5 if is_best else 3,
                alpha=1.0 if is_best else 0.8,
            )

            # Build legend entry
            legend_key = r["tag"]
            if legend_key.startswith("tv"):
                legend_key = "Task vector"
                legend_entries[legend_key] = (TAG_PALETTE["tv"], "^", "Task vector negation")
            else:
                nice_name = r["tag"].replace("bp-", "").replace("e4b-", "").replace("31b-", "")
                legend_entries[legend_key] = (color, marker, r["tag"])

        ax.set_title(f"{label}  ({len(results)} runs)", fontsize=12, fontweight="bold", pad=10)

        # Only show axis labels on edges
        if row == rows - 1:
            ax.set_xlabel("KL Divergence")
        if col_idx == 0:
            ax.set_ylabel("Refusals / 100")

        ax.tick_params(labelsize=9)
        sns.despine(ax=ax)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    # Build legend from collected entries — deduplicate, sort
    handles = []
    seen = set()
    # Baseline first
    if "baseline" in legend_entries:
        c, m, lab = legend_entries.pop("baseline")
        handles.append(plt.scatter([], [], c=c, marker=m, s=60, label=lab, linewidths=1.5))

    # Best marker
    handles.append(plt.scatter([], [], c="white", marker="o", s=80, edgecolors="gold", linewidths=2.5, label="Best result"))

    # Group: biprojection variants
    for key in sorted(legend_entries):
        c, m, lab = legend_entries[key]
        if key not in seen:
            handles.append(plt.scatter([], [], c=c, marker=m, s=50, label=lab, edgecolors="white", linewidths=0.5))
            seen.add(key)

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        fontsize=7,
        frameon=True,
        fancybox=True,
        shadow=False,
        title="Experiments",
        title_fontsize=8,
        columnspacing=1.0,
    )

    fig.suptitle(
        "Abliteration: Refusals vs Model Distortion (KL Divergence)",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Dashboard saved to {output_path}")


def print_summary(by_model: dict[str, list[dict]]) -> None:
    """Print terminal summary."""
    for model_id in MODEL_ORDER:
        if model_id not in by_model:
            continue
        results = by_model[model_id]
        label = MODEL_LABELS.get(model_id, model_id)
        bl = BASELINES.get(model_id, "?")
        print(f"\n  {label} (baseline: {bl}/100)")
        print(f"  {'—' * 50}")
        for r in sorted(results, key=lambda r: (r["refusals"], r["kl_divergence"])):
            print(f"    {r['tag']:<25} {r['refusals']:>3}/100  KL={r['kl_divergence']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--output", default="dashboard.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.exists():
        print(f"Error: {experiments_dir} not found")
        sys.exit(1)

    by_model = load_experiments(experiments_dir)
    if not by_model:
        print("No experiments found")
        sys.exit(1)

    print_summary(by_model)
    generate_dashboard(by_model, Path(args.output))


if __name__ == "__main__":
    main()
