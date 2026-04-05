#!/usr/bin/env python3
"""
Abliteration experiment results dashboard.

Reads all experiments/*.json files, generates a visual summary:
  1. Refusals vs KL divergence scatter plot (Pareto frontier highlighted)
  2. Method comparison bar chart
  3. Summary table printed to terminal

Usage:
    python scripts/dashboard.py                    # Generate PNG + terminal summary
    python scripts/dashboard.py --output dashboard.png  # Custom output path
    python scripts/dashboard.py --no-show          # Don't open the image
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_experiments(experiments_dir: Path) -> list[dict]:
    """Load all experiment JSON files, normalizing different formats."""
    results = []
    for f in sorted(experiments_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        # Skip non-experiment files (audit, sanity check, etc.)
        if "refusals" not in data and "results" not in data:
            continue

        # Handle task vector format (multiple alpha results in one file)
        if "results" in data and isinstance(data["results"], list) and data["results"] and "alpha" in data["results"][0]:
            for r in data["results"]:
                results.append({
                    "tag": f"tv-α={r['alpha']}",
                    "method": data.get("method", "task_vector"),
                    "refusals": r["refusals"],
                    "n_prompts": r["n_prompts"],
                    "kl_divergence": r["kl_divergence"],
                    "file": f.name,
                })
            continue

        # Handle biprojection format
        if "refusals" in data:
            tag = data.get("tag", f.stem)
            # Skip reeval/audit duplicates for cleaner chart
            if "reeval" in tag or "audit" in tag or "combined" in tag:
                continue
            results.append({
                "tag": tag,
                "method": data.get("method", "unknown"),
                "refusals": data["refusals"],
                "n_prompts": data.get("n_prompts", 100),
                "kl_divergence": data.get("kl_divergence", 0),
                "scale": data.get("scale", 1.0),
                "top_pct": data.get("top_pct", 100),
                "winsorize": data.get("winsorize", 0.995),
                "file": f.name,
            })

    return results


def find_pareto_front(results: list[dict]) -> list[int]:
    """Find Pareto-optimal points (minimize refusals AND KL divergence)."""
    pareto = []
    for i, r in enumerate(results):
        dominated = False
        for j, s in enumerate(results):
            if i == j:
                continue
            if s["refusals"] <= r["refusals"] and s["kl_divergence"] <= r["kl_divergence"]:
                if s["refusals"] < r["refusals"] or s["kl_divergence"] < r["kl_divergence"]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(i)
    return pareto


def generate_dashboard(results: list[dict], output_path: Path) -> None:
    """Generate the dashboard PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Abliteration Experiment Dashboard — Gemma 4 E2B", fontsize=14, fontweight="bold")

    # --- Left panel: Refusals vs KL scatter ---
    ax1 = axes[0]

    # Color by method
    method_colors = {
        "biprojection": "#2196F3",
        "task_vector_negation": "#FF5722",
    }

    pareto_idx = find_pareto_front(results)

    for i, r in enumerate(results):
        color = method_colors.get(r["method"], "#9E9E9E")
        marker = "o" if r["method"] == "biprojection" else "^"
        edge = "gold" if i in pareto_idx else "none"
        lw = 2.0 if i in pareto_idx else 0.5

        ax1.scatter(
            r["kl_divergence"], r["refusals"],
            c=color, marker=marker, s=80, edgecolors=edge, linewidths=lw,
            zorder=3 if i in pareto_idx else 2,
        )

        # Label interesting points
        if i in pareto_idx or r["refusals"] <= 2 or r["kl_divergence"] > 5:
            label = r["tag"]
            # Shorten long labels
            if len(label) > 20:
                label = label[:18] + "…"
            ax1.annotate(
                label, (r["kl_divergence"], r["refusals"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, alpha=0.8,
            )

    ax1.set_xlabel("KL Divergence (lower = less model distortion)")
    ax1.set_ylabel("Refusals / 100 (lower = better)")
    ax1.set_title("Refusals vs KL Divergence")
    ax1.axhline(y=5, color="green", linestyle="--", alpha=0.3, label="5% refusal threshold")
    ax1.axvline(x=0.5, color="orange", linestyle="--", alpha=0.3, label="KL=0.5 threshold")

    # Legend
    bp_patch = mpatches.Patch(color="#2196F3", label="Biprojection")
    tv_patch = mpatches.Patch(color="#FF5722", label="Task Vector")
    pareto_patch = mpatches.Patch(facecolor="white", edgecolor="gold", linewidth=2, label="Pareto optimal")
    ax1.legend(handles=[bp_patch, tv_patch, pareto_patch], loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)

    # --- Right panel: Best results bar chart ---
    ax2 = axes[1]

    # Filter to interesting results (exclude baseline-like and destroyed models)
    interesting = [r for r in results if r["refusals"] < 50 and r["kl_divergence"] < 2.0]
    interesting.sort(key=lambda r: (r["refusals"], r["kl_divergence"]))

    if interesting:
        tags = [r["tag"][:25] for r in interesting]
        refusals = [r["refusals"] for r in interesting]
        kls = [r["kl_divergence"] for r in interesting]

        x = np.arange(len(tags))
        width = 0.35

        bars1 = ax2.bar(x - width/2, refusals, width, label="Refusals/100", color="#2196F3", alpha=0.8)
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x + width/2, kls, width, label="KL Divergence", color="#FF9800", alpha=0.8)

        ax2.set_ylabel("Refusals / 100", color="#2196F3")
        ax2_twin.set_ylabel("KL Divergence", color="#FF9800")
        ax2.set_xticks(x)
        ax2.set_xticklabels(tags, rotation=45, ha="right", fontsize=7)
        ax2.set_title("Top Configurations (refusals < 50, KL < 2.0)")

        # Combined legend
        ax2.legend(loc="upper left", fontsize=8)
        ax2_twin.legend(loc="upper right", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No qualifying results", ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Dashboard saved to {output_path}")


def print_summary(results: list[dict]) -> None:
    """Print a terminal-friendly summary table."""
    print("\n" + "=" * 85)
    print(f"{'Tag':<30} {'Method':<15} {'Refusals':>10} {'KL Div':>10} {'File'}")
    print("-" * 85)

    # Sort by refusals, then KL
    sorted_results = sorted(results, key=lambda r: (r["refusals"], r["kl_divergence"]))

    for r in sorted_results:
        ref_str = f"{r['refusals']}/{r['n_prompts']}"
        print(f"{r['tag']:<30} {r['method']:<15} {ref_str:>10} {r['kl_divergence']:>10.4f} {r['file']}")

    print("=" * 85)
    print(f"Total experiments: {len(results)}")

    # Best result
    best = sorted_results[0]
    print(f"Best: {best['tag']} — {best['refusals']}/{best['n_prompts']} refusals, KL={best['kl_divergence']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Abliteration experiment dashboard")
    parser.add_argument("--experiments-dir", default="experiments", help="Directory with experiment JSON files")
    parser.add_argument("--output", default="dashboard.png", help="Output PNG path")
    parser.add_argument("--no-show", action="store_true", help="Don't attempt to open the image")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.exists():
        print(f"Error: {experiments_dir} not found")
        sys.exit(1)

    results = load_experiments(experiments_dir)
    if not results:
        print("No experiment results found")
        sys.exit(1)

    print_summary(results)
    generate_dashboard(results, Path(args.output))

    if not args.no_show:
        # Try to open the image
        try:
            subprocess.run(["xdg-open", args.output], check=False, capture_output=True)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
