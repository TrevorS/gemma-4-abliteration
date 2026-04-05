"""
Sanity-check an abliterated model by generating responses to harmless prompts.

Loads the model, applies biprojection abliteration with best-known settings,
then generates responses to everyday prompts to verify coherence and quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from heretic.progress import patch_tqdm

patch_tqdm()

import torch
import torch.nn.functional as F
import transformers
from rich.console import Console
from rich.table import Table

from heretic.config import Settings
from heretic.model import Model
from heretic.utils import load_prompts

console = Console()

HARMLESS_PROMPTS = [
    "Write a short poem about autumn leaves.",
    "Explain how photosynthesis works in simple terms.",
    "What are three tips for making good scrambled eggs?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Write a haiku about the ocean.",
    "What is the difference between a compiler and an interpreter?",
    "Describe the water cycle for a 10-year-old.",
    "Give me a fun fact about octopuses.",
    "Write a professional email declining a meeting invitation.",
    "What are the main differences between Python and JavaScript?",
]


def apply_biprojection(
    model: Model,
    good_prompts: list,
    bad_prompts: list,
    *,
    winsorize_quantile: float = 0.995,
    top_pct: float = 100.0,
    scale: float = 1.0,
) -> None:
    """Apply biprojection abliteration to the model (same logic as abliterate.py)."""
    from abliterate import compute_refusal_directions, biprojection_abliterate

    refusal_directions, good_means, bad_means = compute_refusal_directions(
        model._settings, model, good_prompts, bad_prompts,
        winsorize_quantile=winsorize_quantile,
    )

    n_layers = len(model.get_layers())
    n_select = max(1, int(n_layers * top_pct / 100))

    # Compute qualities and select layers
    qualities = []
    for i in range(n_layers):
        rd = refusal_directions[i + 1] if (i + 1) < refusal_directions.shape[0] else refusal_directions[i]
        gm = good_means[i + 1] if (i + 1) < good_means.shape[0] else good_means[i]
        bm = bad_means[i + 1] if (i + 1) < bad_means.shape[0] else bad_means[i]
        # SNR-based quality
        harmful_norm = bm.float().norm()
        harmless_norm = gm.float().norm()
        snr = rd.float().norm() / max(harmful_norm, harmless_norm, torch.tensor(1e-8))
        qualities.append((i, snr.item()))

    qualities.sort(key=lambda x: x[1], reverse=True)
    selected_layers = sorted([idx for idx, _ in qualities[:n_select]])

    console.print(f"\nOrthogonalizing and applying biprojection to {len(selected_layers)} layers...")

    # Orthogonalize
    for i in range(refusal_directions.shape[0]):
        r = refusal_directions[i].float()
        g = good_means[i].float() if i < good_means.shape[0] else good_means[-1].float()
        g_hat = F.normalize(g, dim=0)
        proj = (r @ g_hat) * g_hat
        refusal_directions[i] = F.normalize(r - proj, dim=0)

    # Apply abliteration
    params = AbliterationParameters(
        scale=scale,
        direction_index=0,
    )
    for layer_idx in selected_layers:
        rd_idx = layer_idx + 1 if (layer_idx + 1) < refusal_directions.shape[0] else layer_idx
        model.abliterate(refusal_directions, rd_idx, params)

    console.print(f"  Applied to {len(selected_layers)} layers")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check abliterated model quality")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--output", help="Save results JSON to this path")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    transformers.logging.set_verbosity_error()

    # Build settings
    real_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        settings = Settings(model=args.model, print_responses=True)
    finally:
        sys.argv = real_argv

    console.print(f"\n[bold]Sanity Check: {args.model}[/]")
    console.print(f"Generating {len(HARMLESS_PROMPTS)} harmless responses\n")

    # Load model
    console.print("Loading model...")
    model = Model(settings)

    # Set batch size if not auto-detected
    if settings.batch_size == 0:
        settings.batch_size = 4

    # First generate BASELINE (unabliterated) responses
    console.print("\n[bold cyan]═══ BASELINE (unabliterated) ═══[/]\n")
    baseline_responses = []
    for i, prompt in enumerate(HARMLESS_PROMPTS):
        resp = model.get_responses([type("P", (), {"system": "", "user": prompt})()])
        baseline_responses.append(resp[0] if resp else "")
        console.print(f"[bold]Prompt {i+1}:[/] {prompt}")
        console.print(f"[green]{baseline_responses[-1][:300]}[/]\n")

    # Load prompt data for abliteration
    console.print("\nLoading abliteration prompt data...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)

    # Apply biprojection using the same logic as run_biprojection in abliterate.py
    console.print("\nApplying biprojection abliteration (100% layers, scale=1.0, winsorize=0.995)...")

    from abliterate import compute_refusal_directions, compute_layer_quality, modify_weight_norm_preserved

    refusal_directions, good_means, bad_means = compute_refusal_directions(
        settings, model, good_prompts, bad_prompts, winsorize_quantile=0.995,
    )

    n_layers = len(model.get_layers())
    qualities = []
    for i in range(n_layers):
        rd = refusal_directions[i + 1] if (i + 1) < refusal_directions.shape[0] else refusal_directions[i]
        gm = good_means[i + 1] if (i + 1) < good_means.shape[0] else good_means[i]
        bm = bad_means[i + 1] if (i + 1) < bad_means.shape[0] else bad_means[i]
        q = compute_layer_quality(rd, bm, gm)
        qualities.append((i, q))

    qualities.sort(key=lambda x: x[1], reverse=True)
    selected_layers = sorted([idx for idx, _ in qualities])

    # Orthogonalize refusal directions against harmless means (biprojection)
    for i in range(refusal_directions.shape[0]):
        r = refusal_directions[i].float()
        g = good_means[i].float() if i < good_means.shape[0] else good_means[-1].float()
        g_hat = F.normalize(g, dim=0)
        proj = (r @ g_hat) * g_hat
        refusal_directions[i] = F.normalize(r - proj, dim=0)

    # Apply norm-preserving weight modification (same as run_biprojection)
    from peft import PeftModel
    base_model = model.model
    if isinstance(base_model, PeftModel):
        base_model = base_model.base_model.model

    layers = model.get_layers()
    scale = 1.0
    for layer_idx in selected_layers:
        layer = layers[layer_idx]
        dir_idx = min(layer_idx + 1, refusal_directions.shape[0] - 1)
        r = refusal_directions[dir_idx]

        def get_base_weight(module: torch.nn.Module) -> torch.Tensor | None:
            if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
                return module.base_layer.weight
            if hasattr(module, "weight"):
                return module.weight
            return None

        for attr_path in ["self_attn.o_proj", "mlp.down_proj"]:
            parts = attr_path.split(".")
            mod = layer
            for p in parts:
                mod = getattr(mod, p)
            w = get_base_weight(mod)
            if w is not None:
                w.data = modify_weight_norm_preserved(w.data, r, scale)

    console.print(f"  Applied to {len(selected_layers)} layers")

    console.print(f"\n[bold cyan]═══ ABLITERATED (bp-100%, scale=1.0) ═══[/]\n")
    abliterated_responses = []
    for i, prompt in enumerate(HARMLESS_PROMPTS):
        resp = model.get_responses([type("P", (), {"system": "", "user": prompt})()])
        abliterated_responses.append(resp[0] if resp else "")
        console.print(f"[bold]Prompt {i+1}:[/] {prompt}")
        console.print(f"[yellow]{abliterated_responses[-1][:300]}[/]\n")

    # Compare
    console.print("\n[bold]═══ COMPARISON ═══[/]\n")
    table = Table(title="Baseline vs Abliterated Response Lengths")
    table.add_column("Prompt", max_width=40)
    table.add_column("Baseline len", justify="right")
    table.add_column("Abliterated len", justify="right")
    table.add_column("Ratio", justify="right")

    results = []
    for i, prompt in enumerate(HARMLESS_PROMPTS):
        bl = len(baseline_responses[i])
        al = len(abliterated_responses[i])
        ratio = al / bl if bl > 0 else 0
        table.add_row(prompt[:40], str(bl), str(al), f"{ratio:.2f}")
        results.append({
            "prompt": prompt,
            "baseline_response": baseline_responses[i][:500],
            "abliterated_response": abliterated_responses[i][:500],
            "baseline_len": bl,
            "abliterated_len": al,
            "len_ratio": round(ratio, 2),
        })

    console.print(table)

    avg_ratio = sum(r["len_ratio"] for r in results) / len(results)
    console.print(f"\nAverage length ratio (abliterated/baseline): [bold]{avg_ratio:.2f}[/]")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps({
            "model": args.model,
            "method": "biprojection",
            "config": {"top_pct": 100, "scale": 1.0, "winsorize": 0.995},
            "n_prompts": len(HARMLESS_PROMPTS),
            "avg_length_ratio": round(avg_ratio, 2),
            "results": results,
        }, indent=2))
        console.print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
