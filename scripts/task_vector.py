"""
Task vector negation for Gemma 4.

Computes the "instruction-tuning delta" (instruct - base weights), then subtracts
a fraction of it from the instruct model to partially undo safety training.

This is fundamentally different from abliteration — instead of finding refusal
directions in activation space, we directly manipulate the weight delta that
was introduced during instruction tuning.

Reference: Ilharco et al., "Editing Models with Task Arithmetic" (ICLR 2023)

Usage:
    python task_vector.py --base google/gemma-4-E2B --instruct google/gemma-4-E2B-it --alpha 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from heretic.progress import patch_tqdm

patch_tqdm()

import torch
import transformers
from rich.console import Console
from rich.table import Table

from heretic.config import Settings
from heretic.evaluator import Evaluator
from heretic.model import Model
from heretic.utils import load_prompts

console = Console()

# Same topic markers as abliterate.py
TOPIC_MARKERS = {"violat", "prohibit", "illegal", "harmful", "inappropriate", "unethical", "ethical boundaries"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Task vector negation for abliteration")
    parser.add_argument("--base", default="google/gemma-4-E2B", help="Base (pre-train) model")
    parser.add_argument("--instruct", default="google/gemma-4-E2B-it", help="Instruct-tuned model")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.5, 0.75, 1.0],
                        help="Negation scale(s) to try (default: 0.5 0.75 1.0)")
    parser.add_argument("--strip-topic-markers", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--results-dir", default="experiments")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    transformers.logging.set_verbosity_error()

    console.print(f"\n[bold]Task Vector Negation[/]")
    console.print(f"Base: {args.base}")
    console.print(f"Instruct: {args.instruct}")
    console.print(f"Alpha values: {args.alpha}")

    # Load base model weights (just the state dict, don't need full heretic Model)
    console.print("\nLoading base model weights...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    base_sd = {k: v.clone() for k, v in base_model.state_dict().items()}
    del base_model
    torch.cuda.empty_cache()
    console.print(f"  {len(base_sd)} tensors loaded")

    # Load instruct model via heretic (for evaluation)
    console.print("\nLoading instruct model via heretic...")
    real_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        settings = Settings(model=args.instruct, print_responses=args.verbose)
    finally:
        sys.argv = real_argv

    if args.strip_topic_markers:
        original_count = len(settings.refusal_markers)
        settings.refusal_markers = [m for m in settings.refusal_markers if m not in TOPIC_MARKERS]
        console.print(f"[yellow]Stripped {original_count - len(settings.refusal_markers)} topic-word markers[/]")

    if settings.batch_size == 0:
        settings.batch_size = 4

    model = Model(settings)

    # Save original instruct weights for reset between alpha values
    from peft import PeftModel
    inner = model.model
    if isinstance(inner, PeftModel):
        inner = inner.base_model.model

    instruct_sd = {k: v.clone() for k, v in inner.state_dict().items()}

    # Compute task vector (instruct - base)
    console.print("\nComputing task vector (instruct - base)...")
    task_vector: dict[str, torch.Tensor] = {}
    matched = 0
    skipped = 0
    for key in instruct_sd:
        if key in base_sd and instruct_sd[key].shape == base_sd[key].shape:
            delta = instruct_sd[key].float().cpu() - base_sd[key].float().cpu()
            if delta.abs().max() > 1e-8:  # Only store non-zero deltas
                task_vector[key] = delta
                matched += 1
        else:
            skipped += 1

    console.print(f"  {matched} weight tensors with non-zero deltas")
    console.print(f"  {skipped} tensors skipped (missing or shape mismatch)")
    del base_sd
    torch.cuda.empty_cache()

    # Build evaluator for scoring
    console.print("\nBuilding evaluator...")
    evaluator = Evaluator(settings, model)

    results = []

    for alpha in args.alpha:
        console.print(f"\n[bold cyan]═══ Alpha = {alpha} ═══[/]")

        # Reset to original instruct weights
        inner_sd = inner.state_dict()
        for key in instruct_sd:
            inner_sd[key].copy_(instruct_sd[key])

        # Apply task vector negation: new_weight = instruct - alpha * (instruct - base)
        applied = 0
        for key, delta in task_vector.items():
            if key in inner_sd:
                inner_sd[key].sub_(alpha * delta.to(device=inner_sd[key].device, dtype=inner_sd[key].dtype))
                applied += 1

        console.print(f"  Applied negation to {applied} tensors")

        # Evaluate
        console.print("  Evaluating...")
        score, kl_div, refusals = evaluator.get_score()

        console.print(f"  Refusals: [bold]{refusals}/{len(evaluator.bad_prompts)}[/]")
        console.print(f"  KL divergence: [bold]{kl_div:.4f}[/]")

        results.append({
            "alpha": alpha,
            "refusals": refusals,
            "n_prompts": len(evaluator.bad_prompts),
            "kl_divergence": kl_div,
        })

    # Summary table
    console.print("\n")
    table = Table(title="Task Vector Negation Results")
    table.add_column("Alpha", justify="right")
    table.add_column("Refusals", justify="right")
    table.add_column("KL Div", justify="right")

    for r in results:
        ref_style = "green" if r["refusals"] < 5 else ("yellow" if r["refusals"] < 20 else "red")
        kl_style = "green" if r["kl_divergence"] < 0.5 else ("yellow" if r["kl_divergence"] < 1.0 else "red")
        table.add_row(
            str(r["alpha"]),
            f"[{ref_style}]{r['refusals']}/{r['n_prompts']}[/]",
            f"[{kl_style}]{r['kl_divergence']:.4f}[/]",
        )
    console.print(table)

    # Save results
    output_path = Path(args.results_dir) / f"task-vector-{args.instruct.replace('/', '--')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "method": "task_vector_negation",
        "base_model": args.base,
        "instruct_model": args.instruct,
        "strip_topic_markers": args.strip_topic_markers,
        "n_delta_tensors": matched,
        "results": results,
    }, indent=2))
    console.print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
