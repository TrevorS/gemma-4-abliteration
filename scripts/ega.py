"""
Expert-Granular Abliteration (EGA) for Gemma 4 MoE models.

Extends biprojection to also abliterate expert weights by:
1. Hooking MoE routers during probe to capture per-expert routing weights
2. Computing routing-weighted per-expert refusal directions
3. Applying norm-preserving biprojection to each expert's weight slice

The dense pathway (o_proj + shared mlp.down_proj) is abliterated as before.
Expert weights (experts.down_proj [128, 2816, 704]) are abliterated per-slice.

Usage:
    python scripts/ega.py --model google/gemma-4-26B-A4B-it --tag ega-full
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
import torch.nn.functional as F
import transformers
from rich.console import Console

from heretic.config import QuantizationMethod, RowNormalization, Settings
from heretic.evaluator import Evaluator
from heretic.model import Model
from heretic.utils import load_prompts, empty_cache

console = Console()

TOPIC_MARKERS = {"violat", "prohibit", "illegal", "harmful", "inappropriate", "unethical", "ethical boundaries"}


def modify_weight_norm_preserved(
    weight: torch.Tensor,
    refusal_dir: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Norm-preserving biprojected ablation on a 2D weight matrix."""
    W = weight.float()
    W_norms = W.norm(dim=1, keepdim=True)
    W_dirs = F.normalize(W, dim=1)
    r = F.normalize(refusal_dir.float(), dim=0)

    refusal_component = r @ W_dirs
    proj = scale * r.unsqueeze(1) * refusal_component.unsqueeze(0)
    W_dirs = W_dirs - proj
    W_dirs = F.normalize(W_dirs, dim=1)

    refusal_component2 = r @ W_dirs
    proj2 = r.unsqueeze(1) * refusal_component2.unsqueeze(0)
    W_dirs = W_dirs - proj2
    W_dirs = F.normalize(W_dirs, dim=1)

    return (W_norms * W_dirs).to(weight.dtype)


def get_base_weight(module: torch.nn.Module) -> torch.Tensor | None:
    if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
        return module.base_layer.weight
    if hasattr(module, "weight"):
        return module.weight
    return None


def collect_router_weights(
    model: Model,
    prompts: list,
    n_layers: int,
    n_experts: int,
) -> torch.Tensor:
    """Hook routers during forward pass to collect per-expert routing weights.

    Returns: [n_layers, n_experts] tensor of mean routing weights across all prompts.
    """
    layers = model.get_layers()
    routing_sums = torch.zeros(n_layers, n_experts, device="cpu")
    routing_counts = torch.zeros(n_layers, device="cpu")

    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # Router output is the routing weights [batch, seq_len, n_experts]
            # We want to capture which experts are selected and their weights
            if isinstance(output, tuple):
                # Some routers return (routing_weights, expert_indices)
                routing_weights = output[0]
            else:
                routing_weights = output

            if routing_weights is not None and routing_weights.dim() >= 2:
                # Average over batch and sequence dims
                mean_weights = routing_weights.float().mean(dim=list(range(routing_weights.dim() - 1)))
                routing_sums[layer_idx] += mean_weights.cpu()
                routing_counts[layer_idx] += 1

        return hook_fn

    # Register hooks on all routers
    for i, layer in enumerate(layers):
        if hasattr(layer, 'router'):
            hook = layer.router.register_forward_hook(make_hook(i))
            hooks.append(hook)

    # Run forward passes
    console.print(f"  Collecting router weights from {len(prompts)} prompts...")
    try:
        model.get_residuals_batched(prompts)
    finally:
        for h in hooks:
            h.remove()

    # Average
    for i in range(n_layers):
        if routing_counts[i] > 0:
            routing_sums[i] /= routing_counts[i]

    return routing_sums


def compute_expert_safety_scores(
    harmful_routing: torch.Tensor,
    harmless_routing: torch.Tensor,
) -> torch.Tensor:
    """Compute per-expert safety score: how much more an expert is routed for harmful vs harmless.

    Returns: [n_layers, n_experts] — higher = more safety-critical
    """
    return harmful_routing - harmless_routing


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert-Granular Abliteration for MoE models")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--tag", default="ega")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--expert-scale", type=float, default=1.0, help="Scale for expert abliteration (can differ from dense)")
    parser.add_argument("--top-expert-pct", type=float, default=100, help="Percent of experts to abliterate per layer (by safety score)")
    parser.add_argument("--winsorize", type=float, default=0.995)
    parser.add_argument("--strip-topic-markers", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--results-dir", default="experiments")
    parser.add_argument("--dense-only", action="store_true", help="Skip expert abliteration (dense pathway only, for comparison)")
    parser.add_argument("--save", metavar="DIR", help="Save abliterated model weights to DIR")
    parser.add_argument("--skip-prefix", action="store_true", help="Skip prefix detection (saves 30+ min on large models)")
    parser.add_argument("--no-eval", action="store_true", help="Skip baseline + post-abliteration evaluation (for save-only runs)")
    args = parser.parse_args()

    # Survive broken pipes from dead tee/shell wrappers
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    torch.set_grad_enabled(False)
    transformers.logging.set_verbosity_error()

    console.print(f"\n[bold]Expert-Granular Abliteration (EGA)[/]")
    console.print(f"Model: {args.model}")
    console.print(f"Dense scale: {args.scale}, Expert scale: {args.expert_scale}")
    console.print(f"Top expert %: {args.top_expert_pct}")

    # Build settings
    real_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        settings = Settings(
            model=args.model,
            print_responses=args.verbose,
            winsorization_quantile=args.winsorize,
        )
    finally:
        sys.argv = real_argv

    if args.strip_topic_markers:
        settings.refusal_markers = [m for m in settings.refusal_markers if m not in TOPIC_MARKERS]
        console.print(f"Stripped topic markers, {len(settings.refusal_markers)} remaining")

    settings.batch_size = args.batch_size

    # Load model
    model = Model(settings)
    layers = model.get_layers()
    n_layers = len(layers)

    # Check for experts
    has_experts = hasattr(layers[0], 'experts') and hasattr(layers[0].experts, 'down_proj')
    if has_experts:
        n_experts = layers[0].experts.down_proj.shape[0]
        console.print(f"MoE detected: {n_experts} experts per layer, {n_layers} layers")
    else:
        console.print("[yellow]No MoE experts found — falling back to dense-only abliteration[/]")
        args.dense_only = True

    # Load data
    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)

    # Prefix detection
    if args.skip_prefix:
        console.print("\nSkipping prefix detection (--skip-prefix)")
        model.response_prefix = ""
    else:
        from abliterate import setup_model_prefix
        setup_model_prefix(settings, model, good_prompts, bad_prompts)

    # Build evaluator (skip if --no-eval)
    evaluator = None
    if not args.no_eval:
        evaluator = Evaluator(settings, model)

    # Compute refusal directions (same as biprojection)
    from abliterate import compute_refusal_directions, compute_layer_quality
    refusal_directions, good_means, bad_means = compute_refusal_directions(
        settings, model, good_prompts, bad_prompts, winsorize_quantile=args.winsorize,
    )

    # Compute layer qualities
    console.print("\nComputing layer quality metrics...")
    qualities = []
    for i in range(n_layers):
        rd = refusal_directions[i + 1] if (i + 1) < refusal_directions.shape[0] else refusal_directions[i]
        gm = good_means[i + 1] if (i + 1) < good_means.shape[0] else good_means[i]
        bm = bad_means[i + 1] if (i + 1) < bad_means.shape[0] else bad_means[i]
        q = compute_layer_quality(rd, bm, gm)
        qualities.append((i, q))

    qualities.sort(key=lambda x: x[1], reverse=True)
    selected_layers = sorted([idx for idx, _ in qualities])
    console.print(f"  Using all {len(selected_layers)} layers")

    # Orthogonalize refusal directions (biprojection)
    console.print("\nOrthogonalizing refusal directions...")
    projected_dirs = []
    for i in range(refusal_directions.shape[0]):
        r = refusal_directions[i].float()
        h = good_means[i].float() if i < good_means.shape[0] else good_means[-1].float()
        h_hat = F.normalize(h, dim=0)
        r = r - (r @ h_hat) * h_hat
        r = r - (r @ h_hat) * h_hat
        r = F.normalize(r, dim=0)
        projected_dirs.append(r)
    projected_dirs = torch.stack(projected_dirs)

    # --- Phase 1: Abliterate dense pathway (same as biprojection) ---
    console.print("\n[bold]Phase 1: Dense pathway abliteration[/]")
    from peft import PeftModel
    inner = model.model
    if isinstance(inner, PeftModel):
        inner = inner.base_model.model

    dense_modified = 0
    for layer_idx in selected_layers:
        layer = layers[layer_idx]
        dir_idx = min(layer_idx + 1, projected_dirs.shape[0] - 1)
        r = projected_dirs[dir_idx]

        # o_proj
        w = get_base_weight(layer.self_attn.o_proj)
        if w is not None:
            w.data = modify_weight_norm_preserved(w.data, r, args.scale)
            dense_modified += 1

        # shared mlp.down_proj
        w = get_base_weight(layer.mlp.down_proj)
        if w is not None:
            w.data = modify_weight_norm_preserved(w.data, r, args.scale)
            dense_modified += 1

    console.print(f"  Modified {dense_modified} dense weight matrices")

    # --- Phase 2: Abliterate expert weights ---
    expert_modified = 0
    if has_experts and not args.dense_only:
        console.print("\n[bold]Phase 2: Expert abliteration[/]")

        # Collect routing weights for harmful vs harmless prompts
        console.print("  Collecting routing weights for harmful prompts...")
        harmful_routing = collect_router_weights(model, bad_prompts, n_layers, n_experts)
        console.print("  Collecting routing weights for harmless prompts...")
        harmless_routing = collect_router_weights(model, good_prompts, n_layers, n_experts)

        # Compute safety scores
        safety_scores = compute_expert_safety_scores(harmful_routing, harmless_routing)

        n_abliterate = max(1, int(n_experts * args.top_expert_pct / 100))

        for layer_idx in selected_layers:
            layer = layers[layer_idx]
            dir_idx = min(layer_idx + 1, projected_dirs.shape[0] - 1)
            r = projected_dirs[dir_idx]

            # Select top experts by safety score
            layer_scores = safety_scores[layer_idx]
            top_expert_indices = layer_scores.argsort(descending=True)[:n_abliterate]

            # Abliterate experts.down_proj: [128, 2816, 704]
            expert_down = layer.experts.down_proj  # nn.Parameter
            for expert_idx in top_expert_indices:
                ei = expert_idx.item()
                # Extract 2D slice, abliterate, write back
                slice_2d = expert_down.data[ei]  # [2816, 704]
                expert_down.data[ei] = modify_weight_norm_preserved(slice_2d, r, args.expert_scale)
                expert_modified += 1

        console.print(f"  Modified {expert_modified} expert weight slices ({n_abliterate}/{n_experts} experts per layer)")
    else:
        console.print("\n[bold]Phase 2: Skipped (dense-only mode)[/]")

    # --- Evaluate ---
    refusals = -1
    kl_div = -1.0
    if evaluator and not args.no_eval:
        console.print("\n[bold]Evaluating abliterated model...[/]")
        score, kl_div, refusals = evaluator.get_score()
        console.print(f"  Refusals: [bold]{refusals}/{len(evaluator.bad_prompts)}[/]")
        console.print(f"  KL divergence: [bold]{kl_div:.4f}[/]")
    else:
        console.print("\n[bold]Skipping evaluation (--no-eval)[/]")

    console.print(f"  Dense modified: {dense_modified}")
    console.print(f"  Expert modified: {expert_modified}")

    # Save results
    result = {
        "method": "ega" if not args.dense_only else "biprojection",
        "model": args.model,
        "tag": args.tag,
        "refusals": refusals,
        "n_prompts": len(evaluator.bad_prompts) if evaluator else 0,
        "kl_divergence": kl_div,
        "scale": args.scale,
        "expert_scale": args.expert_scale,
        "top_expert_pct": args.top_expert_pct,
        "winsorize": args.winsorize,
        "n_layers": n_layers,
        "n_experts": n_experts if has_experts else 0,
        "dense_modified": dense_modified,
        "expert_modified": expert_modified,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"ega-{args.tag}.json"
    output_path.write_text(json.dumps(result, indent=2))
    console.print(f"  Results saved to {output_path}")

    # Save model if requested
    if args.save:
        save_dir = Path(args.save)
        console.print(f"\nSaving abliterated model to {save_dir}...")
        save_dir.mkdir(parents=True, exist_ok=True)

        from peft import PeftModel as PM
        save_model = model.model
        if isinstance(save_model, PM):
            console.print("  Merging LoRA adapters...")
            save_model = save_model.merge_and_unload()

        save_model.save_pretrained(save_dir)
        model.tokenizer.save_pretrained(save_dir)
        console.print(f"[bold green]Model saved to {save_dir}[/]")


if __name__ == "__main__":
    main()
