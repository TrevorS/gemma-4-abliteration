"""
Memory-efficient 31B export.

The 31B model (62GB bf16) doesn't fit in 121GB RAM with heretic's full pipeline.
Strategy:
  1. Load model in 4-bit (~16GB) to compute refusal directions
  2. Unload 4-bit model
  3. Load bf16 safetensors, apply directions layer-by-layer, save
"""

from __future__ import annotations

import gc
import json
import signal
import sys
from pathlib import Path

from heretic.progress import patch_tqdm
patch_tqdm()

import torch
import torch.nn.functional as F
import transformers
from rich.console import Console
from safetensors import safe_open
from safetensors.torch import save_file

from heretic.config import QuantizationMethod, Settings
from heretic.model import Model
from heretic.utils import load_prompts, empty_cache

console = Console()
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

TOPIC_MARKERS = {"violat", "prohibit", "illegal", "harmful", "inappropriate", "unethical", "ethical boundaries"}
MODEL_ID = "google/gemma-4-31B-it"
SAVE_DIR = Path("models/google--gemma-4-31B-it")


def compute_directions_4bit() -> tuple[torch.Tensor, torch.Tensor]:
    """Load model in 4-bit, compute per-layer refusal directions. Returns (projected_dirs, layer_count)."""
    console.print("\n[bold]Phase 1: Computing refusal directions (4-bit)[/]")

    real_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        settings = Settings(
            model=MODEL_ID,
            quantization=QuantizationMethod.BNB_4BIT,
            winsorization_quantile=0.995,
        )
    finally:
        sys.argv = real_argv

    settings.refusal_markers = [m for m in settings.refusal_markers if m not in TOPIC_MARKERS]
    settings.batch_size = 4
    model = Model(settings)

    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)

    # Skip prefix detection
    model.response_prefix = ""

    # Compute residuals and directions
    console.print("  Computing residuals for 800 prompts...")
    good_residuals = model.get_residuals_batched(good_prompts)
    bad_residuals = model.get_residuals_batched(bad_prompts)

    # Winsorize
    console.print("  Winsorizing at 0.995...")
    for residuals in [good_residuals, bad_residuals]:
        abs_vals = residuals.abs()
        threshold = torch.quantile(abs_vals.float(), 0.995, dim=-1, keepdim=True)
        residuals.clamp_(-threshold, threshold)

    good_means = good_residuals.mean(dim=0)
    bad_means = bad_residuals.mean(dim=0)
    refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    del good_residuals, bad_residuals
    empty_cache()

    # Orthogonalize (biprojection)
    console.print("  Orthogonalizing...")
    projected_dirs = []
    for i in range(refusal_directions.shape[0]):
        r = refusal_directions[i].float()
        h = good_means[i].float() if i < good_means.shape[0] else good_means[-1].float()
        h_hat = F.normalize(h, dim=0)
        r = r - (r @ h_hat) * h_hat
        r = r - (r @ h_hat) * h_hat
        r = F.normalize(r, dim=0)
        projected_dirs.append(r)

    projected_dirs = torch.stack(projected_dirs).cpu()
    n_layers = len(model.get_layers())

    console.print(f"  Got {projected_dirs.shape[0]} directions for {n_layers} layers")

    # Free the 4-bit model
    del model, settings, good_means, bad_means, refusal_directions
    gc.collect()
    torch.cuda.empty_cache()
    console.print("  4-bit model unloaded")

    return projected_dirs, n_layers


def modify_weight_norm_preserved(weight: torch.Tensor, refusal_dir: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Norm-preserving biprojected ablation."""
    W = weight.float()
    W_norms = W.norm(dim=1, keepdim=True)
    W_dirs = F.normalize(W, dim=1)
    r = F.normalize(refusal_dir.float(), dim=0)

    rc = r @ W_dirs
    proj = scale * r.unsqueeze(1) * rc.unsqueeze(0)
    W_dirs = W_dirs - proj
    W_dirs = F.normalize(W_dirs, dim=1)

    rc2 = r @ W_dirs
    proj2 = r.unsqueeze(1) * rc2.unsqueeze(0)
    W_dirs = W_dirs - proj2
    W_dirs = F.normalize(W_dirs, dim=1)

    return (W_norms * W_dirs).to(weight.dtype)


def apply_and_save(projected_dirs: torch.Tensor, n_layers: int) -> None:
    """Load bf16 weights from HF cache, apply abliteration, save."""
    console.print("\n[bold]Phase 2: Applying directions to bf16 weights[/]")

    from huggingface_hub import snapshot_download
    model_path = Path(snapshot_download(MODEL_ID))
    console.print(f"  Model path: {model_path}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files
    for f in model_path.iterdir():
        if not f.name.endswith(".safetensors"):
            import shutil
            dest = SAVE_DIR / f.name
            if not dest.exists():
                shutil.copy2(f, dest)

    # Process each safetensors shard
    shard_files = sorted(model_path.glob("*.safetensors"))
    console.print(f"  Processing {len(shard_files)} shard(s)...")

    modified_count = 0
    for shard_path in shard_files:
        console.print(f"\n  Shard: {shard_path.name}")
        tensors = {}

        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                tensor = sf.get_tensor(key)

                # Check if this is a weight we should abliterate
                should_modify = False
                layer_idx = -1

                if ".self_attn.o_proj.weight" in key or ".mlp.down_proj.weight" in key:
                    # Extract layer index from key like "model.language_model.layers.15.self_attn.o_proj.weight"
                    parts = key.split(".")
                    for i, p in enumerate(parts):
                        if p == "layers" and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                should_modify = True
                            except ValueError:
                                pass
                            break

                if should_modify and 0 <= layer_idx < n_layers:
                    dir_idx = min(layer_idx + 1, projected_dirs.shape[0] - 1)
                    r = projected_dirs[dir_idx]
                    tensor = modify_weight_norm_preserved(tensor, r, 1.0)
                    modified_count += 1

                tensors[key] = tensor

        # Save modified shard
        save_file(tensors, str(SAVE_DIR / shard_path.name))
        console.print(f"    Saved ({modified_count} weights modified so far)")
        del tensors
        gc.collect()

    console.print(f"\n  Total modified: {modified_count} weight matrices across {n_layers} layers")
    console.print(f"  Saved to {SAVE_DIR}")


def main() -> None:
    torch.set_grad_enabled(False)
    transformers.logging.set_verbosity_error()

    console.print("[bold]31B Memory-Efficient Export[/]")

    # Phase 1: Compute directions in 4-bit
    projected_dirs, n_layers = compute_directions_4bit()

    # Phase 2: Apply to bf16 and save
    apply_and_save(projected_dirs, n_layers)

    console.print("\n[bold green]Done![/]")


if __name__ == "__main__":
    main()
