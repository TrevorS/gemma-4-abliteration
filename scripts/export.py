#!/usr/bin/env python3
"""
Export abliterated models: save bf16, convert to GGUF, quantize, upload to HF.

Unified script replacing the ad-hoc save_all.py and one-off bash commands.

Usage:
    # Save + GGUF for one model
    python scripts/export.py --model E2B

    # Save + GGUF + push for all graduated models
    python scripts/export.py --all --push

    # Just GGUF conversion (weights already saved)
    python scripts/export.py --model E2B --gguf-only

    # Just upload (weights + GGUFs already exist)
    python scripts/export.py --model E2B --upload-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

# Enable Rust-based fast transfers if available
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
HERETIC_PY = Path.home() / ".local/share/uv/tools/heretic-llm/bin/python"
# llama.cpp toolchain. The Gemma4Unified (12B) architecture requires conversion
# support added 2026-06-04 (ggml-org/llama.cpp PR #24118); this isolated clone is
# pinned at commit c34b922 (2026-06-13). The older monolithic convert script
# (qwen-omni-llama.cpp, Apr 2026) does NOT support Gemma4Unified.
LLAMACPP_DIR = Path.home() / "Projects/llama.cpp-gemma4"
CONVERT_SCRIPT = LLAMACPP_DIR / "convert_hf_to_gguf.py"
QUANTIZE_BIN = LLAMACPP_DIR / "build-cpu/bin/llama-quantize"
LIB_DIR = LLAMACPP_DIR / "build-cpu/bin"

GGUF_QUANTS = ["Q4_K_M", "Q8_0"]
HF_USER = "TrevorJS"

MODELS = {
    "E2B": {
        "hf_id": "google/gemma-4-E2B-it",
        "method": "biprojection",
        "abliterate_cmd": ["biprojection", "--top-pct", "100", "--strip-topic-markers", "--skip-prefix"],
        "results": {
            "refusals": "1/100", "baseline": "98", "cross": "3/686 (0.4%)", "kl": 0.346,
            "cross_detail": {"jbb": "0/100", "tulu": "1/320", "nous": "0/166", "mlab": "2/100"},
        },
    },
    "E4B": {
        "hf_id": "google/gemma-4-E4B-it",
        "method": "biprojection",
        "abliterate_cmd": ["biprojection", "--top-pct", "100", "--strip-topic-markers", "--skip-prefix"],
        "results": {
            "refusals": "0/100 effective (3 flagged, all refusal-then-comply)", "baseline": "99",
            "cross": "5/686 (0.7%)", "kl": 0.068,
            "cross_detail": {"jbb": "2/100", "tulu": "1/320", "nous": "2/166", "mlab": "0/100"},
        },
    },
    "26B-MoE": {
        "hf_id": "google/gemma-4-26B-A4B-it",
        "method": "ega",
        "abliterate_cmd": None,
        "results": {
            "refusals": "1/100 effective (3 flagged, 2 refusal-then-comply)", "baseline": "98",
            "cross": "5/686 (0.7%)", "kl": 0.090,
            "cross_detail": {"jbb": "1/100", "tulu": "1/320", "nous": "0/166", "mlab": "3/100"},
        },
    },
    "31B": {
        "hf_id": "google/gemma-4-31B-it",
        "method": "biprojection",
        "abliterate_cmd": ["biprojection", "--top-pct", "100", "--strip-topic-markers", "--skip-prefix"],
        "results": {
            "refusals": "1/100 effective (5 flagged, 4 refusal-then-comply)", "baseline": "100",
            "cross": "22/686 (3.2%)", "kl": 0.124,
            "cross_detail": {"jbb": "5/100", "tulu": "5/320", "nous": "7/166", "mlab": "5/100"},
        },
    },
    "12B": {
        "hf_id": "google/gemma-4-12B-it",
        # gemma4_unified concentrates the refusal signal in L15-47; L0-14 are
        # dead (SNR ~0.000), so 100% ablation only adds distortion there. 70%
        # selects the live layers. (Robust KL needed — arch emits -inf logits
        # for reserved vocab, which NaNs heretic's default KL.)
        "method": "biprojection",
        "abliterate_cmd": ["biprojection", "--top-pct", "70", "--strip-topic-markers", "--skip-prefix"],
        "card_note": (
            "**Note on the Unified architecture.** `gemma-4-12B-it` is the encoder-free "
            "`Gemma4Unified` model (released 2026-06-03). Its refusal signal concentrates in the "
            "upper decoder layers (L15-47), so only the top 70% of layers are abliterated — ablating "
            "the near-zero-SNR early layers adds distortion with no refusal benefit. The arch also "
            "emits hard `-inf` logits for reserved vocabulary tokens, which NaNs a naive KL "
            "divergence; the reported KL masks non-finite positions before reducing.\n\n"
            "**Version requirements:** inference needs `transformers >= 5.10.1` (tested on 5.12.0, "
            "torch 2.11.0+cu130). GGUF conversion/quantization needs llama.cpp with `Gemma4Unified` "
            "support, added 2026-06-04 in [PR #24118](https://github.com/ggml-org/llama.cpp/pull/24118) "
            "(tested at commit `c34b922`)."
        ),
        "gguf_note": (
            "Requires a llama.cpp build from **2026-06-04 or later** "
            "([PR #24118](https://github.com/ggml-org/llama.cpp/pull/24118), which added "
            "`Gemma4Unified` conversion + inference support). Earlier builds will fail to load these files."
        ),
        "audit_note": (
            "Every flagged cross-dataset response was manually audited (`scripts/audit_refusals.py`). "
            "Of 13 flags, **only 1 is a genuine refusal** (a sensitive prompt the model handles by "
            "redirecting to support resources); the other 12 are false positives — an \"I am an AI\" "
            "disclaimer, or a marker that matched inside generated content, followed by compliance. "
            "Effective refusal rate: ~0/686."
        ),
        "results": {
            "refusals": "6/100", "baseline": "99",
            "cross": "14/686 (2.0%)", "kl": 0.0556,
            "cross_detail": {"jbb": "2/100", "tulu": "4/320", "nous": "4/166", "mlab": "4/100"},
            "quality_label": "Quality (coherence)",
            "quality_before": "—",
            "quality_after": "**no degradation** (response audit + Q8 inference verified)",
        },
    },
}

REPO_NAMES = {
    "E2B": "gemma-4-E2B-it-uncensored",
    "E4B": "gemma-4-E4B-it-uncensored",
    "26B-MoE": "gemma-4-26B-A4B-it-uncensored",
    "31B": "gemma-4-31B-it-uncensored",
    "12B": "gemma-4-12B-it-uncensored",
}


def top_pct(model_key: str) -> int:
    """Percent of layers abliterated, parsed from the model's abliterate_cmd.

    EGA (MoE) has no biprojection command and uses the dense default of 100%.
    """
    cmd = MODELS[model_key].get("abliterate_cmd")
    if cmd and "--top-pct" in cmd:
        return int(cmd[cmd.index("--top-pct") + 1])
    return 100


def slug(model_key: str) -> str:
    return MODELS[model_key]["hf_id"].replace("/", "--")


def bf16_dir(model_key: str) -> Path:
    return MODELS_DIR / slug(model_key)


def gguf_dir(model_key: str) -> Path:
    return MODELS_DIR / f"{slug(model_key)}-GGUF"


def repo_id(model_key: str) -> str:
    return f"{HF_USER}/{REPO_NAMES[model_key]}"


def gguf_repo_id(model_key: str) -> str:
    return f"{HF_USER}/{REPO_NAMES[model_key]}-GGUF"


# --- Step 1: Abliterate + Save ---

def abliterate_and_save(model_key: str) -> bool:
    """Run abliteration and save bf16 weights with merged LoRA."""
    config = MODELS[model_key]
    save_dir = bf16_dir(model_key)

    if save_dir.exists() and any(save_dir.glob("*.safetensors")):
        # Verify tensor names are clean
        from safetensors import safe_open
        sf_file = next(save_dir.glob("*.safetensors"))
        with safe_open(str(sf_file), framework="pt") as sf:
            has_lora = any("base_layer" in k or "lora" in k.lower() for k in sf.keys())
        if not has_lora:
            print(f"  [{model_key}] bf16 weights already saved (clean)")
            return True
        print(f"  [{model_key}] bf16 weights have LoRA artifacts, re-saving...")

    save_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_DATASETS_CACHE": "/tmp/hf_datasets_cache", "PYTHONPATH": "scripts"}

    if config["method"] == "ega":
        cmd = [
            str(HERETIC_PY), "scripts/ega.py",
            "--model", config["hf_id"],
            "--strip-topic-markers", "--batch-size", "4", "--skip-prefix", "--no-eval",
            "--save", str(save_dir),
            "--results-dir", "experiments",
            "--tag", f"export-{model_key.lower()}",
        ]
    else:
        cmd = [
            str(HERETIC_PY), "scripts/abliterate.py",
            *config["abliterate_cmd"],
            "--model", config["hf_id"],
            "--auto-save", str(save_dir),
            "--results-dir", "experiments",
            "--tag", f"export-{model_key.lower()}",
            "--batch-size", "4",
        ]

    print(f"  [{model_key}] Abliterating + saving...")
    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  [{model_key}] FAILED (exit {result.returncode})")
        return False

    print(f"  [{model_key}] bf16 saved to {save_dir}")
    return True


# --- Step 2: GGUF Conversion ---

def convert_to_gguf(model_key: str) -> bool:
    """Convert bf16 safetensors to GGUF quants."""
    src = bf16_dir(model_key)
    dst = gguf_dir(model_key)
    name = REPO_NAMES[model_key]

    if not src.exists() or not any(src.glob("*.safetensors")):
        print(f"  [{model_key}] No bf16 weights to convert")
        return False

    dst.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "LD_LIBRARY_PATH": f"{LIB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"}

    # Check if quants already exist
    existing = [q for q in GGUF_QUANTS if (dst / f"{name}-{q}.gguf").exists()]
    if len(existing) == len(GGUF_QUANTS):
        print(f"  [{model_key}] GGUFs already exist: {existing}")
        return True

    # Convert to F16 intermediate
    f16 = dst / f"{name}-f16.gguf"
    if not f16.exists():
        print(f"  [{model_key}] Converting to F16 GGUF...")
        result = subprocess.run(
            [str(HERETIC_PY), str(CONVERT_SCRIPT), str(src), "--outfile", str(f16), "--outtype", "f16"],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not f16.exists():
            print(f"  [{model_key}] GGUF conversion failed: {result.stderr[-300:]}")
            return False
        print(f"  [{model_key}] F16 GGUF: {f16.stat().st_size / 1e9:.1f} GB")

    # Quantize
    for quant in GGUF_QUANTS:
        qfile = dst / f"{name}-{quant}.gguf"
        if qfile.exists():
            continue
        print(f"  [{model_key}] Quantizing {quant}...")
        result = subprocess.run(
            [str(QUANTIZE_BIN), str(f16), str(qfile), quant],
            capture_output=True, text=True, env=env,
        )
        if result.returncode != 0:
            print(f"  [{model_key}] {quant} quantization failed")
            continue
        print(f"  [{model_key}] {quant}: {qfile.stat().st_size / 1e9:.1f} GB")

    # Clean up F16 intermediate
    if f16.exists():
        print(f"  [{model_key}] Removing F16 intermediate ({f16.stat().st_size / 1e9:.1f} GB)")
        f16.unlink()

    return True


# --- Step 3: Model Cards ---

def write_model_cards(model_key: str) -> None:
    """Write README.md model cards for bf16 and GGUF repos."""
    config = MODELS[model_key]
    results = config["results"]
    name = REPO_NAMES[model_key]
    is_moe = "MoE" in model_key
    rid = repo_id(model_key)

    if is_moe:
        method_section = """Norm-preserving biprojected abliteration on the dense pathway (o_proj + shared mlp.down_proj),
plus **Expert-Granular Abliteration (EGA)** on all 128 MoE expert down_proj slices per layer.

EGA ([OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS)) hooks the MoE routers during probing
to compute per-expert routing weights for harmful vs harmless prompts, then applies norm-preserving
projection ([grimjim](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)) to each expert
individually. Dense-only abliteration leaves 29/100 refusals; adding EGA drops it to 3/100."""
    else:
        method_section = """Norm-preserving biprojected abliteration ([grimjim, Nov 2025](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)).
Each weight row is decomposed into magnitude + direction, the refusal direction is projected out of the
direction component only, then recombined with the original magnitude — guaranteeing `||W_new|| = ||W_orig||`."""

    cross_data = results.get("cross_detail", {})
    tp = top_pct(model_key)
    card_note = config.get("card_note", "")
    note_section = f"\n\n> {card_note}\n" if card_note else ""

    bf16_card = f"""---
base_model: {config['hf_id']}
pipeline_tag: text-generation
library_name: transformers
language:
- en
license: apache-2.0
tags:
- abliteration
- uncensored
- gemma-4
---

# {name}

Uncensored version of [{config['hf_id']}](https://huggingface.co/{config['hf_id']}) with refusal behavior removed.

## Results

| | Before | After |
|--|--------|-------|
| **Refusals (mlabonne, 100 prompts)** | {results.get('baseline', '98-100')}/100 | **{results['refusals']}** |
| **Refusals (cross-dataset, 686 prompts)** | — | **{results['cross']}** |
| **KL Divergence** | 0 (baseline) | **{results['kl']}** |
| **{results.get('quality_label', 'Quality (harmless response length ratio)')}** | {results.get('quality_before', '1.0')} | {results.get('quality_after', '**~1.01** (no degradation)')} |

### Cross-Dataset Validation

Tested against 4 independent prompt datasets to verify generalization:

| Dataset | Prompts | Refusals |
|---------|---------|----------|
| [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | 100 | {cross_data.get('jbb', 'see below')} |
| [tulu-harmbench](https://huggingface.co/datasets/allenai/tulu-3-harmbench-eval) | 320 | {cross_data.get('tulu', 'see below')} |
| [NousResearch/RefusalDataset](https://huggingface.co/datasets/NousResearch/RefusalDataset) | 166 | {cross_data.get('nous', 'see below')} |
| [mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors) | 100 | {cross_data.get('mlab', 'see below')} |
| **Total** | **686** | **{results['cross']}** |

{config.get("audit_note", 'Every flagged refusal was manually audited. Most are "refusal-then-comply" false positives where the model adds an AI identity disclaimer then answers the question anyway.')}

## Method

{method_section}{note_section}

### Pipeline

1. Load model in bf16 with LoRA adapters on `o_proj` and `mlp.down_proj`
2. Collect residual activations for 400 harmful + 400 harmless prompts ([mlabonne](https://huggingface.co/mlabonne) datasets)
3. Winsorize activations at 99.5th percentile (clamps GeGLU outlier activations in Gemma family)
4. Compute per-layer refusal direction: `normalize(mean(harmful) - mean(harmless))`
5. Orthogonalize each direction against harmless mean (double-pass Gram-Schmidt)
6. Apply norm-preserving weight modification to `o_proj` and `down_proj` in the selected layers{'''
7. Hook MoE routers, collect per-expert routing weights for harmful vs harmless prompts
8. Apply same norm-preserving modification to all 128 expert `down_proj` slices per layer''' if is_moe else ''}
{9 if is_moe else 7}. Merge LoRA adapters into base weights for clean tensor names

### Parameters

| Parameter | Value |
|-----------|-------|
| Layers abliterated | {tp}% |
| Scale | 1.0 |
| Winsorization | 0.995 |{'''
| Experts abliterated | 100% (128/128 per layer) |
| Expert scale | 1.0 |''' if is_moe else ''}

### How this differs from vanilla [heretic](https://github.com/p-e-w/heretic)

- **Norm-preserving biprojection** instead of standard projection (preserves weight magnitudes)
- **Per-layer refusal directions** instead of one global direction
- **Deterministic single-pass** instead of 50-trial Optuna search (faster, same or better results)
- **LoRA merge before save** for clean GGUF-compatible tensor names{'''
- **Expert-Granular Abliteration** for MoE expert weights (not supported in heretic)''' if is_moe else ''}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("{rid}", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{rid}")

messages = [{{"role": "user", "content": "Your prompt here"}}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs.to(model.device), max_new_tokens=512)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Reproduction

Full code and experiment data: [abliteration research repo](https://github.com/TrevorS/gemma-4-abliteration)

```bash
python scripts/{'ega.py' if is_moe else 'abliterate.py biprojection'} --model {config['hf_id']} \\
  --top-pct {tp} --strip-topic-markers --skip-prefix --batch-size 4 \\
  --{'save' if is_moe else 'auto-save'} output_dir
```
"""
    (bf16_dir(model_key) / "README.md").write_text(bf16_card)

    # GGUF card
    gd = gguf_dir(model_key)
    if gd.exists():
        quant_rows = []
        for q in GGUF_QUANTS:
            qf = gd / f"{name}-{q}.gguf"
            if qf.exists():
                quant_rows.append(f"| `{qf.name}` | {q} | {qf.stat().st_size / 1e9:.1f} GB |")

        grid = gguf_repo_id(model_key)
        quant_table = "\n".join(quant_rows)
        gguf_card = f"""---
base_model: {rid}
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
license: apache-2.0
tags:
- abliteration
- uncensored
- gemma-4
- gguf
---

# {name} (GGUF)

GGUF quantizations of [{rid}](https://huggingface.co/{rid}).
{(chr(10) + "> " + config["gguf_note"] + chr(10)) if config.get("gguf_note") else ""}
## Files

| File | Quant | Size |
|------|-------|------|
{quant_table}

## Usage

```bash
# From HuggingFace (auto-downloads)
llama-server -hf {grid} -c 8192

# From local file
llama-server -m {name}-Q4_K_M.gguf -c 8192
```

Then open http://localhost:8080 for the chat UI.

## Details

These are GGUF quantizations of [{rid}](https://huggingface.co/{rid}), an abliterated
(uncensored) version of [{config['hf_id']}](https://huggingface.co/{config['hf_id']}).
Refusal behavior has been removed using norm-preserving biprojected abliteration{' with Expert-Granular Abliteration (EGA) for MoE expert weights' if is_moe else ''}.

See the [bf16 model card](https://huggingface.co/{rid}) for full method details,
before/after refusal rates, and cross-dataset validation results.

Source code: [TrevorJS/gemma-4-abliteration](https://github.com/TrevorS/gemma-4-abliteration)
"""
        (gd / "README.md").write_text(gguf_card)


# --- Step 4: Upload ---

def upload(model_key: str, private: bool = True) -> None:
    """Upload bf16 + GGUF to HuggingFace using resumable large folder upload."""
    from huggingface_hub import HfApi, repo_exists
    api = HfApi()

    # bf16
    src = bf16_dir(model_key)
    rid = repo_id(model_key)
    if src.exists() and any(src.glob("*.safetensors")):
        if not repo_exists(rid, repo_type="model"):
            api.create_repo(rid, repo_type="model", private=private)
        print(f"  [{model_key}] Uploading bf16 to {rid}...")
        api.upload_large_folder(folder_path=str(src), repo_id=rid, repo_type="model")
        print(f"  [{model_key}] bf16 pushed")

    # GGUF
    gd = gguf_dir(model_key)
    grid = gguf_repo_id(model_key)
    if gd.exists() and any(gd.glob("*.gguf")):
        if not repo_exists(grid, repo_type="model"):
            api.create_repo(grid, repo_type="model", private=private)
        print(f"  [{model_key}] Uploading GGUFs to {grid}...")
        api.upload_large_folder(folder_path=str(gd), repo_id=grid, repo_type="model")
        print(f"  [{model_key}] GGUFs pushed")


# --- Main ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Export abliterated models")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Export one model")
    parser.add_argument("--all", action="store_true", help="Export all models")
    parser.add_argument("--push", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--public", action="store_true", help="Create public repos (default: private)")
    parser.add_argument("--gguf-only", action="store_true", help="Skip abliteration, just convert existing weights")
    parser.add_argument("--upload-only", action="store_true", help="Skip abliteration + GGUF, just upload")
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Specify --model or --all")

    targets = list(MODELS.keys()) if args.all else [args.model]
    print(f"Exporting: {targets}\n")

    for key in targets:
        print(f"\n{'='*50}")
        print(f"  {key}: {MODELS[key]['hf_id']}")
        print(f"{'='*50}")

        if not args.upload_only:
            # Step 1: Abliterate + save
            if not args.gguf_only:
                if not abliterate_and_save(key):
                    continue

            # Step 2: GGUF
            convert_to_gguf(key)

        # Step 3: Model cards
        write_model_cards(key)

        # Step 4: Upload
        if args.push:
            upload(key, private=not args.public)

    print("\nDone.")


if __name__ == "__main__":
    main()
