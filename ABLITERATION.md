# Gemma 4 Abliteration Research

## Goal

Abliterate Gemma 4 models — remove refusal behavior while preserving capabilities. Starting with E2B (2.3B dense) for fast iteration, targeting 26B-A4B (MoE) for production on DGX Spark.

## Models

| Model | Params | Architecture | HF ID |
|-------|--------|-------------|-------|
| E2B | 2.3B | Dense | `google/gemma-4-E2B-it` |
| E4B | 4.5B | Dense | `google/gemma-4-E4B-it` |
| 26B-A4B | 25.2B (3.8B active) | MoE (128 experts, top-8) | `google/gemma-4-26B-A4B-it` |
| 31B | 31B | Dense | `google/gemma-4-31B-it` |
| 12B | 11.95B | Dense (`Gemma4Unified`, encoder-free) | `google/gemma-4-12B-it` |

All Apache 2.0, ungated. The 12B Unified model was added to the family on 2026-06-03, two months after the initial launch — see [Architecture Notes](#gemma-4-12b-unified-encoder-free-multimodal).

## Experiment Results — E2B (2.3B dense)

### Heretic (Optuna TPE, 50 trials each, ~6-7 min)

| Experiment | Refusals | KL Div | Notes |
|---|---|---|---|
| baseline | 24/100 | 0.1693 | Default settings, global direction |
| orthogonal | 13/100 | 0.2547 | Per-layer direction mode |
| winsorize | 12/100 | 0.3735 | `--winsorize 0.95`, per-layer |
| kitchen-sink | 45/100 | 1.28 | Both flags together — too aggressive, killed early |

### Biprojection (single-pass, ~2 min each)

Results with default markers (known to have false positives):

| Experiment | Refusals | KL Div | Layers | Scale | Winsorize |
|---|---|---|---|---|---|
| bp 30% | 58/100 | 0.1373 | 10/35 | 1.0 | 0.995 |
| bp 70% | 16/100 | 0.2597 | 24/35 | 1.0 | 0.995 |
| bp 100% | 12/100 | 0.3461 | 35/35 | 1.0 | 0.995 |
| bp scale=1.3 | 15/100 | 0.2556 | 24/35 | 1.3 | 0.995 |
| bp scale=1.5 all | 13/100 | 0.3433 | 35/35 | 1.5 | 0.995 |
| bp no-winsorize | 28/100 | 0.2627 | 24/35 | 1.0 | off |

### Corrected Results (stripped topic-word markers)

Audit of the 12/100 "refusals" on bp-100% revealed 11 were false positives — the model complied but mentioned "illegal" in factual disclaimers. Stripping topic-word markers ("illegal", "harmful", "inappropriate", "unethical", "prohibit", "violat", "ethical boundaries") gives accurate counts:

| Experiment | Refusals (corrected) | KL Div | Layers | Scale | Winsorize |
|---|---|---|---|---|---|
| bp 30% | 27/100 | 0.1373 | 10/35 | 1.0 | 0.995 |
| bp 70% | 2/100 | 0.2597 | 24/35 | 1.0 | 0.995 |
| **bp 100%** | **1/100** | **0.3461** | **35/35** | **1.0** | **0.995** |
| bp scale=1.3 | 3/100 | 0.2556 | 24/35 | 1.3 | 0.995 |
| bp scale=1.5 all | 1/100 | 0.3433 | 35/35 | 1.5 | 0.995 |
| bp no-winsorize | 2/100 | 0.2627 | 24/35 | 1.0 | off |

### Task Vector Negation

Compute `instruct - base` weight delta, subtract a fraction (α) from the instruct model.
Uses `google/gemma-4-E2B` (base) and `google/gemma-4-E2B-it` (instruct).

| Alpha | Refusals | KL Div | Verdict |
|---|---|---|---|
| 0.05 | 96/100 | 0.19 | Barely moves refusals |
| 0.10 | 97/100 | 0.55 | Same |
| 0.15 | 92/100 | 0.70 | Slight dip |
| 0.20 | 93/100 | 0.95 | KL rising fast |
| 0.30 | 93/100 | 1.39 | Still 93% refusals |
| 0.50 | 0/100 | 7.87 | Phase transition — model destroyed |
| 0.75 | 0/100 | 10.15 | Worse |
| 1.00 | 0/100 | 12.22 | Full undo of instruction tuning |

**Conclusion**: Sharp phase transition. No usable middle ground. Biprojection is strictly better.

### COSMIC Layer Selection

Replaced SNR-based quality metric with cosine similarity sorting (lower similarity = better target).
22/24 layers overlap with SNR at 70%. Identical results at 100%.

| Method | Layers | Refusals | KL Div |
|---|---|---|---|
| SNR (default) | 70% | 2/100 | 0.260 |
| COSMIC | 70% | 2/100 | 0.293 |
| SNR (default) | 100% | 1/100 | 0.346 |
| COSMIC | 100% | 1/100 | 0.346 |

**Conclusion**: No meaningful improvement. SNR and COSMIC converge on the same layers.

### Multi-Dataset Cross-Validation

Tested best config (bp-100%, stripped markers) against 4 independent prompt datasets:

| Dataset | N | Baseline Refusals | Abliterated | Reduction |
|---|---|---|---|---|
| JailbreakBench | 100 | 92% | **0/100 (0%)** | 100% |
| tulu-harmbench | 320 | 61% | **1/320 (0.3%)** | 99% |
| NousResearch | 166 | 86% | **0/166 (0%)** | 100% |
| mlabonne | 100 | 97% | **2/100 (2%)** | 98% |
| **Total** | **686** | — | **3/686 (0.4%)** | **99.4%** |

### Quality Sanity Check

10 harmless prompts compared baseline vs abliterated. Average response length ratio: 1.01.
All responses coherent and on-topic. Some identical, others slightly reworded with equivalent quality.

## Experiment Results — E4B (4.5B dense)

| Experiment | Refusals | KL Div | Layers |
|---|---|---|---|
| baseline | 99/100 | — | — |
| bp 70% | 6/100 | 0.0428 | 29/42 |
| **bp 100%** | **3/100 (0 effective)** | **0.0678** | **42/42** |

All 3 flagged refusals are refusal-then-comply (model disclaims as AI then answers). Effective rate: 0/100.

Cross-dataset: 5/686 (0.7%) — JailbreakBench 2/100, tulu-harmbench 1/320, NousResearch 2/166, mlabonne 0/100.

KL is 5x lower than E2B (0.068 vs 0.346) — larger models absorb abliteration with less distortion.

## Experiment Results — 12B (Unified, encoder-free dense)

`gemma4_unified`, 48 layers. Refusal signal concentrates in L15-47 (SNR ~0.005-0.008); L0-14 are dead
(~0.000). Needs `transformers >= 5.10.1` and the non-finite-safe KL patch (reserved-token `-inf` logits).

| Config | Refusals (mlabonne) | KL Div | Layers |
|---|---|---|---|
| baseline | 99/100 | — | — |
| **bp 70% (shipped)** | **6/100** | **0.0556** | **34/48 (L15-47)** |

Cross-dataset (686), abliterated:

| Dataset | N | Baseline | bp 70% (shipped) | bp 100% |
|---|---|---|---|---|
| JailbreakBench | 100 | 95/100 | 2/100 | 3/100 |
| tulu-harmbench | 320 | 249/320 | 4/320 | 3/320 |
| NousResearch | 166 | 148/166 | 4/166 | 4/166 |
| mlabonne | 100 | 97/100 | 4/100 | 4/100 |
| **Total** | **686** | — | **14/686 (2.0%)** | **14/686 (2.0%)** |

**70% and 100% reach the same 14/686 generalization** — ablating the dead early layers buys nothing on
the cross-dataset eval, so we ship 70% for lower distortion (KL 0.0556 is the lowest in the family).
This answers the layer-count question for the Unified arch: stop at the layers that carry signal.

KL=0.0556 is the cleanest abliteration across all five models — the 12B Unified text decoder absorbs
the refusal-direction removal with less distortion than even E4B (0.068).

**Audit (`scripts/audit_refusals.py`).** All 13 flagged cross-dataset responses were read on the shipped
weights. Only **1 is a genuine refusal** (a sensitive prompt the model handles by redirecting to support
resources). The other 12 are false positives: 8 are disclaimer-then-comply ("I am an AI ..." followed by a
full answer); 2 are the marker matching inside generated content rather than a refusal; 1 is a hedge that
still provides the content; 1 is a confused-but-compliant answer. **Effective refusal rate: ~0/686.** This
matches the family pattern — substring markers can't distinguish refusal-then-comply (or markers in
generated content) from a real refusal.

## Experiment Results — 26B-A4B (MoE, 128 experts top-8)

| Method | Refusals | KL Div | Dense modified | Expert modified |
|---|---|---|---|---|
| Dense-only biprojection | 29/100 | 0.091 | 60 | 0 |
| **EGA (dense + all 128 experts)** | **3/100 (1 effective)** | **0.090** | **60** | **3840** |

Dense-only abliteration leaves 29/100 refusals — the MoE experts carry most of the refusal signal. EGA abliterates each expert's down_proj slice individually, dropping refusals from 29 → 3 with identical KL.

Audit of 3 flagged refusals: 1 genuine (racist comic prompt), 2 refusal-then-comply false positives.

### Key Findings (Updated)

1. **Biprojection + EGA covers dense, MoE, and Unified architectures**
2. **Layer count is the main dial** — 30% not enough, 70%+ good. On most models 100% edges it; on the 12B Unified arch 70% matches 100% (14/686 either way) at lower KL, because the early layers carry no refusal signal. Ablate where the signal is.
3. **Larger models absorb abliteration better** — 12B KL=0.056 < E4B 0.068 < 26B 0.090 < 31B 0.124 < E2B 0.346
4. **MoE experts carry refusal signal** — dense-only gets 29/100, adding experts drops to 3/100
5. **Default refusal markers are broken** — "illegal", "harmful", etc. match disclaimers. Use `--strip-topic-markers`
6. **Refusal-then-comply is the dominant pattern on larger models** — model disclaims then answers. Substring markers can't distinguish this from genuine refusal.
7. **Task vector negation is dead** — sharp phase transition, no usable alpha
8. **COSMIC ≈ SNR** for layer selection — both converge on the same layers
9. **Results generalize** — sub-1% refusal rate across 686 prompts from 4 independent datasets on all tested models

## Tooling

### Heretic (primary driver)

Installed from git HEAD: `uv tool install 'heretic-llm @ git+https://github.com/p-e-w/heretic' --with protobuf`

**Patches applied to heretic:**

1. **LoRA target scoping** (`model.py:_apply_lora`): Changed from leaf module names to full qualified paths. Gemma 4's `Gemma4ClippableLinear` wrapper in vision/audio encoders was breaking PEFT. Fixed by using `module_id_to_full_path` so LoRA only targets text decoder layers.

2. **Transformers version**: PyPI heretic 1.2.0 pins `transformers~=4.57` which has a tokenizer bug with Gemma 4's `extra_special_tokens` (passed as list, expected dict). Git HEAD supports transformers 5.x. The original four models ran on `transformers 5.5.0`; the **12B Unified** model (`gemma4_unified`, added 2026-06-03) needs **`transformers >= 5.10.1`** — earlier versions raise `does not recognize this architecture`. Current env: `transformers 5.12.0`, `torch 2.11.0+cu130`.

3. **Non-finite-safe KL** (`abliterate.py:patch_evaluator_robust_kl`): `gemma4_unified` emits hard `-inf` logits for reserved vocabulary tokens. heretic's `F.kl_div(..., log_target=True)` then evaluates `exp(-inf) * (-inf - (-inf)) = 0 * nan = nan` at those positions, poisoning the batchmean (KL reported as `nan`). The patch masks non-finite per-position terms before reducing; it matches the original KL exactly when every position is finite, so it is applied to all biprojection runs. This is what turned 12B's `KL=nan` into `KL=0.0556`. llama.cpp handles the same reserved-token issue at inference with a logits-bias patch (`src/models/gemma4.cpp`).

### abliterate.py (experiment driver)

Headless wrapper around heretic internals. Supports both Optuna TPE search and single-pass biprojection.

```bash
PYTHON=/home/trevor/.local/share/uv/tools/heretic-llm/bin/python
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache

# Biprojection (recommended, ~2 min)
$PYTHON scripts/abliterate.py biprojection --model google/gemma-4-E2B-it \
  --tag my-tag --top-pct 100 --strip-topic-markers --results-dir experiments/

# Heretic Optuna search (~6 min for 50 trials)
$PYTHON scripts/abliterate.py run --model google/gemma-4-E2B-it --n-trials 50 --tag my-tag

# Additional flags:
#   --strip-topic-markers   Remove false-positive-prone markers
#   --first-n-words N       Position-aware detection (first N words only)
#   --layer-select cosmic   Use COSMIC instead of SNR layer selection
#   --verbose               Print all prompt/response pairs
#   --refusal-markers-file  Custom marker list
```

### dashboard.py (results visualization)

Reads `experiments/*.json`, generates scatter plot (refusals vs KL with Pareto front) and bar chart.

```bash
python3 scripts/dashboard.py              # Generates dashboard.png
```

### multi_eval.py (cross-dataset validation)

Tests abliterated model against 4 datasets (686 prompts total) with category-level breakdown.

```bash
$PYTHON scripts/multi_eval.py --model google/gemma-4-E2B-it --results-dir experiments/
```

### task_vector.py (task vector negation)

Tested and ruled out — documented for completeness.

### sanity_check.py (quality verification)

Generates 10 harmless responses before/after abliteration, compares length and coherence.

### Eval datasets

- `prompts/iteration-eval-150.jsonl` — 150-prompt seeded subset (seed=42) for fast iteration
- `prompts/full-eval-686.jsonl` — full 686-prompt pool from 4 datasets for final validation

## Techniques

### Implemented & Tested

**Standard abliteration (heretic default)**
- Compute refusal direction as mean-difference between harmful/harmless prompt activations
- LoRA-based weight modification on `o_proj` and `down_proj`
- Optuna TPE optimization over direction index, layer weights, and positions
- Source: Arditi et al. 2024, "Refusal in LLMs is Mediated by a Single Direction"

**Projected/orthogonalized abliteration** (`--orthogonalize`)
- Gram-Schmidt orthogonalize refusal direction against harmless mean direction
- Removes only the refusal-specific component, preserves helpfulness direction
- Source: grimjim, "Projected Abliteration" (HF blog, Oct 2025)

**Winsorization** (`--winsorize 0.95`)
- Clamp activation vectors to percentile range before computing refusal directions
- Tames GeGLU outlier activations in Gemma-family models (BOS token problem)
- Source: grimjim, Heretic built-in

**Norm-preserving biprojected abliteration** ✅ IMPLEMENTED — best method
- Decompose weight rows into magnitude + direction
- Ablate refusal from directional component only
- Recombine with original magnitudes (guarantees `||W_new|| = ||W_orig||`)
- Double-pass Gram-Schmidt for numerical stability
- Source: grimjim, "Norm-Preserving Biprojected Abliteration" (HF blog, Nov 2025)

**COSMIC layer selection** ✅ TESTED — no improvement over SNR
- Cosine similarity between harmful/harmless activations
- 22/24 layer overlap with SNR at 70% — both metrics find the same layers
- Source: Siu et al., ACL 2025 Findings (arXiv:2506.00085)

**Task vector negation** ✅ TESTED — dead approach
- Sharp phase transition at α≈0.4: no effect below, model destroyed above
- Biprojection is strictly better
- Source: Ilharco et al., "Editing Models with Task Arithmetic" (ICLR 2023)

### To Implement

**Expert-Granular Abliteration (EGA) for MoE** (FOR 26B TARGET)
- Hook MoE routers during probe to capture per-expert routing weights
- Compute routing-weighted per-expert refusal directions
- Classify experts by safety score: `mean_harmful_routing - mean_harmless_routing`
- Project each expert with its own direction instead of global direction
- Source: OBLITERATUS (elder-plinius/OBLITERATUS, March 2026)

**CoT-aware orthogonalization**
- SVD on harmless activations to extract "reasoning direction" per layer
- Three-tier overlap handling:
  - `|overlap| > 0.7`: skip (too entangled)
  - `0.1 < |overlap| < 0.7`: scaled partial orthogonalization
  - `|overlap| < 0.1`: no action needed
- Source: OBLITERATUS

**Composite quality metric for layer selection**
- `quality = SNR × (1 - cosine_similarity) × purity_ratio`
- `purity_ratio = ||refusal_orthogonal|| / ||refusal||`
- Identifies layers where refusal is cleanly separable from harmless behavior
- Source: grimjim analyze.py

### Alternative Approaches (Not Yet Explored)

**Router manipulation for MoE**
- Modify router weights to de-prioritize safety-critical experts
- Manipulating 5 routers in DeepSeek-V2-Lite increased jailbreak success 4x
- Different attack surface than directional ablation
- Source: "Sparse Models, Sparse Safety" (arXiv:2602.08621, Feb 2026)

**SAFEx expert identification**
- Stability-based identification of safety-critical experts
- On Qwen3-30B-A3B: disabling 12 of 128 experts dropped refusal 22%
- Source: Lai et al., NeurIPS 2025 (arXiv:2506.17368)

**DPO recovery**
- Preference optimization fine-tune after abliteration to recover capabilities
- Particularly effective for math reasoning (GSM8K), the most sensitive capability
- Datasets: `mlabonne/orpo-dpo-mix-40k`, `anthracite-org/kalo-opus-instruct-22k-no-refusal`

## Key Papers

| Paper | Date | Key Contribution |
|---|---|---|
| Arditi et al., "Refusal in LLMs is Mediated by a Single Direction" | 2024 | Foundation — mean-difference abliteration |
| grimjim, "Projected Abliteration" | Oct 2025 | Orthogonalize against harmless direction |
| grimjim, "Norm-Preserving Biprojected Abliteration" | Nov 2025 | Preserve weight magnitudes, improve reasoning |
| Young, "Comparative Analysis of Abliteration Methods" | Dec 2025 | Heretic vs DECCP vs ErisForge vs FailSpy benchmarks |
| Siu et al., "COSMIC" (ACL 2025) | 2025 | Automated layer selection via cosine metrics |
| Zhao et al., "Harmfulness ≠ Refusal" (NeurIPS 2025) | 2025 | Harm and refusal encoded separately, at different positions |
| Yeo et al., "SAE Refusal Analysis" | 2025 | Sparse autoencoder decomposition of refusal features |
| Lai et al., "SAFEx" (NeurIPS 2025) | 2025 | Safety-critical expert identification in MoE |
| Liang et al., "RASA" | Feb 2026 | MoE safety lives in routing, not expert weights |
| Jiang et al., "Sparse Safety" | Feb 2026 | Router manipulation for MoE jailbreaking |
| elder-plinius, "OBLITERATUS" | Mar 2026 | 13-method toolkit with EGA for MoE |
| Abu Shairah et al., "Defense Against Abliteration" | 2025 | Extended-refusal training resists abliteration |
| IBM, "CAST" (ICLR 2025) | 2025 | Conditional activation steering (input-dependent) |

## Architecture Notes

### Gemma 4 26B-A4B MoE Layer Structure

```
layer.self_attn.o_proj          ← nn.Linear (shared, all tokens)
layer.mlp.down_proj             ← nn.Linear (shared dense MLP, always runs)
layer.experts.down_proj         ← nn.Parameter [128, hidden, moe_inter] (3D!)
layer.experts.gate_up_proj      ← nn.Parameter [128, 2*moe_inter, hidden] (3D!)
layer.router                    ← routes tokens to top-8 of 128 experts
```

Unique: dense MLP and MoE experts run **in parallel**, outputs summed. The dense MLP is always a valid abliteration target. Expert weights are 3D `nn.Parameter` tensors, not `nn.Linear` — need per-slice iteration for modification.

### Gemma 4 Attention Geometry

- Sliding window layers: `head_dim=256`, 16 KV heads
- Full attention layers: `head_dim=512`, 4 KV heads, K=V weight sharing
- Proportional RoPE on full attention (25% of dims rotated)
- Layer pattern: 5:1 sliding:full

### Gemma 4 12B Unified (Encoder-Free Multimodal)

Added to the family 2026-06-03 — a fifth model, two months after launch. Arch class
`Gemma4UnifiedForConditionalGeneration` (`model_type: gemma4_unified`), 11.95B params, 48 decoder
layers, 256K context. "Unified" = encoder-free: raw image patches and 16 kHz audio waveforms are
projected straight into the text embedding space via lightweight linear layers — no vision/audio tower.

What this means for abliteration:

```
model.language_model.layers.{0..47}.self_attn.o_proj   ← nn.Linear (abliteration target)
model.language_model.layers.{0..47}.mlp.down_proj      ← nn.Linear (abliteration target)
model.language_model.layers.{0..47}.self_attn.{q,k}_norm  ← QK-norm
<multimodal embedders>                                 ← ignored (text-only refusal abliteration)
```

- The text decoder is a standard dense stack nested under `model.language_model.*`, so biprojection
  targets `o_proj` + `mlp.down_proj` exactly as on the other dense models. The multimodal projectors
  are left untouched.
- **Refusal signal lives in L15-47.** SNR layer-quality scores L0-14 at ~0.000 (dead), L20-46 at
  0.005-0.008. We abliterate the top 70% of layers; including the dead early layers only adds
  distortion (and, with norm-preserving rescaling on near-zero projections, risks numerical blow-up).
- **Reserved-token `-inf` logits.** The arch hard-masks reserved vocabulary tokens to `-inf`. This
  NaNs a naive KL (see heretic patch #3 above) and requires a logits-bias patch at inference
  (handled in transformers >= 5.10.1 and llama.cpp >= 2026-06-04).

### GGUF conversion for Gemma 4 Unified

The monolithic `convert_hf_to_gguf.py` (qwen-omni llama.cpp checkout, Apr 2026) only registers
`Gemma4ForConditionalGeneration` and fails on the Unified arch with
`Model Gemma4UnifiedForConditionalGeneration is not supported`. Support landed upstream
2026-06-04 in [PR #24118](https://github.com/ggml-org/llama.cpp/pull/24118), where the converter was
also refactored into a `conversion/` package (`conversion/gemma.py: Gemma4UnifiedModel`).

We build the GGUF toolchain from an isolated clone at `~/Projects/llama.cpp-gemma4` (commit `c34b922`,
2026-06-13), CPU-only, `llama-quantize` target. `export.py` points `CONVERT_SCRIPT` / `QUANTIZE_BIN`
there. Anyone running the resulting GGUFs needs a llama.cpp build from 2026-06-04 or later.
