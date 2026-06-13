# Research State

## Current Model Under Test
`google/gemma-4-12B-it` (11.95B dense, `Gemma4Unified` encoder-free arch)

## 12B Results (graduated, June 2026)
- **Method**: Biprojection, 70% layers (L15-47), scale=1.0, winsorize=0.995
- **Refusals (mlabonne)**: 6/100 (baseline 99/100)
- **Cross-dataset**: 14/686 (2.0%) flagged — **audit: 1 genuine refusal (redirected to support resources), 12 false positives → ~0/686 effective**
- **KL Divergence**: 0.0556 — **cleanest in the family** (< E4B 0.068)
- **70% == 100%** on the 686 eval (both 14/686); ship 70% for lower distortion
- **Arch gotchas**: needs `transformers >= 5.10.1`; emits `-inf` logits for reserved vocab → naive KL = nan, use the non-finite-safe KL patch; GGUF needs llama.cpp >= 2026-06-04 (PR #24118)

## 26B-A4B Results (graduated)
| Method | Refusals | KL Div | Dense modified | Expert modified |
|--------|----------|--------|----------------|-----------------|
| Dense-only biprojection | 29/100 | 0.091 | 60 | 0 |
| **EGA (dense + all experts)** | **3/100** | **0.090** | **60** | **3840** |
| **EGA cross-dataset** | **5/686 (0.7%)** | — | — | — |

## Best Result So Far (E4B)
- **Method**: Biprojection, 100% layers, scale=1.0, winsorize=0.995
- **Refusals (mlabonne)**: 0/100 (cross-dataset: 5/686, 0.7%)
- **KL Divergence**: 0.0678 (5x lower than E2B!)
- **Quality**: Sanity check passed — avg length ratio 1.01
- **Cross-dataset**: JailbreakBench 2/100, tulu-harmbench 1/320, NousResearch 2/166, mlabonne 0/100
- **Time**: ~3 min single pass

## E2B Best (graduated)
- **Refusals (cross-dataset)**: 3/686 (0.4%)
- **KL Divergence**: 0.3461

## Baseline
- Model: `google/gemma-4-E2B-it`
- Refusals before abliteration: 98/100

## What We Know Works
- Biprojection (norm-preserving) is the best approach — fast and effective on both E2B and E4B
- Layer count is the primary control (30% not enough, 70%+ good)
- Per-layer direction mode beats global in Heretic
- `--strip-topic-markers` flag fixes false-positive refusal detection
- Larger models absorb abliteration with less distortion (E4B KL=0.07 vs E2B KL=0.35)
- Larger models show refusal-then-comply instead of full refusal — need better eval than substring matching
- The dense biprojection recipe ports straight to the `Gemma4Unified` arch — heretic targets `model.language_model.layers.*`, the multimodal projectors are ignored
- For Unified, abliterate only the layers with refusal signal (70%, L15-47) — 100% adds distortion without improving the 686 eval

## What Doesn't Work
- Combining orthogonalize + winsorize in Heretic's Optuna framework (KL explodes >1.0)
- Scale >1.0 with norm-preserving biprojection (already at ceiling)
- Default refusal markers: "illegal", "harmful", "inappropriate" etc. cause massive false positives (11/12 flagged refusals were the model complying but adding factual disclaimers)
- Task vector negation (instruct - base weights): 0/100 refusals but KL=7.87-12.22 — destroys model quality. Too blunt vs biprojection's targeted approach (KL=0.35)

## Model Progression Plan
1. ~~E2B (2.3B dense)~~ — **done**, 3/686 refusals (0.4%), KL=0.346
2. ~~E4B (4.5B dense)~~ — **done**, 5/686 refusals (0.7%), KL=0.068
3. ~~26B-A4B (MoE)~~ — **done**, EGA 5/686 (0.7%), KL=0.090
4. ~~31B (dense)~~ — **done**, 22/686 (3.2%), KL=0.124. Strongest safety training (100/100 baseline). Most residual refusals are refusal-then-comply.
5. ~~12B (Unified, encoder-free dense)~~ — **done**, 14/686 (2.0%), KL=0.056 (cleanest in family). New arch added 2026-06-03; needed transformers>=5.10.1 + non-finite-safe KL. Shipped at 70% layers (signal in L15-47).

## Experiment Log
See `experiments/` directory for JSON results and `ABLITERATION.md` for the full table.
