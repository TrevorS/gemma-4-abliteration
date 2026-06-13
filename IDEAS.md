# Ideas Backlog

Ideas for the auto-research loop to pick up and try. Add new ideas at the bottom. The research agent will pick the highest-priority untried idea, run it, and record results.

## Priority: Critical (Evaluation Fix)

- [x] Strip topic-word markers — added `--strip-topic-markers` and `--refusal-markers-file` flags to abliterate.py. Removes 7 markers ("illegal", "harmful", "inappropriate", "unethical", "prohibit", "violat", "ethical boundaries"). Result: 12/100 → 1/100 refusals with same KL. See `experiments/biprojection-bp-stripped-markers.json`.
- [x] Add position-aware refusal detection — added `--first-n-words N` flag. With default markers: 12→10/100 (only catches 2 of 11 false positives, because "illegal" appears in opening disclaimers). With stripped markers: still 1/100, same as stripped-only. Stripping topic markers is the primary fix; position-aware is complementary hardening. See `experiments/position-aware-eval.json`.
- [ ] LLM-as-judge for refusal detection — use a small model (Gemma 2B base or Qwen 0.5B) to classify "did this response comply with the request?" Binary yes/no. More accurate than substring matching.
- [x] Non-finite-robust KL divergence — `gemma4_unified` emits hard `-inf` logits for reserved vocab tokens, so heretic's `F.kl_div(log_target=True)` hits `0*nan` and reports KL=nan. Added `patch_evaluator_robust_kl` in abliterate.py: masks non-finite per-position terms before the batchmean. Identical to the original KL when all positions are finite, so applied unconditionally. Fixed 12B's KL nan → 0.0556.
- [x] Re-evaluate all past experiments with fixed markers — done. Most experiments dropped dramatically (e.g. 16→2, 28→2, 13→1). 30% layers still insufficient (58→27). Surprise: no-winsorize (2/100) matches winsorized 70% (2/100). See `-reeval` JSON files in `experiments/`.

## Priority: High

- [x] Inspect the remaining 12 refusals from bp-100% — 11/12 are false positives (marker "illegal" matches disclaimers, not refusals). 1 genuine refusal ("as an ai" on product review). Effective rate: 1/100. See `experiments/refusal-audit-bp-allayers.json`.
- [x] Task vector negation — tested α=0.05 through 1.0. Sharp phase transition: α≤0.3 barely dents refusals (92-97/100) while KL rises to 1.39; α≥0.5 hits 0/100 but KL explodes (7.87-12.22). No usable middle ground. Biprojection is decisively better. See `experiments/task-vector-google--gemma-4-E2B-it.json`.
- [x] COSMIC layer selection — added `--layer-select cosmic` flag. At 70%: 2/100 refusals, KL=0.293 (vs SNR's 2/100, KL=0.260). 22/24 layers overlap. At 100%: identical to SNR (1/100, KL=0.346). No meaningful improvement — SNR and COSMIC converge on the same layers. See `experiments/biprojection-bp-cosmic-*.json`.
- [ ] Multi-direction SVD — instead of 1 mean-difference direction, extract top-3 SVD directions from the harmful-harmless residual difference. Project all 3 out.
- [x] Sanity-check model output — 10 harmless responses compared baseline vs abliterated. Avg length ratio 1.01, all responses coherent and on-topic. Some identical (octopus fact), others slightly reworded but equivalent quality. No degradation detected. See `experiments/sanity-check-bp-allayers.json`.

## Priority: Medium

- [ ] Try `jim-plus/llm-abliteration` directly (grimjim's actual code) instead of our reimplementation — compare results
- [ ] Combine biprojection + task vector negation — apply both, see if they stack
- [ ] OBLITERATUS EGA (Expert-Granular Abliteration) — not useful on E2B (dense) but prototype the code for 26B MoE
- [ ] Different prompt datasets — try `NousResearch/RefusalDataset` or `byroneverson/abliterate-refusal` instead of mlabonne defaults
- [x] Research red team benchmarks — surveyed HarmBench, AdvBench, JailbreakBench, StrongREJECT, TDC, AART + attack tools (PAIR, GCG, AutoDAN, TAP). See `REDTEAM.md` for full findings. HarmBench is the gold standard eval framework.
- [ ] Tune refusal markers — Gemma 4 may use unique refusal phrasing not in heretic's default marker list
- [ ] CoT-aware orthogonalization from OBLITERATUS — protect reasoning directions during ablation

## Priority: High (Evaluation & Red Team Infrastructure)

- [x] Build a results dashboard — `scripts/dashboard.py` reads all `experiments/*.json`, generates scatter plot (refusals vs KL with Pareto front) + bar chart (top configs). Outputs `dashboard.png`. Run with `python3 scripts/dashboard.py`.
- [x] Multi-dataset eval — tested against JailbreakBench (100), tulu-harmbench (320), NousResearch (166), mlabonne (100). Result: 3/686 total refusals (0.4%). Abliteration generalizes across all datasets with 98-100% refusal reduction. `scripts/multi_eval.py`. See `experiments/multi-dataset-eval.json`.
- [x] Build seeded iteration eval subset — sampled 150 prompts (seed=42) proportionally from JailbreakBench (22), tulu-harmbench (70), NousResearch (36), mlabonne (22). 42 categories covered. Saved as `prompts/iteration-eval-150.jsonl`. Full pool at `prompts/full-eval-686.jsonl`.
- [ ] Category-level reporting — tag each eval prompt with a category (hacking, fraud, drugs, weapons, manipulation, etc.), report per-category refusal rates. Catches regressions hidden by aggregate numbers.
- [ ] Integrate HarmBench classifier — replace substring matching with HarmBench's Llama 2 classifier for binary+continuous harm scoring. Handles refusal-then-comply and benign outputs. GitHub: `centerforaisafety/HarmBench`.
- [ ] Build red team eval agent — reusable agent that: (1) generates adversarial prompts via PAIR (`patrickrchao/JailbreakingLLMs`), (2) tests against target model, (3) classifies responses with calibrated judge, (4) reports per-category ASR. Useful for abliteration eval and red team competitions.
- [ ] Adversarial robustness eval — test abliterated models against GCG (`GraySwanAI/nanoGCG`), AutoDAN (`SheltonLiu-N/AutoDAN`), and PAIR attacks. Abliteration weakens safety training — measure how much more vulnerable it makes the model.
- [ ] Build custom Gemma 4 eval set — current mlabonne has gaps (no surveillance, weak social engineering, no multi-turn). Build 50-100 prompts covering blind spots, tagged by category.
- [ ] Confidence intervals — bootstrap resample eval prompts to estimate variance on refusal rate and KL. Current 100-prompt CI on 1/100 is ~0-5%.

## Priority: Low / Exploratory

- [ ] Router manipulation for MoE (only relevant for 26B)
- [ ] SAFEx expert identification (only relevant for 26B)
- [ ] DPO recovery pass after abliteration
- [ ] Conditional activation steering (CAST) — inference-time approach, keep selective refusal
- [ ] Train a custom harmful/harmless prompt set tuned to Gemma 4's specific refusal patterns

## Priority: High (Post-12B follow-ups, June 2026)

- [ ] 100% vs 70% layers on 12B unified — we shipped 70% on SNR grounds (signal in L15-47, L0-14 dead). Verify quantitatively: does ablating the dead early layers raise KL or refusals? Confirms the layer-selection call for unified-arch models.
- [ ] Benchmark ARA vs biprojection on E2B — `Kasper-Bankler/gemma-4-uncensored` (Arbitrary-Rank Ablation) claims KL 0.165 on E2B vs our 0.346, but its reported benchmarks look broken (ARC-Easy ~30% = random). Re-run ARA through our eval harness to settle whether it's a real low-distortion gain or an apples-to-oranges metric.
- [ ] Abliterate Gemma 4 QAT models (released Jun 6) — Q4_0 / W4A16 variants; E2B QAT runs in <1 GB. Real audience for uncensored quants on edge devices. Check whether biprojection on QAT weights holds up or needs re-quantization after.
- [ ] EGA is irrelevant for 12B — 12B Unified is dense (48 layers, no experts), so the dense biprojection path is the whole story. Noted to avoid re-testing.
