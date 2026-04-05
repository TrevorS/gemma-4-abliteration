# Red Team Eval Research

## Key Benchmarks (HuggingFace Available)

| Benchmark | Dataset ID | Size | Scoring | Notes |
|-----------|-----------|------|---------|-------|
| HarmBench | `walledai/HarmBench` | 510 behaviors | Llama 2 classifier, binary+continuous | Gold standard, modular framework |
| AdvBench | `walledai/AdvBench` | 520 behaviors | ASR only (refusal counting) | Simple but limited |
| JailbreakBench | `JailbreakBench/JBB-Behaviors` | 100 misuse + 100 benign | ASR with artifacts | MIT licensed, precomputed jailbreaks |
| StrongREJECT | (embedded in other benchmarks) | — | 0-1 continuous harm score | Best human-judge agreement |

## Attack Tools

| Tool | Strategy | Repo | Notes |
|------|----------|------|-------|
| PAIR | LLM-as-attacker, iterative refinement | `patrickrchao/JailbreakingLLMs` | Black-box, <20 queries |
| GCG | Gradient-based suffix optimization | `llm-attacks/llm-attacks`, `GraySwanAI/nanoGCG` | White-box, transferable |
| AutoDAN | Genetic algorithm for stealthy jailbreaks | `SheltonLiu-N/AutoDAN` | Coherent attacks |
| TAP | Tree search extending PAIR | Integrated in Giskard/DeepTeam | Query-efficient |

## Eval Frameworks

- **HarmBench** (`centerforaisafety/HarmBench`) — modular, extensible, our best bet
- **Promptfoo** (`promptfoo/promptfoo`) — NIST/OWASP presets, 50+ providers
- **DeepTeam** (`confident-ai/deepteam`) — 40+ vulnerability types

## What We Need Beyond Refusal Counting

1. **Refusal-aware classification** — detect compliance after initial refusal
2. **Harm quantization** — continuous scoring of response usefulness to attacker
3. **Multi-turn robustness** — test defense persistence across turns
4. **Behavioral coverage** — map attacks to risk category taxonomy
5. **Calibrated judge model** — LLM classifier instead of keyword matching
