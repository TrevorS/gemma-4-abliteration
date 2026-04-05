# Abliteration Research Loop

This directory contains an automated research loop for abliterating Gemma 4 models.

## How It Works

1. Read `STATE.md` for current best results and model under test
2. Read `IDEAS.md` for the prioritized backlog of techniques to try
3. Pick the highest-priority untried idea
4. Run the experiment using scripts in `scripts/`
5. Record results in `experiments/` as JSON
6. Update `STATE.md` if the result is a new best
7. Mark the idea as done in `IDEAS.md` with results summary
8. Add any new ideas that came up during the experiment
9. Update `ABLITERATION.md` results table

## Directory Layout

```
abliteration/
├── CLAUDE.md           # This file — instructions for the research agent
├── STATE.md            # Current best results, model under test, what works/doesn't
├── IDEAS.md            # Prioritized backlog of techniques to try
├── ABLITERATION.md     # Full research doc with results tables and paper references
├── scripts/
│   └── abliterate.py   # Experiment driver (heretic Optuna + biprojection)
├── experiments/        # JSON results from each experiment run
├── checkpoints/        # Heretic Optuna checkpoints
├── prompts/            # Custom prompt datasets
└── models/             # Saved abliterated model weights
```

## Running Experiments

All experiments use heretic's Python environment:

```bash
PYTHON=/home/trevor/.local/share/uv/tools/heretic-llm/bin/python
HF_DATASETS_CACHE=/tmp/hf_datasets_cache

# Biprojection (single-pass, ~2 min)
$PYTHON scripts/abliterate.py biprojection --model google/gemma-4-E2B-it --tag my-tag --top-pct 70 --results-dir experiments/

# Heretic Optuna (multi-trial, ~6 min for 50 trials)
$PYTHON scripts/abliterate.py run --model google/gemma-4-E2B-it --tag my-tag --n-trials 50

# View heretic results
$PYTHON scripts/abliterate.py results --checkpoint checkpoints/my-tag/google--gemma-4-E2B-it.jsonl
```

## Research Loop Guidelines

- **One experiment at a time** on the GPU — parallel runs cause contention and slow everything down
- **Record everything** — every experiment gets a JSON in `experiments/`, every finding goes in ABLITERATION.md
- **Think before running** — read STATE.md and IDEAS.md, form a hypothesis, predict the outcome, then run
- **Compare fairly** — same model, same evaluation prompts (mlabonne defaults), same metrics (refusals/100, KL divergence)
- **Know when to move on** — if 3 experiments in a row don't beat the current best on a model size, graduate to the next model
- **Journal decisions** — use `mcp__cj__journal_add` to log reasoning, not just results
