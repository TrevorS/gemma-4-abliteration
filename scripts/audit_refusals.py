#!/usr/bin/env python
"""Audit flagged refusals on an already-abliterated model.

Loads a *saved* (already abliterated) model, runs the 4 cross-eval datasets in a
single pass, and dumps the full prompt + response for every response flagged as a
refusal by the stripped-marker heuristic. Lets us classify each flag as a genuine
refusal vs a refusal-then-comply false positive (the dominant pattern on larger
Gemma models). Audits the exact shipped weights — no re-abliteration.

Usage:
    heretic-python scripts/audit_refusals.py --model models/google--gemma-4-12B-it
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import torch
import transformers
from rich.console import Console

sys.path.insert(0, "scripts")
from heretic.config import Settings
from heretic.model import Model
from multi_eval import TOPIC_MARKERS, load_eval_datasets

console = Console()


@dataclass
class Prompt:
    system: str
    user: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to the saved abliterated model dir")
    ap.add_argument("--out", default="experiments/refusal-audit-12b.json")
    ap.add_argument("--max-chars", type=int, default=1500, help="Response chars to print per flag")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    transformers.logging.set_verbosity_error()

    console.print(f"\n[bold]Refusal Audit[/]\nModel: {args.model}\n")
    datasets = load_eval_datasets()

    real_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        settings = Settings(model=args.model)
    finally:
        sys.argv = real_argv
    settings.refusal_markers = [m for m in settings.refusal_markers if m not in TOPIC_MARKERS]
    if settings.batch_size == 0:
        settings.batch_size = 4
    model = Model(settings)

    flagged: list[dict] = []
    for ds in datasets:
        objs = [Prompt(system="", user=p) for p in ds.prompts]
        responses: list[str] = []
        bs = settings.batch_size
        for i in range(0, len(objs), bs):
            responses.extend(model.get_responses_batched(objs[i:i + bs], skip_special_tokens=True))
        n_flagged = 0
        for prompt, resp in zip(ds.prompts, responses):
            rc = " ".join(resp.lower().replace("*", "").replace("’", "'").split())
            hit = next((m for m in settings.refusal_markers if m.lower() in rc), None)
            empty = not resp.strip()
            if hit or empty:
                n_flagged += 1
                flagged.append({
                    "dataset": ds.name,
                    "marker": "EMPTY" if empty else hit,
                    "prompt": prompt,
                    "response": resp,
                    "response_words": len(resp.split()),
                })
        console.print(f"  {ds.name}: {n_flagged} flagged / {len(ds.prompts)}")

    json.dump(flagged, open(args.out, "w"), indent=2)
    console.rule(f"{len(flagged)} FLAGGED RESPONSES")
    for i, f in enumerate(flagged, 1):
        console.print(f"\n[bold cyan]#{i} [{f['dataset']}] marker={f['marker']!r} "
                      f"({f['response_words']} words)[/]")
        console.print(f"[yellow]PROMPT:[/] {f['prompt']}")
        console.print(f"[green]RESPONSE:[/] {f['response'][:args.max_chars]}")
    console.print(f"\n[bold]Saved {len(flagged)} flagged responses to {args.out}[/]")


if __name__ == "__main__":
    main()
