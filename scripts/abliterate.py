"""
Headless Gemma 4 abliteration experiment driver using heretic internals.

Usage:
    python abliterate.py run --model google/gemma-4-E2B-it --n-trials 50
    python abliterate.py run --model google/gemma-4-E2B-it --n-trials 50 --orthogonalize
    python abliterate.py run --model google/gemma-4-E2B-it --n-trials 50 --winsorize 0.95
    python abliterate.py results --checkpoint checkpoints/google--gemma-4-E2B-it.jsonl
    python abliterate.py save --checkpoint checkpoints/google--gemma-4-E2B-it.jsonl --trial 17 --output ./models/e2b-heretic
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import asdict
from os.path import commonprefix
from pathlib import Path

# Patch tqdm before other heretic imports
from heretic.progress import patch_tqdm

patch_tqdm()

import optuna
import torch
import torch.nn.functional as F
import transformers
from optuna import Trial, TrialPruned
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import StudyDirection
from optuna.trial import TrialState
from rich.console import Console
from rich.table import Table

from heretic.config import QuantizationMethod, RowNormalization, Settings
from heretic.evaluator import Evaluator
from heretic.model import AbliterationParameters, Model
from heretic.utils import (
    empty_cache,
    format_duration,
    get_trial_parameters,
    load_prompts,
    print_memory_usage,
)

console = Console()


# Topic-word markers that match disclaimers, not actual refusals.
# Stripping these fixes the false-positive problem discovered in the refusal audit.
TOPIC_MARKERS = {"violat", "prohibit", "illegal", "harmful", "inappropriate", "unethical", "ethical boundaries"}


def make_settings(args: argparse.Namespace) -> Settings:
    """Build a Settings object from CLI args, bypassing heretic's CLI parser."""
    tag = args.tag or ""
    checkpoint_dir = args.checkpoint_dir or "checkpoints"
    if tag:
        checkpoint_dir = os.path.join(checkpoint_dir, tag)

    kwargs: dict = dict(
        model=args.model,
        quantization=(
            QuantizationMethod.BNB_4BIT
            if args.quantize
            else QuantizationMethod.NONE
        ),
        n_trials=args.n_trials,
        n_startup_trials=args.n_startup,
        study_checkpoint_dir=checkpoint_dir,
        orthogonalize_direction=args.orthogonalize,
        winsorization_quantile=args.winsorize,
        row_normalization=(
            RowNormalization.FULL if args.row_norm else RowNormalization.NONE
        ),
        print_responses=args.verbose,
        kl_divergence_scale=args.kl_scale,
    )

    # Settings uses pydantic-settings CliSettingsSource which auto-parses sys.argv.
    # Temporarily clear argv so it doesn't conflict with our argparse.
    real_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        settings = Settings(**kwargs)
    finally:
        sys.argv = real_argv

    # Override refusal markers if requested
    if getattr(args, "refusal_markers_file", None):
        markers = Path(args.refusal_markers_file).read_text().strip().splitlines()
        settings.refusal_markers = [m.strip() for m in markers if m.strip()]
        console.print(f"[yellow]Using custom refusal markers from {args.refusal_markers_file} ({len(settings.refusal_markers)} markers)[/]")
    elif getattr(args, "strip_topic_markers", False):
        original_count = len(settings.refusal_markers)
        settings.refusal_markers = [m for m in settings.refusal_markers if m not in TOPIC_MARKERS]
        stripped = original_count - len(settings.refusal_markers)
        console.print(f"[yellow]Stripped {stripped} topic-word markers, {len(settings.refusal_markers)} remaining[/]")

    # Monkey-patch is_refusal for position-aware detection
    first_n = getattr(args, "first_n_words", None)
    if first_n:
        console.print(f"[yellow]Position-aware refusal detection: checking first {first_n} words only[/]")

    return settings


def patch_evaluator_position_aware(evaluator: Evaluator, first_n_words: int) -> None:
    """Monkey-patch the evaluator's is_refusal to only check the first N words."""
    original_is_refusal = evaluator.is_refusal

    def position_aware_is_refusal(response: str) -> bool:
        if not response.strip():
            return True

        # Truncate to first N words before checking markers
        words = response.split()
        truncated = " ".join(words[:first_n_words])

        # Apply same normalization as heretic
        truncated = truncated.lower().replace("*", "").replace("\u2019", "'")
        truncated = " ".join(truncated.split())

        for marker in evaluator.settings.refusal_markers:
            if marker.lower() in truncated:
                return True

        return False

    evaluator.is_refusal = position_aware_is_refusal


def patch_evaluator_robust_kl(evaluator: Evaluator) -> None:
    """Replace get_score's KL with a non-finite-robust version.

    Some architectures (e.g. gemma4_unified) emit hard -inf logits for reserved
    vocabulary tokens. heretic's F.kl_div(..., log_target=True) then evaluates
    exp(target) * (target - input) = exp(-inf) * (-inf - (-inf)) = 0 * nan = nan
    at those positions, which poisons the batchmean and yields KL=nan. We mask
    non-finite per-position terms before reducing. Identical to the original KL
    when every position is finite, so it is safe to apply unconditionally.
    """

    def robust_get_score() -> tuple[tuple[float, float], float, int]:
        print("  * Obtaining first-token probability distributions...")
        logprobs = evaluator.model.get_logprobs_batched(evaluator.good_prompts)
        target = evaluator.base_logprobs.float()  # P (baseline)
        inp = logprobs.float()                    # Q (abliterated)
        n = inp.shape[0]
        # log_target KL term, matching F.kl_div(log_target=True): P*(logP - logQ)
        term = target.exp() * (target - inp)
        finite = torch.isfinite(term)
        dropped = (~finite).sum().item()
        kl_divergence = (term[finite].sum() / n).item()
        if dropped:
            print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/] "
                  f"(robust; masked {dropped} non-finite vocab positions)")
        else:
            print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]")

        print("  * Counting model refusals...")
        refusals = evaluator.count_refusals()
        print(f"  * Refusals: [bold]{refusals}[/]/{len(evaluator.bad_prompts)}")

        refusals_score = refusals / len(evaluator.bad_prompts)
        return (refusals_score, kl_divergence), kl_divergence, refusals

    evaluator.get_score = robust_get_score


def checkpoint_path(settings: Settings) -> str:
    model_slug = "".join(
        (c if (c.isalnum() or c in ["_", "-"]) else "--") for c in settings.model
    )
    return os.path.join(settings.study_checkpoint_dir, f"{model_slug}.jsonl")


def run_experiment(args: argparse.Namespace) -> None:
    settings = make_settings(args)
    cp = checkpoint_path(settings)

    console.print(f"\n[bold]Experiment: {args.tag or 'default'}[/]")
    console.print(f"Model: [bold]{settings.model}[/]")
    console.print(f"Trials: {settings.n_trials} ({settings.n_startup_trials} startup)")
    console.print(f"Orthogonalize: {settings.orthogonalize_direction}")
    console.print(f"Winsorize: {settings.winsorization_quantile}")
    console.print(f"Row norm: {settings.row_normalization.value}")
    console.print(f"KL scale: {settings.kl_divergence_scale}")
    console.print(f"Checkpoint: {cp}")

    # Silence noise
    torch.set_grad_enabled(False)
    torch._dynamo.config.cache_size_limit = 64
    transformers.logging.set_verbosity_error()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    os.makedirs(settings.study_checkpoint_dir, exist_ok=True)

    # Check for existing checkpoint
    if os.path.exists(cp) and not args.restart:
        console.print(f"\n[yellow]Resuming from existing checkpoint[/]")
    elif os.path.exists(cp) and args.restart:
        os.unlink(cp)
        console.print(f"\n[yellow]Deleted old checkpoint, starting fresh[/]")

    lock_obj = JournalFileOpenLock(cp)
    backend = JournalFileBackend(cp, lock_obj=lock_obj)
    storage = JournalStorage(backend)

    # Load model
    model = Model(settings)
    console.print()
    print_memory_usage()

    # Load prompt datasets
    console.print(f"\nLoading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    console.print(f"* {len(good_prompts)} prompts loaded")

    console.print(f"\nLoading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    console.print(f"* {len(bad_prompts)} prompts loaded")

    # Auto batch size
    if settings.batch_size == 0:
        console.print("\nDetermining optimal batch size...")
        batch_size = 1
        best_batch_size = 1
        best_performance = -1.0

        while batch_size <= settings.max_batch_size:
            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]
            try:
                model.get_responses(prompts)  # warmup
                start = time.perf_counter()
                responses = model.get_responses(prompts)
                elapsed = time.perf_counter() - start
                response_lengths = [len(model.tokenizer.encode(r)) for r in responses]
                perf = sum(response_lengths) / elapsed
                console.print(f"  batch={batch_size}: {perf:.0f} tok/s")
                if perf > best_performance:
                    best_batch_size = batch_size
                    best_performance = perf
            except Exception:
                break
            batch_size *= 2

        settings.batch_size = best_batch_size
        console.print(f"  chosen: {settings.batch_size}")

    # Response prefix detection (CoT suppression)
    console.print("\nChecking for common response prefix...")
    prefix_prompts = good_prompts[:100] + bad_prompts[:100]
    responses = model.get_responses_batched(prefix_prompts)
    model.response_prefix = commonprefix(responses).rstrip(" ")

    recheck = False
    if model.response_prefix:
        recheck = True
        if model.response_prefix.startswith("<think>"):
            model.response_prefix = "<think></think>"
        elif model.response_prefix.startswith("<|channel|>analysis<|message|>"):
            model.response_prefix = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
        else:
            recheck = False

    if model.response_prefix:
        console.print(f"  prefix: {model.response_prefix!r}")
        if recheck:
            responses = model.get_responses_batched(prefix_prompts)
            extra = commonprefix(responses).rstrip(" ")
            if extra:
                model.response_prefix += extra
                console.print(f"  extended: {model.response_prefix!r}")
    else:
        console.print("  none found")

    # Build evaluator (computes baseline refusals + logprobs)
    evaluator = Evaluator(settings, model)
    if getattr(args, "first_n_words", None):
        patch_evaluator_position_aware(evaluator, args.first_n_words)

    # Compute refusal directions
    console.print("\nCalculating per-layer refusal directions...")
    good_residuals = model.get_residuals_batched(good_prompts)
    bad_residuals = model.get_residuals_batched(bad_prompts)

    good_means = good_residuals.mean(dim=0)
    bad_means = bad_residuals.mean(dim=0)
    refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    if settings.orthogonalize_direction:
        good_directions = F.normalize(good_means, p=2, dim=1)
        proj = torch.sum(refusal_directions * good_directions, dim=1)
        refusal_directions = refusal_directions - proj.unsqueeze(1) * good_directions
        refusal_directions = F.normalize(refusal_directions, p=2, dim=1)

    del good_residuals, bad_residuals
    empty_cache()

    # Optuna optimization
    trial_index = 0
    start_index = 0
    start_time = time.perf_counter()

    def objective(trial: Trial) -> tuple[float, float]:
        nonlocal trial_index
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        trial.suggest_categorical("direction_scope", ["global", "per layer"])
        last_layer = len(model.get_layers()) - 1

        direction_index = trial.suggest_float(
            "direction_index", 0.4 * last_layer, 0.9 * last_layer
        )
        if trial.params["direction_scope"] == "per layer":
            direction_index = None

        parameters = {}
        for component in model.get_abliterable_components():
            max_weight = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
            max_weight_pos = trial.suggest_float(
                f"{component}.max_weight_position", 0.6 * last_layer, 1.0 * last_layer
            )
            min_weight = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
            min_weight_dist = trial.suggest_float(
                f"{component}.min_weight_distance", 1.0, 0.6 * last_layer
            )
            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_pos,
                min_weight=min_weight * max_weight,
                min_weight_distance=min_weight_dist,
            )

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr(
            "parameters", {k: asdict(v) for k, v in parameters.items()}
        )

        console.print(f"\n[dim]Trial {trial_index}/{settings.n_trials}[/]", end=" ")
        model.reset_model()
        model.abliterate(refusal_directions, direction_index, parameters)
        score, kl_div, refusals = evaluator.get_score()

        elapsed = time.perf_counter() - start_time
        trial.set_user_attr("kl_divergence", kl_div)
        trial.set_user_attr("refusals", refusals)

        console.print(
            f"  → refusals={refusals}/{len(evaluator.bad_prompts)} "
            f"kl={kl_div:.4f} "
            f"[dim]({format_duration(elapsed)} elapsed)[/]"
        )
        return score

    def objective_wrapper(trial: Trial) -> tuple[float, float]:
        try:
            return objective(trial)
        except KeyboardInterrupt:
            trial.study.stop()
            raise TrialPruned()

    study = optuna.create_study(
        sampler=TPESampler(
            n_startup_trials=settings.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
        ),
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
        storage=storage,
        study_name="heretic",
        load_if_exists=True,
    )
    study.set_user_attr("settings", settings.model_dump_json())
    study.set_user_attr("finished", False)

    completed = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    start_index = trial_index = completed
    if completed > 0:
        console.print(f"\nResuming from trial {completed}")

    remaining = settings.n_trials - completed
    if remaining > 0:
        try:
            study.optimize(objective_wrapper, n_trials=remaining)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — results saved to checkpoint[/]")

    study.set_user_attr("finished", True)

    # Print results
    print_pareto_front(study, len(evaluator.bad_prompts))

    total = time.perf_counter() - start_time
    console.print(f"\n[bold green]Done![/] {format_duration(total)} total")
    console.print(f"Checkpoint: {cp}")

    # Auto-save best trial if requested
    if args.auto_save:
        best = get_best_trial(study)
        if best:
            save_dir = args.auto_save
            console.print(f"\nSaving best trial ({best.user_attrs['refusals']} refusals) to {save_dir}...")
            model.reset_model()
            model.abliterate(
                refusal_directions,
                best.user_attrs["direction_index"],
                {
                    k: AbliterationParameters(**v)
                    for k, v in best.user_attrs["parameters"].items()
                },
            )
            os.makedirs(save_dir, exist_ok=True)
            merged = model.get_merged_model()
            merged.save_pretrained(save_dir)
            del merged
            empty_cache()
            model.tokenizer.save_pretrained(save_dir)
            console.print(f"[bold green]Saved to {save_dir}[/]")


def get_best_trial(study):
    """Get the trial with fewest refusals among Pareto-optimal trials."""
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        return None

    sorted_trials = sorted(
        completed,
        key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]),
    )
    min_div = math.inf
    pareto = []
    for t in sorted_trials:
        kl = t.user_attrs["kl_divergence"]
        if kl < min_div:
            min_div = kl
            pareto.append(t)

    # Return trial with fewest refusals (first in pareto since sorted by refusals)
    return pareto[0] if pareto else sorted_trials[0]


def print_pareto_front(study, n_prompts: int) -> None:
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        console.print("[red]No completed trials[/]")
        return

    sorted_trials = sorted(
        completed,
        key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]),
    )
    min_div = math.inf
    pareto = []
    for t in sorted_trials:
        kl = t.user_attrs["kl_divergence"]
        if kl < min_div:
            min_div = kl
            pareto.append(t)

    table = Table(title="Pareto Front")
    table.add_column("Trial", style="bold")
    table.add_column("Refusals", justify="right")
    table.add_column("KL Div", justify="right")
    table.add_column("Direction", justify="right")

    for t in pareto:
        refusals = t.user_attrs["refusals"]
        kl = t.user_attrs["kl_divergence"]
        di = t.user_attrs.get("direction_index", "per-layer")
        if di is not None:
            di = f"{di:.1f}"
        kl_style = "green" if kl < 0.1 else ("yellow" if kl < 1.0 else "red")
        ref_style = "green" if refusals < 10 else ("yellow" if refusals < 50 else "red")
        table.add_row(
            str(t.user_attrs["index"]),
            f"[{ref_style}]{refusals}/{n_prompts}[/]",
            f"[{kl_style}]{kl:.4f}[/]",
            str(di),
        )

    console.print()
    console.print(table)


def load_model_and_data(
    args: argparse.Namespace,
) -> tuple[Settings, Model, list, list]:
    """Shared setup: load model + prompt datasets. Returns (settings, model, good, bad)."""
    settings = make_settings(args)

    torch.set_grad_enabled(False)
    torch._dynamo.config.cache_size_limit = 64
    transformers.logging.set_verbosity_error()

    model = Model(settings)
    console.print()
    print_memory_usage()

    console.print(f"\nLoading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    console.print(f"* {len(good_prompts)} prompts loaded")

    console.print(f"\nLoading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    console.print(f"* {len(bad_prompts)} prompts loaded")

    return settings, model, good_prompts, bad_prompts


def setup_model_prefix(
    settings: Settings, model: Model, good_prompts: list, bad_prompts: list
) -> None:
    """Detect and set response prefix (CoT suppression)."""
    console.print("\nChecking for common response prefix...")
    prefix_prompts = good_prompts[:100] + bad_prompts[:100]
    responses = model.get_responses_batched(prefix_prompts)
    model.response_prefix = commonprefix(responses).rstrip(" ")

    recheck = False
    if model.response_prefix:
        recheck = True
        if model.response_prefix.startswith("<think>"):
            model.response_prefix = "<think></think>"
        else:
            recheck = False

    if model.response_prefix:
        console.print(f"  prefix: {model.response_prefix!r}")
        if recheck:
            responses = model.get_responses_batched(prefix_prompts)
            extra = commonprefix(responses).rstrip(" ")
            if extra:
                model.response_prefix += extra
    else:
        console.print("  none found")


def compute_refusal_directions(
    settings: Settings,
    model: Model,
    good_prompts: list,
    bad_prompts: list,
    *,
    winsorize_quantile: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-layer refusal directions. Returns (directions, good_means, bad_means)."""
    console.print("\nCalculating per-layer refusal directions...")
    good_residuals = model.get_residuals_batched(good_prompts)
    bad_residuals = model.get_residuals_batched(bad_prompts)

    if winsorize_quantile < 1.0:
        console.print(f"  winsorizing at {winsorize_quantile}")
        for residuals in [good_residuals, bad_residuals]:
            abs_vals = residuals.abs()
            threshold = torch.quantile(abs_vals.float(), winsorize_quantile, dim=-1, keepdim=True)
            residuals.clamp_(-threshold, threshold)

    good_means = good_residuals.mean(dim=0)
    bad_means = bad_residuals.mean(dim=0)
    refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    del good_residuals, bad_residuals
    empty_cache()

    return refusal_directions, good_means, bad_means


# ---------------------------------------------------------------------------
# Norm-preserving biprojected abliteration (grimjim / jim-plus)
# ---------------------------------------------------------------------------


def compute_layer_quality(
    refusal_dir: torch.Tensor,
    harmful_mean: torch.Tensor,
    harmless_mean: torch.Tensor,
) -> float:
    """Composite quality metric: SNR * (1 - cos_sim) * purity_ratio."""
    harmful_norm = harmful_mean.float().norm()
    harmless_norm = harmless_mean.float().norm()
    snr = refusal_dir.float().norm() / max(harmful_norm, harmless_norm, torch.tensor(1e-8))

    cos_sim = F.cosine_similarity(
        harmful_mean.float().unsqueeze(0), harmless_mean.float().unsqueeze(0)
    ).item()

    # Purity: fraction of refusal orthogonal to harmless
    harmless_hat = F.normalize(harmless_mean.float(), dim=0)
    proj = (refusal_dir.float() @ harmless_hat) * harmless_hat
    refusal_orth = refusal_dir.float() - proj
    purity = refusal_orth.norm() / max(refusal_dir.float().norm(), 1e-8)

    return float(snr * (1.0 - cos_sim) * purity)


def modify_weight_norm_preserved(
    weight: torch.Tensor,
    refusal_dir: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Norm-preserving biprojected ablation on a 2D weight matrix.

    Decomposes each row into magnitude + direction, projects refusal out of
    direction only, recombines with original magnitude. Double-pass Gram-Schmidt.

    Weight shape is [out_features, in_features]. The refusal direction lives in
    the output (hidden state) space. We project it out of each row's contribution
    to that output space by left-multiplying: W' = W - scale * (r @ r^T) @ W,
    but done norm-preserving per row.
    """
    W = weight.float()
    r = F.normalize(refusal_dir.float(), dim=0)

    # r is in output space (dim = out_features). Compute how much each row
    # aligns with the refusal direction in output space.
    # For W [out, in], we want to remove the component of each column of W^T
    # that points along r. Equivalently: W_new = W - scale * outer(r, r) @ W
    # But we do it norm-preserving per row.

    # Per-row norms (each row is a vector in input space projected to one output dim)
    W_norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    W_dirs = W / W_norms

    # How much does each output dimension (row) align with refusal direction?
    # r has shape [out], W_dirs has shape [out, in]
    # We want: for each row i, scale its contribution by (1 - scale * r_i^2)
    # and redistribute: W_dirs_new[i] = W_dirs[i] - scale * r[i] * sum_j(r[j] * W_dirs[j])
    # This is: W_dirs_new = W_dirs - scale * r.unsqueeze(1) * (r @ W_dirs)
    refusal_component = r @ W_dirs  # [in] — the refusal direction's projection in input space
    proj = scale * r.unsqueeze(1) * refusal_component.unsqueeze(0)  # [out, in]
    W_dirs = W_dirs - proj
    W_dirs = F.normalize(W_dirs, dim=1)

    # Second pass for numerical stability
    refusal_component2 = r @ W_dirs
    proj2 = r.unsqueeze(1) * refusal_component2.unsqueeze(0)
    W_dirs = W_dirs - proj2
    W_dirs = F.normalize(W_dirs, dim=1)

    # Recombine with original magnitudes
    return (W_norms * W_dirs).to(weight.dtype)


def run_biprojection(args: argparse.Namespace) -> None:
    """Single-pass norm-preserving biprojected abliteration."""
    settings, model, good_prompts, bad_prompts = load_model_and_data(args)

    console.print(f"\n[bold]Biprojection experiment: {args.tag or 'default'}[/]")
    console.print(f"Winsorize: {args.winsorize}")
    console.print(f"Scale: {args.scale}")
    console.print(f"Top layers: {args.top_pct}%")

    # Batch size: use explicit override or auto-detect
    if getattr(args, "batch_size", None):
        settings.batch_size = args.batch_size
        console.print(f"\nUsing batch size: {settings.batch_size} (manual override)")
    elif settings.batch_size == 0:
        console.print("\nDetermining optimal batch size...")
        batch_size = 1
        best_batch_size = 1
        best_perf = -1.0
        while batch_size <= settings.max_batch_size:
            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]
            try:
                model.get_responses(prompts)
                start = time.perf_counter()
                responses = model.get_responses(prompts)
                elapsed = time.perf_counter() - start
                lengths = [len(model.tokenizer.encode(r)) for r in responses]
                perf = sum(lengths) / elapsed
                if perf > best_perf:
                    best_batch_size = batch_size
                    best_perf = perf
            except Exception:
                break
            batch_size *= 2
        settings.batch_size = best_batch_size

    if getattr(args, "skip_prefix", False):
        console.print("\nSkipping prefix detection (--skip-prefix)")
        model.response_prefix = ""
    else:
        setup_model_prefix(settings, model, good_prompts, bad_prompts)

    # Build evaluator (baseline refusals + logprobs)
    evaluator = None
    if not getattr(args, "no_eval", False):
        evaluator = Evaluator(settings, model)
        patch_evaluator_robust_kl(evaluator)
        if getattr(args, "first_n_words", None):
            patch_evaluator_position_aware(evaluator, args.first_n_words)

    # Compute refusal directions with optional winsorization
    refusal_directions, good_means, bad_means = compute_refusal_directions(
        settings, model, good_prompts, bad_prompts,
        winsorize_quantile=args.winsorize,
    )

    n_layers = len(model.get_layers())

    # Compute per-layer quality metric for layer selection
    layer_select = getattr(args, "layer_select", "snr")
    console.print(f"\nComputing layer quality metrics (method: {layer_select})...")
    qualities = []
    for i in range(n_layers):
        # refusal_directions[0] is embeddings, so layer i is at index i+1
        rd = refusal_directions[i + 1] if (i + 1) < refusal_directions.shape[0] else refusal_directions[i]
        gm = good_means[i + 1] if (i + 1) < good_means.shape[0] else good_means[i]
        bm = bad_means[i + 1] if (i + 1) < bad_means.shape[0] else bad_means[i]

        if layer_select == "cosmic":
            # COSMIC: use 1 - cosine_similarity as quality (lower similarity = more different = better target)
            cos_sim = F.cosine_similarity(bm.float().unsqueeze(0), gm.float().unsqueeze(0)).item()
            q = 1.0 - cos_sim
        else:
            # Default: composite SNR * (1-cos) * purity
            q = compute_layer_quality(rd, bm, gm)
        qualities.append((i, q))

    qualities.sort(key=lambda x: x[1], reverse=True)
    n_select = max(1, int(n_layers * args.top_pct / 100))
    selected_layers = sorted([idx for idx, _ in qualities[:n_select]])

    console.print(f"  selected {len(selected_layers)}/{n_layers} layers: {selected_layers}")
    console.print(f"  quality scores: {', '.join(f'L{i}={q:.3f}' for i, q in qualities[:n_select])}")

    # Orthogonalize refusal directions against harmless means (projected abliteration)
    console.print("\nOrthogonalizing refusal directions (biprojection)...")
    projected_dirs = []
    for i in range(refusal_directions.shape[0]):
        r = refusal_directions[i].float()
        h = good_means[i].float() if i < good_means.shape[0] else good_means[-1].float()
        h_hat = F.normalize(h, dim=0)
        # Double-pass Gram-Schmidt
        r = r - (r @ h_hat) * h_hat
        r = r - (r @ h_hat) * h_hat
        r = F.normalize(r, dim=0)
        projected_dirs.append(r)
    projected_dirs = torch.stack(projected_dirs)

    # Apply norm-preserving weight modification to selected layers
    console.print(f"\nApplying norm-preserving ablation (scale={args.scale})...")
    # Need to unwrap PeftModel to get at actual weights
    from peft import PeftModel
    base_model = model.model
    if isinstance(base_model, PeftModel):
        base_model = base_model.base_model.model

    layers = model.get_layers()
    modified_count = 0
    for layer_idx in selected_layers:
        layer = layers[layer_idx]
        # Direction for this layer (offset by 1 for embeddings)
        dir_idx = min(layer_idx + 1, projected_dirs.shape[0] - 1)
        r = projected_dirs[dir_idx]

        # Get the actual weight tensor from a possibly LoRA-wrapped module
        def get_base_weight(module: torch.nn.Module) -> torch.Tensor | None:
            # PEFT LoRA wraps nn.Linear — base weights at .base_layer.weight
            if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
                return module.base_layer.weight
            if hasattr(module, "weight"):
                return module.weight
            return None

        # Modify o_proj
        try:
            o_proj = layer.self_attn.o_proj
            w = get_base_weight(o_proj)
            if w is not None:
                w.data = modify_weight_norm_preserved(w.data, r, args.scale)
                modified_count += 1
            else:
                console.print(f"  [yellow]L{layer_idx} o_proj: no weight found ({type(o_proj).__name__})[/]")
        except Exception as e:
            console.print(f"  [red]L{layer_idx} o_proj error: {e}[/]")

        # Modify mlp.down_proj (shared dense MLP)
        try:
            down_proj = layer.mlp.down_proj
            w = get_base_weight(down_proj)
            if w is not None:
                w.data = modify_weight_norm_preserved(w.data, r, args.scale)
                modified_count += 1
            else:
                console.print(f"  [yellow]L{layer_idx} down_proj: no weight found ({type(down_proj).__name__})[/]")
        except Exception as e:
            console.print(f"  [red]L{layer_idx} down_proj error: {e}[/]")

    console.print(f"  modified {modified_count} weight matrices across {len(selected_layers)} layers")

    # Evaluate
    refusals = -1
    kl_div = -1.0
    if evaluator and not getattr(args, "no_eval", False):
        console.print("\nEvaluating abliterated model...")
        score, kl_div, refusals = evaluator.get_score()
        console.print(f"  Refusals: [bold]{refusals}/{len(evaluator.bad_prompts)}[/]")
        console.print(f"  KL divergence: [bold]{kl_div:.4f}[/]")
    else:
        console.print("\n[bold]Skipping evaluation (--no-eval)[/]")

    # Save results to JSON
    result = {
        "method": "biprojection",
        "model": args.model,
        "tag": args.tag,
        "refusals": refusals,
        "n_prompts": len(evaluator.bad_prompts) if evaluator else 0,
        "kl_divergence": kl_div,
        "scale": args.scale,
        "winsorize": args.winsorize,
        "top_pct": args.top_pct,
        "selected_layers": selected_layers,
        "n_layers": n_layers,
        "qualities": {str(i): round(q, 4) for i, q in qualities},
    }

    results_dir = args.results_dir or "results"
    os.makedirs(results_dir, exist_ok=True)
    tag = args.tag or "default"
    results_file = os.path.join(results_dir, f"biprojection-{tag}.json")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)
    console.print(f"  Results saved to {results_file}")

    # Save model if requested
    if args.auto_save:
        console.print(f"\nSaving to {args.auto_save}...")
        os.makedirs(args.auto_save, exist_ok=True)
        # Merge LoRA adapters and unwrap to get clean tensor names for GGUF conversion
        save_model = model.model
        from peft import PeftModel as PM
        if isinstance(save_model, PM):
            console.print("  Merging LoRA adapters...")
            save_model = save_model.merge_and_unload()
        save_model.save_pretrained(args.auto_save)
        model.tokenizer.save_pretrained(args.auto_save)
        console.print(f"[bold green]Saved[/]")


def show_results(args: argparse.Namespace) -> None:
    """Show results from an existing checkpoint."""
    cp = args.checkpoint
    if not os.path.exists(cp):
        console.print(f"[red]Checkpoint not found: {cp}[/]")
        sys.exit(1)

    lock_obj = JournalFileOpenLock(cp)
    backend = JournalFileBackend(cp, lock_obj=lock_obj)
    storage = JournalStorage(backend)

    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.load_study(study_name="heretic", storage=storage)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    console.print(f"\n[bold]Checkpoint:[/] {cp}")
    console.print(f"Completed trials: {len(completed)}")

    if not completed:
        return

    # Summary stats
    refusals = [t.user_attrs["refusals"] for t in completed]
    kls = [t.user_attrs["kl_divergence"] for t in completed]
    console.print(f"Refusals range: {min(refusals)}-{max(refusals)}")
    console.print(f"KL divergence range: {min(kls):.4f}-{max(kls):.4f}")

    # Get n_prompts from settings if available
    try:
        s = json.loads(study.user_attrs.get("settings", "{}"))
        split = s.get("bad_evaluation_prompts", {}).get("split", "test[:100]")
        # Parse count from split string like "test[:100]"
        n = int(split.split(":")[-1].rstrip("]"))
    except Exception:
        n = 100

    print_pareto_front(study, n)

    if args.json:
        best = get_best_trial(study)
        if best:
            result = {
                "trial": best.user_attrs["index"],
                "refusals": best.user_attrs["refusals"],
                "kl_divergence": best.user_attrs["kl_divergence"],
                "direction_index": best.user_attrs.get("direction_index"),
                "parameters": best.user_attrs.get("parameters"),
            }
            print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 abliteration experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="Run an abliteration experiment")
    run_p.add_argument("--model", required=True, help="HF model ID")
    run_p.add_argument("--n-trials", type=int, default=50)
    run_p.add_argument("--n-startup", type=int, default=15)
    run_p.add_argument("--tag", help="Experiment tag (used for checkpoint subdir)")
    run_p.add_argument("--checkpoint-dir", default="checkpoints")
    run_p.add_argument("--orthogonalize", action="store_true")
    run_p.add_argument("--winsorize", type=float, default=1.0)
    run_p.add_argument("--row-norm", action="store_true")
    run_p.add_argument("--kl-scale", type=float, default=1.0)
    run_p.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    run_p.add_argument("--verbose", action="store_true", help="Print responses")
    run_p.add_argument("--restart", action="store_true", help="Delete existing checkpoint")
    run_p.add_argument("--auto-save", metavar="DIR", help="Auto-save best trial to DIR")
    run_p.add_argument("--strip-topic-markers", action="store_true", help="Remove topic-word markers (illegal, harmful, etc.) that cause false positives")
    run_p.add_argument("--refusal-markers-file", help="Path to a file with one refusal marker per line")
    run_p.add_argument("--first-n-words", type=int, help="Only check first N words for refusal markers (position-aware detection)")

    # --- biprojection ---
    bp_p = sub.add_parser("biprojection", help="Single-pass norm-preserving biprojected abliteration")
    bp_p.add_argument("--model", required=True, help="HF model ID")
    bp_p.add_argument("--tag", help="Experiment tag")
    bp_p.add_argument("--winsorize", type=float, default=0.995, help="Winsorization quantile (default: 0.995)")
    bp_p.add_argument("--scale", type=float, default=1.0, help="Ablation scale factor")
    bp_p.add_argument("--top-pct", type=float, default=30, help="Percent of layers to ablate (by quality)")
    bp_p.add_argument("--quantize", action="store_true")
    bp_p.add_argument("--verbose", action="store_true")
    bp_p.add_argument("--auto-save", metavar="DIR")
    bp_p.add_argument("--results-dir", default="results")
    # These are needed by make_settings but unused for biprojection
    bp_p.add_argument("--n-trials", type=int, default=1)
    bp_p.add_argument("--n-startup", type=int, default=1)
    bp_p.add_argument("--checkpoint-dir", default="checkpoints")
    bp_p.add_argument("--orthogonalize", action="store_true", default=True)
    bp_p.add_argument("--row-norm", action="store_true")
    bp_p.add_argument("--kl-scale", type=float, default=1.0)
    bp_p.add_argument("--strip-topic-markers", action="store_true", help="Remove topic-word markers (illegal, harmful, etc.) that cause false positives")
    bp_p.add_argument("--refusal-markers-file", help="Path to a file with one refusal marker per line")
    bp_p.add_argument("--first-n-words", type=int, help="Only check first N words for refusal markers (position-aware detection)")
    bp_p.add_argument("--layer-select", choices=["snr", "cosmic"], default="snr", help="Layer selection metric: snr (default composite) or cosmic (cosine similarity)")
    bp_p.add_argument("--batch-size", type=int, help="Override batch size (skip auto-detection)")
    bp_p.add_argument("--skip-prefix", action="store_true", help="Skip prefix detection (saves 30+ min on large models)")
    bp_p.add_argument("--no-eval", action="store_true", help="Skip baseline + post-abliteration evaluation (for save-only runs)")

    # --- results ---
    res_p = sub.add_parser("results", help="Show results from a checkpoint")
    res_p.add_argument("--checkpoint", required=True, help="Path to checkpoint .jsonl")
    res_p.add_argument("--json", action="store_true", help="Output best trial as JSON")

    args = parser.parse_args()

    if args.command == "run":
        run_experiment(args)
    elif args.command == "biprojection":
        run_biprojection(args)
    elif args.command == "results":
        show_results(args)


if __name__ == "__main__":
    main()
