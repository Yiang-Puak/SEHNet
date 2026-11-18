#!/usr/bin/env python3
"""Fixed-weight BP SEHNet Top/Random/Bottom local-segment occlusion.

The cohort and segment ranking are fixed before any masking.  This is a
post-hoc mechanism analysis: it makes no training, threshold tuning, or DAG
propagation, and reports every sampled protein--term pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

from sehnet.evaluate import build_model  # noqa: E402
from sehnet.metrics import Ontology  # noqa: E402


def final_checkpoint_epochs(run_dir: Path) -> tuple[int, ...]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    final_epoch = int(config["epochs"])
    if final_epoch < 5:
        raise ValueError("At least five training epochs are required")
    epochs = tuple(range(final_epoch - 4, final_epoch + 1))
    for epoch in epochs:
        path = run_dir / "trajectory_checkpoints" / f"epoch_{epoch:03d}.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Missing final-five checkpoint: {path.name}")
    return epochs

DEPTH_BINS = ("1-2", "3-4", "5-6", "7+")
FREQUENCY_BINS = ("1-4", "5-19", "20-99", "100+")
BOOTSTRAP_REPLICATES = 10000


def parse_args() -> argparse.Namespace:
    default_run = (
        ROOT / "results" / "bp_analysis"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_run / "bp_analysis" / "occlusion",
    )
    parser.add_argument("--max-per-stratum", type=int, default=20)
    parser.add_argument("--minimum-raw-probability", type=float, default=0.5)
    parser.add_argument("--random-control-draws", type=int, default=20)
    parser.add_argument(
        "--rank-mode",
        choices=("routing", "signed_support"),
        default="signed_support",
        help="Pre-mask segment score; signed_support follows SEHNet child-parent evidence.",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_pids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and line.strip().upper() != "PROTEINS"
    ]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No rows were produced")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def direct_parent_indices(terms: list[str], go_obo: Path) -> list[np.ndarray]:
    term_index = {term: i for i, term in enumerate(terms)}
    ontology = Ontology(go_obo)
    return [
        np.asarray(
            sorted(
                {
                    term_index[parent]
                    for parent in ontology.ont.get(term, {}).get("is_a", [])
                    if parent in term_index and parent != term
                }
            ),
            dtype=np.int64,
        )
        for term in terms
    ]


def choose_cohort(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    pids: list[str],
    manifest: list[dict[str, str]],
    parents: list[np.ndarray],
    valid_counts: np.ndarray,
    rng: np.random.Generator,
    max_per_stratum: int,
    minimum_raw_probability: float,
) -> list[dict[str, object]]:
    """Sample only correct, confident predictions; no masking result is read."""

    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for protein_i, term_i in zip(*np.nonzero(labels)):
        row = manifest[int(term_i)]
        key = (row["depth_bin"], row["frequency_bin"])
        if (
            key[0] in DEPTH_BINS
            and key[1] in FREQUENCY_BINS
            and len(parents[int(term_i)]) > 0
            and valid_counts[int(protein_i)] >= 3
            and raw_scores[int(protein_i), int(term_i)] >= minimum_raw_probability
        ):
            groups[key].append((int(protein_i), int(term_i)))

    chosen: list[dict[str, object]] = []
    used_proteins: set[int] = set()
    for depth in DEPTH_BINS:
        for frequency in FREQUENCY_BINS:
            key = (depth, frequency)
            candidates = groups[key]
            if candidates:
                candidates = [candidates[i] for i in rng.permutation(len(candidates))]
            taken = 0
            for protein_i, term_i in candidates:
                if protein_i in used_proteins:
                    continue
                used_proteins.add(protein_i)
                chosen.append(
                    {
                        "pair_id": f"bp_{len(chosen) + 1:03d}",
                        "protein_index": protein_i,
                        "pid": pids[protein_i],
                        "term_index": term_i,
                        "target_term": manifest[term_i]["term"],
                        "depth_bin": depth,
                        "frequency_bin": frequency,
                        "train_positive_count": int(manifest[term_i]["train_positive_count"]),
                    }
                )
                taken += 1
                if taken == max_per_stratum:
                    break
    if not chosen:
        raise RuntimeError("No eligible correct, confident BP predictions")
    return chosen


@torch.inference_mode()
def segment_importance(
    model: torch.nn.Module,
    local_segments: torch.Tensor,
    local_mask: torch.Tensor,
    target_indices: torch.Tensor,
    rank_mode: str,
    parent_child_indices: torch.Tensor | None = None,
    parent_indices: torch.Tensor | None = None,
    parent_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a pre-mask SEHNet route score or signed child-parent support proxy."""

    module = model.latent_domain_fusion
    if module is None or not module.label_readout or not module.competitive_segment_assignment:
        raise ValueError("This analysis requires the full competitive SEHNet label readout")
    mask = local_mask.bool()
    valid = mask[:, None, :]
    normalized_layers = torch.stack(
        [module.layer_norms[i](local_segments[:, i]) for i in range(module.num_local_layers)],
        dim=1,
    )
    projected = module.local_projection(normalized_layers.mean(dim=1))
    queries = module.domain_query_norm(module.domain_queries)
    logits = torch.einsum("md,bsd->bms", queries.to(projected.dtype), projected)
    logits = logits * (module.attention_dim ** -0.5)
    segment_weights = torch.softmax(logits.float().masked_fill(~valid, -1e4), dim=-2)
    segment_weights = segment_weights * valid.float()
    segment_weights = segment_weights / segment_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    tokens = torch.einsum("bms,bsd->bmd", segment_weights.to(projected.dtype), projected)
    keys = module.domain_norm(tokens)
    locators = module.prototype_proj(model.classifier.weight.detach().float())
    target_locators = locators[target_indices]
    term_to_domain = torch.softmax(
        torch.einsum("bmd,bd->bm", keys, target_locators.to(keys.dtype))
        * (module.attention_dim ** -0.5),
        dim=-1,
    )
    routing = torch.einsum("bm,bms->bs", term_to_domain, segment_weights)
    if rank_mode == "routing":
        return routing
    parent_child_indices = (
        module.parent_child_indices
        if parent_child_indices is None
        else parent_child_indices
    )
    parent_indices = module.parent_indices if parent_indices is None else parent_indices
    parent_counts = module.parent_counts if parent_counts is None else parent_counts
    parent_sums = torch.zeros_like(locators)
    parent_sums.index_add_(0, parent_child_indices, locators[parent_indices])
    discriminators = torch.nn.functional.normalize(
        locators - parent_sums / parent_counts.clamp_min(1).to(locators.dtype)[:, None],
        dim=-1,
        eps=1e-6,
    )
    discriminators = discriminators * parent_counts.gt(0)[:, None].to(discriminators.dtype)
    segment_directions = torch.einsum(
        "bsd,bd->bs",
        torch.nn.functional.normalize(projected.float(), dim=-1, eps=1e-6),
        discriminators[target_indices],
    )
    gates = torch.tanh(module.depth_gates)[module.label_depth_groups[target_indices]]
    return routing * segment_directions * gates[:, None]


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    draws = rng.choice(values, size=(BOOTSTRAP_REPLICATES, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def sign_test(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = min(int(np.count_nonzero(nonzero > 0)), int(np.count_nonzero(nonzero < 0)))
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def summarize(name: str, values: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    low, high = bootstrap_mean_ci(values, rng)
    return {
        "comparison": name,
        "n_pairs": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "top_larger_fraction": float(np.mean(values > 0)),
        "bootstrap95_mean_low": low,
        "bootstrap95_mean_high": high,
        "two_sided_sign_test_p": sign_test(values),
    }


def main() -> int:
    args = parse_args()
    if min(args.max_per_stratum, args.batch_size, args.random_control_draws) <= 0:
        raise ValueError("--max-per-stratum, --batch-size, and --random-control-draws must be positive")
    run_dir = args.run_dir.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    analysis_dir = run_dir / "bp_analysis"
    with (run_dir / "last_five_test_occlusion_scores.pkl").open("rb") as handle:
        scores = pickle.load(handle)
    terms = [str(x) for x in scores["terms"]]
    labels = np.asarray(scores["labels"], dtype=np.uint8)
    raw_scores = np.asarray(scores["scores_raw"], dtype=np.float32)
    manifest = read_rows(analysis_dir / "term_group_manifest.csv")
    if [row["term"] for row in manifest] != terms:
        raise ValueError("Term manifest and score bundle disagree")

    with (data_root / "features_3B" / "bp_test_esm2.pkl").open("rb") as handle:
        esm_features = pickle.load(handle)
    pids = [pid for pid in read_pids(data_root / "real_dataset" / "splits" / "bp_test_pid_list.txt") if pid in esm_features]
    if len(pids) != len(labels):
        raise ValueError("Test PID order does not match score bundle")
    local_dir = data_root / "features_150M_multilayer_segments" / "bp"
    local_index = {pid: i for i, pid in enumerate(read_pids(local_dir / "test_pids.txt"))}
    if any(pid not in local_index for pid in pids):
        raise ValueError("Some BP test proteins lack local features")
    segment_mem = np.load(local_dir / "test_segments.npy", mmap_mode="r")
    global_mem = np.load(local_dir / "test_global.npy", mmap_mode="r")
    mask_mem = np.load(local_dir / "test_segment_mask.npy", mmap_mode="r")
    valid_counts = np.asarray([mask_mem[local_index[pid]].sum() for pid in pids])
    parents = direct_parent_indices(terms, data_root / "go.obo")
    rng = np.random.default_rng(args.seed)
    cohort = choose_cohort(
        labels, raw_scores, pids, manifest, parents, valid_counts, rng,
        args.max_per_stratum, args.minimum_raw_probability,
    )
    print(f"[cohort] sampled {len(cohort)} fixed pairs", flush=True)

    local_segments = []
    local_masks = []
    esms = []
    for row in cohort:
        local_row = local_index[str(row["pid"])]
        mask = np.asarray(mask_mem[local_row], dtype=np.bool_)
        segments = np.asarray(segment_mem[local_row], dtype=np.float32).copy()
        segments[:, mask] -= np.asarray(global_mem[local_row], dtype=np.float32)[:, None, :]
        segments[:, ~mask] = 0.0
        local_segments.append(segments)
        local_masks.append(mask)
        value = esm_features[str(row["pid"])]
        esms.append(np.asarray(value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value, dtype=np.float32).reshape(-1))
    local_segments_t = torch.from_numpy(np.stack(local_segments)).to(device)
    local_masks_t = torch.from_numpy(np.stack(local_masks)).to(device)
    esm_t = torch.from_numpy(np.stack(esms)).to(device)
    target_t = torch.tensor([int(row["term_index"]) for row in cohort], device=device)

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model = build_model(config, len(terms), terms, data_root, "bp", device).to(device)
    importance = np.zeros((len(cohort), 32), dtype=np.float64)
    epochs = final_checkpoint_epochs(run_dir)
    probabilities = np.zeros((len(epochs), 3 + args.random_control_draws, len(cohort), 2), dtype=np.float64)
    checkpoint_paths = [run_dir / "trajectory_checkpoints" / f"epoch_{epoch:03d}.pth" for epoch in epochs]
    if any(not path.is_file() for path in checkpoint_paths):
        raise FileNotFoundError("The final five checkpoints are required")

    for checkpoint_i, checkpoint_path in enumerate(checkpoint_paths):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        importance += segment_importance(
            model, local_segments_t, local_masks_t, target_t, args.rank_mode
        ).cpu().numpy()
    importance /= len(epochs)
    top_segments = np.argmax(np.where(np.stack(local_masks), importance, -np.inf), axis=1)
    bottom_segments = np.argmin(np.where(np.stack(local_masks), importance, np.inf), axis=1)
    random_segments = np.empty((args.random_control_draws, len(cohort)), dtype=np.int64)
    for pair_i, (mask, top, bottom) in enumerate(zip(local_masks, top_segments, bottom_segments)):
        candidates = np.flatnonzero(mask & (np.arange(32) != top) & (np.arange(32) != bottom))
        random_segments[:, pair_i] = rng.choice(candidates, size=args.random_control_draws, replace=True)

    for checkpoint_i, checkpoint_path in enumerate(checkpoint_paths):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        selections = (top_segments, *random_segments, bottom_segments)
        variants = local_segments_t.repeat((1 + len(selections), 1, 1, 1))
        masks = local_masks_t.repeat((1 + len(selections), 1))
        for variant_i, selected in enumerate(selections, start=1):
            positions = torch.arange(len(cohort), device=device) + variant_i * len(cohort)
            masks[positions, torch.from_numpy(selected).to(device)] = False
            variants[positions, :, torch.from_numpy(selected).to(device), :] = 0.0
        esm_variants = esm_t.repeat((1 + len(selections), 1))
        for start in range(0, len(esm_variants), args.batch_size):
            end = min(start + args.batch_size, len(esm_variants))
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(esm_variants[start:end], local_segments=variants[start:end], local_mask=masks[start:end]).probs.float().detach().cpu().numpy()
            for local_i, global_i in enumerate(range(start, end)):
                variant_i, pair_i = divmod(global_i, len(cohort))
                target_i = int(cohort[pair_i]["term_index"])
                probabilities[checkpoint_i, variant_i, pair_i, 0] = out[local_i, target_i]
                probabilities[checkpoint_i, variant_i, pair_i, 1] = out[local_i, parents[target_i]].mean()
        print(f"[checkpoint] {checkpoint_path.name}", flush=True)

    mean_probabilities = probabilities.mean(axis=0)
    baseline = mean_probabilities[0]
    margins = logit(mean_probabilities[:, :, 0]) - logit(mean_probabilities[:, :, 1])
    top_drop = baseline[:, 0] - mean_probabilities[1, :, 0]
    random_drops = baseline[None, :, 0] - mean_probabilities[2:-1, :, 0]
    bottom_drop = baseline[:, 0] - mean_probabilities[-1, :, 0]
    top_margin_drop = margins[0] - margins[1]
    random_margin_drops = margins[0][None, :] - margins[2:-1]
    bottom_margin_drop = margins[0] - margins[-1]
    random_drop_mean = random_drops.mean(axis=0)
    random_margin_drop_mean = random_margin_drops.mean(axis=0)
    rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    for i, row in enumerate(cohort):
        for draw_i in range(args.random_control_draws):
            random_rows.append({
                "pair_id": row["pair_id"],
                "draw": draw_i + 1,
                "segment_0based": int(random_segments[draw_i, i]),
                "target_probability_drop": float(random_drops[draw_i, i]),
                "child_parent_logit_margin_drop": float(random_margin_drops[draw_i, i]),
            })
        rows.append({
            **{k: v for k, v in row.items() if k not in {"protein_index", "term_index"}},
            "valid_segments": int(local_masks[i].sum()),
            "top_segment_0based": int(top_segments[i]),
            "random_segments_0based": ";".join(str(x) for x in random_segments[:, i]),
            "bottom_segment_0based": int(bottom_segments[i]),
            "baseline_target_raw_probability": float(baseline[i, 0]),
            "baseline_parent_mean_raw_probability": float(baseline[i, 1]),
            "top_target_probability_drop": float(top_drop[i]),
            "mean_random_target_probability_drop": float(random_drop_mean[i]),
            "random_target_probability_drop_sd": float(random_drops[:, i].std(ddof=0)),
            "bottom_target_probability_drop": float(bottom_drop[i]),
            "top_child_parent_logit_margin_drop": float(top_margin_drop[i]),
            "mean_random_child_parent_logit_margin_drop": float(random_margin_drop_mean[i]),
            "random_child_parent_logit_margin_drop_sd": float(random_margin_drops[:, i].std(ddof=0)),
            "bottom_child_parent_logit_margin_drop": float(bottom_margin_drop[i]),
        })
    write_rows(output_dir / "pair_results.csv", rows)
    write_rows(output_dir / "random_control_draws.csv", random_rows)
    comparisons = []
    for metric, top, random_mean, bottom in (
        ("target_probability_drop", top_drop, random_drop_mean, bottom_drop),
        ("child_parent_logit_margin_drop", top_margin_drop, random_margin_drop_mean, bottom_margin_drop),
    ):
        comparisons += [
            summarize(f"top_minus_random_{metric}", top - random_mean, rng),
            summarize(f"top_minus_bottom_{metric}", top - bottom, rng),
        ]
    protocol = {
        "analysis_type": "post-hoc fixed-weight BP mechanism analysis; no training, threshold tuning, or DAG propagation",
        "checkpoints": list(epochs),
        "cohort": "one randomly sampled, correct BP annotation per protein within each depth-frequency stratum; raw mean probability >= configured fixed threshold; all sampled pairs reported",
        "rank_mode": args.rank_mode,
        "segment_rank": (
            "mean over checkpoints of sum_m(term-to-domain attention[c,m] * competitive domain-to-segment weight[m,i])"
            if args.rank_mode == "routing"
            else "mean over checkpoints of route score times the segment's normalized child-minus-direct-parent discriminator compatibility and its learned depth gate"
        ),
        "controls": "independent uniform draws from valid segments excluding Top and Bottom, plus the lowest pre-mask-ranked valid segment",
        "random_control_draws": args.random_control_draws,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_interval": "two-sided 95% percentile interval",
        "acceptance": "top-minus-random target-probability-drop has bootstrap 95% CI entirely above zero and two-sided paired sign-test p < 0.05",
        "seed": args.seed,
        "max_per_stratum": args.max_per_stratum,
        "minimum_raw_probability": args.minimum_raw_probability,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summary = {
        **protocol,
        "n_pairs": len(rows),
        "stratum_counts": {f"{d}|{f}": sum(r["depth_bin"] == d and r["frequency_bin"] == f for r in rows) for d in DEPTH_BINS for f in FREQUENCY_BINS},
        "comparisons": comparisons,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[complete] {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
