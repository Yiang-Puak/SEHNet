#!/usr/bin/env python3
"""Evaluate the final five training checkpoints without retrieval evidence."""

from __future__ import annotations

import argparse
import json
import pickle as pkl
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import get_path
from .data import SEHNetDataset, load_terms, sehnet_collate
from .metrics import DAG_propagation, Ontology, calculate_cafa_fmax_aupr
from .model import FunctionalHypothesisEnsemble
from .training_utils import (
    load_go_depth_groups,
    load_go_direct_parent_edges,
    load_go_shuffled_parent_edges,
    load_go_semantics,
    load_parent_contrast_go_semantics,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--ont", choices=("mf", "bp", "cc"))
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="test",
        help="Dataset split to predict; test preserves the original output names.",
    )
    parser.add_argument("--epochs", type=int, nargs="+")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-tag",
        default="",
        help=(
            "Optional suffix for output files; the default "
            "preserves the formal protocol filenames."
        ),
    )
    parser.add_argument(
        "--disable-local-evidence",
        action="store_true",
        help=(
            "Same-checkpoint mechanistic ablation: zero learned local depth "
            "gates after loading each checkpoint"
        ),
    )
    return parser.parse_args()


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def discover_epochs(checkpoint_dir: Path) -> List[int]:
    epochs = []
    for path in checkpoint_dir.glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def build_model(
    config: Dict[str, object],
    class_nums: int,
    terms: Sequence[str],
    data_root: Path,
    ontology: str,
    device: torch.device,
) -> torch.nn.Module:
    ensemble_size = int(config.get("ensemble_size", 1))
    hidden_dim = int(config.get("hidden_dim", 1024))
    dropout = float(config.get("dropout", 0.2))
    if ensemble_size == 8:
        semantic_router = bool(config.get("semantic_router", False))
        local_semantic_attention = bool(
            config.get("local_semantic_attention", False)
        )
        local_query_source = str(
            config.get("local_query_source", "semantic")
        )
        local_multilayer_segments = bool(
            config.get("local_multilayer_segments", False)
        )
        latent_domain_fusion = bool(
            config.get("latent_domain_fusion", False)
        )
        latent_domain_layer_contrast = bool(
            config.get("latent_domain_layer_contrast", False)
        )
        latent_domain_label_readout = bool(
            config.get("latent_domain_label_readout", False)
        )
        latent_domain_normalize_label_queries = bool(
            config.get("latent_domain_normalize_label_queries", False)
        )
        latent_domain_adaptive_label_query_norm = bool(
            config.get("latent_domain_adaptive_label_query_norm", False)
        )
        latent_domain_confidence_direction_routing = bool(
            config.get(
                "latent_domain_confidence_direction_routing",
                False,
            )
        )
        latent_domain_ic_semantic_residual = bool(
            config.get("latent_domain_ic_semantic_residual", False)
        )
        latent_domain_readout_mode = str(
            config.get("latent_domain_readout_mode", "term_specific")
        )
        latent_domain_discriminator_mode = str(
            config.get(
                "latent_domain_discriminator_mode", "child_minus_parent"
            )
        )
        latent_parent_mapping = str(
            config.get("latent_parent_mapping", "direct")
        )
        local_classifier_discriminator = str(
            config.get(
                "local_classifier_discriminator",
                "parent_contrast",
            )
        )
        go_semantics = (
            load_go_semantics(
                ontology,
                terms,
                data_root,
            ).to(device)
            if semantic_router
            else None
        )
        local_go_semantics = None
        local_parent_contrast = None
        local_depth_groups = None
        local_parent_child_indices = None
        local_parent_indices = None
        local_parent_counts = None
        latent_label_depth_groups = None
        latent_parent_child_indices = None
        latent_parent_indices = None
        latent_parent_counts = None
        latent_label_semantics = None
        latent_label_semantic_weights = None
        if local_semantic_attention:
            if local_query_source == "classifier":
                local_depth_groups, _ = load_go_depth_groups(
                    ontology,
                    terms,
                    data_root,
                )
                local_go_semantics = torch.zeros(
                    len(terms),
                    1,
                    dtype=torch.float32,
                )
                local_parent_contrast = torch.zeros_like(
                    local_go_semantics
                )
            else:
                (
                    local_go_semantics,
                    local_parent_contrast,
                    local_depth_groups,
                    _,
                ) = load_parent_contrast_go_semantics(
                    ontology,
                    terms,
                    data_root,
                )
            local_go_semantics = local_go_semantics.to(device)
            local_parent_contrast = local_parent_contrast.to(device)
            local_depth_groups = local_depth_groups.to(device)
            if (
                local_query_source == "classifier"
                and local_classifier_discriminator == "parent_contrast"
            ):
                (
                    local_parent_child_indices,
                    local_parent_indices,
                    local_parent_counts,
                ) = load_go_direct_parent_edges(
                    ontology,
                    terms,
                    data_root,
                )
                local_parent_child_indices = (
                    local_parent_child_indices.to(device)
                )
                local_parent_indices = local_parent_indices.to(device)
                local_parent_counts = local_parent_counts.to(device)
        if latent_domain_label_readout:
            latent_label_depth_groups, _ = load_go_depth_groups(
                ontology,
                terms,
                data_root,
            )
            if latent_parent_mapping == "shuffled":
                (
                    latent_parent_child_indices,
                    latent_parent_indices,
                    latent_parent_counts,
                    _,
                ) = load_go_shuffled_parent_edges(
                    ontology,
                    terms,
                    data_root,
                    int(config.get("parent_shuffle_seed", config.get("seed", 3407))),
                )
            else:
                (
                    latent_parent_child_indices,
                    latent_parent_indices,
                    latent_parent_counts,
                ) = load_go_direct_parent_edges(
                    ontology,
                    terms,
                    data_root,
                )
            latent_label_depth_groups = latent_label_depth_groups.to(device)
            latent_parent_child_indices = latent_parent_child_indices.to(
                device
            )
            latent_parent_indices = latent_parent_indices.to(device)
            latent_parent_counts = latent_parent_counts.to(device)
            if latent_domain_ic_semantic_residual:
                latent_label_semantics = load_go_semantics(
                    ontology,
                    terms,
                    data_root,
                ).to(device).float()
                latent_label_semantics = (
                    latent_label_semantics
                    - latent_label_semantics.mean(dim=0, keepdim=True)
                )
                latent_label_semantics = torch.nn.functional.normalize(
                    latent_label_semantics,
                    dim=-1,
                )
                # The persistent checkpoint buffer overwrites these values.
                latent_label_semantic_weights = torch.ones(
                    len(terms),
                    dtype=torch.float32,
                    device=device,
                )
        return FunctionalHypothesisEnsemble(
            class_nums=class_nums,
            hidden_dim=hidden_dim,
            dropout=dropout,
            ensemble_size=ensemble_size,
            classifier_rank=int(config.get("classifier_rank", 0)),
            independent_classifier=bool(
                config.get("independent_classifier", False)
            ),
            functional_breadth=bool(
                config.get("functional_breadth", False)
            ),
            semantic_router=semantic_router,
            go_semantics=go_semantics,
            router_temperature=float(config.get("router_temperature", 1.0)),
            function_alignment=bool(
                config.get("function_alignment", False)
            ),
            function_alignment_dim=int(
                config.get("function_alignment_dim", 128)
            ),
            function_semantic_dim=768,
            gated_refinement=bool(
                config.get("gated_refinement", False)
            ),
            gated_refinement_expansion=float(
                config.get("gated_refinement_expansion", 1.0)
            ),
            local_semantic_attention=local_semantic_attention,
            local_segment_dim=640,
            local_attention_dim=int(
                config.get("local_attention_dim", 64)
            ),
            local_label_chunk_size=int(
                config.get("local_label_chunk_size", 512)
            ),
            local_task_query_residual=bool(
                config.get("local_task_query_residual", False)
            ),
            local_query_source=local_query_source,
            local_num_layers=(3 if local_multilayer_segments else 1),
            local_classifier_parent_contrast=(
                local_classifier_discriminator == "parent_contrast"
            ),
            local_go_semantics=local_go_semantics,
            local_parent_contrast=local_parent_contrast,
            local_depth_groups=local_depth_groups,
            local_parent_child_indices=local_parent_child_indices,
            local_parent_indices=local_parent_indices,
            local_parent_counts=local_parent_counts,
            latent_domain_fusion=latent_domain_fusion,
            latent_domain_count=int(
                config.get("latent_domain_count", 8)
            ),
            latent_domain_dim=int(
                config.get("latent_domain_dim", 64)
            ),
            latent_domain_num_layers=3,
            latent_domain_layer_contrast=(
                latent_domain_layer_contrast
            ),
            latent_domain_competition=bool(
                config.get("latent_domain_competition", False)
            ),
            latent_domain_label_readout=latent_domain_label_readout,
            latent_domain_normalize_label_queries=(
                latent_domain_normalize_label_queries
            ),
            latent_domain_adaptive_label_query_norm=(
                latent_domain_adaptive_label_query_norm
            ),
            latent_domain_confidence_direction_routing=(
                latent_domain_confidence_direction_routing
            ),
            latent_domain_ic_semantic_residual=(
                latent_domain_ic_semantic_residual
            ),
            latent_label_semantics=latent_label_semantics,
            latent_label_semantic_weights=(
                latent_label_semantic_weights
            ),
            latent_semantic_dim=(
                int(latent_label_semantics.shape[1])
                if latent_label_semantics is not None
                else 768
            ),
            latent_label_chunk_size=int(
                config.get("local_label_chunk_size", 512)
            ),
            latent_label_depth_groups=latent_label_depth_groups,
            latent_parent_child_indices=latent_parent_child_indices,
            latent_parent_indices=latent_parent_indices,
            latent_parent_counts=latent_parent_counts,
            latent_domain_readout_mode=latent_domain_readout_mode,
            latent_domain_discriminator_mode=(
                latent_domain_discriminator_mode
            ),
            anchor_preserving_local_gradients=bool(
                config.get("anchor_preserving_local_gradients", False)
            ),
        )
    raise ValueError("The retained evaluator supports only K=8 SEHNet runs")


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    iterator = (
        tqdm(loader, desc=description, leave=False)
        if tqdm is not None
        else loader
    )
    for batch in iterator:
        esm = batch["esm"].to(device, non_blocking=True)
        local_segments = (
            batch["local_segments"].to(device, non_blocking=True)
            if "local_segments" in batch
            else None
        )
        local_mask = (
            batch["local_mask"].to(device, non_blocking=True)
            if "local_mask" in batch
            else None
        )
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            output = (
                model(
                    esm,
                    local_segments=local_segments,
                    local_mask=local_mask,
                )
                if local_segments is not None
                else model(esm)
            )
        probabilities.append(output.probs.float().cpu().numpy())
        labels.append(batch["label"].to(dtype=torch.uint8).numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    data_root = (
        args.data_root
        if args.data_root is not None
        else Path(str(config["data_root"]))
    ).expanduser().resolve()
    ontology = args.ont or str(config["ont"])
    device = torch.device(args.device)

    checkpoint_dir = run_dir / "trajectory_checkpoints"
    available_epochs = discover_epochs(checkpoint_dir)
    if args.epochs:
        selected_epochs = sorted(set(args.epochs))
    else:
        # Early stopping defines the completed trajectory; the configured
        # maximum epoch is not necessarily present.
        final_epoch = max(available_epochs)
        selected_epochs = list(range(max(1, final_epoch - 4), final_epoch + 1))
    missing = [
        epoch for epoch in selected_epochs if epoch not in available_epochs
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing requested checkpoint epochs {missing}; "
            f"available epochs are {available_epochs}"
        )

    label_dir = get_path("label_dir", data_root)
    terms = load_terms(label_dir, ontology)
    max_samples_key = f"max_{args.split}_samples"
    split_seed_offset = {"train": 0, "valid": 1, "test": 2}[args.split]
    dataset = SEHNetDataset(
        ont=ontology,
        split=args.split,
        data_root=data_root,
        terms=terms,
        label_chunksize=int(config.get("label_chunksize", 512)),
        max_samples=config.get(max_samples_key),
        sample_seed=(
            int(config.get("subset_seed", config.get("seed", 3407)))
            + split_seed_offset
        ),
        local_segments_150m=bool(
            config.get("local_semantic_attention", False)
            and not config.get("local_multilayer_segments", False)
        ),
        local_multilayer_segments_150m=bool(
            (
                config.get("local_semantic_attention", False)
                and config.get("local_multilayer_segments", False)
            )
            or config.get("latent_domain_fusion", False)
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=(
            args.eval_batch_size
            if args.eval_batch_size is not None
            else int(config.get("eval_batch_size", 256))
        ),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=sehnet_collate,
    )
    model = build_model(
        config,
        len(terms),
        terms,
        data_root,
        ontology,
        device,
    ).to(device)

    members: List[np.ndarray] = []
    labels = None
    checkpoint_iterator = (
        tqdm(
            selected_epochs,
            desc=f"{args.ont.upper()} {args.split} final-five checkpoints",
            unit="checkpoint",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else selected_epochs
    )
    for epoch in checkpoint_iterator:
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if list(checkpoint.get("terms", terms)) != terms:
            raise ValueError(f"GO term order mismatch in {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state"])
        if args.disable_local_evidence:
            local_module = getattr(
                model,
                "local_semantic_attention",
                None,
            )
            latent_module = getattr(
                model,
                "latent_domain_fusion",
                None,
            )
            if local_module is None and latent_module is None:
                raise ValueError(
                    "Selected run has no local evidence module to disable"
                )
            with torch.no_grad():
                if local_module is not None:
                    local_module.depth_gates.zero_()
                if latent_module is not None:
                    latent_module.zero_evidence_gate()
        member_probs, member_labels = predict(
            model,
            loader,
            device,
            f"{args.split} epoch {epoch}",
            (
                device.type == "cuda"
                and not bool(config.get("disable_amp", False))
            ),
        )
        members.append(member_probs)
        if labels is None:
            labels = member_labels
        elif not np.array_equal(labels, member_labels):
            raise ValueError(
                f"{args.split.capitalize()} label order differs across checkpoints"
            )

    assert labels is not None
    raw_scores = np.mean(members, axis=0, dtype=np.float64).astype(np.float32)
    go_obj = Ontology(get_path("go_obo", data_root))
    propagated = DAG_propagation(
        np.array(raw_scores, copy=True),
        go_obj,
        terms,
    )
    fmax, aupr, threshold = calculate_cafa_fmax_aupr(
        labels,
        propagated,
        terms,
    )
    selection_protocol = (
        "validation_selected_rolling_five"
        if args.output_tag == "best_validation_window"
        else "explicit_final_training_checkpoints"
    )
    bundle = {
        "scores": propagated.astype(np.float32),
        "scores_raw": raw_scores,
        "labels": labels.astype(np.uint8),
        "terms": list(terms),
        "ensemble_epochs": selected_epochs,
        "checkpoint_protocol": selection_protocol,
        "selection_split": (
            "valid"
            if selection_protocol == "validation_selected_rolling_five"
            else None
        ),
        "selection_metric": (
            "Fmax+AUPR_after_raw_probability_average_and_one_DAG_pass"
            if selection_protocol == "validation_selected_rolling_five"
            else None
        ),
    }
    suffix = "_local_disabled" if args.disable_local_evidence else ""
    if args.output_tag:
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.output_tag).strip(
            "_."
        )
        if not safe_tag:
            raise ValueError("--output-tag must contain a filename-safe token")
        suffix += f"_{safe_tag}"
    score_path = run_dir / (
        f"last_five_{args.split}{suffix}_scores.pkl"
    )
    with score_path.open("wb") as handle:
        pkl.dump(bundle, handle, protocol=pkl.HIGHEST_PROTOCOL)
    summary = {
        "run_dir": str(run_dir),
        "ontology": ontology,
        "split": args.split,
        "checkpoint_protocol": selection_protocol,
        "selection_split": (
            "valid"
            if selection_protocol == "validation_selected_rolling_five"
            else None
        ),
        "selection_metric": (
            "Fmax+AUPR_after_raw_probability_average_and_one_DAG_pass"
            if selection_protocol == "validation_selected_rolling_five"
            else None
        ),
        "ensemble_epochs": selected_epochs,
        "fmax": float(fmax),
        "aupr": float(aupr),
        "threshold": float(threshold),
        "num_proteins": len(dataset),
        "num_terms": len(terms),
        "pure_model": True,
        "retrieval_or_homology": False,
        "local_evidence_disabled": bool(args.disable_local_evidence),
    }
    output_path = run_dir / (
        f"last_five_{args.split}{suffix}_summary.json"
    )
    save_json(output_path, summary)
    print(
        f"[last-five] {ontology.upper()}-{args.split} epochs: {selected_epochs}\n"
        f"[last-five] Fmax/AUPR: {fmax:.4f}/{aupr:.4f}\n"
        f"[last-five] Summary: {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
