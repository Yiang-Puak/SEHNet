from __future__ import annotations

import json
import os
import pickle as pkl
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import get_path
from .data import (
    SEHNetDataset,
    load_labels_for_pids,
    load_terms,
    read_pids,
    sehnet_collate,
)
from .metrics import DAG_propagation, Ontology, calculate_cafa_fmax_aupr
from .model import FunctionalHypothesisEnsemble
from .training_utils import (
    AsymmetricLoss,
    load_go_depth_groups,
    load_go_direct_parent_edges,
    load_go_shuffled_parent_edges,
    load_go_semantics,
    load_parent_contrast_go_semantics,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def make_result_dir(args) -> Path:
    result_dir = Path(args.result_dir).expanduser().resolve(strict=False)
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"Result directory is not empty: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def cafa_metrics(
    raw_probs: np.ndarray,
    labels: np.ndarray,
    go_obj: Ontology,
    terms: Sequence[str],
) -> Tuple[np.ndarray, float, float, float]:
    propagated = DAG_propagation(np.array(raw_probs, copy=True), go_obj, terms)
    fmax, aupr, threshold = calculate_cafa_fmax_aupr(
        labels,
        propagated,
        terms,
    )
    return propagated, fmax, aupr, threshold


def amp_autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def forward_with_optional_local(
    model: torch.nn.Module,
    esm: torch.Tensor,
    local_segments: torch.Tensor | None,
    local_mask: torch.Tensor | None,
):
    if local_segments is None:
        return model(esm)
    return model(
        esm,
        local_segments=local_segments,
        local_mask=local_mask,
    )


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def supervised_classification_loss(
    *,
    output,
    target: torch.Tensor,
    criterion: AsymmetricLoss,
    ensemble_size: int,
    class_nums: int,
    semantic_router: bool,
    router_member_loss_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    member_loss = output.logits.new_zeros(())
    if ensemble_size > 1 and semantic_router:
        classification_loss = criterion(output.logits, target)
        member_target = target.unsqueeze(1).expand(
            -1,
            ensemble_size,
            -1,
        )
        member_loss = criterion(
            output.member_logits.reshape(-1, class_nums),
            member_target.reshape(-1, class_nums),
        )
        classification_loss = (
            classification_loss
            + router_member_loss_ratio * member_loss
        )
    elif ensemble_size > 1:
        member_target = target.unsqueeze(1).expand(
            -1,
            ensemble_size,
            -1,
        )
        classification_loss = criterion(
            output.member_logits.reshape(-1, class_nums),
            member_target.reshape(-1, class_nums),
        )
    else:
        classification_loss = criterion(output.logits, target)
    return classification_loss, member_loss


def anchor_classification_loss(
    *,
    output,
    target: torch.Tensor,
    criterion: AsymmetricLoss,
    ensemble_size: int,
    class_nums: int,
) -> torch.Tensor:

    if output.anchor_member_logits is None:
        return output.logits.new_zeros(())
    if ensemble_size <= 1:
        raise ValueError(
            "Anchor-preserving local gradients require an ensemble"
        )
    member_target = target.unsqueeze(1).expand(
        -1,
        ensemble_size,
        -1,
    )
    return criterion(
        output.anchor_member_logits.reshape(-1, class_nums),
        member_target.reshape(-1, class_nums),
    )


@torch.no_grad()
def sam_perturb_parameters(
    model: torch.nn.Module,
    *,
    rho: float,
    gradient_scale: float,
) -> List[Tuple[torch.nn.Parameter, torch.Tensor]]:
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not parameters:
        return []
    gradient_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                parameter.grad.detach().float().norm()
                / float(gradient_scale)
                for parameter in parameters
            ]
        )
    )
    perturbation_scale = float(rho) / float(
        gradient_norm.clamp_min(1e-12)
    )
    perturbations = []
    for parameter in parameters:
        original = parameter.detach().clone()
        perturbation = (
            parameter.grad.detach()
            * (perturbation_scale / float(gradient_scale))
        )
        parameter.add_(perturbation)
        perturbations.append((parameter, original))
    return perturbations


@torch.no_grad()
def restore_sam_parameters(
    perturbations: Sequence[Tuple[torch.nn.Parameter, torch.Tensor]],
) -> None:
    for parameter, original in perturbations:
        parameter.copy_(original)


def pause_batch_norm_updates(
    model: torch.nn.Module,
) -> List[Tuple[torch.nn.modules.batchnorm._BatchNorm, float | None]]:
    states = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            states.append((module, module.momentum))
            module.momentum = 0.0
    return states


def restore_batch_norm_updates(
    states: Sequence[
        Tuple[torch.nn.modules.batchnorm._BatchNorm, float | None]
    ],
) -> None:
    for module, momentum in states:
        module.momentum = momentum


@torch.no_grad()
def evaluate(
    model: ESMMLPBaseline,
    loader: DataLoader,
    criterion: AsymmetricLoss,
    go_obj: Ontology,
    terms: Sequence[str],
    device: torch.device,
    amp_enabled: bool,
    desc: str,
    direct_children: torch.Tensor | None = None,
    direct_parents: torch.Tensor | None = None,
) -> Dict[str, object]:
    model.eval()
    losses: List[float] = []
    member_variances: List[float] = []
    routing_entropies: List[float] = []
    routing_top1_shares: List[float] = []
    latent_domain_segment_entropies: List[float] = []
    latent_domain_selection_entropies: List[float] = []
    latent_domain_member_cosines: List[float] = []
    latent_domain_label_evidence_abs_means: List[float] = []
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    iterator = (
        tqdm(loader, desc=desc, leave=False)
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
        target = batch["label"].to(device, non_blocking=True)
        with amp_autocast(amp_enabled):
            output = forward_with_optional_local(
                model,
                esm,
                local_segments,
                local_mask,
            )
            loss = criterion(output.logits, target)
        losses.append(float(loss))
        if hasattr(output, "member_probs"):
            member_variances.append(
                float(
                    output.member_probs.float().var(
                        dim=1,
                        unbiased=False,
                    ).mean()
                )
            )
        if getattr(output, "routing_weights", None) is not None:
            weights = output.routing_weights.float()
            routing_entropies.append(
                float(
                    (
                        -weights * weights.clamp_min(1e-8).log()
                    ).sum(dim=-1).mean()
                )
            )
            routing_top1_shares.append(
                float(weights.max(dim=-1).values.mean())
            )
        if (
            getattr(output, "latent_domain_segment_entropy", None)
            is not None
        ):
            latent_domain_segment_entropies.append(
                float(output.latent_domain_segment_entropy)
            )
        if (
            getattr(output, "latent_domain_selection_entropy", None)
            is not None
        ):
            latent_domain_selection_entropies.append(
                float(output.latent_domain_selection_entropy)
            )
        if getattr(output, "latent_domain_evidence", None) is not None:
            evidence = F.normalize(
                output.latent_domain_evidence.float(),
                dim=-1,
            )
            similarities = torch.bmm(
                evidence,
                evidence.transpose(1, 2),
            )
            member_count = similarities.shape[-1]
            if member_count > 1:
                off_diagonal = ~torch.eye(
                    member_count,
                    dtype=torch.bool,
                    device=similarities.device,
                )
                latent_domain_member_cosines.append(
                    float(similarities[:, off_diagonal].mean())
                )
        if (
            getattr(output, "latent_domain_label_evidence", None)
            is not None
        ):
            latent_domain_label_evidence_abs_means.append(
                float(output.latent_domain_label_evidence.float().abs().mean())
            )
        probabilities.append(output.probs.float().cpu().numpy())
        labels.append(target.to(dtype=torch.uint8).cpu().numpy())
    raw_probs = np.concatenate(probabilities, axis=0)
    target_array = np.concatenate(labels, axis=0)
    propagated, fmax, aupr, threshold = cafa_metrics(
        raw_probs,
        target_array,
        go_obj,
        terms,
    )
    hierarchy_violation_rate = 0.0
    hierarchy_violation_margin = 0.0
    if (
        direct_children is not None
        and direct_parents is not None
        and direct_children.numel() > 0
    ):
        child_scores = raw_probs[:, direct_children.cpu().numpy()]
        parent_scores = raw_probs[:, direct_parents.cpu().numpy()]
        margins = child_scores - parent_scores
        hierarchy_violation_rate = float(np.mean(margins > 0.0))
        hierarchy_violation_margin = float(
            np.maximum(margins, 0.0).mean()
        )
    return {
        "loss": float(np.mean(losses)),
        "fmax": fmax,
        "aupr": aupr,
        "threshold": threshold,
        "preds": propagated,
        "preds_raw": raw_probs,
        "labels": target_array,
        "member_probability_variance": (
            float(np.mean(member_variances))
            if member_variances
            else 0.0
        ),
        "routing_entropy": (
            float(np.mean(routing_entropies))
            if routing_entropies
            else 0.0
        ),
        "routing_top1_share": (
            float(np.mean(routing_top1_shares))
            if routing_top1_shares
            else 0.0
        ),
        "latent_domain_segment_entropy": (
            float(np.mean(latent_domain_segment_entropies))
            if latent_domain_segment_entropies
            else 0.0
        ),
        "latent_domain_selection_entropy": (
            float(np.mean(latent_domain_selection_entropies))
            if latent_domain_selection_entropies
            else 0.0
        ),
        "latent_domain_member_cosine": (
            float(np.mean(latent_domain_member_cosines))
            if latent_domain_member_cosines
            else (
                float("nan")
                if bool(
                    getattr(
                        getattr(model, "latent_domain_fusion", None),
                        "label_readout",
                        False,
                    )
                )
                else 0.0
            )
        ),
        "latent_domain_label_evidence_abs_mean": (
            float(np.mean(latent_domain_label_evidence_abs_means))
            if latent_domain_label_evidence_abs_means
            else 0.0
        ),
        "hierarchy_violation_rate": hierarchy_violation_rate,
        "hierarchy_violation_margin": hierarchy_violation_margin,
    }


def build_datasets(args):
    requested_splits = ("train", "valid") if args.skip_final_test else ("train", "valid", "test")
    dataset_kwargs = {
        "ont": args.ont,
        "data_root": args.data_root,
        "label_chunksize": args.label_chunksize,
        "local_segments_150m": (
            args.local_semantic_attention
            and not args.local_multilayer_segments
        ),
        "local_multilayer_segments_150m": (
            (
                args.local_semantic_attention
                and args.local_multilayer_segments
            )
            or args.latent_domain_fusion
        ),
    }
    if all(
        getattr(args, f"max_{split}_samples") is None
        for split in requested_splits
    ):
        data_root = Path(args.data_root)
        label_dir = get_path("label_dir", data_root)
        split_dir = get_path("split_dir", data_root)
        terms = load_terms(label_dir, args.ont)
        all_pids: List[str] = []
        seen = set()
        for split in requested_splits:
            for pid in read_pids(
                split_dir / f"{args.ont}_{split}_pid_list.txt"
            ):
                if pid not in seen:
                    seen.add(pid)
                    all_pids.append(pid)
        labels = load_labels_for_pids(
            label_dir / f"{args.ont}_data_separate_{args.ont}_labels.csv",
            all_pids,
            terms=terms,
            chunksize=args.label_chunksize,
        )
        dataset_kwargs.update(terms=terms, shared_labels=labels)
    split_seed_offsets = {"train": 0, "valid": 1, "test": 2}
    datasets = {
        split: SEHNetDataset(
            split=split,
            max_samples=getattr(args, f"max_{split}_samples"),
            sample_seed=args.subset_seed + split_seed_offsets[split],
            **dataset_kwargs,
        )
        for split in requested_splits
    }
    return datasets["train"], datasets["valid"], datasets.get("test")


def functional_breadth_statistics(
    dataset: SEHNetDataset,
) -> Tuple[float, float]:
    root_indices = {
        index
        for index, term in enumerate(dataset.terms)
        if term in {"GO:0003674", "GO:0008150", "GO:0005575"}
    }
    values = []
    for pid in dataset.proteins:
        labels = dataset.labels[pid]
        count = int(labels.sum())
        count -= sum(int(labels[index]) for index in root_indices)
        values.append(np.log1p(max(count, 0)))
    return float(np.mean(values)), float(max(np.std(values), 1e-6))


def protein_set_f1_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    root_indices: Sequence[int],
) -> torch.Tensor:

    probabilities = probabilities.float()
    targets = targets.float()
    if root_indices:
        keep = torch.ones(
            probabilities.shape[1],
            dtype=torch.bool,
            device=probabilities.device,
        )
        keep[list(root_indices)] = False
        probabilities = probabilities[:, keep]
        targets = targets[:, keep]
    true_positive = (probabilities * targets).sum(dim=1)
    soft_precision = (
        true_positive / probabilities.sum(dim=1).clamp_min(1e-6)
    ).mean()
    soft_recall = (
        true_positive / targets.sum(dim=1).clamp_min(1.0)
    ).mean()
    soft_f1 = (
        2.0
        * soft_precision
        * soft_recall
        / (soft_precision + soft_recall).clamp_min(1e-6)
    )
    return 1.0 - soft_f1


def maybe_freeze_latent_depth_gates(
    model: torch.nn.Module,
    threshold: float,
    epoch: int,
    already_frozen: bool,
) -> tuple[bool, int, float | None]:
    if threshold <= 0.0 or already_frozen:
        return already_frozen, 0, None
    latent_module = getattr(model, "latent_domain_fusion", None)
    depth_gates = (
        getattr(latent_module, "depth_gates", None)
        if latent_module is not None
        else None
    )
    if depth_gates is None:
        raise RuntimeError(
            "GO-depth gate freezing requested without depth gates"
        )
    current_gate_abs_mean = float(
        torch.tanh(depth_gates.detach()).abs().mean()
    )
    if current_gate_abs_mean < threshold:
        return False, 0, current_gate_abs_mean
    depth_gates.requires_grad_(False)
    depth_gates.grad = None
    return True, epoch, current_gate_abs_mean


def run_training(args) -> None:
    os.environ["PROTEIN_FUNCTION_DATA"] = str(
        Path(args.data_root).expanduser().resolve(strict=False)
    )
    seed_everything(args.seed)
    if args.subset_seed is None:
        args.subset_seed = args.seed
    if args.early_stop_monitor not in {"single_epoch", "rolling_five"}:
        raise ValueError(
            "early_stop_monitor must be single_epoch or rolling_five"
        )
    if args.sam_rho < 0.0:
        raise ValueError("sam_rho must be non-negative")
    if args.sam_rho > 0.0 and (
        args.semantic_router
        or args.functional_breadth
        or args.set_loss_ratio > 0.0
        or args.pu_rank_ratio > 0.0
        or args.max_constraint_ratio > 0.0
        or args.function_alignment
        or args.gated_refinement
        or args.local_semantic_attention
        or args.latent_domain_fusion
        or args.classifier_rank > 0
        or args.independent_classifier
    ):
        raise ValueError(
            "SAM is a controlled single-variable experiment and cannot "
            "be combined with optional architecture or auxiliary-loss flags"
        )
    if args.local_semantic_attention and (
        args.ensemble_size <= 1
        or args.semantic_router
        or args.functional_breadth
        or args.set_loss_ratio > 0.0
        or args.pu_rank_ratio > 0.0
        or args.max_constraint_ratio > 0.0
        or args.function_alignment
        or args.gated_refinement
        or args.classifier_rank > 0
        or args.independent_classifier
    ):
        raise ValueError(
            "Local semantic attention is a controlled single-variable "
            "K>1 experiment and cannot be combined with other optional "
            "architecture or auxiliary-loss flags"
        )
    if args.latent_domain_fusion and (
        args.ensemble_size <= 1
        or args.semantic_router
        or args.functional_breadth
        or args.set_loss_ratio > 0.0
        or args.pu_rank_ratio > 0.0
        or args.max_constraint_ratio > 0.0
        or args.function_alignment
        or args.gated_refinement
        or args.local_semantic_attention
        or args.classifier_rank > 0
        or args.independent_classifier
    ):
        raise ValueError(
            "Latent-domain fusion is a controlled single-variable K>1 "
            "experiment and cannot be combined with other optional "
            "architecture or auxiliary-loss flags"
        )
    if args.latent_domain_count <= 0:
        raise ValueError("latent_domain_count must be positive")
    if args.latent_domain_dim <= 0:
        raise ValueError("latent_domain_dim must be positive")
    if (
        args.latent_domain_layer_contrast
        and (
            not args.latent_domain_fusion
            or not args.latent_domain_label_readout
        )
    ):
        raise ValueError(
            "latent_domain_layer_contrast requires SEHNet-style "
            "latent-domain GO-label readout"
        )
    if args.latent_domain_layer_contrast and (
        args.latent_domain_normalize_label_queries
        or args.latent_domain_adaptive_label_query_norm
        or args.latent_domain_confidence_direction_routing
        or args.latent_domain_ic_semantic_residual
    ):
        raise ValueError(
            "Layer contrast cannot be combined with label-query "
            "normalization in a controlled single-variable experiment"
        )
    if (
        args.latent_domain_competition
        and not args.latent_domain_fusion
    ):
        raise ValueError(
            "latent_domain_competition requires latent_domain_fusion"
        )
    if args.latent_domain_label_readout and (
        not args.latent_domain_fusion
        or (
            not args.latent_domain_competition
            and args.ablation != "independent_segments"
        )
        or (
            not args.anchor_preserving_local_gradients
            and args.ablation != "joint_gradient"
        )
    ):
        raise ValueError(
            "SEHNet GO-label readout requires its declared fusion, competition, "
            "and anchoring configuration"
        )
    if not 0.0 <= args.freeze_latent_depth_gates_at_abs_mean < 1.0:
        raise ValueError(
            "freeze_latent_depth_gates_at_abs_mean must be in [0, 1)"
        )
    if (
        args.freeze_latent_depth_gates_at_abs_mean > 0.0
        and not args.latent_domain_label_readout
    ):
        raise ValueError(
            "freeze_latent_depth_gates_at_abs_mean requires "
            "latent_domain_label_readout"
        )
    if (
        args.latent_domain_normalize_label_queries
        and not args.latent_domain_label_readout
    ):
        raise ValueError(
            "latent_domain_normalize_label_queries requires "
            "latent_domain_label_readout"
        )
    if (
        args.latent_domain_adaptive_label_query_norm
        and not args.latent_domain_label_readout
    ):
        raise ValueError(
            "latent_domain_adaptive_label_query_norm requires "
            "latent_domain_label_readout"
        )
    if (
        args.latent_domain_confidence_direction_routing
        and not args.latent_domain_label_readout
    ):
        raise ValueError(
            "latent_domain_confidence_direction_routing requires "
            "latent_domain_label_readout"
        )
    if (
        args.latent_domain_adaptive_label_query_norm
        and args.latent_domain_normalize_label_queries
    ):
        raise ValueError(
            "Adaptive and fixed latent-domain query normalization cannot "
            "be combined"
        )
    if args.latent_domain_confidence_direction_routing and (
        args.latent_domain_normalize_label_queries
        or args.latent_domain_adaptive_label_query_norm
    ):
        raise ValueError(
            "Confidence-direction routing and existing label-query "
            "normalizations cannot be combined"
        )
    if (
        args.latent_domain_ic_semantic_residual
        and (
            not args.latent_domain_label_readout
            or args.latent_domain_normalize_label_queries
            or args.latent_domain_adaptive_label_query_norm
            or args.latent_domain_confidence_direction_routing
            or args.freeze_latent_depth_gates_at_abs_mean > 0.0
        )
    ):
        raise ValueError(
            "latent_domain_ic_semantic_residual is a single SEHNet variable "
            "and cannot be combined with routing or gate interventions"
        )
    if args.anchor_preserving_local_gradients and not args.latent_domain_fusion:
        raise ValueError(
            "anchor_preserving_local_gradients requires latent-domain fusion"
        )
    if (
        args.anchor_preserving_local_gradients
        and args.init_checkpoint is not None
    ):
        raise ValueError(
            "Anchor-preserving local-gradient experiments must start from "
            "random initialization"
        )
    if args.local_task_query_residual and not args.local_semantic_attention:
        raise ValueError(
            "local_task_query_residual requires local_semantic_attention"
        )
    if (
        args.local_task_query_residual
        and args.local_query_source != "semantic"
    ):
        raise ValueError(
            "local_task_query_residual requires semantic local queries"
        )
    if (
        not args.local_semantic_attention
        and args.local_query_source != "semantic"
    ):
        raise ValueError(
            "local_query_source is only used with local_semantic_attention"
        )
    if (
        args.local_multilayer_segments
        and not args.local_semantic_attention
    ):
        raise ValueError(
            "local_multilayer_segments requires local_semantic_attention"
        )
    if (
        args.local_classifier_discriminator != "parent_contrast"
        and args.local_query_source != "classifier"
    ):
        raise ValueError(
            "local_classifier_discriminator is only used with classifier "
            "local queries"
        )
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    result_dir = make_result_dir(args)
    save_json(
        result_dir / "config.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )

    train_ds, valid_ds, test_ds = build_datasets(args)
    available_datasets = [
        ("train", train_ds),
        ("valid", valid_ds),
    ]
    if test_ds is not None:
        available_datasets.append(("test", test_ds))
    for split, dataset in available_datasets:
        (result_dir / f"{split}_pids.txt").write_text(
            "\n".join(dataset.proteins) + "\n",
            encoding="utf-8",
        )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": sehnet_collate,
        "persistent_workers": args.num_workers > 0,
    }
    train_order_generator = torch.Generator(device="cpu")
    train_order_generator.manual_seed(
        (int(args.seed) + 0x51E71A) % (2**63 - 1)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_order_generator,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        if test_ds is not None
        else None
    )

    breadth_target_mean, breadth_target_std = functional_breadth_statistics(
        train_ds
    )
    needs_go_semantics = (
        args.semantic_router
        or args.pu_rank_ratio > 0.0
        or args.function_alignment
        or args.latent_domain_ic_semantic_residual
    )
    term_semantics = (
        load_go_semantics(
            args.ont,
            train_ds.terms,
            Path(args.data_root),
        ).to(device)
        if needs_go_semantics
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
    latent_parent_mapping_stats: Dict[str, object] = {}
    local_semantic_stats: Dict[str, object] = {}
    if args.local_semantic_attention:
        if args.local_query_source == "classifier":
            (
                local_depth_groups,
                local_semantic_stats,
            ) = load_go_depth_groups(
                args.ont,
                train_ds.terms,
                Path(args.data_root),
            )
            local_go_semantics = torch.zeros(
                len(train_ds.terms),
                1,
                dtype=torch.float32,
            )
            local_parent_contrast = torch.zeros_like(local_go_semantics)
        else:
            (
                local_go_semantics,
                local_parent_contrast,
                local_depth_groups,
                local_semantic_stats,
            ) = load_parent_contrast_go_semantics(
                args.ont,
                train_ds.terms,
                Path(args.data_root),
            )
        local_go_semantics = local_go_semantics.to(device)
        local_parent_contrast = local_parent_contrast.to(device)
        local_depth_groups = local_depth_groups.to(device)
        if (
            args.local_query_source == "classifier"
            and args.local_classifier_discriminator == "parent_contrast"
        ):
            (
                local_parent_child_indices,
                local_parent_indices,
                local_parent_counts,
            ) = load_go_direct_parent_edges(
                args.ont,
                train_ds.terms,
                Path(args.data_root),
            )
            local_parent_child_indices = (
                local_parent_child_indices.to(device)
            )
            local_parent_indices = local_parent_indices.to(device)
            local_parent_counts = local_parent_counts.to(device)
    if args.latent_domain_label_readout:
        (
            latent_label_depth_groups,
            latent_semantic_stats,
        ) = load_go_depth_groups(
            args.ont,
            train_ds.terms,
            Path(args.data_root),
        )
        local_semantic_stats = {
            **local_semantic_stats,
            **{
                f"latent_label_{key}": value
                for key, value in latent_semantic_stats.items()
            },
        }
        if args.latent_parent_mapping == "shuffled":
            (
                latent_parent_child_indices,
                latent_parent_indices,
                latent_parent_counts,
                latent_parent_mapping_stats,
            ) = load_go_shuffled_parent_edges(
                args.ont,
                train_ds.terms,
                Path(args.data_root),
                args.parent_shuffle_seed,
            )
        else:
            (
                latent_parent_child_indices,
                latent_parent_indices,
                latent_parent_counts,
            ) = load_go_direct_parent_edges(
                args.ont,
                train_ds.terms,
                Path(args.data_root),
            )
            latent_parent_mapping_stats = {"mapping": "direct_go_parent"}
        latent_label_depth_groups = latent_label_depth_groups.to(device)
        latent_parent_child_indices = latent_parent_child_indices.to(device)
        latent_parent_indices = latent_parent_indices.to(device)
        latent_parent_counts = latent_parent_counts.to(device)
        if args.latent_domain_ic_semantic_residual:
            if term_semantics is None:
                raise RuntimeError(
                    "IC semantic residual requires GO semantics"
                )
            latent_label_semantics = term_semantics.float()
            latent_label_semantics = (
                latent_label_semantics
                - latent_label_semantics.mean(dim=0, keepdim=True)
            )
            latent_label_semantics = F.normalize(
                latent_label_semantics,
                dim=-1,
            )
            label_frequency = torch.as_tensor(
                (train_ds.label_counts.astype(np.float64) + 1.0)
                / (len(train_ds) + 1.0),
                dtype=torch.float32,
                device=device,
            )
            latent_label_semantic_weights = (
                -label_frequency.clamp_min(1e-8).log()
            )
            latent_label_semantic_weights = (
                latent_label_semantic_weights
                / latent_label_semantic_weights.mean().clamp_min(1e-6)
            )
    ranking_semantics = None
    information_content = None
    if args.pu_rank_ratio > 0.0 or args.function_alignment:
        if term_semantics is None:
            raise RuntimeError("Semantic supervision requires GO semantics")
        ranking_semantics = term_semantics.float()
        ranking_semantics = (
            ranking_semantics
            - ranking_semantics.mean(dim=0, keepdim=True)
        )
        ranking_semantics = F.normalize(ranking_semantics, dim=-1)
        label_frequency = torch.as_tensor(
            (train_ds.label_counts.astype(np.float64) + 1.0)
            / (len(train_ds) + 1.0),
            dtype=torch.float32,
            device=device,
        )
        information_content = -label_frequency.clamp_min(1e-8).log()
        information_content = (
            information_content
            / information_content.mean().clamp_min(1e-6)
        )
    if args.ensemble_size > 1:
        go_semantics = (
            term_semantics
            if args.semantic_router
            else None
        )
        model = FunctionalHypothesisEnsemble(
            class_nums=train_ds.class_nums,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            ensemble_size=args.ensemble_size,
            classifier_rank=args.classifier_rank,
            independent_classifier=args.independent_classifier,
            functional_breadth=args.functional_breadth,
            breadth_target_mean=breadth_target_mean,
            breadth_target_std=breadth_target_std,
            semantic_router=args.semantic_router,
            go_semantics=go_semantics,
            router_temperature=args.router_temperature,
            function_alignment=args.function_alignment,
            function_alignment_dim=args.function_alignment_dim,
            function_semantic_dim=(
                int(term_semantics.shape[1])
                if term_semantics is not None
                else 768
            ),
            gated_refinement=args.gated_refinement,
            gated_refinement_expansion=(
                args.gated_refinement_expansion
            ),
            local_semantic_attention=args.local_semantic_attention,
            local_segment_dim=640,
            local_attention_dim=args.local_attention_dim,
            local_label_chunk_size=args.local_label_chunk_size,
            local_task_query_residual=args.local_task_query_residual,
            local_query_source=args.local_query_source,
            local_num_layers=(3 if args.local_multilayer_segments else 1),
            local_classifier_parent_contrast=(
                args.local_classifier_discriminator == "parent_contrast"
            ),
            local_go_semantics=local_go_semantics,
            local_parent_contrast=local_parent_contrast,
            local_depth_groups=local_depth_groups,
            local_parent_child_indices=local_parent_child_indices,
            local_parent_indices=local_parent_indices,
            local_parent_counts=local_parent_counts,
            latent_domain_fusion=args.latent_domain_fusion,
            latent_domain_count=args.latent_domain_count,
            latent_domain_dim=args.latent_domain_dim,
            latent_domain_num_layers=3,
            latent_domain_layer_contrast=(
                args.latent_domain_layer_contrast
            ),
            latent_domain_competition=args.latent_domain_competition,
            latent_domain_label_readout=args.latent_domain_label_readout,
            latent_domain_normalize_label_queries=(
                args.latent_domain_normalize_label_queries
            ),
            latent_domain_adaptive_label_query_norm=(
                args.latent_domain_adaptive_label_query_norm
            ),
            latent_domain_confidence_direction_routing=(
                args.latent_domain_confidence_direction_routing
            ),
            latent_domain_ic_semantic_residual=(
                args.latent_domain_ic_semantic_residual
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
            latent_label_chunk_size=args.local_label_chunk_size,
            latent_label_depth_groups=latent_label_depth_groups,
            latent_parent_child_indices=latent_parent_child_indices,
            latent_parent_indices=latent_parent_indices,
            latent_parent_counts=latent_parent_counts,
            latent_domain_readout_mode=args.latent_domain_readout_mode,
            latent_domain_discriminator_mode=(
                args.latent_domain_discriminator_mode
            ),
            anchor_preserving_local_gradients=(
                args.anchor_preserving_local_gradients
            ),
        ).to(device)
        architecture = (
            (
                "GO-DAG label readout over layer-contrastive "
                "anchor-preserving competitive latent domains"
                if args.latent_domain_layer_contrast
                else (
                    "IC-weighted GO-semantic residual over SEHNet "
                    "task-prototype latent-domain readout"
                    if args.latent_domain_ic_semantic_residual
                    else (
                    "confidence-direction disentangled GO-prototype "
                    "routing over anchor-preserving competitive latent "
                    "domains"
                    if args.latent_domain_confidence_direction_routing
                    else (
                        "GO-depth-adaptive prototype normalization over "
                        "anchor-preserving competitive latent domains"
                        if args.latent_domain_adaptive_label_query_norm
                        else (
                            "GO-DAG-aligned label readout over "
                            "anchor-preserving competitive latent domains"
                        )
                    )
                    )
                )
            )
            if args.latent_domain_label_readout
            else
            (
                (
                    (
                        "anchor-preserving gradient-decoupled competitive "
                        "latent-domain local ESM fusion"
                    )
                    if args.anchor_preserving_local_gradients
                    else (
                        "competitive latent-domain-guided multi-layer "
                        "local ESM fusion"
                    )
                )
                if args.latent_domain_competition
                else "latent-domain-guided multi-layer local ESM fusion"
            )
            if args.latent_domain_fusion
            else
            (
                "GO-depth-aligned multi-layer local ESM evidence decoder"
            )
            if (
                args.local_semantic_attention
                and args.local_multilayer_segments
                and args.local_query_source == "classifier"
                and args.local_classifier_discriminator == "prototype"
            )
            else
            (
                "classifier-anchored DAG-contrastive attention over "
                "centered local ESM evidence"
            )
            if (
                args.local_semantic_attention
                and args.local_query_source == "classifier"
            )
            else (
                "task-adapted parent-contrastive GO attention over "
                "centered local ESM evidence"
            )
            if args.local_task_query_residual
            else "parent-contrastive GO attention over centered local ESM evidence"
            if args.local_semantic_attention
            else (
                "GO-conditioned functional hypothesis router"
                if args.semantic_router
                else "parameter-efficient functional hypothesis ensemble"
            )
        )
    else:
        raise RuntimeError("The retained entry point supports only K=8 SEHNet")
    if args.init_checkpoint is not None:
        checkpoint_path = args.init_checkpoint.expanduser().resolve()
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_state = dict(checkpoint["model_state"])
        if (
            args.independent_classifier
            and "classifier.weight" in checkpoint_state
        ):
            shared_weight = checkpoint_state.pop("classifier.weight")
            shared_bias = checkpoint_state.pop("classifier.bias")
            input_scale = checkpoint_state.pop("classifier.input_scale")
            output_scale = checkpoint_state.pop("classifier.output_scale")
            old_member_bias = checkpoint_state.pop(
                "classifier.member_bias"
            )
            checkpoint_state.pop("classifier.residual_left", None)
            checkpoint_state.pop("classifier.residual_right", None)
            checkpoint_state["classifier.member_weight"] = (
                output_scale[:, :, None]
                * shared_weight[None, :, :]
                * input_scale[:, None, :]
            )
            checkpoint_state["classifier.member_bias"] = (
                output_scale * shared_bias[None, :] + old_member_bias
            )
        incompatible = model.load_state_dict(
            checkpoint_state,
            strict=False,
        )
        allowed_missing = set()
        if args.functional_breadth:
            allowed_missing.update(
                name
                for name, _ in model.named_parameters()
                if name.startswith(("breadth_head.", "breadth_gain"))
            )
        if args.classifier_rank > 0:
            allowed_missing.update(
                name
                for name, _ in model.named_parameters()
                if name.startswith(
                    (
                        "classifier.residual_left",
                        "classifier.residual_right",
                    )
                )
            )
        if args.semantic_router:
            allowed_missing.update(
                name
                for name, _ in model.named_parameters()
                if name.startswith(("protein_router.", "go_router."))
            )
        if args.function_alignment:
            allowed_missing.update(
                name
                for name, _ in model.named_parameters()
                if name.startswith(
                    (
                        "protein_alignment_head.",
                        "function_alignment_head.",
                    )
                )
            )
        if args.gated_refinement:
            allowed_missing.update(
                name
                for name, _ in model.named_parameters()
                if name.startswith("gated_refinement.")
            )
        if args.local_semantic_attention:
            allowed_missing.update(
                name
                for name, _ in model.named_parameters()
                if name.startswith("local_semantic_attention.")
            )
        if args.latent_domain_fusion:
            allowed_missing.update(
                name
                for name, _ in model.named_parameters()
                if name.startswith("latent_domain_fusion.")
            )
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing:
            raise ValueError(
                f"Unexpected missing checkpoint keys: "
                f"{sorted(unexpected_missing)}"
            )
        if incompatible.unexpected_keys:
            raise ValueError(
                f"Unexpected checkpoint keys: {incompatible.unexpected_keys}"
            )
        if list(checkpoint.get("terms", train_ds.terms)) != train_ds.terms:
            raise ValueError("GO term order differs in init checkpoint")
        print(f"[SEHNet] Loaded initial checkpoint: {checkpoint_path}")
    criterion = AsymmetricLoss(
        gamma_neg=args.gamma_neg,
        gamma_pos=args.gamma_pos,
        clip=args.clip,
    ).to(device)
    if args.ensemble_size > 1 and (
        args.ensemble_adapter_lr is not None
        or args.new_component_lr is not None
    ):
        adapter_tokens = ("input_scale", "output_scale", "member_bias")
        adapter_parameters = []
        shared_parameters = []
        new_parameters = []
        for name, parameter in model.named_parameters():
            is_new_parameter = name.startswith(
                (
                    "breadth_head.",
                    "breadth_gain",
                    "classifier.residual_left",
                    "classifier.residual_right",
                    "protein_router.",
                    "go_router.",
                    "protein_alignment_head.",
                    "function_alignment_head.",
                    "gated_refinement.",
                    "local_semantic_attention.",
                    "latent_domain_fusion.",
                )
            )
            if args.independent_classifier and name.startswith(
                ("classifier.member_weight", "classifier.member_bias")
            ):
                is_new_parameter = True
            if is_new_parameter:
                target = new_parameters
            elif any(token in name for token in adapter_tokens):
                target = adapter_parameters
            else:
                target = shared_parameters
            target.append(parameter)
        parameter_groups = [{"params": shared_parameters, "lr": args.lr}]
        if adapter_parameters:
            parameter_groups.append(
                {
                    "params": adapter_parameters,
                    "lr": (
                        args.lr
                        if args.ensemble_adapter_lr is None
                        else args.ensemble_adapter_lr
                    ),
                }
            )
        if new_parameters:
            parameter_groups.append(
                {
                    "params": new_parameters,
                    "lr": (
                        args.lr
                        if args.new_component_lr is None
                        else args.new_component_lr
                    ),
                }
            )
        optimizer = optim.AdamW(
            parameter_groups,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    scaler = make_grad_scaler(amp_enabled)
    go_obj = Ontology(get_path("go_obo", args.data_root))
    direct_children, direct_parents, _ = load_go_direct_parent_edges(
        args.ont,
        train_ds.terms,
        Path(args.data_root),
    )
    max_constraint_stats = {"enabled": False}

    parameter_count = int(sum(p.numel() for p in model.parameters()))
    save_json(
        result_dir / "model_summary.json",
        {
            "architecture": architecture,
            "ontology": args.ont,
            "esm_dim": 2560,
            "hidden_dim": args.hidden_dim,
            "ensemble_size": args.ensemble_size,
            "classifier_rank": args.classifier_rank,
            "independent_classifier": args.independent_classifier,
            "semantic_router": args.semantic_router,
            "router_temperature": args.router_temperature,
            "router_member_loss_ratio": args.router_member_loss_ratio,
            "ensemble_adapter_lr": args.ensemble_adapter_lr,
            "functional_breadth": args.functional_breadth,
            "breadth_loss_ratio": args.breadth_loss_ratio,
            "set_loss_ratio": args.set_loss_ratio,
            "pu_rank_ratio": args.pu_rank_ratio,
            "pu_rank_until_epoch": args.pu_rank_until_epoch,
            "pu_rank_top_k": args.pu_rank_top_k,
            "pu_rank_temperature": args.pu_rank_temperature,
            "pu_rank_min_margin": args.pu_rank_min_margin,
            "pu_rank_max_margin": args.pu_rank_max_margin,
            "pu_rank_positive_mode": args.pu_rank_positive_mode,
            "pu_rank_ic_power": args.pu_rank_ic_power,
            "max_constraint_ratio": args.max_constraint_ratio,
            "max_constraint_until_epoch": args.max_constraint_until_epoch,
            "max_constraint_graph": max_constraint_stats,
            "function_alignment": args.function_alignment,
            "function_alignment_dim": args.function_alignment_dim,
            "function_alignment_ratio": args.function_alignment_ratio,
            "function_alignment_temperature": (
                args.function_alignment_temperature
            ),
            "function_alignment_target": args.function_alignment_target,
            "function_alignment_until_epoch": (
                args.function_alignment_until_epoch
            ),
            "gated_refinement": args.gated_refinement,
            "gated_refinement_expansion": (
                args.gated_refinement_expansion
            ),
            "local_semantic_attention": args.local_semantic_attention,
            "local_segment_dim": (
                640
                if (
                    args.local_semantic_attention
                    or args.latent_domain_fusion
                )
                else 0
            ),
            "local_attention_dim": args.local_attention_dim,
            "local_label_chunk_size": args.local_label_chunk_size,
            "local_task_query_residual": (
                args.local_task_query_residual
            ),
            "local_query_source": args.local_query_source,
            "local_multilayer_segments": args.local_multilayer_segments,
            "local_num_layers": (
                3
                if (
                    args.local_multilayer_segments
                    or args.latent_domain_fusion
                )
                else 1
            ),
            "local_representation_layers": (
                [10, 20, 30]
                if (
                    args.local_multilayer_segments
                    or args.latent_domain_fusion
                )
                else [30]
            ),
            "local_classifier_discriminator": (
                args.local_classifier_discriminator
            ),
            "local_semantic_stats": local_semantic_stats,
            "latent_domain_fusion": args.latent_domain_fusion,
            "latent_domain_count": args.latent_domain_count,
            "latent_domain_dim": args.latent_domain_dim,
            "latent_domain_layer_contrast": (
                args.latent_domain_layer_contrast
            ),
            "latent_domain_layer_contrast_parameters": (
                640 * args.latent_domain_dim
                if args.latent_domain_layer_contrast
                else 0
            ),
            "latent_domain_competition": (
                args.latent_domain_competition
            ),
            "latent_domain_label_readout": (
                args.latent_domain_label_readout
            ),
            "ablation": args.ablation,
            "latent_domain_readout_mode": args.latent_domain_readout_mode,
            "latent_domain_discriminator_mode": (
                args.latent_domain_discriminator_mode
            ),
            "latent_parent_mapping": args.latent_parent_mapping,
            "parent_shuffle_seed": (
                args.parent_shuffle_seed
                if args.latent_parent_mapping == "shuffled"
                else None
            ),
            "latent_parent_mapping_stats": latent_parent_mapping_stats,
            "latent_domain_normalize_label_queries": (
                args.latent_domain_normalize_label_queries
            ),
            "latent_domain_adaptive_label_query_norm": (
                args.latent_domain_adaptive_label_query_norm
            ),
            "latent_domain_confidence_direction_routing": (
                args.latent_domain_confidence_direction_routing
            ),
            "latent_domain_ic_semantic_residual": (
                args.latent_domain_ic_semantic_residual
            ),
            "latent_domain_ic_semantic_residual_parameters": (
                768 * args.latent_domain_dim + 1
                if args.latent_domain_ic_semantic_residual
                else 0
            ),
            "latent_domain_semantic_preprocessing": (
                "global_center_then_row_l2_normalize"
                if args.latent_domain_ic_semantic_residual
                else None
            ),
            "latent_domain_semantic_weighting": (
                "train_label_laplace_information_content_mean_normalized"
                if args.latent_domain_ic_semantic_residual
                else None
            ),
            "latent_domain_confidence_normalization": (
                "detached_label_norm_over_GO_depth_RMS"
                if args.latent_domain_confidence_direction_routing
                else None
            ),
            "latent_domain_query_norm_exponents": (
                4 if args.latent_domain_adaptive_label_query_norm else 0
            ),
            "latent_label_readout_source": (
                "detached_shared_classifier_prototype"
                if args.latent_domain_discriminator_mode == "prototype"
                else "detached_shared_classifier_child_minus_pseudo_parent"
                if args.latent_parent_mapping == "shuffled"
                else "detached_shared_classifier_child_minus_direct_parent"
                if args.latent_domain_label_readout
                else None
            ),
            "latent_label_depth_gates": (
                4 if args.latent_domain_label_readout else 0
            ),
            "latent_depth_gate_freeze_abs_mean_threshold": (
                args.freeze_latent_depth_gates_at_abs_mean
                if args.freeze_latent_depth_gates_at_abs_mean > 0.0
                else None
            ),
            "anchor_preserving_local_gradients": (
                args.anchor_preserving_local_gradients
            ),
            "latent_domain_num_layers": (
                3 if args.latent_domain_fusion else 0
            ),
            "latent_domain_layer_mix": (
                (
                    "uniform_after_layer_norm_plus_zero_initialized_"
                    "final_minus_first_residual"
                )
                if args.latent_domain_layer_contrast
                else "uniform_after_layer_norm"
                if args.latent_domain_fusion
                else None
            ),
            "latent_domain_global_query_stop_gradient": (
                bool(
                    args.latent_domain_fusion
                    and not args.latent_domain_label_readout
                )
            ),
            "latent_label_prototype_stop_gradient": (
                bool(args.latent_domain_label_readout)
            ),
            "sam_rho": args.sam_rho,
            "breadth_target_mean": breadth_target_mean,
            "breadth_target_std": breadth_target_std,
            "num_terms": train_ds.class_nums,
            "total_parameters": parameter_count,
        },
    )
    print(f"[SEHNet] Result dir: {result_dir}")
    print(f"[SEHNet] Variant: {args.ablation}")
    print(
        f"[SEHNet] Train/Valid/Test: "
        f"{len(train_ds)}/{len(valid_ds)}/"
        f"{len(test_ds) if test_ds is not None else 'not loaded'}"
    )
    if args.max_constraint_ratio > 0.0:
        print(
            "[SEHNet] GO max constraint: "
            f"{max_constraint_stats['nonself_closure_pairs']:,} "
            "non-self closure pairs | "
            f"{max_constraint_stats['direct_edges']:,} direct edges | "
            f"relative weight {args.max_constraint_ratio:g}"
        )
    if args.function_alignment:
        print(
            "[SEHNet] Function-set alignment: "
            f"dim {args.function_alignment_dim} | "
            f"temperature {args.function_alignment_temperature:g} | "
            f"target {args.function_alignment_target} | "
            f"relative weight {args.function_alignment_ratio:g}"
        )
    if args.gated_refinement:
        print(
            "[SEHNet] ReZero SwiGLU refinement: "
            f"expansion {args.gated_refinement_expansion:g}"
        )
    if args.latent_domain_fusion:
        print(
            "[SEHNet] Latent-domain local fusion: "
            f"{args.latent_domain_count} domains | "
            f"dim {args.latent_domain_dim} | fixed uniform layer mix | "
            + (
                "competitive segment assignment"
                if args.latent_domain_competition
                else "independent segment attention"
            )
        )
        if args.anchor_preserving_local_gradients:
            print(
                "[SEHNet] Anchor preservation: disjoint global and local "
                "classification gradients"
            )
        if args.latent_domain_label_readout:
            print(
                "[SEHNet] GO-DAG label readout: detached shared-classifier "
                + (
                    "raw prototypes"
                    if args.latent_domain_discriminator_mode == "prototype"
                    else "child-minus-pseudo-parent locators"
                    if args.latent_parent_mapping == "shuffled"
                    else "child-minus-parent locators"
                )
                + " | four zero-initialized depth gates"
            )
            if args.latent_domain_readout_mode == "uniform_domains":
                print("[SEHNet] GO-label domain selection: uniform domain mean")
            if args.latent_domain_normalize_label_queries:
                print(
                    "[SEHNet] GO-label domain routing: parameter-free "
                    "query LayerNorm"
                )
            if args.latent_domain_adaptive_label_query_norm:
                print(
                    "[SEHNet] GO-label domain routing: four zero-initialized "
                    "GO-depth adaptive query-normalization exponents"
                )
            if args.latent_domain_confidence_direction_routing:
                print(
                    "[SEHNet] GO-label domain routing: normalized prototype "
                    "direction with detached depth-RMS confidence"
                )
            if args.latent_domain_ic_semantic_residual:
                print(
                    "[SEHNet] GO-label semantics: zero-initialized global "
                    "residual scale with train-only IC weighting"
                )
            if args.freeze_latent_depth_gates_at_abs_mean > 0.0:
                print(
                    "[SEHNet] GO-depth gate stabilization: freeze at bounded "
                    "absolute mean "
                    f"{args.freeze_latent_depth_gates_at_abs_mean:g}"
                )
    if args.sam_rho > 0.0:
        print(f"[SEHNet] Sharpness-aware minimization rho: {args.sam_rho:g}")
    print(
        f"[SEHNet] Terms: {train_ds.class_nums} | "
        f"Parameters: {parameter_count:,} | Device: {device} | AMP: {amp_enabled}"
    )

    best_score = -1.0
    best_epoch = 0
    best_monitor_fmax = float("nan")
    best_monitor_aupr = float("nan")
    best_monitor_threshold = float("nan")
    best_window_epochs: List[int] = []
    patience_counter = 0
    rows: List[Dict[str, object]] = []
    rolling_valid_raw: List[np.ndarray] = []
    rolling_valid_labels: np.ndarray | None = None
    best_valid_aupr = float("-inf")
    best_valid_aupr_fmax = float("nan")
    best_valid_aupr_threshold = float("nan")
    best_valid_aupr_window_epochs: List[int] = []
    breadth_root_indices = [
        index
        for index, term in enumerate(train_ds.terms)
        if term in {"GO:0003674", "GO:0008150", "GO:0005575"}
    ]
    trajectory_dir = result_dir / "trajectory_checkpoints"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    saved_epochs: Dict[int, Path] = {}
    latent_depth_gates_frozen = False
    latent_depth_gate_freeze_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.time()
        losses: List[float] = []
        classification_losses: List[float] = []
        anchor_classification_losses: List[float] = []
        member_losses: List[float] = []
        routing_entropies: List[float] = []
        breadth_losses: List[float] = []
        set_losses: List[float] = []
        pu_rank_losses: List[float] = []
        pu_rank_weights: List[float] = []
        max_constraint_losses: List[float] = []
        max_constraint_weights: List[float] = []
        function_alignment_losses: List[float] = []
        function_alignment_weights: List[float] = []
        iterator = (
            tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{args.epochs} [train]",
                leave=False,
            )
            if tqdm is not None
            else train_loader
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
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(amp_enabled):
                output = forward_with_optional_local(
                    model,
                    esm,
                    local_segments,
                    local_mask,
                )
                classification_loss, member_loss = (
                    supervised_classification_loss(
                        output=output,
                        target=target,
                        criterion=criterion,
                        ensemble_size=args.ensemble_size,
                        class_nums=train_ds.class_nums,
                        semantic_router=args.semantic_router,
                        router_member_loss_ratio=(
                            args.router_member_loss_ratio
                        ),
                    )
                )
                anchor_loss = anchor_classification_loss(
                    output=output,
                    target=target,
                    criterion=criterion,
                    ensemble_size=args.ensemble_size,
                    class_nums=train_ds.class_nums,
                )
                breadth_loss = classification_loss.new_zeros(())
                effective_breadth_weight = classification_loss.new_zeros(())
                if args.functional_breadth:
                    if output.breadth_score is None:
                        raise RuntimeError("Missing functional breadth output")
                    breadth_count = target.sum(dim=1)
                    if breadth_root_indices:
                        breadth_count = breadth_count - target[
                            :, breadth_root_indices
                        ].sum(dim=1)
                    breadth_target = (
                        torch.log1p(breadth_count.clamp_min(0.0))
                        - breadth_target_mean
                    ) / breadth_target_std
                    breadth_loss = F.smooth_l1_loss(
                        output.breadth_score.float(),
                        breadth_target.float(),
                    )
                    effective_breadth_weight = (
                        args.breadth_loss_ratio
                        * classification_loss.detach()
                        / breadth_loss.detach().clamp_min(1e-8)
                    )
                set_loss = classification_loss.new_zeros(())
                effective_set_weight = classification_loss.new_zeros(())
                if args.set_loss_ratio > 0.0:
                    set_loss = protein_set_f1_loss(
                        output.probs,
                        target,
                        breadth_root_indices,
                    )
                    effective_set_weight = (
                        args.set_loss_ratio
                        * classification_loss.detach()
                        / set_loss.detach().clamp_min(1e-8)
                    )
                pu_rank_loss = classification_loss.new_zeros(())
                effective_pu_rank_weight = classification_loss.new_zeros(())
                pu_rank_active = (
                    args.pu_rank_ratio > 0.0
                    and (
                        args.pu_rank_until_epoch <= 0
                        or epoch <= args.pu_rank_until_epoch
                    )
                )
                if pu_rank_active:
                    if (
                        ranking_semantics is None
                        or information_content is None
                    ):
                        raise RuntimeError("Missing PU ranking state")
                    pu_rank_loss = dynamic_pu_label_ranking_loss(
                        logits=output.logits,
                        labels=target,
                        normalized_semantics=ranking_semantics,
                        information_content=information_content,
                        root_indices=breadth_root_indices,
                        top_k=args.pu_rank_top_k,
                        temperature=args.pu_rank_temperature,
                        min_margin=args.pu_rank_min_margin,
                        max_margin=args.pu_rank_max_margin,
                        positive_mode=args.pu_rank_positive_mode,
                        ic_power=args.pu_rank_ic_power,
                    )
                    effective_pu_rank_weight = (
                        args.pu_rank_ratio
                        * classification_loss.detach()
                        / pu_rank_loss.detach().clamp_min(1e-8)
                    )
                max_constraint_loss = classification_loss.new_zeros(())
                effective_max_constraint_weight = (
                    classification_loss.new_zeros(())
                )
                max_constraint_active = (
                    args.max_constraint_ratio > 0.0
                    and (
                        args.max_constraint_until_epoch <= 0
                        or epoch <= args.max_constraint_until_epoch
                    )
                )
                if max_constraint_active:
                    max_constraint_loss = max_constraint_asymmetric_loss(
                        probabilities=output.probs,
                        labels=target,
                        closure_sources=closure_sources,
                        closure_targets=closure_targets,
                        criterion=criterion,
                    )
                    effective_max_constraint_weight = (
                        args.max_constraint_ratio
                        * classification_loss.detach()
                        / max_constraint_loss.detach().clamp_min(1e-8)
                    )
                function_alignment_loss = classification_loss.new_zeros(())
                effective_function_alignment_weight = (
                    classification_loss.new_zeros(())
                )
                function_alignment_active = (
                    args.function_alignment
                    and args.function_alignment_ratio > 0.0
                    and (
                        args.function_alignment_until_epoch <= 0
                        or epoch <= args.function_alignment_until_epoch
                    )
                )
                if function_alignment_active:
                    if (
                        ranking_semantics is None
                        or information_content is None
                    ):
                        raise RuntimeError(
                            "Missing function-alignment semantics"
                        )
                    function_alignment_loss = function_set_alignment_loss(
                        model=model,
                        protein_features=output.features.mean(dim=1),
                        labels=target,
                        normalized_semantics=ranking_semantics,
                        information_content=information_content,
                        root_indices=breadth_root_indices,
                        temperature=(
                            args.function_alignment_temperature
                        ),
                        target_mode=args.function_alignment_target,
                    )
                    effective_function_alignment_weight = (
                        args.function_alignment_ratio
                        * classification_loss.detach()
                        / function_alignment_loss.detach().clamp_min(1e-8)
                    )
                loss = (
                    classification_loss
                    + anchor_loss
                    + effective_breadth_weight * breadth_loss
                    + effective_set_weight * set_loss
                    + effective_pu_rank_weight * pu_rank_loss
                    + effective_max_constraint_weight
                    * max_constraint_loss
                    + effective_function_alignment_weight
                    * function_alignment_loss
                )
            scaler.scale(loss).backward()
            if args.sam_rho > 0.0:
                perturbations = sam_perturb_parameters(
                    model,
                    rho=args.sam_rho,
                    gradient_scale=float(scaler.get_scale()),
                )
                optimizer.zero_grad(set_to_none=True)
                batch_norm_states = pause_batch_norm_updates(model)
                try:
                    with amp_autocast(amp_enabled):
                        output = forward_with_optional_local(
                            model,
                            esm,
                            local_segments,
                            local_mask,
                        )
                        classification_loss, member_loss = (
                            supervised_classification_loss(
                                output=output,
                                target=target,
                                criterion=criterion,
                                ensemble_size=args.ensemble_size,
                                class_nums=train_ds.class_nums,
                                semantic_router=args.semantic_router,
                                router_member_loss_ratio=(
                                    args.router_member_loss_ratio
                                ),
                            )
                        )
                        loss = classification_loss
                    scaler.scale(loss).backward()
                finally:
                    restore_batch_norm_updates(batch_norm_states)
                    restore_sam_parameters(perturbations)
            scaler.unscale_(optimizer)
            local_grad_norm = loss.new_zeros(())
            if args.anchor_preserving_local_gradients:
                global_parameters = []
                local_parameters = []
                for name, parameter in model.named_parameters():
                    (
                        local_parameters
                        if name.startswith("latent_domain_fusion.")
                        else global_parameters
                    ).append(parameter)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    global_parameters,
                    args.grad_clip,
                )
                local_grad_norm = torch.nn.utils.clip_grad_norm_(
                    local_parameters,
                    args.grad_clip,
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            classification_losses.append(
                float(classification_loss.detach())
            )
            anchor_classification_losses.append(
                float(anchor_loss.detach())
            )
            member_losses.append(float(member_loss.detach()))
            if output.routing_entropy is not None:
                routing_entropies.append(
                    float(output.routing_entropy.detach())
                )
            breadth_losses.append(float(breadth_loss.detach()))
            set_losses.append(float(set_loss.detach()))
            pu_rank_losses.append(float(pu_rank_loss.detach()))
            pu_rank_weights.append(
                float(effective_pu_rank_weight.detach())
            )
            max_constraint_losses.append(
                float(max_constraint_loss.detach())
            )
            max_constraint_weights.append(
                float(effective_max_constraint_weight.detach())
            )
            function_alignment_losses.append(
                float(function_alignment_loss.detach())
            )
            function_alignment_weights.append(
                float(effective_function_alignment_weight.detach())
            )
            if tqdm is not None:
                iterator.set_postfix(loss=f"{float(loss.detach()):.4f}")
        train_seconds = time.time() - started
        valid_started = time.time()
        valid = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            go_obj=go_obj,
            terms=train_ds.terms,
            device=device,
            amp_enabled=amp_enabled,
            desc=f"Epoch {epoch}/{args.epochs} [valid]",
            direct_children=direct_children,
            direct_parents=direct_parents,
        )
        single_epoch_score = float(valid["fmax"]) + float(valid["aupr"])
        window_fmax = float("nan")
        window_aupr = float("nan")
        window_threshold = float("nan")
        window_score = float("nan")
        window_epochs: List[int] = []
        if args.early_stop_monitor == "rolling_five":
            current_labels = np.asarray(valid["labels"])
            if rolling_valid_labels is None:
                rolling_valid_labels = current_labels.copy()
            elif not np.array_equal(rolling_valid_labels, current_labels):
                raise ValueError(
                    "Validation label order changed across rolling checkpoints"
                )
            rolling_valid_raw.append(
                np.asarray(valid["preds_raw"], dtype=np.float32)
            )
            if len(rolling_valid_raw) > 5:
                rolling_valid_raw.pop(0)
            if len(rolling_valid_raw) == 5:
                averaged_raw = np.zeros_like(
                    rolling_valid_raw[0],
                    dtype=np.float64,
                )
                for member_raw in rolling_valid_raw:
                    averaged_raw += member_raw
                averaged_raw = (averaged_raw / 5.0).astype(np.float32)
                (
                    _,
                    window_fmax,
                    window_aupr,
                    window_threshold,
                ) = cafa_metrics(
                    averaged_raw,
                    rolling_valid_labels,
                    go_obj,
                    train_ds.terms,
                )
                window_score = float(window_fmax) + float(window_aupr)
                window_epochs = list(range(epoch - 4, epoch + 1))
                if window_aupr > best_valid_aupr:
                    best_valid_aupr = float(window_aupr)
                    best_valid_aupr_fmax = float(window_fmax)
                    best_valid_aupr_threshold = float(window_threshold)
                    best_valid_aupr_window_epochs = list(window_epochs)
        monitor_score = (
            single_epoch_score
            if args.early_stop_monitor == "single_epoch"
            else (window_score if window_epochs else None)
        )
        scheduler.step(
            single_epoch_score if monitor_score is None else monitor_score
        )
        was_latent_depth_gates_frozen = latent_depth_gates_frozen
        (
            latent_depth_gates_frozen,
            new_freeze_epoch,
            current_gate_abs_mean,
        ) = maybe_freeze_latent_depth_gates(
            model,
            args.freeze_latent_depth_gates_at_abs_mean,
            epoch,
            latent_depth_gates_frozen,
        )
        if (
            latent_depth_gates_frozen
            and not was_latent_depth_gates_frozen
        ):
            latent_depth_gate_freeze_epoch = new_freeze_epoch
            print(
                "[SEHNet] Froze four latent GO-depth gates at epoch "
                f"{epoch} (bounded abs mean "
                f"{current_gate_abs_mean:.6f})."
            )
        peak_vram_mib = (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_classification_loss": float(
                np.mean(classification_losses)
            ),
            "train_anchor_classification_loss": float(
                np.mean(anchor_classification_losses)
            ),
            "train_member_loss": float(np.mean(member_losses)),
            "train_routing_entropy": (
                float(np.mean(routing_entropies))
                if routing_entropies
                else 0.0
            ),
            "train_breadth_loss": float(np.mean(breadth_losses)),
            "train_set_loss": float(np.mean(set_losses)),
            "train_pu_rank_loss": float(np.mean(pu_rank_losses)),
            "train_pu_rank_effective_weight": float(
                np.mean(pu_rank_weights)
            ),
            "pu_rank_active": bool(
                args.pu_rank_ratio > 0.0
                and (
                    args.pu_rank_until_epoch <= 0
                    or epoch <= args.pu_rank_until_epoch
                )
            ),
            "train_max_constraint_loss": float(
                np.mean(max_constraint_losses)
            ),
            "train_max_constraint_effective_weight": float(
                np.mean(max_constraint_weights)
            ),
            "max_constraint_active": bool(
                args.max_constraint_ratio > 0.0
                and (
                    args.max_constraint_until_epoch <= 0
                    or epoch <= args.max_constraint_until_epoch
                )
            ),
            "train_function_alignment_loss": float(
                np.mean(function_alignment_losses)
            ),
            "train_function_alignment_effective_weight": float(
                np.mean(function_alignment_weights)
            ),
            "function_alignment_active": bool(
                args.function_alignment
                and args.function_alignment_ratio > 0.0
                and (
                    args.function_alignment_until_epoch <= 0
                    or epoch <= args.function_alignment_until_epoch
                )
            ),
            "valid_loss": float(valid["loss"]),
            "valid_fmax": float(valid["fmax"]),
            "valid_aupr": float(valid["aupr"]),
            "valid_threshold": float(valid["threshold"]),
            "valid_member_probability_variance": float(
                valid["member_probability_variance"]
            ),
            "valid_routing_entropy": float(valid["routing_entropy"]),
            "valid_routing_top1_share": float(valid["routing_top1_share"]),
            "valid_hierarchy_violation_rate": float(
                valid["hierarchy_violation_rate"]
            ),
            "valid_hierarchy_violation_margin": float(
                valid["hierarchy_violation_margin"]
            ),
            "checkpoint_score": single_epoch_score,
            "valid_window_fmax": window_fmax,
            "valid_window_aupr": window_aupr,
            "valid_window_threshold": window_threshold,
            "valid_window_score": window_score,
            "valid_window_start_epoch": (
                window_epochs[0] if window_epochs else 0
            ),
            "valid_window_end_epoch": (
                window_epochs[-1] if window_epochs else 0
            ),
            "early_stop_score": (
                float(monitor_score)
                if monitor_score is not None
                else float("nan")
            ),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "lr_adapter": float(
                optimizer.param_groups[1]["lr"]
                if len(optimizer.param_groups) > 1
                else optimizer.param_groups[0]["lr"]
            ),
            "lr_new_component": float(
                optimizer.param_groups[-1]["lr"]
                if (
                    args.functional_breadth
                    or args.classifier_rank > 0
                    or args.semantic_router
                    or args.independent_classifier
                    or args.function_alignment
                    or args.gated_refinement
                    or args.local_semantic_attention
                    or args.latent_domain_fusion
                )
                else optimizer.param_groups[0]["lr"]
            ),
            "breadth_gain": float(
                model.breadth_gain.detach()
                if getattr(model, "breadth_gain", None) is not None
                else 0.0
            ),
            "gated_refinement_scale_abs_mean": float(
                model.gated_refinement.member_scale.detach().abs().mean()
                if getattr(model, "gated_refinement", None) is not None
                else 0.0
            ),
            "local_gate_depth_1_2": float(
                model.local_semantic_attention.depth_gates[0].detach()
                if getattr(model, "local_semantic_attention", None)
                is not None
                else 0.0
            ),
            "local_gate_depth_3_4": float(
                model.local_semantic_attention.depth_gates[1].detach()
                if getattr(model, "local_semantic_attention", None)
                is not None
                else 0.0
            ),
            "local_gate_depth_5_6": float(
                model.local_semantic_attention.depth_gates[2].detach()
                if getattr(model, "local_semantic_attention", None)
                is not None
                else 0.0
            ),
            "local_gate_depth_7_plus": float(
                model.local_semantic_attention.depth_gates[3].detach()
                if getattr(model, "local_semantic_attention", None)
                is not None
                else 0.0
            ),
            "local_query_residual_norm_mean": float(
                model.local_semantic_attention.query_residual.detach()
                .norm(dim=-1)
                .mean()
                if (
                    getattr(model, "local_semantic_attention", None)
                    is not None
                    and model.local_semantic_attention.query_residual
                    is not None
                )
                else 0.0
            ),
            "latent_domain_gate_abs_mean": float(
                model.latent_domain_fusion.gate_abs_mean()
                if getattr(model, "latent_domain_fusion", None)
                is not None
                else 0.0
            ),
            "latent_depth_gates_frozen": bool(
                latent_depth_gates_frozen
            ),
            "latent_depth_gate_freeze_epoch": int(
                latent_depth_gate_freeze_epoch
            ),
            "latent_label_gate_depth_1_2": float(
                torch.tanh(
                    model.latent_domain_fusion.depth_gates[0].detach()
                )
                if (
                    getattr(model, "latent_domain_fusion", None) is not None
                    and model.latent_domain_fusion.depth_gates is not None
                )
                else 0.0
            ),
            "latent_label_gate_depth_3_4": float(
                torch.tanh(
                    model.latent_domain_fusion.depth_gates[1].detach()
                )
                if (
                    getattr(model, "latent_domain_fusion", None) is not None
                    and model.latent_domain_fusion.depth_gates is not None
                )
                else 0.0
            ),
            "latent_label_gate_depth_5_6": float(
                torch.tanh(
                    model.latent_domain_fusion.depth_gates[2].detach()
                )
                if (
                    getattr(model, "latent_domain_fusion", None) is not None
                    and model.latent_domain_fusion.depth_gates is not None
                )
                else 0.0
            ),
            "latent_label_gate_depth_7_plus": float(
                torch.tanh(
                    model.latent_domain_fusion.depth_gates[3].detach()
                )
                if (
                    getattr(model, "latent_domain_fusion", None) is not None
                    and model.latent_domain_fusion.depth_gates is not None
                )
                else 0.0
            ),
            "latent_domain_query_cosine_mean": float(
                model.latent_domain_fusion.mean_query_cosine()
                if getattr(model, "latent_domain_fusion", None)
                is not None
                else 0.0
            ),
            "valid_latent_domain_segment_entropy": float(
                valid["latent_domain_segment_entropy"]
            ),
            "valid_latent_domain_selection_entropy": float(
                valid["latent_domain_selection_entropy"]
            ),
            "valid_latent_domain_member_cosine": float(
                valid["latent_domain_member_cosine"]
            ),
            "valid_latent_domain_label_evidence_abs_mean": float(
                valid["latent_domain_label_evidence_abs_mean"]
            ),
            "latent_query_norm_exponent_depth_1_2": float(
                torch.tanh(
                    model.latent_domain_fusion.query_norm_exponents[0]
                    .detach()
                )
                if (
                    getattr(model, "latent_domain_fusion", None) is not None
                    and model.latent_domain_fusion.query_norm_exponents
                    is not None
                )
                else 0.0
            ),
            "latent_query_norm_exponent_depth_3_4": float(
                torch.tanh(
                    model.latent_domain_fusion.query_norm_exponents[1]
                    .detach()
                )
                if (
                    getattr(model, "latent_domain_fusion", None) is not None
                    and model.latent_domain_fusion.query_norm_exponents
                    is not None
                )
                else 0.0
            ),
            "latent_query_norm_exponent_depth_5_6": float(
                torch.tanh(
                    model.latent_domain_fusion.query_norm_exponents[2]
                    .detach()
                )
                if (
                    getattr(model, "latent_domain_fusion", None) is not None
                    and model.latent_domain_fusion.query_norm_exponents
                    is not None
                )
                else 0.0
            ),
            "latent_query_norm_exponent_depth_7_plus": float(
                torch.tanh(
                    model.latent_domain_fusion.query_norm_exponents[3]
                    .detach()
                )
                if (
                    getattr(model, "latent_domain_fusion", None) is not None
                    and model.latent_domain_fusion.query_norm_exponents
                    is not None
                )
                else 0.0
            ),
            "latent_semantic_residual_scale": float(
                model.latent_domain_fusion.semantic_residual_scale()
                if getattr(model, "latent_domain_fusion", None) is not None
                else 0.0
            ),
            "latent_layer_contrast_weight_norm": float(
                model.latent_domain_fusion.layer_contrast_projection.weight
                .detach()
                .norm()
                if (
                    getattr(model, "latent_domain_fusion", None) is not None
                    and model.latent_domain_fusion
                    .layer_contrast_projection is not None
                )
                else 0.0
            ),
            "grad_norm": float(grad_norm),
            "local_grad_norm": float(local_grad_norm),
            "train_seconds": train_seconds,
            "valid_seconds": time.time() - valid_started,
            "peak_vram_mib": peak_vram_mib,
        }
        local_module = getattr(model, "local_semantic_attention", None)
        if (
            local_module is not None
            and local_module.depth_layer_logits is not None
        ):
            depth_layer_weights = torch.softmax(
                local_module.depth_layer_logits.detach(),
                dim=-1,
            ).cpu()
            for depth_index, depth_name in enumerate(
                ("1_2", "3_4", "5_6", "7_plus")
            ):
                for layer_index, layer_name in enumerate(("10", "20", "30")):
                    row[
                        f"local_depth_{depth_name}_layer_{layer_name}_weight"
                    ] = float(
                        depth_layer_weights[depth_index, layer_index]
                    )
        rows.append(row)
        pd.DataFrame(rows).to_csv(result_dir / "metrics_log.csv", index=False)
        trajectory_path = trajectory_dir / f"epoch_{epoch:03d}.pth"
        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "terms": train_ds.terms,
            },
            trajectory_path,
        )
        saved_epochs[epoch] = trajectory_path
        print(
            f"Epoch {epoch:03d} | Train {row['train_loss']:.4f} | "
            f"Valid {valid['fmax']:.4f}/{valid['aupr']:.4f} | "
            + (
                f"Window5 {window_fmax:.4f}/{window_aupr:.4f} | "
                if window_epochs
                else ""
            )
            +
            (
                ""
            )
            +
            f"LR {row['lr']:.2e} | Time {train_seconds:.1f}s | "
            f"VRAM {peak_vram_mib:.0f} MiB"
        )

        if monitor_score is not None and monitor_score > best_score:
            best_score = float(monitor_score)
            best_epoch = epoch
            best_monitor_fmax = (
                float(valid["fmax"])
                if args.early_stop_monitor == "single_epoch"
                else window_fmax
            )
            best_monitor_aupr = (
                float(valid["aupr"])
                if args.early_stop_monitor == "single_epoch"
                else window_aupr
            )
            best_monitor_threshold = (
                float(valid["threshold"])
                if args.early_stop_monitor == "single_epoch"
                else window_threshold
            )
            best_window_epochs = (
                list(window_epochs)
                if window_epochs
                else [epoch]
            )
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": epoch,
                    "terms": train_ds.terms,
                },
                result_dir / "best_model.pth",
            )
            save_json(result_dir / "best_valid_metrics.json", row)
        elif monitor_score is not None:
            patience_counter += 1

        required_epochs = set(best_window_epochs)
        required_epochs.update(best_valid_aupr_window_epochs)
        if args.early_stop_monitor == "single_epoch" and best_epoch > 0:
            required_epochs.update(
                range(max(1, best_epoch - 2), best_epoch + 3)
            )
        required_epochs.update(range(max(1, epoch - 4), epoch + 1))
        for saved_epoch in list(saved_epochs):
            if saved_epoch not in required_epochs:
                saved_epochs[saved_epoch].unlink(missing_ok=True)
                del saved_epochs[saved_epoch]
        if args.patience > 0 and patience_counter >= args.patience:
            print(
                f"[SEHNet] Early stopping at epoch {epoch}; "
                f"best monitor endpoint {best_epoch}."
            )
            break

    final_epoch = int(rows[-1]["epoch"])
    last_five_epochs = [
        epoch
        for epoch in range(max(1, final_epoch - 4), final_epoch + 1)
        if epoch in saved_epochs
    ]
    if not last_five_epochs:
        raise RuntimeError("No checkpoints available for final evaluation")

    if args.skip_final_test:
        save_json(
            result_dir / "training_summary.json",
            {
                "best_epoch": best_epoch,
                "early_stop_monitor": args.early_stop_monitor,
                "best_monitor_fmax": best_monitor_fmax,
                "best_monitor_aupr": best_monitor_aupr,
                "best_monitor_threshold": best_monitor_threshold,
                "best_validation_window_epochs": best_window_epochs,
                "best_validation_aupr_window_fmax": (
                    best_valid_aupr_fmax
                    if best_valid_aupr_window_epochs
                    else None
                ),
                "best_validation_aupr_window_aupr": (
                    best_valid_aupr
                    if best_valid_aupr_window_epochs
                    else None
                ),
                "best_validation_aupr_window_threshold": (
                    best_valid_aupr_threshold
                    if best_valid_aupr_window_epochs
                    else None
                ),
                "best_validation_aupr_window_epochs": (
                    best_valid_aupr_window_epochs
                ),
                "final_epoch": final_epoch,
                "final_five_epochs": last_five_epochs,
                "checkpoint_protocol": "last_five_training_epochs",
                "checkpoint_score": best_score,
                "num_train_proteins": len(train_ds),
                "num_valid_proteins": len(valid_ds),
                "num_terms": train_ds.class_nums,
                "total_parameters": parameter_count,
                "latent_depth_gate_freeze_abs_mean_threshold": (
                    args.freeze_latent_depth_gates_at_abs_mean
                    if args.freeze_latent_depth_gates_at_abs_mean > 0.0
                    else None
                ),
                "latent_depth_gate_freeze_epoch": (
                    latent_depth_gate_freeze_epoch
                    if latent_depth_gates_frozen
                    else None
                ),
                "latent_depth_gates_frozen": bool(
                    latent_depth_gates_frozen
                ),
                "final_latent_semantic_residual_scale": float(
                    rows[-1]["latent_semantic_residual_scale"]
                ),
                "mean_train_seconds": float(
                    np.mean([row["train_seconds"] for row in rows])
                ),
                "max_peak_vram_mib": float(
                    max(row["peak_vram_mib"] for row in rows)
                ),
                "test_evaluation_skipped": True,
            },
        )
        print(
            f"\n====== SEHNet {args.ont.upper()} Training Stage ======\n"
            f"Best validation epoch: {best_epoch}\n"
            f"Early-stop monitor: {args.early_stop_monitor}\n"
            f"Best validation window: {best_window_epochs}\n"
            f"Final five checkpoints: {last_five_epochs}\n"
            + (
                'Test evaluation: skipped by configuration\n'
            )
            +
            "=========================================="
        )
        return

    if test_ds is None or test_loader is None:
        raise RuntimeError("Test evaluation requested without a test dataset")
    test_members: Dict[int, Dict[str, np.ndarray]] = {}
    for epoch in last_five_epochs:
        checkpoint = torch.load(
            saved_epochs[epoch],
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        member = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            go_obj=go_obj,
            terms=train_ds.terms,
            device=device,
            amp_enabled=amp_enabled,
            desc=f"test checkpoint epoch {epoch}",
            direct_children=direct_children,
            direct_parents=direct_parents,
        )
        test_members[epoch] = {
            "preds_raw": np.asarray(member["preds_raw"]),
            "labels": np.asarray(member["labels"]),
        }

    def evaluate_checkpoint_window(
        epochs: Sequence[int],
    ) -> Dict[str, object]:
        labels = np.asarray(test_members[epochs[0]]["labels"])
        raw_probs = np.mean(
            [test_members[epoch]["preds_raw"] for epoch in epochs],
            axis=0,
        )
        propagated, fmax, aupr, threshold = cafa_metrics(
            raw_probs,
            labels,
            go_obj,
            train_ds.terms,
        )
        return {
            "epochs": list(epochs),
            "labels": labels,
            "scores_raw": raw_probs,
            "scores": propagated,
            "fmax": fmax,
            "aupr": aupr,
            "threshold": threshold,
        }

    final_result = evaluate_checkpoint_window(last_five_epochs)
    labels = np.asarray(final_result["labels"])
    raw_probs = np.asarray(final_result["scores_raw"])
    propagated = np.asarray(final_result["scores"])
    fmax = float(final_result["fmax"])
    aupr = float(final_result["aupr"])
    threshold = float(final_result["threshold"])
    ensemble_epochs = list(last_five_epochs)

    score_bundle = {
        "scores": propagated.astype(np.float32),
        "scores_raw": raw_probs.astype(np.float32),
        "labels": labels.astype(np.uint8),
        "terms": list(train_ds.terms),
        "ensemble_epochs": ensemble_epochs,
        "checkpoint_protocol": "last_five_training_epochs",
    }
    with (result_dir / "final_test_scores.pkl").open("wb") as handle:
        pkl.dump(score_bundle, handle, protocol=pkl.HIGHEST_PROTOCOL)
    summary = {
        "best_epoch": best_epoch,
        "early_stop_monitor": args.early_stop_monitor,
        "best_monitor_fmax": best_monitor_fmax,
        "best_monitor_aupr": best_monitor_aupr,
        "best_monitor_threshold": best_monitor_threshold,
        "best_validation_window_epochs": best_window_epochs,
        "final_epoch": final_epoch,
        "ensemble_epochs": ensemble_epochs,
        "checkpoint_protocol": "last_five_training_epochs",
        "checkpoint_score": best_score,
        "fmax": fmax,
        "aupr": aupr,
        "threshold": threshold,
        "num_test_proteins": len(test_ds),
        "num_terms": train_ds.class_nums,
        "total_parameters": parameter_count,
        "mean_train_seconds": float(
            np.mean([row["train_seconds"] for row in rows])
        ),
        "max_peak_vram_mib": float(
            max(row["peak_vram_mib"] for row in rows)
        ),
    }
    save_json(result_dir / "test_summary.json", summary)
    (result_dir / "best_metrics.txt").write_text(
        f"{fmax:.4f},{aupr:.4f}\n",
        encoding="utf-8",
    )
    print(
        f"\n====== SEHNet {args.ont.upper()} Test ======\n"
        f"Best epoch: {best_epoch}\n"
        f"Final five epochs: {ensemble_epochs}\n"
        f"Fmax/AUPR: {fmax:.4f}/{aupr:.4f}\n"
        f"=========================================="
    )
