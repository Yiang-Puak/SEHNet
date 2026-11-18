from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .config import ONTOLOGIES, get_path
from .training import run_training


ABLATIONS = {
    "full": "full",
    "global_only": "global_only",
    "prototype_only": "raw_prototype",
    "pseudo_parent": "shuffled_parent",
    "uniform_readout": "uniform_domain_readout",
    "independent_allocation": "independent_segments",
    "joint_gradient": "joint_gradient",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train SEHNet or one of its six operation-defined ablations.')
    parser.add_argument("--config", type=Path, help="Ontology configuration JSON.")
    parser.add_argument("--ont", choices=ONTOLOGIES)
    parser.add_argument("--data_root", type=Path, default=get_path("data_root"))
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--ablation", choices=tuple(ABLATIONS), default="full")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--subset_seed", type=int)
    parser.add_argument(
        "--parent_shuffle_seed",
        type=int,
        help="Fixed pseudo-parent map seed; defaults to --seed for shuffled_parent.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Defaults to the manuscript endpoint for the chosen ontology.")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument(
        "--early_stop_monitor",
        choices=("single_epoch", "rolling_five"),
        default="rolling_five",
    )
    parser.add_argument("--skip_final_test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--label_chunksize", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ensemble_adapter_lr", type=float, default=1e-4)
    parser.add_argument("--new_component_lr", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_factor", type=float, default=0.6)
    parser.add_argument("--scheduler_patience", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--gamma_neg", type=float, default=2.0)
    parser.add_argument("--gamma_pos", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    probe, _ = parser.parse_known_args()
    if probe.config is not None:
        defaults = json.loads(probe.config.read_text(encoding="utf-8"))
        allowed = {action.dest for action in parser._actions} - {"help", "config"}
        unknown = set(defaults) - allowed
        if unknown:
            parser.error(f"Unknown configuration keys: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    if args.ont not in ONTOLOGIES:
        parser.error("Select --ont mf/bp/cc or provide --config.")
    if args.ablation not in ABLATIONS:
        parser.error("Unknown ablation configuration.")
    if args.epochs is None:
        args.epochs = {"mf": 50, "cc": 45, "bp": 75}[args.ont]
    if args.epochs < 5:
        parser.error("At least five epochs are required for final-five averaging.")
    return args


def apply_model_config(args: argparse.Namespace) -> None:

    variant = ABLATIONS[args.ablation]
    uses_local = variant != "global_only"
    args.ensemble_size = 8
    args.classifier_rank = 0
    args.independent_classifier = False
    args.semantic_router = False
    args.router_temperature = 1.0
    args.router_member_loss_ratio = 0.25
    args.max_train_samples = None
    args.max_valid_samples = None
    args.max_test_samples = None
    args.init_checkpoint = None
    args.functional_breadth = False
    args.breadth_loss_ratio = 0.0
    args.set_loss_ratio = 0.0
    args.pu_rank_ratio = 0.0
    args.pu_rank_until_epoch = 0
    args.pu_rank_top_k = 16
    args.pu_rank_temperature = 0.5
    args.pu_rank_min_margin = 0.0
    args.pu_rank_max_margin = 0.2
    args.pu_rank_positive_mode = "prototype_mean"
    args.pu_rank_ic_power = 1.0
    args.max_constraint_ratio = 0.0
    args.max_constraint_until_epoch = 0
    args.function_alignment = False
    args.function_alignment_dim = 128
    args.function_alignment_ratio = 0.0
    args.function_alignment_temperature = 0.1
    args.function_alignment_target = "identity"
    args.function_alignment_until_epoch = 0
    args.gated_refinement = False
    args.gated_refinement_expansion = 1.0
    args.local_semantic_attention = False
    args.local_attention_dim = 64
    args.local_label_chunk_size = 512
    args.local_task_query_residual = False
    args.local_query_source = "semantic"
    args.local_multilayer_segments = False
    args.local_classifier_discriminator = "parent_contrast"
    args.latent_domain_fusion = uses_local
    args.latent_domain_count = 8
    args.latent_domain_dim = 64
    args.latent_domain_layer_contrast = False
    args.latent_domain_competition = uses_local and variant != "independent_segments"
    args.latent_domain_label_readout = uses_local
    args.latent_domain_normalize_label_queries = False
    args.latent_domain_adaptive_label_query_norm = False
    args.latent_domain_confidence_direction_routing = False
    args.latent_domain_ic_semantic_residual = False
    args.freeze_latent_depth_gates_at_abs_mean = 0.0
    args.anchor_preserving_local_gradients = (
        uses_local
        and variant != "joint_gradient"
    )
    args.latent_domain_readout_mode = (
        "uniform_domains" if variant == "uniform_domain_readout" else "term_specific"
    )
    args.latent_domain_discriminator_mode = (
        "prototype" if variant == "raw_prototype" else "child_minus_parent"
    )
    args.latent_parent_mapping = (
        "shuffled" if variant == "shuffled_parent" else "direct"
    )
    args.parent_shuffle_seed = (
        args.parent_shuffle_seed
        if args.parent_shuffle_seed is not None
        else args.seed
    )
    args.sam_rho = 0.0


def main() -> int:
    args = parse_args()
    apply_model_config(args)
    os.environ["PROTEIN_FUNCTION_DATA"] = str(
        args.data_root.expanduser().resolve(strict=False)
    )
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
