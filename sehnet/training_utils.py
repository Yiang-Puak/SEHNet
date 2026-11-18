from __future__ import annotations

import hashlib
import pickle as pkl
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import get_path


class AsymmetricLoss(nn.Module):

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        probs = torch.sigmoid(logits)
        negative_probs = F.relu(probs - self.clip)
        positive_loss = (
            -targets
            * (1.0 - probs).pow(self.gamma_pos)
            * F.logsigmoid(logits)
        )
        negative_loss = (
            -(1.0 - targets)
            * negative_probs.pow(self.gamma_neg)
            * torch.log1p(
                -negative_probs.clamp_max(1.0 - self.eps)
            )
        )
        return (positive_loss + negative_loss).mean()


def load_go_semantics(
    ont: str,
    terms: Sequence[str],
    data_root: str | Path,
) -> torch.Tensor:

    path = (
        get_path("go_semantics_dir", data_root)
        / f"{ont}_go_semantics_768.pkl"
    )
    with path.open("rb") as handle:
        semantics = pkl.load(handle)
    rows: List[torch.Tensor] = []
    missing = 0
    for term in terms:
        value = semantics.get(term)
        if value is None:
            missing += 1
            rows.append(torch.zeros(768, dtype=torch.float32))
        else:
            rows.append(
                torch.as_tensor(value, dtype=torch.float32).reshape(-1)
            )
    if missing:
        raise ValueError(f"Missing semantics for {missing} GO terms in {path}")
    result = torch.stack(rows)
    if result.shape != (len(terms), 768):
        raise ValueError(f"Unexpected GO semantics shape: {tuple(result.shape)}")
    return result


def load_parent_contrast_go_semantics(
    ont: str,
    terms: Sequence[str],
    data_root: str | Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:

    from .metrics import Ontology

    semantics = load_go_semantics(ont, terms, data_root).float()
    semantics = semantics - semantics.mean(dim=0, keepdim=True)
    semantics = F.normalize(semantics, dim=-1)
    ontology = Ontology(get_path("go_obo", data_root))
    term_to_index = {term: index for index, term in enumerate(terms)}
    parent_contrast = torch.zeros_like(semantics)
    parent_counts = torch.zeros(len(terms), dtype=torch.long)

    for child_index, term in enumerate(terms):
        parents = ontology.ont.get(term, {}).get("is_a", [])
        parent_indices = sorted(
            {
                term_to_index[parent]
                for parent in parents
                if parent in term_to_index and parent != term
            }
        )
        if not parent_indices:
            continue
        parent_counts[child_index] = len(parent_indices)
        parent_mean = semantics[parent_indices].mean(dim=0)
        parent_contrast[child_index] = semantics[child_index] - parent_mean
    parent_contrast = F.normalize(parent_contrast, dim=-1)

    roots = {
        "mf": "GO:0003674",
        "bp": "GO:0008150",
        "cc": "GO:0005575",
    }
    root = roots[ont]

    @lru_cache(maxsize=None)
    def depth(term: str) -> int:
        if term == root:
            return 0
        parents = ontology.ont.get(term, {}).get("is_a", [])
        if not parents:
            return 0
        return 1 + max(depth(parent) for parent in parents)

    depths = torch.tensor(
        [depth(term) for term in terms],
        dtype=torch.long,
    )
    depth_groups = torch.zeros_like(depths)
    depth_groups[(depths >= 3) & (depths <= 4)] = 1
    depth_groups[(depths >= 5) & (depths <= 6)] = 2
    depth_groups[depths >= 7] = 3
    stats: Dict[str, object] = {
        "relations": "is_a_and_part_of_as_parsed_by_DPGOK_ontology",
        "terms": len(terms),
        "terms_with_selected_parents": int((parent_counts > 0).sum()),
        "max_selected_parent_count": int(parent_counts.max()),
        "depth_group_counts": [
            int((depth_groups == index).sum()) for index in range(4)
        ],
    }
    return semantics, parent_contrast, depth_groups, stats


def load_go_direct_parent_edges(
    ont: str,
    terms: Sequence[str],
    data_root: str | Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    from .metrics import Ontology

    ontology = Ontology(get_path("go_obo", data_root))
    term_to_index = {term: index for index, term in enumerate(terms)}
    child_indices: List[int] = []
    parent_indices: List[int] = []
    parent_counts = torch.zeros(len(terms), dtype=torch.long)
    for child_index, term in enumerate(terms):
        parents = sorted(
            {
                term_to_index[parent]
                for parent in ontology.ont.get(term, {}).get("is_a", [])
                if parent in term_to_index and parent != term
            }
        )
        parent_counts[child_index] = len(parents)
        child_indices.extend([child_index] * len(parents))
        parent_indices.extend(parents)
    return (
        torch.tensor(child_indices, dtype=torch.long),
        torch.tensor(parent_indices, dtype=torch.long),
        parent_counts,
    )


def load_go_shuffled_parent_edges(
    ont: str,
    terms: Sequence[str],
    data_root: str | Path,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:

    from .metrics import Ontology

    ontology = Ontology(get_path("go_obo", data_root))
    term_to_index = {term: index for index, term in enumerate(terms)}
    roots = {"mf": "GO:0003674", "bp": "GO:0008150", "cc": "GO:0005575"}
    root = roots[ont]

    @lru_cache(maxsize=None)
    def depth(term: str) -> int:
        if term == root:
            return 0
        parents = ontology.ont.get(term, {}).get("is_a", [])
        return 0 if not parents else 1 + max(depth(parent) for parent in parents)

    term_depths = [depth(term) for term in terms]
    candidates_by_depth: Dict[int, List[int]] = {}
    for index, value in enumerate(term_depths):
        candidates_by_depth.setdefault(value, []).append(index)

    direct_parents: List[List[int]] = []
    ancestors: List[set[int]] = []
    descendants: List[set[int]] = [set() for _ in terms]
    for child_index, term in enumerate(terms):
        parents = sorted(
            {
                term_to_index[parent]
                for parent in ontology.ont.get(term, {}).get("is_a", [])
                if parent in term_to_index and parent != term
            }
        )
        direct_parents.append(parents)
        ancestor_indices = {
            term_to_index[value]
            for value in ontology.get_anchestors(term)
            if value in term_to_index
        }
        ancestors.append(ancestor_indices)
        for ancestor_index in ancestor_indices:
            descendants[ancestor_index].add(child_index)

    rng = random.Random(int(seed))
    child_indices: List[int] = []
    parent_indices: List[int] = []
    parent_counts = torch.zeros(len(terms), dtype=torch.long)
    nearest_depth_fallback_links = 0
    maximum_depth_shift = 0
    for child_index, parents in enumerate(direct_parents):
        forbidden = ancestors[child_index] | descendants[child_index] | {child_index}
        replacements: List[int] = []
        for parent_index in parents:
            pool = [
                candidate
                for candidate in candidates_by_depth[term_depths[parent_index]]
                if candidate not in forbidden
            ]
            if not pool:
                safe = [
                    candidate
                    for candidate in range(len(terms))
                    if candidate not in forbidden
                ]
                if not safe:
                    raise ValueError(
                        f"No topology-safe pseudo-parent for {terms[child_index]}"
                    )
                shift = min(
                    abs(term_depths[candidate] - term_depths[parent_index])
                    for candidate in safe
                )
                pool = [
                    candidate
                    for candidate in safe
                    if abs(term_depths[candidate] - term_depths[parent_index])
                    == shift
                ]
                nearest_depth_fallback_links += 1
                maximum_depth_shift = max(maximum_depth_shift, shift)
            replacements.append(rng.choice(pool))
        parent_counts[child_index] = len(replacements)
        child_indices.extend([child_index] * len(replacements))
        parent_indices.extend(replacements)

    digest = hashlib.sha256(
        ("|".join(map(str, child_indices)) + "#" + "|".join(map(str, parent_indices))).encode()
    ).hexdigest()
    return (
        torch.tensor(child_indices, dtype=torch.long),
        torch.tensor(parent_indices, dtype=torch.long),
        parent_counts,
        {
            "mapping": "topology_safe_depth_matched_pseudo_parent",
            "seed": int(seed),
            "sha256": digest,
            "nearest_depth_fallback_links": nearest_depth_fallback_links,
            "maximum_parent_depth_shift": maximum_depth_shift,
        },
    )


def load_go_depth_groups(
    ont: str,
    terms: Sequence[str],
    data_root: str | Path,
) -> Tuple[torch.Tensor, Dict[str, object]]:

    from .metrics import Ontology

    ontology = Ontology(get_path("go_obo", data_root))
    roots = {
        "mf": "GO:0003674",
        "bp": "GO:0008150",
        "cc": "GO:0005575",
    }
    root = roots[ont]

    @lru_cache(maxsize=None)
    def depth(term: str) -> int:
        if term == root:
            return 0
        parents = ontology.ont.get(term, {}).get("is_a", [])
        if not parents:
            return 0
        return 1 + max(depth(parent) for parent in parents)

    depths = torch.tensor([depth(term) for term in terms], dtype=torch.long)
    depth_groups = torch.zeros_like(depths)
    depth_groups[(depths >= 3) & (depths <= 4)] = 1
    depth_groups[(depths >= 5) & (depths <= 6)] = 2
    depth_groups[depths >= 7] = 3
    stats: Dict[str, object] = {
        "relations": "is_a_and_part_of_as_parsed_by_DPGOK_ontology",
        "terms": len(terms),
        "max_depth": int(depths.max()) if len(depths) else 0,
        "depth_group_counts": [
            int((depth_groups == index).sum()) for index in range(4)
        ],
        "text_semantics_loaded": False,
    }
    return depth_groups, stats
