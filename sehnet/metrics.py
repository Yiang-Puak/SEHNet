"""CAFA metrics matching the official DPGOK evaluation implementation.

Reference:
https://github.com/CSUBioGroup/DPGOK/blob/4f2716cecde42a8918995168509403b6aef7a4cd/src/evaluation.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Sequence, Set, Tuple

import numpy as np
import scipy.sparse as ssp


ROOT_GO_TERMS = {"GO:0003674", "GO:0008150", "GO:0005575"}


class Ontology:
    def __init__(self, filename: str | Path):
        self.ont = {}
        self._ancestor_index_cache = {}
        with Path(filename).open("r", encoding="utf-8", errors="ignore") as handle:
            obj = None
            for line in handle:
                line = line.strip()
                if line == "[Term]":
                    if obj is not None:
                        self.ont[obj["id"]] = obj
                    obj = {"is_a": [], "is_obsolete": False}
                elif line == "[Typedef]":
                    if obj is not None:
                        self.ont[obj["id"]] = obj
                    obj = None
                elif obj is not None and ": " in line:
                    key, value = line.split(": ", 1)
                    if key == "id":
                        obj["id"] = value
                    elif key == "is_a":
                        obj["is_a"].append(value.split(" ! ")[0])
                    elif key == "relationship":
                        parts = value.split()
                        if parts and parts[0] == "part_of":
                            obj["is_a"].append(parts[1])
                    elif key == "is_obsolete" and value == "true":
                        obj["is_obsolete"] = True
            if obj is not None:
                self.ont[obj["id"]] = obj
        for term_id in list(self.ont.keys()):
            if self.ont[term_id].get("is_obsolete", False):
                del self.ont[term_id]

    def get_anchestors(self, term_id: str) -> Set[str]:
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque([term_id])
        while q:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]["is_a"]:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set


def DAG_propagation(
    preds: np.ndarray,
    go_obj: Ontology,
    valid_terms: Sequence[str],
) -> np.ndarray:
    """Propagate each term score to all ancestors in one batch.

    This is mathematically identical to the original score-sorted protein loop:
    every output term receives the maximum score of all descendants. Ancestor
    indices are cached because the ontology and term order stay fixed.
    """
    # Official DPGOK converts prediction dictionaries back into a NumPy
    # matrix, which evaluates the stored float32 values in float64.
    preds = np.asarray(preds, dtype=np.float64)
    cache_key = tuple(valid_terms)
    ancestor_indices = go_obj._ancestor_index_cache.get(cache_key)
    if ancestor_indices is None:
        go_dict = {term: idx for idx, term in enumerate(valid_terms)}
        ancestor_indices = tuple(
            np.fromiter(
                (
                    go_dict[ancestor]
                    for ancestor in go_obj.get_anchestors(term)
                    if ancestor in go_dict
                ),
                dtype=np.int64,
            )
            for term in valid_terms
        )
        go_obj._ancestor_index_cache[cache_key] = ancestor_indices

    original = np.array(preds, copy=True)
    for term_idx, indices in enumerate(ancestor_indices):
        if indices.size:
            preds[:, indices] = np.maximum(
                preds[:, indices],
                original[:, term_idx, None],
            )
    return preds


def calculate_cafa_fmax_aupr(targets: np.ndarray, scores: np.ndarray, valid_terms: Sequence[str]) -> Tuple[float, float, float]:
    targets = np.array(targets, copy=True)
    scores = np.array(scores, copy=True)

    # DPGOK removes ontology roots from the evaluated GO namespace instead of
    # retaining zero-valued root columns.
    keep = np.fromiter(
        (term not in ROOT_GO_TERMS for term in valid_terms),
        dtype=bool,
        count=len(valid_terms),
    )
    targets = targets[:, keep]
    scores = scores[:, keep]

    targets_csr = ssp.csr_matrix(targets)
    fmax_, t_max = 0.0, 0.0
    precisions, recalls = [], []

    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets_csr).sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            protein_precision = np.asarray(
                correct / cut_sc.sum(axis=1)
            ).ravel()
            protein_recall = np.asarray(
                correct / targets_csr.sum(axis=1)
            ).ravel()
        valid_precision = protein_precision[~np.isnan(protein_precision)]
        p = (
            float(np.average(valid_precision))
            if valid_precision.size
            else float("nan")
        )
        r = float(np.average(protein_recall))

        precisions.append(0.0 if np.isnan(p) else p)
        recalls.append(r)

        if not np.isnan(p) and p + r > 0:
            f1 = 2 * p * r / (p + r)
            # Official code uses tuple max, so an exact F1 tie selects the
            # larger threshold.
            if f1 >= fmax_:
                fmax_ = f1
                t_max = cut

    precisions, recalls = np.array(precisions), np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls, precisions = recalls[sorted_index], precisions[sorted_index]
    trapezoid = getattr(np, "trapezoid", None)
    aupr_ = (
        trapezoid(precisions, recalls)
        if trapezoid is not None
        else np.trapz(precisions, recalls)
    )

    return float(fmax_), float(aupr_), float(t_max)
