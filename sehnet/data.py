from __future__ import annotations

import pickle as pkl
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import torch
from torch.utils import data

from .config import ONTOLOGIES, SPLITS, get_data_root, get_path


def read_pids(path: Path) -> List[str]:
    pids: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            pid = line.strip()
            if pid and pid.upper() != "PROTEINS":
                pids.append(pid)
    return pids


def read_pickle(path: Path):
    with path.open("rb") as handle:
        return pkl.load(handle)


def load_terms(label_dir: Path, ont: str) -> List[str]:
    path = label_dir / f"select_min_count_1_{ont}_labels.csv"
    frame = pd.read_csv(path, usecols=["functions"])
    terms = frame["functions"].astype(str).tolist()
    if not terms:
        raise ValueError(f"No GO terms found in {path}")
    if len(set(terms)) != len(terms):
        raise ValueError(f"Duplicate GO terms found in {path}")
    return terms


def load_labels_for_pids(
    label_csv: Path,
    target_pids: Sequence[str],
    *,
    terms: Sequence[str],
    chunksize: int = 512,
) -> Dict[str, np.ndarray]:
    target_set: Set[str] = set(target_pids)
    labels: Dict[str, np.ndarray] = {}
    if not target_set:
        return labels

    for chunk in pd.read_csv(label_csv, chunksize=chunksize):
        if "proteins" not in chunk.columns:
            raise ValueError(f"Missing 'proteins' column in {label_csv}")
        label_columns = [str(column) for column in chunk.columns[1:]]
        if label_columns != list(terms):
            raise ValueError(
                f"GO term order mismatch between {label_csv} and the "
                "selected-term file"
            )
        selected = chunk[chunk["proteins"].isin(target_set)]
        if selected.empty:
            continue
        values = selected.iloc[:, 1:].to_numpy(dtype=np.uint8, copy=True)
        for pid, row in zip(selected["proteins"].astype(str), values):
            labels[pid] = row
        if len(labels) == len(target_set):
            break
    return labels


class SEHNetDataset(data.Dataset):

    def __init__(
        self,
        ont: str,
        split: str,
        data_root: Optional[str | Path] = None,
        max_samples: Optional[int] = None,
        sample_seed: Optional[int] = None,
        label_chunksize: int = 512,
        terms: Optional[Sequence[str]] = None,
        shared_labels: Optional[Dict[str, np.ndarray]] = None,
        local_segments_150m: bool = False,
        local_multilayer_segments_150m: bool = False,
    ) -> None:
        if ont not in ONTOLOGIES:
            raise ValueError(f"Unsupported ontology: {ont}")
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")

        self.ont = ont
        self.split = split
        self.data_root = get_data_root(data_root)
        self.label_dir = get_path("label_dir", self.data_root)
        self.feature_dir = get_path("feature_dir", self.data_root)
        self.split_dir = get_path("split_dir", self.data_root)
        self.local_segments_150m = bool(local_segments_150m)
        self.local_multilayer_segments_150m = bool(
            local_multilayer_segments_150m
        )
        if (
            self.local_segments_150m
            and self.local_multilayer_segments_150m
        ):
            raise ValueError(
                "Single-layer and multi-layer local features are mutually "
                "exclusive"
            )
        self.terms = (
            list(terms)
            if terms is not None
            else load_terms(self.label_dir, ont)
        )
        self.class_nums = len(self.terms)
        split_path = self.split_dir / f"{ont}_{split}_pid_list.txt"
        split_pids = read_pids(split_path)
        feature_path = self.feature_dir / f"{ont}_{split}_esm2.pkl"
        self.esm_features: Dict[str, object] = read_pickle(feature_path)
        missing_features = [
            pid for pid in split_pids if pid not in self.esm_features
        ]
        if missing_features:
            raise ValueError(
                f"Missing ESM2-3B features for {len(missing_features)} "
                f"{ont}-{split} proteins; sample: {missing_features[:10]}"
            )
        self.missing_feature_pids = []
        proteins = list(split_pids)
        if max_samples is not None and max_samples < len(proteins):
            if sample_seed is None:
                proteins = proteins[:max_samples]
            else:
                generator = np.random.default_rng(sample_seed)
                indices = np.sort(
                    generator.choice(
                        len(proteins),
                        size=max_samples,
                        replace=False,
                    )
                )
                proteins = [proteins[int(index)] for index in indices]
        if not proteins:
            raise ValueError(f"No usable proteins found for {ont}-{split}")
        label_path = self.label_dir / f"{ont}_data_separate_{ont}_labels.csv"
        self.labels = (
            shared_labels
            if shared_labels is not None
            else load_labels_for_pids(
                label_path,
                proteins,
                terms=self.terms,
                chunksize=label_chunksize,
            )
        )
        missing = [pid for pid in proteins if pid not in self.labels]
        if missing:
            raise ValueError(
                f"Missing labels for {len(missing)} {ont}-{split} proteins; "
                f"sample: {missing[:10]}"
            )

        self.proteins = proteins
        self.local_segment_features = None
        self.local_global_features = None
        self.local_segment_mask = None
        self.local_pid_to_row: Dict[str, int] = {}
        if (
            self.local_segments_150m
            or self.local_multilayer_segments_150m
        ):
            local_dir = (
                self.data_root
                / (
                    "features_150M_multilayer_segments"
                    if self.local_multilayer_segments_150m
                    else "features_150M_segments"
                )
                / self.ont
            )
            local_pids = read_pids(local_dir / f"{split}_pids.txt")
            self.local_pid_to_row = {
                pid: index for index, pid in enumerate(local_pids)
            }
            missing_local = [
                pid for pid in self.proteins if pid not in self.local_pid_to_row
            ]
            if missing_local:
                raise ValueError(
                    f"Missing 150M segment rows for {len(missing_local)} "
                    f"{ont}-{split} proteins; sample: {missing_local[:10]}"
                )
            self.local_segment_features = np.load(
                local_dir / f"{split}_segments.npy",
                mmap_mode="r",
            )
            self.local_global_features = np.load(
                local_dir / f"{split}_global.npy",
                mmap_mode="r",
            )
            self.local_segment_mask = np.load(
                local_dir / f"{split}_segment_mask.npy",
                mmap_mode="r",
            )
            completed = np.load(
                local_dir / f"{split}_completed.npy",
                mmap_mode="r",
            )
            expected_segment_shape = (
                (len(local_pids), 3, 32, 640)
                if self.local_multilayer_segments_150m
                else (len(local_pids), 32, 640)
            )
            expected_global_shape = (
                (len(local_pids), 3, 640)
                if self.local_multilayer_segments_150m
                else (len(local_pids), 640)
            )
            if (
                self.local_segment_features.shape != expected_segment_shape
                or self.local_global_features.shape != expected_global_shape
                or self.local_segment_mask.shape != (len(local_pids), 32)
                or completed.shape != (len(local_pids),)
            ):
                raise ValueError(
                    f"Invalid 150M segment feature shapes in {local_dir}"
                )
            incomplete = [
                pid
                for pid in self.proteins
                if not bool(completed[self.local_pid_to_row[pid]])
            ]
            if incomplete:
                raise ValueError(
                    f"Incomplete 150M features for {len(incomplete)} "
                    f"{ont}-{split} proteins; sample: {incomplete[:10]}"
                )
        self.label_counts = np.zeros(self.class_nums, dtype=np.int64)
        for pid in self.proteins:
            self.label_counts += self.labels[pid].astype(np.int64, copy=False)
        self.label_prevalence = (
            (self.label_counts.astype(np.float64) + 1.0)
            / (len(self.proteins) + 2.0)
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.proteins)

    def __getitem__(self, index: int) -> Dict[str, object]:
        pid = self.proteins[index]
        value = self.esm_features[pid]
        if isinstance(value, torch.Tensor):
            esm = value.detach().to(dtype=torch.float32).reshape(-1)
        else:
            esm = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if esm.numel() != 2560:
            raise ValueError(
                f"Expected a 2560-dimensional ESM feature for {pid}, "
                f"found {esm.numel()}"
            )
        item = {
            "pid": pid,
            "esm": esm,
            "label": torch.as_tensor(self.labels[pid], dtype=torch.float32),
        }
        if (
            self.local_segments_150m
            or self.local_multilayer_segments_150m
        ):
            if (
                self.local_segment_features is None
                or self.local_global_features is None
                or self.local_segment_mask is None
            ):
                raise RuntimeError(
                    "150M segment feature arrays are not initialised"
                )
            local_row = self.local_pid_to_row[pid]
            local_mask = np.asarray(
                self.local_segment_mask[local_row],
                dtype=np.bool_,
            )
            local_segments = np.asarray(
                self.local_segment_features[local_row],
                dtype=np.float32,
            ).copy()
            local_global = np.asarray(
                self.local_global_features[local_row],
                dtype=np.float32,
            )
            if self.local_multilayer_segments_150m:
                local_segments[:, local_mask] -= local_global[:, None, :]
                local_segments[:, ~local_mask] = 0.0
            else:
                local_segments[local_mask] -= local_global[None, :]
                local_segments[~local_mask] = 0.0
            item["local_segments"] = torch.from_numpy(local_segments)
            item["local_mask"] = torch.from_numpy(local_mask.copy())
        return item


def sehnet_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    collated = {
        "pid": [str(item["pid"]) for item in batch],
        "esm": torch.stack([item["esm"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }
    if "local_segments" in batch[0]:
        collated["local_segments"] = torch.stack(
            [item["local_segments"] for item in batch]
        )
        collated["local_mask"] = torch.stack(
            [item["local_mask"] for item in batch]
        )
    return collated
