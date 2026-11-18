"""Path configuration for the sequence-only SEHNet project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("PROTEIN_FUNCTION_DATA", PACKAGE_ROOT / "data")
)

ONTOLOGIES = ("mf", "bp", "cc")
SPLITS = ("train", "valid", "test")


def _directory_or_flat_root(root: Path, directory: str, probe: str) -> Path:
    candidate = root / directory
    if candidate.is_dir() or not (root / probe).is_file():
        return candidate
    return root


def get_data_root(data_root: Optional[str | Path] = None) -> Path:
    value = data_root if data_root is not None else os.environ.get(
        "PROTEIN_FUNCTION_DATA", str(DEFAULT_DATA_ROOT)
    )
    return Path(value).expanduser().resolve(strict=False)


def get_path(name: str, data_root: Optional[str | Path] = None) -> Path:
    root = get_data_root(data_root)
    mapping = {
        "data_root": root,
        "go_obo": root / "go.obo",
        "label_dir": _directory_or_flat_root(
            root,
            "labels",
            "select_min_count_1_mf_labels.csv",
        ),
        "go_semantics_dir": _directory_or_flat_root(
            root,
            "go_semantics",
            "mf_go_semantics_768.pkl",
        ),
        "feature_dir": root / "features_3B",
        "split_dir": root / "real_dataset" / "splits",
    }
    if name not in mapping:
        raise KeyError(f"Unknown SEHNet path key: {name}")
    return mapping[name]
