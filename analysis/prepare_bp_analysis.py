from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FREQUENCY_BINS = (
    ("1-4", 1, 4),
    ("5-19", 5, 19),
    ("20-99", 20, 99),
    ("100+", 100, None),
)


def read_pids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip() and line.strip().upper() != "PROTEINS"]


def load_terms(label_dir: Path) -> list[str]:
    terms = pd.read_csv(label_dir / "select_min_count_1_bp_labels.csv", usecols=["functions"])["functions"].astype(str).tolist()
    if not terms or len(terms) != len(set(terms)):
        raise ValueError("BP term list is empty or duplicated")
    return terms


def data_path(data_root: Path, name: str) -> Path:
    paths = {
        "label_dir": data_root / "labels",
        "split_dir": data_root / "real_dataset" / "splits",
        "go_obo": data_root / "go.obo",
    }
    return paths[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare BP term strata and checkpoint metadata.\n\nThis script performs no training and no prediction.  It fixes the selected\ncheckpoint set, derives GO-depth and training-frequency strata exclusively\nfrom the supplied training labels, and records test-label coverage needed by\nthe later per-term error table and local-segment occlusion analysis.\n')
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def depth_map(go_obo: Path, terms: list[str]) -> tuple[np.ndarray, list[str], list[int]]:
    ontology: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None
    with go_obo.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line == "[Term]":
                if current and "id" in current and not current.get("is_obsolete", False):
                    ontology[str(current["id"])] = current
                current = {"is_a": [], "is_obsolete": False}
            elif line == "[Typedef]":
                if current and "id" in current and not current.get("is_obsolete", False):
                    ontology[str(current["id"])] = current
                current = None
            elif current is not None and ": " in line:
                key, value = line.split(": ", 1)
                if key == "id":
                    current["id"] = value
                elif key == "is_a":
                    current["is_a"].append(value.split(" ! ")[0])
                elif key == "relationship" and value.split() and value.split()[0] == "part_of":
                    current["is_a"].append(value.split()[1])
                elif key == "is_obsolete" and value == "true":
                    current["is_obsolete"] = True
    if current and "id" in current and not current.get("is_obsolete", False):
        ontology[str(current["id"])] = current
    root = "GO:0008150"
    memo: dict[str, int] = {}

    def depth(term: str, visiting: set[str] | None = None) -> int:
        if term == root:
            return 0
        if term in memo:
            return memo[term]
        visiting = set() if visiting is None else visiting
        if term in visiting:
            return 0
        parents = ontology.get(term, {}).get("is_a", [])
        if not parents:
            memo[term] = 0
        else:
            visiting.add(term)
            memo[term] = 1 + max(depth(parent, visiting) for parent in parents)
            visiting.remove(term)
        return memo[term]

    values = np.asarray([depth(term) for term in terms], dtype=np.int16)
    bins = ["1-2" if value <= 2 else "3-4" if value <= 4 else "5-6" if value <= 6 else "7+" for value in values]
    parent_counts = [len(ontology.get(term, {}).get("is_a", [])) for term in terms]
    return values, bins, parent_counts


def frequency_bin(value: int) -> str:
    for name, low, high in FREQUENCY_BINS:
        if value >= low and (high is None or value <= high):
            return name
    return "0"


def label_counts(label_path: Path, terms: list[str], train_pids: set[str], test_pids: set[str]) -> tuple[np.ndarray, np.ndarray, int]:
    train_counts = np.zeros(len(terms), dtype=np.int64)
    test_counts = np.zeros(len(terms), dtype=np.int64)
    chunks = 0
    for chunk in pd.read_csv(label_path, chunksize=512):
        columns = [str(value) for value in chunk.columns[1:]]
        if columns != terms:
            raise ValueError(f"GO term order mismatch in {label_path}")
        proteins = chunk["proteins"].astype(str)
        train = chunk.loc[proteins.isin(train_pids)]
        test = chunk.loc[proteins.isin(test_pids)]
        if not train.empty:
            train_counts += train.iloc[:, 1:].to_numpy(dtype=np.uint8).sum(axis=0, dtype=np.int64)
        if not test.empty:
            test_counts += test.iloc[:, 1:].to_numpy(dtype=np.uint8).sum(axis=0, dtype=np.int64)
        chunks += 1
        if chunks % 10 == 0:
            print(f"[labels] processed {chunks} chunks", flush=True)
    return train_counts, test_counts, chunks


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("ont") != "bp":
        raise ValueError("This script is defined only for a BP run")
    final_epoch = int(config["epochs"])
    if final_epoch < 5:
        raise ValueError("At least five training epochs are required")
    epochs = list(range(final_epoch - 4, final_epoch + 1))
    checkpoint_dir = run_dir / "trajectory_checkpoints"
    checkpoint_paths = [checkpoint_dir / f"epoch_{epoch:03d}.pth" for epoch in epochs]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing selected checkpoints: " + ", ".join(missing))
    output_dir = (args.output_dir or run_dir / "bp_analysis").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_dir = data_path(data_root, "label_dir")
    split_dir = data_path(data_root, "split_dir")
    terms = load_terms(label_dir)
    train_pids = set(read_pids(split_dir / "bp_train_pid_list.txt"))
    test_pids = set(read_pids(split_dir / "bp_test_pid_list.txt"))
    print(f"[setup] terms={len(terms)} train={len(train_pids)} test={len(test_pids)}", flush=True)
    train_counts, test_counts, chunks = label_counts(
        label_dir / "bp_data_separate_bp_labels.csv", terms, train_pids, test_pids
    )
    depths, depth_bins, parent_counts = depth_map(data_path(data_root, "go_obo"), terms)

    term_rows: list[dict[str, object]] = []
    for index, term in enumerate(terms):
        term_rows.append({
            "term": term,
            "go_depth": int(depths[index]),
            "depth_bin": depth_bins[index],
            "train_positive_count": int(train_counts[index]),
            "frequency_bin": frequency_bin(int(train_counts[index])),
            "test_positive_count": int(test_counts[index]),
            "direct_parent_count": int(parent_counts[index]),
            "eligible_for_term_ap": bool(test_counts[index] > 0),
            "eligible_for_occlusion_target": bool(test_counts[index] > 0 and train_counts[index] >= 5 and parent_counts[index] > 0),
        })
    write_csv(output_dir / "term_group_manifest.csv", term_rows)

    group_rows: list[dict[str, object]] = []
    for depth_bin in ("1-2", "3-4", "5-6", "7+"):
        for freq_bin, _, _ in FREQUENCY_BINS:
            indices = [i for i, row in enumerate(term_rows) if row["depth_bin"] == depth_bin and row["frequency_bin"] == freq_bin]
            if not indices:
                continue
            group_rows.append({
                "depth_bin": depth_bin,
                "frequency_bin": freq_bin,
                "term_count": len(indices),
                "terms_with_test_positive": int(sum(test_counts[i] > 0 for i in indices)),
                "train_positive_total": int(train_counts[indices].sum()),
                "test_positive_total": int(test_counts[indices].sum()),
                "occlusion_eligible_terms": int(sum(term_rows[i]["eligible_for_occlusion_target"] for i in indices)),
            })
    write_csv(output_dir / "depth_frequency_group_manifest.csv", group_rows)

    candidate_rows = [row for row in term_rows if row["eligible_for_occlusion_target"]]
    candidate_rows.sort(key=lambda row: (row["depth_bin"], row["frequency_bin"], -int(row["test_positive_count"]), row["term"]))
    write_csv(output_dir / "occlusion_eligible_term_pool.csv", candidate_rows)

    checkpoint_manifest = {
        "run_dir": str(run_dir),
        "ontology": "bp",
        "selected_epochs": epochs,
        "checkpoints": [{"epoch": epoch, "path": str(path), "sha256": sha256(path)} for epoch, path in zip(epochs, checkpoint_paths)],
        "raw_probability_rule": "arithmetic mean over the final five training checkpoints before one GO DAG propagation",
    }
    (output_dir / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[complete] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
