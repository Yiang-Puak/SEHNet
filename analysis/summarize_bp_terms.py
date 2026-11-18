from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize BP SEHNet term-wise errors from a saved five-checkpoint score bundle.')
    parser.add_argument("--scores", type=Path, required=True, help="last_five_test_scores.pkl")
    parser.add_argument("--term-manifest", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Fmax threshold from matching summary JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = y[order].astype(np.int64, copy=False)
    ranks = np.arange(1, ranked.size + 1, dtype=np.float64)
    return float((np.cumsum(ranked) / ranks * ranked).sum() / positives)


def main() -> int:
    args = parse_args()
    with args.scores.open("rb") as handle:
        bundle = pickle.load(handle)
    terms = [str(term) for term in bundle["terms"]]
    scores = np.asarray(bundle["scores"], dtype=np.float32)
    labels = np.asarray(bundle["labels"], dtype=np.uint8)
    if scores.shape != labels.shape or scores.shape[1] != len(terms):
        raise ValueError("Score, label, and term dimensions do not agree")
    manifest = read_csv(args.term_manifest)
    by_term = {row["term"]: row for row in manifest}
    if list(by_term) != terms:
        raise ValueError("Term order/content mismatch between score bundle and manifest")
    predicted = scores >= args.threshold
    rows: list[dict[str, object]] = []
    for index, term in enumerate(terms):
        y = labels[:, index]
        pred = predicted[:, index]
        tp = int(np.logical_and(pred, y == 1).sum())
        fp = int(np.logical_and(pred, y == 0).sum())
        fn = int(np.logical_and(~pred, y == 1).sum())
        meta = by_term[term]
        rows.append({
            "term": term,
            "go_depth": meta["go_depth"],
            "depth_bin": meta["depth_bin"],
            "train_positive_count": meta["train_positive_count"],
            "frequency_bin": meta["frequency_bin"],
            "test_positive_count": int(y.sum()),
            "term_ap": average_precision(y, scores[:, index]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "fnr": float(fn / (tp + fn)) if tp + fn else float("nan"),
            "fdr": float(fp / (tp + fp)) if tp + fp else float("nan"),
            "predicted_positive_count": int(pred.sum()),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "term_level_error_table.csv", rows)
    group_rows: list[dict[str, object]] = []
    for depth in ("1-2", "3-4", "5-6", "7+"):
        for frequency in ("1-4", "5-19", "20-99", "100+"):
            group = [row for row in rows if row["depth_bin"] == depth and row["frequency_bin"] == frequency]
            if not group:
                continue
            valid_ap = [float(row["term_ap"]) for row in group if np.isfinite(float(row["term_ap"]))]
            tp, fp, fn = (sum(int(row[key]) for row in group) for key in ("tp", "fp", "fn"))
            group_rows.append({
                "depth_bin": depth,
                "frequency_bin": frequency,
                "term_count": len(group),
                "evaluated_term_count": len(valid_ap),
                "macro_term_ap": float(np.mean(valid_ap)) if valid_ap else float("nan"),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "micro_fnr": float(fn / (tp + fn)) if tp + fn else float("nan"),
                "micro_fdr": float(fp / (tp + fp)) if tp + fp else float("nan"),
            })
    write_csv(args.output_dir / "depth_frequency_error_table.csv", group_rows)
    def paper_value(value: object) -> str:
        number = float(value)
        return "—" if not np.isfinite(number) else f"{number:.3f}"

    paper_rows = [
        {
            "GO_depth": row["depth_bin"],
            "training_label_frequency": row["frequency_bin"],
            "macro_term_AP": paper_value(row["macro_term_ap"]),
            "micro_FNR": paper_value(row["micro_fnr"]),
            "micro_FDR": paper_value(row["micro_fdr"]),
        }
        for row in group_rows
    ]
    write_csv(args.output_dir / "paper_depth_frequency_error_table.csv", paper_rows)
    print(f"[complete] terms={len(rows)} groups={len(group_rows)} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
