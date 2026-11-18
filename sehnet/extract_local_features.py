"""Extract reproducible whole-protein ESM-2 global and segment features.

The extractor keeps every residue by splitting long proteins into non-overlapping
windows. Residue representations are pooled into fixed whole-protein coordinate
bins, so downstream training can use local evidence without loading the PLM.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np
import pandas as pd
import torch

from .config import get_data_root, get_path
from .data import read_pids

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


MODEL_NAME = "esm2_t30_150M_UR50D"
DEFAULT_MODEL_LAYERS = (10, 20, 30)
MODEL_DIM = 640


@dataclass(frozen=True)
class WindowTask:
    row: int
    pid: str
    protein_length: int
    start: int
    sequence: str
    is_last: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ont", default="mf", choices=("mf", "cc", "bp"))
    parser.add_argument("--data_root", type=Path, default=get_data_root())
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Default: <data_root>/features_150M_segments",
    )
    parser.add_argument("--splits", nargs="+", default=("train", "valid", "test"))
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument(
        "--repr_layers",
        type=int,
        nargs="+",
        default=DEFAULT_MODEL_LAYERS,
        help=(
            "ESM representation layers to save. Multiple layers are stored "
            "with shape (protein,layer,segment,dimension)."
        ),
    )
    parser.add_argument("--window_length", type=int, default=512)
    parser.add_argument("--token_budget", type=int, default=1536)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--flush_every", type=int, default=500)
    parser.add_argument(
        "--max_proteins",
        type=int,
        default=None,
        help="Smoke-test only; use all proteins when omitted.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch_hub_dir",
        type=Path,
        default=None,
        help="Default: <data_root>/model_cache/torch_hub",
    )
    return parser.parse_args()


def load_target_pids(data_root: Path, ont: str, split: str) -> List[str]:
    feature_dir = get_path("feature_dir", data_root)
    split_dir = get_path("split_dir", data_root)
    split_pids = read_pids(split_dir / f"{ont}_{split}_pid_list.txt")
    feature_path = feature_dir / f"{ont}_{split}_esm2.pkl"
    with feature_path.open("rb") as handle:
        existing_features = pickle.load(handle)
    usable = [pid for pid in split_pids if pid in existing_features]
    del existing_features
    return usable


def load_sequences(sequence_tsv: Path, target_pids: Sequence[str]) -> Dict[str, str]:
    target_set = set(target_pids)
    frame = pd.read_csv(
        sequence_tsv,
        sep="\t",
        usecols=["Entry", "Entry Name", "Sequence"],
        dtype=str,
    )
    sequences: Dict[str, str] = {}
    for entry, entry_name, sequence in frame.itertuples(index=False, name=None):
        sequence = str(sequence).strip().upper()
        if entry_name in target_set:
            sequences[entry_name] = sequence
        if entry in target_set:
            sequences[entry] = sequence
    return sequences


def iter_window_tasks(
    pids: Sequence[str],
    sequences: Dict[str, str],
    completed: np.ndarray,
    window_length: int,
) -> Iterator[WindowTask]:
    for row, pid in enumerate(pids):
        if bool(completed[row]):
            continue
        sequence = sequences.get(pid, "")
        if not sequence:
            yield WindowTask(row, pid, 0, 0, "", True)
            continue
        length = len(sequence)
        for start in range(0, length, window_length):
            end = min(start + window_length, length)
            yield WindowTask(
                row=row,
                pid=pid,
                protein_length=length,
                start=start,
                sequence=sequence[start:end],
                is_last=end == length,
            )


def pack_batches(
    tasks: Iterable[WindowTask],
    token_budget: int,
    max_batch_size: int,
) -> Iterator[List[WindowTask]]:
    batch: List[WindowTask] = []
    max_length = 0
    for task in tasks:
        if not task.sequence:
            if batch:
                yield batch
                batch = []
                max_length = 0
            yield [task]
            continue
        proposed_max = max(max_length, len(task.sequence) + 2)
        proposed_size = len(batch) + 1
        exceeds_budget = proposed_max * proposed_size > token_budget
        if batch and (exceeds_budget or proposed_size > max_batch_size):
            yield batch
            batch = []
            max_length = 0
        batch.append(task)
        max_length = max(max_length, len(task.sequence) + 2)
    if batch:
        yield batch


def open_or_create_memmap(
    path: Path,
    *,
    shape: Sequence[int],
    dtype: np.dtype | str,
) -> np.memmap:
    if path.exists():
        array = np.lib.format.open_memmap(path, mode="r+")
        if tuple(array.shape) != tuple(shape) or array.dtype != np.dtype(dtype):
            raise ValueError(
                f"Existing array mismatch for {path}: "
                f"shape={array.shape}, dtype={array.dtype}"
            )
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def finalise_row(
    row: int,
    sums: np.ndarray,
    counts: np.ndarray,
    segments_out: np.memmap,
    masks_out: np.memmap,
    global_out: np.memmap,
    completed_out: np.memmap,
) -> None:
    valid = counts > 0
    pooled = np.zeros_like(sums, dtype=np.float32)
    pooled[:, valid] = sums[:, valid] / counts[None, valid, None]
    if segments_out.ndim == 3:
        segments_out[row] = pooled[0].astype(np.float16)
    else:
        segments_out[row] = pooled.astype(np.float16)
    masks_out[row] = valid
    total_count = int(counts.sum())
    if total_count:
        global_values = (sums.sum(axis=1) / total_count).astype(np.float16)
        global_out[row] = (
            global_values[0] if global_out.ndim == 2 else global_values
        )
    else:
        global_out[row] = 0
    completed_out[row] = True


def flush_arrays(arrays: Sequence[np.memmap]) -> None:
    for array in arrays:
        array.flush()


def extract_split(
    *,
    split: str,
    pids: Sequence[str],
    sequences: Dict[str, str],
    output_dir: Path,
    model: torch.nn.Module,
    alphabet,
    device: torch.device,
    repr_layers: Sequence[int],
    segments: int,
    window_length: int,
    token_budget: int,
    max_batch_size: int,
    flush_every: int,
) -> dict:
    prefix = output_dir / f"{split}"
    num_layers = len(repr_layers)
    shape = (
        (len(pids), segments, MODEL_DIM)
        if num_layers == 1
        else (len(pids), num_layers, segments, MODEL_DIM)
    )
    segments_out = open_or_create_memmap(
        prefix.with_name(prefix.name + "_segments.npy"),
        shape=shape,
        dtype=np.float16,
    )
    masks_out = open_or_create_memmap(
        prefix.with_name(prefix.name + "_segment_mask.npy"),
        shape=(len(pids), segments),
        dtype=np.bool_,
    )
    global_out = open_or_create_memmap(
        prefix.with_name(prefix.name + "_global.npy"),
        shape=(
            (len(pids), MODEL_DIM)
            if num_layers == 1
            else (len(pids), num_layers, MODEL_DIM)
        ),
        dtype=np.float16,
    )
    completed_out = open_or_create_memmap(
        prefix.with_name(prefix.name + "_completed.npy"),
        shape=(len(pids),),
        dtype=np.bool_,
    )
    pid_path = prefix.with_name(prefix.name + "_pids.txt")
    expected_pid_text = "".join(f"{pid}\n" for pid in pids)
    if pid_path.exists() and pid_path.read_text(encoding="utf-8") != expected_pid_text:
        raise ValueError(f"PID order mismatch in {pid_path}")
    if not pid_path.exists():
        pid_path.write_text(expected_pid_text, encoding="utf-8")

    batch_converter = alphabet.get_batch_converter()
    accum_sums: Dict[int, np.ndarray] = {}
    accum_counts: Dict[int, np.ndarray] = {}
    already_completed = int(np.count_nonzero(completed_out))
    newly_completed = 0
    missing_sequences = 0
    windows = 0
    peak_vram = 0.0
    started = time.perf_counter()

    tasks = iter_window_tasks(pids, sequences, completed_out, window_length)
    progress = (
        tqdm(
            total=len(pids),
            initial=already_completed,
            desc=f"{split} ESM2-150M extraction",
            unit="protein",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )
    try:
        for batch in pack_batches(tasks, token_budget, max_batch_size):
            if not batch[0].sequence:
                task = batch[0]
                finalise_row(
                    task.row,
                    np.zeros(
                        (num_layers, segments, MODEL_DIM),
                        dtype=np.float32,
                    ),
                    np.zeros(segments, dtype=np.int64),
                    segments_out,
                    masks_out,
                    global_out,
                    completed_out,
                )
                missing_sequences += 1
                newly_completed += 1
                if progress is not None:
                    progress.update(1)
                continue

            converted = [(task.pid, task.sequence) for task in batch]
            _, _, tokens = batch_converter(converted)
            tokens = tokens.to(device, non_blocking=True)
            with torch.inference_mode():
                representations = model(
                    tokens,
                    repr_layers=list(repr_layers),
                    return_contacts=False,
                )["representations"]
            peak_vram = max(
                peak_vram,
                torch.cuda.max_memory_allocated(device) / (1024.0**2)
                if device.type == "cuda"
                else 0.0,
            )

            for batch_index, task in enumerate(batch):
                positions = torch.arange(
                    task.start,
                    task.start + len(task.sequence),
                    device=device,
                    dtype=torch.long,
                )
                segment_index = torch.div(
                    positions * segments,
                    task.protein_length,
                    rounding_mode="floor",
                ).clamp_max(segments - 1)
                chunk_sums = []
                for layer in repr_layers:
                    residue_repr = representations[layer][
                        batch_index, 1 : len(task.sequence) + 1
                    ].float()
                    layer_sums = torch.zeros(
                        (segments, MODEL_DIM),
                        device=device,
                        dtype=torch.float32,
                    )
                    layer_sums.index_add_(0, segment_index, residue_repr)
                    chunk_sums.append(layer_sums)
                chunk_sums_tensor = torch.stack(chunk_sums, dim=0)
                chunk_counts = torch.bincount(
                    segment_index, minlength=segments
                ).to(torch.int64)

                if task.row not in accum_sums:
                    accum_sums[task.row] = np.zeros(
                        (num_layers, segments, MODEL_DIM),
                        dtype=np.float32,
                    )
                    accum_counts[task.row] = np.zeros(segments, dtype=np.int64)
                accum_sums[task.row] += chunk_sums_tensor.cpu().numpy()
                accum_counts[task.row] += chunk_counts.cpu().numpy()
                windows += 1

                if task.is_last:
                    finalise_row(
                        task.row,
                        accum_sums.pop(task.row),
                        accum_counts.pop(task.row),
                        segments_out,
                        masks_out,
                        global_out,
                        completed_out,
                    )
                    newly_completed += 1
                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(
                            windows=windows,
                            vram_mib=f"{peak_vram:.0f}",
                        )
                    if newly_completed % flush_every == 0:
                        flush_arrays(
                            (segments_out, masks_out, global_out, completed_out)
                        )
                        elapsed = time.perf_counter() - started
                        rate = newly_completed / max(elapsed, 1e-9)
                        print(
                            f"{split}: {already_completed + newly_completed}/"
                            f"{len(pids)} proteins, {windows} windows, "
                            f"{rate:.2f} proteins/s, peak {peak_vram:.1f} MiB",
                            flush=True,
                        )
            del representations, tokens
    finally:
        if progress is not None:
            progress.close()

    if accum_sums:
        raise RuntimeError(f"Unfinished protein accumulators: {len(accum_sums)}")
    flush_arrays((segments_out, masks_out, global_out, completed_out))
    elapsed = time.perf_counter() - started
    summary = {
        "split": split,
        "proteins": len(pids),
        "already_completed": already_completed,
        "newly_completed": newly_completed,
        "missing_sequences": missing_sequences,
        "windows": windows,
        "elapsed_seconds": elapsed,
        "proteins_per_second": newly_completed / max(elapsed, 1e-9),
        "peak_vram_mib": peak_vram,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    repr_layers = tuple(dict.fromkeys(int(layer) for layer in args.repr_layers))
    if not repr_layers or any(layer < 0 or layer > 30 for layer in repr_layers):
        raise ValueError("repr_layers must contain unique values in [0, 30]")
    data_root = args.data_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            data_root
            / (
                "features_150M_segments"
                if len(repr_layers) == 1
                else "features_150M_multilayer_segments"
            )
            / args.ont
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_hub_dir = (
        args.torch_hub_dir.resolve()
        if args.torch_hub_dir is not None
        else data_root / "model_cache" / "torch_hub"
    )
    torch_hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_hub_dir))

    split_pids: Dict[str, List[str]] = {}
    all_pids: List[str] = []
    for split in args.splits:
        pids = load_target_pids(data_root, args.ont, split)
        if args.max_proteins is not None:
            pids = pids[: args.max_proteins]
        split_pids[split] = pids
        all_pids.extend(pids)

    sequence_tsv = (
        data_root / "sequence_features" / "uniprot_sequences.tsv"
    )
    sequences = load_sequences(sequence_tsv, list(dict.fromkeys(all_pids)))
    print(
        f"Loaded {len(sequences)}/{len(set(all_pids))} requested sequences",
        flush=True,
    )

    import esm

    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    device = torch.device(args.device)
    model = model.eval().to(device)
    if device.type == "cuda":
        model = model.half()
        torch.cuda.reset_peak_memory_stats(device)

    summaries = []
    for split in args.splits:
        summaries.append(
            extract_split(
                split=split,
                pids=split_pids[split],
                sequences=sequences,
                output_dir=output_dir,
                model=model,
                alphabet=alphabet,
                device=device,
                repr_layers=repr_layers,
                segments=args.segments,
                window_length=args.window_length,
                token_budget=args.token_budget,
                max_batch_size=args.max_batch_size,
                flush_every=args.flush_every,
            )
        )

    metadata_path = output_dir / "metadata.json"
    existing_summaries = {}
    if metadata_path.exists():
        existing_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        existing_summaries = {
            str(summary["split"]): summary
            for summary in existing_metadata.get("splits", [])
        }
    existing_summaries.update(
        {str(summary["split"]): summary for summary in summaries}
    )
    metadata = {
        "ontology": args.ont,
        "model": MODEL_NAME,
        "layers": list(repr_layers),
        "embedding_dim": MODEL_DIM,
        "layer_axis": len(repr_layers) > 1,
        "segments": args.segments,
        "window_length": args.window_length,
        "token_budget": args.token_budget,
        "max_batch_size": args.max_batch_size,
        "full_sequence_coverage": True,
        "window_overlap": 0,
        "pooling": "whole-protein-coordinate segment mean",
        "dtype": "float16",
        "fair_esm_version": "2.0.0",
        "splits": [
            existing_summaries[split]
            for split in ("train", "valid", "test")
            if split in existing_summaries
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
