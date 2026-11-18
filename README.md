# SEHNet

Reference code for term-conditioned local evidence and parent-contrast decoding in protein GO prediction.

## Contents

- `sehnet/`: model, local feature extraction, data loading, training and evaluation.
- `analysis/`: BP depth/frequency summaries, local-slot occlusion and attribution controls.
- `configs/`: portable MF, CC and BP training configurations.

This release contains code and configuration files.

## Setup and data

Use Python 3.10 or later. Install PyTorch for your hardware, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Obtain the original [DPGOK Temporal Hold-out benchmark](https://github.com/CSUBioGroup/DPGOK), retaining its protein partitions, GO labels and ontology version. Supply the global ESM2-3B representations in the benchmark protein-ID order. The expected data layout is:

```text
data/
  go.obo
  real_dataset/splits/{mf,bp,cc}_{train,valid,test}_pid_list.txt
  labels/select_min_count_1_{mf,bp,cc}_labels.csv
  labels/{mf,bp,cc}_data_separate_{mf,bp,cc}_labels.csv
  features_3B/{mf,bp,cc}_{train,valid,test}_esm2.pkl
  sequence_features/uniprot_sequences.tsv
  features_150M_multilayer_segments/{mf,bp,cc}/
    {train,valid,test}_pids.txt
    {train,valid,test}_segments.npy
    {train,valid,test}_global.npy
    {train,valid,test}_segment_mask.npy
    {train,valid,test}_completed.npy
```

Each global-feature pickle maps a protein ID to a 2560-dimensional vector. Global features follow the DPGOK ESM2-3B protocol: layer 36, mean pooling, and a maximum of 1022 residues. The sequence TSV has `Entry`, `Entry Name` and `Sequence` columns and uses the same sequence release as the global features. GO-label columns follow the selected-term CSV order.

Extract local features with frozen ESM2-150M:

```bash
python -m sehnet.extract_local_features --ont bp --data_root data
```

Run separately for `mf`, `cc` and `bp`. Defaults use non-overlapping 512-residue windows covering the full sequence, layers 10/20/30 and 32 whole-protein relative-coordinate slots. Local arrays have shapes `(N,3,32,640)` for segments, `(N,3,640)` for global anchors and `(N,32)` for validity masks. Training centers valid local features by their layer-specific anchor.

## Training and evaluation

Run commands from the repository root:

```bash
python -m sehnet.train --config configs/bp.json --data_root data --result_dir results/bp_full
python -m sehnet.evaluate --run-dir results/bp_full --data-root data --device cuda
```

The public profiles use 8 BatchEnsemble members, a 1024-dimensional global hidden layer, 8 local tokens, a 64-dimensional local projection, dropout 0.2, AdamW and the manuscript asymmetric loss. MF/CC/BP train for 50/45/75 epochs. The epoch-based scheduler has factor 0.6 and patience 1000; the learning rate remains 0.0001 over these budgets. Early stopping is disabled. Evaluation averages probabilities from the final five checkpoints, then applies GO maximum propagation before Fmax/AUPR calculation.

The result directory stores the resolved configuration, metrics and trajectory checkpoints. A nonempty result directory is never overwritten.

Paper seeds:

| Ontology | Seeds |
| --- | --- |
| MF | 1644393137, 956266548, 1680496530 |
| CC | 1445141706, 1717527647, 535261970 |
| BP | 771476079, 1455114607, 1124400099 |

Use `--seed` to override a configuration's seed and a separate result directory for each run.

## Six ablations

Use the same ontology, seed and training budget, selecting `--ablation`:

| Option | Operation |
| --- | --- |
| `full` | Full SEHNet |
| `global_only` | Remove the local path |
| `prototype_only` | Replace child-minus-parent contrast with the term prototype |
| `pseudo_parent` | Use a fixed seeded pseudo-parent map |
| `uniform_readout` | Give local tokens equal term-readout weight |
| `independent_allocation` | Normalize assignment over segments rather than competing tokens |
| `joint_gradient` | Allow local supervision to update the global path |

```bash
python -m sehnet.train --config configs/bp.json --data_root data --result_dir results/bp_global --ablation global_only --seed 771476079
```

## BP analysis

BP occlusion averages predictions from the final five training epochs. Produce the matching probability bundle and term strata:

```bash
python -m sehnet.evaluate --run-dir results/bp_full --data-root data --output-tag occlusion
python -m analysis.prepare_bp_analysis --run-dir results/bp_full --data-root data
python -m analysis.occlude_bp --run-dir results/bp_full --data-root data --output-dir results/bp_full/occlusion
python -m analysis.attribution_baselines --run-dir results/bp_full --data-root data --pairs results/bp_full/occlusion/pair_results.csv --output-dir results/bp_full/attribution
```

Occlusion pre-ranks valid slots, removes Top/Bottom slots separately, and draws 20 random valid slots with replacement after excluding the original Top/Bottom. The cohort includes one annotated high-confidence target per selected protein, stratified by GO depth and training frequency. Attribution controls compare Routing and signed Gradient × Input on this same cohort and reuse its random-control effects.

For the BP term-level summaries, pass a probability bundle, the generated `term_group_manifest.csv` and its matching Fmax threshold:

```bash
python -m analysis.summarize_bp_terms --scores results/bp_full/last_five_test_occlusion_scores.pkl --term-manifest results/bp_full/bp_analysis/term_group_manifest.csv --threshold THRESHOLD --output-dir results/bp_full/term_summary
```

Replace `THRESHOLD` with the `threshold` value in the corresponding evaluation summary. Use `--help` on each module for input and output options.
