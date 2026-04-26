# snptx-repro-chembl

Public reproducibility artifact for the **ChEMBL bioactivity GCN** result reported on the [SNPTX academic site](https://snptx1.github.io/snptx-academic/).

This is a carve-out of the private `snptx-core` codebase containing only the files needed to reproduce a single headline number. Fusion, NLP/vision modalities, autonomy, deployment infrastructure, and clinical data adapters live in the private repo.

## Headline result

```
dataset:  chembl_bioactivity (sample, ~4,685 graphs)
model:    GCN (3 layers, hidden=64, batch=32, lr=1e-3)
seed:     42
n_train:  3,748
n_test:   937
accuracy: 0.9850586979722519
```

Bit-equal across runs at `seed=42`; ~15 s on CPU, no GPU required.

### Note on the historical 92.1% figure

An earlier internal run on 2026-04-03 recorded `accuracy = 0.9210` using an in-progress version of the trainer that was never committed in isolation (it was rolled into a single Phase C squash commit on 2026-04-04). The committed trainer in this repo adds an internal validation split with best-checkpoint selection, which improves held-out accuracy to **0.9797** on the same data and same seed. The 0.9797 number is the canonical, deterministic, publicly reproducible result.

## Reproduce in 5 minutes

```bash
# prerequisites: python3.10+, pip, make, git
# (Ubuntu/Debian: sudo apt install python3 python3-venv make git)
git clone https://github.com/snptx1/snptx-repro-chembl.git
cd snptx-repro-chembl
make install         # create .venv, install CPU-only deps (~1 min, ~200 MB)
make repro-chembl    # ≈ 15s on CPU; no GPU required
cat results/metrics/drug_discovery_result.json   # expect accuracy = 0.9850586979722519
```

The script pins seed 42 and writes a result summary to `results/metrics/drug_discovery_result.json`.

## What's in this repo

| Path | Purpose |
| --- | --- |
| `repro_chembl.py` | One-shot orchestrator: adapter → split → graph build → train → eval |
| `src/adapters/chembl.py` | ChEMBL bioactivity → tabular features (log_value, units, activity) |
| `src/adapters/{base,registry}.py` | Adapter framework (BaseAdapter, GraphAdapter, registry decorator) |
| `src/models/gnn.py` | 6 GNN architectures (GCN, GAT, GIN, GINE, MPNN, R-GCN) + factory |
| `workflow/scripts/train_gnn.py` | Generic GNN trainer with MLflow logging, deterministic seeding |
| `workflow/scripts/evaluate_gnn.py` | Eval harness producing accuracy/F1/precision/recall + embeddings |
| `data/raw/chembl/*.json` | ChEMBL approved drugs + bioactivity sample (CC-BY-SA 3.0) |
| `tests/test_gnn.py` | Unit tests for all 6 GNN architectures |

Total: ~16 source files, ~2,150 LOC, ~9 MB on disk.

## Requirements

- Python 3.11
- PyTorch 2.5.1 + CUDA 12.1 (recommended) or CPU
- PyTorch Geometric 2.7.0
- See [`requirements.txt`](requirements.txt) for full lock

## Data license

ChEMBL is distributed by the European Bioinformatics Institute under [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The two JSON files under `data/raw/chembl/` are sample slices; for the full database see [chembl.gitbook.io](https://chembl.gitbook.io/chembl-interface-documentation).

## Limitations

- Single headline number; no calibration, no bootstrap CIs, no leakage gates (those live in `snptx-core`).
- Single-node molecular graphs with self-loops (the GCN is essentially operating as an MLP on the bioactivity-derived feature vector). This is a baseline; richer molecular graph featurizations are part of the broader benchmark.
- Bit-equal across CPU runs at `seed=42`; GPU runs may differ in the 4th decimal due to cuDNN non-determinism.

## Citation

If you use this code, please cite the SNPTX academic page:

> SNPTX. *An Experimentation Layer for Multi-Modal Biomedical Machine Learning.* https://snptx1.github.io/snptx-academic/

## License

Code: MIT. Data: CC-BY-SA 3.0 (ChEMBL upstream).
