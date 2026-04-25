"""snptx-repro-chembl — minimal end-to-end orchestrator.

Reproduces the headline ChEMBL bioactivity GCN result reported in the
snptx-academic site (https://snptx1.github.io/snptx-academic/):

    accuracy = 0.9797225186766275
    n_train  = 3748
    n_test   = 937
    seed     = 42

Bit-equal across runs at seed=42; ~15s on CPU.

Run with:

    make repro-chembl

or directly:

    python repro_chembl.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from src.adapters.chembl import ChemblAdapter
from workflow.scripts.evaluate_gnn import evaluate_gnn
from workflow.scripts.train_gnn import train_gnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("repro_chembl")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
METRICS = RESULTS / "metrics"
EMBEDDINGS = RESULTS / "embeddings"

SEED = 42
EPOCHS = 10
DEVICE = "auto"


def main() -> dict:
    for d in (MODELS, METRICS, EMBEDDINGS):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ChEMBL adapter (data/raw/chembl)")
    adapter = ChemblAdapter(raw_dir=DATA / "raw" / "chembl")
    df = adapter.build("bioactivity", activity_threshold=10000)

    target_col = "active"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    logger.info(f"Splits: n_train={len(X_train)}, n_test={len(X_test)}, dim={X.shape[1]}")

    drug_dir = RESULTS / "data" / "drug_discovery"
    drug_dir.mkdir(parents=True, exist_ok=True)

    def _make_mol_graphs(X_set: np.ndarray, y_set: np.ndarray) -> list[Data]:
        graphs: list[Data] = []
        for i in range(len(X_set)):
            x = torch.tensor(X_set[i:i + 1], dtype=torch.float32)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # self-loop
            graphs.append(
                Data(x=x, edge_index=edge_index, y=torch.tensor([y_set[i]], dtype=torch.long))
            )
        return graphs

    torch.save(_make_mol_graphs(X_train, y_train), drug_dir / "train_graphs.pt")
    torch.save(_make_mol_graphs(X_test, y_test), drug_dir / "test_graphs.pt")

    model_path = str(MODELS / "drug_gnn.pt")
    train_metrics_path = str(METRICS / "drug_train_metrics.json")
    eval_metrics_path = str(METRICS / "drug_eval_metrics.json")
    eval_emb_path = str(EMBEDDINGS / "drug_discovery_embeddings.npy")

    logger.info("Training GCN (10 epochs, hidden=64, layers=3, batch=32, lr=1e-3)")
    train_gnn(
        data_dir=str(drug_dir),
        output_model=model_path,
        output_metrics=train_metrics_path,
        arch="gcn",
        task="graph",
        in_channels=X_train.shape[1],
        hidden_channels=64,
        num_classes=2,
        num_layers=3,
        epochs=EPOCHS,
        batch_size=32,
        lr=1e-3,
        seed=SEED,
        device_cfg=DEVICE,
    )

    logger.info("Evaluating checkpoint on held-out test set")
    eval_result = evaluate_gnn(
        checkpoint_path=model_path,
        data_dir=str(drug_dir),
        output_metrics=eval_metrics_path,
        output_embeddings=eval_emb_path,
        batch_size=32,
        device_cfg=DEVICE,
    )
    np.save(EMBEDDINGS / "drug_discovery_labels.npy", y_test)

    metrics = eval_result if isinstance(eval_result, dict) else {}
    summary = {
        "modality": "drug_discovery",
        "dataset": "chembl_bioactivity",
        "model": "gcn",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_classes": 2,
        "metrics": metrics,
        "seed": SEED,
        "epochs": EPOCHS,
        "expected_accuracy": 0.9797225186766275,
    }
    summary_path = METRICS / "drug_discovery_result.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    acc = metrics.get("accuracy")
    print()
    print("=" * 64)
    print(f" ChEMBL bioactivity GCN — accuracy = {acc!r}")
    print(f" Expected (deterministic, seed=42): 0.9797225186766275")
    print(f" Summary written to: {summary_path}")
    print("=" * 64)
    return summary


if __name__ == "__main__":
    main()
