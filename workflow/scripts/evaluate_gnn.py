#!/usr/bin/env python3
"""evaluate_gnn.py - Evaluate trained GNN models and extract graph embeddings.

Loads a GNN checkpoint, computes node-level or graph-level metrics,
and extracts embeddings for the EmbeddingRegistry / fusion pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.gnn import build_gnn_model

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def load_test_graphs(data_dir: str) -> list:
    data_path = Path(data_dir)
    if (data_path / "test_graphs.pt").exists():
        return torch.load(data_path / "test_graphs.pt", weights_only=False)

    pt_files = sorted(data_path.glob("test_graph_*.pt"))
    if pt_files:
        return [torch.load(f, weights_only=False) for f in pt_files]

    # Fall back to synthetic
    from workflow.scripts.train_gnn import _create_synthetic_graphs

    return _create_synthetic_graphs(num_graphs=30)


def evaluate_gnn(
    checkpoint_path: str,
    data_dir: str,
    output_metrics: str,
    output_embeddings: str | None = None,
    batch_size: int = 32,
    device_cfg: str = "auto",
    registry_dir: str = "",
) -> dict:
    """Evaluate a GNN model checkpoint."""
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for GNN evaluation")

    device = _resolve_device(device_cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    arch = ckpt["arch"]
    task = ckpt["task"]
    in_channels = ckpt["in_channels"]
    hidden_channels = ckpt["hidden_channels"]
    num_classes = ckpt["num_classes"]
    num_layers = ckpt["num_layers"]
    seed = ckpt.get("seed", 42)
    set_deterministic(seed)

    model = build_gnn_model(
        model_type=arch,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        num_layers=num_layers,
        task=task,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    graphs = load_test_graphs(data_dir)
    loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)  # type: ignore[reportPossiblyUnboundVariable]

    all_preds = []
    all_labels = []
    all_embeddings = []

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch=getattr(batch, "batch", None))

            if task == "graph":
                y = batch.y.view(-1)
            else:
                y = batch.y

            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

            emb = model.extract_embeddings(
                batch.x, batch.edge_index, batch=getattr(batch, "batch", None)
            )
            all_embeddings.append(emb.cpu())

    inference_time = time.time() - t0

    preds_np = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    embeddings_np = torch.cat(all_embeddings).numpy()

    metrics = {
        "modality": "gnn",
        "arch": arch,
        "task": task,
        "num_classes": num_classes,
        "test_graphs": len(graphs),
        "total_predictions": len(preds_np),
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "precision": float(precision_score(labels_np, preds_np, average="macro", zero_division=0)),  # type: ignore[reportArgumentType]
        "recall": float(recall_score(labels_np, preds_np, average="macro", zero_division=0)),  # type: ignore[reportArgumentType]
        "f1": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),  # type: ignore[reportArgumentType]
        "inference_seconds": round(inference_time, 3),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    Path(output_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(output_metrics).write_text(json.dumps(metrics, indent=2))

    if output_embeddings:
        Path(output_embeddings).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_embeddings, embeddings_np)

    # Register in EmbeddingRegistry for lineage tracking
    if registry_dir:
        try:
            from src.embeddings import EmbeddingRegistry

            registry = EmbeddingRegistry(registry_dir)
            key = registry.register(
                embeddings=embeddings_np,
                metadata={
                    "model_type": arch,
                    "modality": "graph",
                    "dataset_path": data_dir,
                    "checkpoint_path": checkpoint_path,
                    "device": str(device),
                    "torch_version": torch.__version__,
                },
                labels=labels_np,
            )
            metrics["embedding_registry_key"] = key
        except Exception:
            pass

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GNN model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-embeddings", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--registry-dir", default="", help="EmbeddingRegistry directory")
    args = parser.parse_args()

    evaluate_gnn(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_metrics=args.output_metrics,
        output_embeddings=args.output_embeddings,
        batch_size=args.batch_size,
        device_cfg=args.device,
        registry_dir=args.registry_dir,
    )


if __name__ == "__main__":
    main()
