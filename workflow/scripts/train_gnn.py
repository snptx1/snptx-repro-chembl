#!/usr/bin/env python3
"""train_gnn.py - Train graph neural network models on biological graphs.

Supports GCN, GAT, MPNN, and RGCN architectures for node-level or
graph-level classification tasks (protein-protein interactions,
drug-target interactions, knowledge graphs).

References:
    Kipf & Welling. "Semi-Supervised Classification with GCNs" (ICLR 2017)
    Veličković et al. "Graph Attention Networks" (ICLR 2018)
    Gilmer et al. "Neural Message Passing for Quantum Chemistry" (ICML 2017)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.gnn import build_gnn_model

try:
    from torch_geometric.data import Data
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
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True)


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _create_synthetic_graphs(
    num_graphs: int = 100,
    avg_nodes: int = 20,
    node_features: int = 16,
    num_classes: int = 2,
    task: str = "graph",
) -> list:
    """Create synthetic graph data for development/testing."""
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for GNN training")

    graphs = []
    for _ in range(num_graphs):
        n = max(5, avg_nodes + random.randint(-5, 5))
        # Random edges
        num_edges = n * 3
        src = torch.randint(0, n, (num_edges,))
        dst = torch.randint(0, n, (num_edges,))
        edge_index = torch.stack([src, dst], dim=0)

        x = torch.randn(n, node_features)
        if task == "graph":
            y = torch.randint(0, num_classes, (1,))
        else:
            y = torch.randint(0, num_classes, (n,))

        data = Data(x=x, edge_index=edge_index, y=y)  # type: ignore[reportPossiblyUnboundVariable]
        graphs.append(data)
    return graphs


def load_graph_data(data_dir: str, task: str = "graph") -> list:
    """Load graph data from PyG .pt files or create synthetic data."""
    data_path = Path(data_dir)

    # Try loading pre-saved PyG dataset
    if (data_path / "train_graphs.pt").exists():
        return torch.load(data_path / "train_graphs.pt", weights_only=False)

    # Try loading individual graph files
    pt_files = sorted(data_path.glob("graph_*.pt"))
    if pt_files:
        return [torch.load(f, weights_only=False) for f in pt_files]

    # Fall back to synthetic
    return _create_synthetic_graphs(task=task)


def train_gnn(
    data_dir: str,
    output_model: str,
    output_metrics: str,
    arch: str = "gat",
    task: str = "graph",
    in_channels: int = 16,
    hidden_channels: int = 128,
    num_classes: int = 2,
    num_layers: int = 3,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 5e-4,
    seed: int = 42,
    device_cfg: str = "auto",
    use_center_readout: bool = False,
    val_ratio: float = 0.15,
    model_kwargs: dict | None = None,
) -> dict:
    """Train a GNN model."""
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for GNN training")

    set_deterministic(seed)
    device = _resolve_device(device_cfg)
    t0 = time.time()

    # Load data
    graph_list = load_graph_data(data_dir, task=task)
    if graph_list:
        in_channels = graph_list[0].x.shape[1]

    # Train/val split with stratification to preserve class balance.
    n_total = len(graph_list)
    indices = np.arange(n_total)
    graph_labels = np.array([int(g.y.view(-1)[0].item()) for g in graph_list])
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=graph_labels if len(np.unique(graph_labels)) > 1 else None,
    )
    val_graphs = [graph_list[int(i)] for i in val_idx]
    train_graphs = [graph_list[int(i)] for i in train_idx]

    loader = PyGDataLoader(train_graphs, batch_size=batch_size, shuffle=True)  # type: ignore[reportPossiblyUnboundVariable]
    val_loader = PyGDataLoader(val_graphs, batch_size=batch_size, shuffle=False)  # type: ignore[reportPossiblyUnboundVariable]

    # Build model
    model = build_gnn_model(
        model_type=arch,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        num_layers=num_layers,
        task=task,
        **(model_kwargs or {}),
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Check if graphs have center_idx for center-node readout
    has_center = use_center_readout and hasattr(train_graphs[0], "center_idx")
    if has_center:
        print(f"  Using center-node readout (task=graph)")

    # R-GCN needs an edge_type tensor; other archs use the standard signature.
    needs_edge_type = arch == "rgcn"
    if needs_edge_type and not hasattr(train_graphs[0], "edge_type"):
        raise ValueError(
            "arch='rgcn' requires graphs with an `edge_type` tensor. "
            "Re-run prepare_hetionet.py to regenerate the dataset."
        )

    def _forward(mdl, batch_data):
        """Forward pass with optional center-node readout."""
        if needs_edge_type:
            extra = {"edge_type": batch_data.edge_type}
        else:
            extra = {}
        if has_center and task == "graph":
            # Get per-node embeddings (pass batch=None to skip global pooling)
            node_embs = mdl.extract_embeddings(
                batch_data.x, batch_data.edge_index, batch=None, **extra
            )
            # Extract center node embeddings using PyG ptr offsets
            center_indices = batch_data.center_idx.view(-1)
            ptr = batch_data.ptr  # cumulative node counts per graph
            global_center = ptr[:-1] + center_indices
            center_emb = node_embs[global_center]
            return mdl.classifier(center_emb)
        else:
            return mdl(
                batch_data.x,
                batch_data.edge_index,
                batch=getattr(batch_data, "batch", None),
                **extra,
            )

    # Compute class weights from training data to handle imbalanced targets.
    # Inverse-frequency weighting (Japkowicz & Stephen, 2002).
    all_labels = torch.cat([g.y.view(-1) for g in train_graphs])
    class_counts = torch.bincount(all_labels, minlength=num_classes).float()
    class_weights = torch.where(class_counts > 0, 1.0 / class_counts, torch.zeros_like(class_counts))
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    loss_history = []
    val_acc_history = []
    val_macro_f1_history = []
    avg_loss = 0.0
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    final_train_accuracy = 0.0
    final_train_macro_f1 = 0.0
    best_state = None
    patience_counter = 0
    patience = 20

    model.train()
    for _epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        train_preds_epoch: list[int] = []
        train_labels_epoch: list[int] = []
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = _forward(model, batch)

            if task == "graph":
                y = batch.y.view(-1)
            else:
                y = batch.y

            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * y.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            train_preds_epoch.extend(preds.detach().cpu().numpy().tolist())
            train_labels_epoch.extend(y.detach().cpu().numpy().tolist())

        avg_loss = epoch_loss / max(total, 1)
        final_train_accuracy = correct / max(total, 1)
        final_train_macro_f1 = (
            f1_score(train_labels_epoch, train_preds_epoch, average="macro")
            if train_labels_epoch
            else 0.0
        )
        loss_history.append(avg_loss)
        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds_all: list[int] = []
        val_labels_all: list[int] = []
        with torch.no_grad():
            for vbatch in val_loader:
                vbatch = vbatch.to(device)
                vout = _forward(model, vbatch)
                if task == "graph":
                    vy = vbatch.y.view(-1)
                else:
                    vy = vbatch.y
                vpreds = vout.argmax(1)
                val_correct += (vpreds == vy).sum().item()
                val_total += vy.size(0)
                val_preds_all.extend(vpreds.detach().cpu().numpy().tolist())
                val_labels_all.extend(vy.detach().cpu().numpy().tolist())
        val_acc = val_correct / max(val_total, 1)
        val_macro_f1 = f1_score(val_labels_all, val_preds_all, average="macro") if val_labels_all else 0.0
        val_acc_history.append(val_acc)
        val_macro_f1_history.append(val_macro_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    duration = time.time() - t0

    # Save checkpoint
    Path(output_model).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "arch": arch,
        "task": task,
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "num_classes": num_classes,
        "num_layers": num_layers,
        "embed_dim": model.embed_dim,
        "seed": seed,
        "epochs_trained": epochs,
    }
    torch.save(checkpoint, output_model)

    # Save metrics
    metrics = {
        "modality": "gnn",
        "framework": "pytorch_geometric",
        "arch": arch,
        "task": task,
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "num_classes": num_classes,
        "num_layers": num_layers,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "training_accuracy_last_epoch": final_train_accuracy,
        "training_macro_f1_last_epoch": final_train_macro_f1,
        "validation_accuracy": best_val_acc,
        "validation_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_val_acc,
        "best_val_f1": best_val_macro_f1,
        "final_loss": avg_loss,
        "loss_history": loss_history,
        "val_acc_history": val_acc_history,
        "val_macro_f1_history": val_macro_f1_history,
        "train_graphs": len(train_graphs),
        "validation_graphs": len(val_graphs),
        "duration_seconds": round(duration, 2),
        "device": str(device),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit_hash(),
        "torch_version": torch.__version__,
    }
    Path(output_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(output_metrics).write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GNN model")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--arch", default="gin", choices=["gcn", "gat", "gin", "gine", "mpnn", "rgcn"])
    parser.add_argument("--task", default="graph", choices=["graph", "node"])
    parser.add_argument("--in-channels", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    train_gnn(
        data_dir=args.data_dir,
        output_model=args.output_model,
        output_metrics=args.output_metrics,
        arch=args.arch,
        task=args.task,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device_cfg=args.device,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
