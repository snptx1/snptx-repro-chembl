# pyright: reportMissingImports=false
"""Tests for Phase C2: GNN model architectures and training/evaluation scripts."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.gnn import (
    GATModel,
    GCNModel,
    GINEModel,
    GINModel,
    MPNNModel,
    RGCNModel,
    build_gnn_model,
)

# ── Shared constants ────────────────────────────────────────────────────

NUM_NODES = 20
NUM_EDGES = 60
IN_CHANNELS = 16
HIDDEN_CHANNELS = 32
NUM_CLASSES = 3


def _make_graph_tensors(
    num_nodes: int = NUM_NODES,
    num_edges: int = NUM_EDGES,
    in_channels: int = IN_CHANNELS,
):
    """Create synthetic graph tensors for testing."""
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.stack([
        torch.randint(0, num_nodes, (num_edges,)),
        torch.randint(0, num_nodes, (num_edges,)),
    ])
    return x, edge_index


# ══════════════════════════════════════════════════════════════════════
# Model Construction
# ══════════════════════════════════════════════════════════════════════


class TestGNNModelConstruction:
    """Verify GNN model factory and basic properties."""

    def test_build_gcn(self):
        model = build_gnn_model("gcn", in_channels=IN_CHANNELS, out_channels=NUM_CLASSES)
        assert isinstance(model, GCNModel)
        assert hasattr(model, "embed_dim")

    def test_build_gat(self):
        model = build_gnn_model("gat", in_channels=IN_CHANNELS, out_channels=NUM_CLASSES)
        assert isinstance(model, GATModel)
        assert hasattr(model, "embed_dim")

    def test_build_mpnn(self):
        model = build_gnn_model("mpnn", in_channels=IN_CHANNELS, out_channels=NUM_CLASSES)
        assert isinstance(model, MPNNModel)
        assert hasattr(model, "embed_dim")

    def test_build_rgcn(self):
        model = build_gnn_model(
            "rgcn", in_channels=IN_CHANNELS, out_channels=NUM_CLASSES, num_relations=5,
        )
        assert isinstance(model, RGCNModel)
        assert hasattr(model, "embed_dim")

    def test_build_gin(self):
        model = build_gnn_model("gin", in_channels=IN_CHANNELS, out_channels=NUM_CLASSES)
        assert isinstance(model, GINModel)
        assert hasattr(model, "embed_dim")
        # PairNorm + DropEdge wired in for over-smoothing control.
        assert hasattr(model, "pair_norm")
        assert model.edge_dropout > 0.0

    def test_build_gine(self):
        model = build_gnn_model(
            "gine", in_channels=IN_CHANNELS, out_channels=NUM_CLASSES, edge_dim=4,
        )
        assert isinstance(model, GINEModel)
        assert hasattr(model, "embed_dim")
        assert model.edge_dim == 4

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_gnn_model("nonexistent_gnn", in_channels=IN_CHANNELS)


# ══════════════════════════════════════════════════════════════════════
# Forward Pass - Node Classification
# ══════════════════════════════════════════════════════════════════════


class TestGNNNodeClassification:
    """Verify forward pass for node-level tasks."""

    @pytest.mark.parametrize("arch", ["gcn", "gat", "gin"])
    def test_node_forward_shape(self, arch):
        model = build_gnn_model(
            arch, in_channels=IN_CHANNELS, out_channels=NUM_CLASSES,
            hidden_channels=HIDDEN_CHANNELS, task="node",
        )
        model.eval()
        x, edge_index = _make_graph_tensors()
        with torch.no_grad():
            out = model(x, edge_index)
        assert out.shape == (NUM_NODES, NUM_CLASSES)

    def test_rgcn_node_forward(self):
        model = build_gnn_model(
            "rgcn", in_channels=IN_CHANNELS, out_channels=NUM_CLASSES,
            hidden_channels=HIDDEN_CHANNELS, num_relations=3, task="node",
        )
        model.eval()
        x, edge_index = _make_graph_tensors()
        edge_type = torch.randint(0, 3, (NUM_EDGES,))
        with torch.no_grad():
            out = model(x, edge_index, edge_type)
        assert out.shape == (NUM_NODES, NUM_CLASSES)


# ══════════════════════════════════════════════════════════════════════
# Forward Pass - Graph Classification
# ══════════════════════════════════════════════════════════════════════


class TestGNNGraphClassification:
    """Verify forward pass for graph-level tasks with batch index."""

    @pytest.mark.parametrize("arch", ["gcn", "gat", "gin"])
    def test_graph_forward_shape(self, arch):
        model = build_gnn_model(
            arch, in_channels=IN_CHANNELS, out_channels=NUM_CLASSES,
            hidden_channels=HIDDEN_CHANNELS, task="graph",
        )
        model.eval()
        x, edge_index = _make_graph_tensors()
        batch = torch.zeros(NUM_NODES, dtype=torch.long)  # single graph
        with torch.no_grad():
            out = model(x, edge_index, batch=batch)
        assert out.shape == (1, NUM_CLASSES)

    def test_gine_graph_forward(self):
        model = build_gnn_model(
            "gine", in_channels=IN_CHANNELS, out_channels=NUM_CLASSES,
            hidden_channels=HIDDEN_CHANNELS, edge_dim=4, task="graph",
        )
        model.eval()
        x, edge_index = _make_graph_tensors()
        edge_attr = torch.randn(NUM_EDGES, 4)
        batch = torch.zeros(NUM_NODES, dtype=torch.long)
        with torch.no_grad():
            out = model(x, edge_index, edge_attr=edge_attr, batch=batch)
        assert out.shape == (1, NUM_CLASSES)

    def test_mpnn_graph_forward(self):
        model = build_gnn_model(
            "mpnn", in_channels=IN_CHANNELS, out_channels=NUM_CLASSES,
            hidden_channels=HIDDEN_CHANNELS, edge_dim=4, task="graph",
        )
        model.eval()
        x, edge_index = _make_graph_tensors()
        edge_attr = torch.randn(NUM_EDGES, 4)
        batch = torch.zeros(NUM_NODES, dtype=torch.long)
        with torch.no_grad():
            out = model(x, edge_index, edge_attr=edge_attr, batch=batch)
        assert out.shape == (1, NUM_CLASSES)


# ══════════════════════════════════════════════════════════════════════
# Embedding Extraction
# ══════════════════════════════════════════════════════════════════════


class TestGNNEmbeddings:
    """Verify embedding extraction dimensions."""

    @pytest.mark.parametrize("arch", ["gcn", "gat", "gin"])
    def test_node_embedding_shape(self, arch):
        model = build_gnn_model(
            arch, in_channels=IN_CHANNELS, out_channels=NUM_CLASSES,
            hidden_channels=HIDDEN_CHANNELS, task="node",
        )
        model.eval()
        x, edge_index = _make_graph_tensors()
        with torch.no_grad():
            emb = model.extract_embeddings(x, edge_index)
        assert emb.shape == (NUM_NODES, HIDDEN_CHANNELS)

    def test_gin_dropedge_active_only_in_training(self):
        """DropEdge must be a no-op at eval time (deterministic forward)."""
        model = build_gnn_model(
            "gin", in_channels=IN_CHANNELS, out_channels=NUM_CLASSES,
            hidden_channels=HIDDEN_CHANNELS, edge_dropout=0.5, task="node",
        )
        model.eval()
        x, edge_index = _make_graph_tensors()
        with torch.no_grad():
            out_a = model(x, edge_index)
            out_b = model(x, edge_index)
        assert torch.allclose(out_a, out_b)

    def test_gin_pair_norm_preserves_variance(self):
        """PairNorm should keep embedding variance non-trivial across layers."""
        model = build_gnn_model(
            "gin", in_channels=IN_CHANNELS, out_channels=NUM_CLASSES,
            hidden_channels=HIDDEN_CHANNELS, num_layers=6,
            edge_dropout=0.0, task="node",
        )
        model.eval()
        x, edge_index = _make_graph_tensors()
        with torch.no_grad():
            emb = model.extract_embeddings(x, edge_index)
        # Without PairNorm, deep stacks collapse toward a constant; require
        # per-feature variance well above zero as a sanity check.
        assert emb.var(dim=0).mean().item() > 1e-3

    def test_graph_embedding_shape(self):
        model = build_gnn_model(
            "gcn", in_channels=IN_CHANNELS, out_channels=NUM_CLASSES,
            hidden_channels=HIDDEN_CHANNELS, task="graph",
        )
        model.eval()
        x, edge_index = _make_graph_tensors()
        batch = torch.zeros(NUM_NODES, dtype=torch.long)
        with torch.no_grad():
            emb = model.extract_embeddings(x, edge_index, batch=batch)
        assert emb.shape == (1, HIDDEN_CHANNELS)


# ══════════════════════════════════════════════════════════════════════
# Training Script
# ══════════════════════════════════════════════════════════════════════


class TestTrainGNN:
    """Verify GNN training script writes checkpoint and metrics."""

    def test_train_produces_checkpoint(self, tmp_path: Path):
        from workflow.scripts.train_gnn import train_gnn

        data_dir = str(tmp_path / "data")
        Path(data_dir).mkdir()
        model_path = str(tmp_path / "model.pt")
        metrics_path = str(tmp_path / "metrics.json")

        train_gnn(
            data_dir=data_dir,
            output_model=model_path,
            output_metrics=metrics_path,
            arch="gcn",
            task="graph",
            in_channels=IN_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            num_classes=NUM_CLASSES,
            epochs=2,
            batch_size=16,
            lr=1e-3,
            seed=42,
            device_cfg="cpu",
        )

        assert Path(model_path).exists()
        assert Path(metrics_path).exists()

        metrics = json.loads(Path(metrics_path).read_text())
        assert "train_accuracy" in metrics
        assert metrics["epochs"] == 2
        assert metrics["modality"] == "gnn"


# ══════════════════════════════════════════════════════════════════════
# Evaluation Script
# ══════════════════════════════════════════════════════════════════════


class TestEvaluateGNN:
    """Verify GNN evaluation produces metrics and embeddings."""

    def test_evaluate_produces_metrics(self, tmp_path: Path):
        from workflow.scripts.evaluate_gnn import evaluate_gnn
        from workflow.scripts.train_gnn import train_gnn

        data_dir = str(tmp_path / "data")
        Path(data_dir).mkdir()
        model_path = str(tmp_path / "model.pt")

        train_gnn(
            data_dir=data_dir,
            output_model=model_path,
            output_metrics=str(tmp_path / "train_metrics.json"),
            arch="gcn",
            task="graph",
            in_channels=IN_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            num_classes=NUM_CLASSES,
            epochs=2,
            batch_size=16,
            seed=42,
            device_cfg="cpu",
        )

        eval_metrics_path = str(tmp_path / "eval_metrics.json")
        emb_path = str(tmp_path / "embeddings.npy")

        evaluate_gnn(
            checkpoint_path=model_path,
            data_dir=data_dir,
            output_metrics=eval_metrics_path,
            output_embeddings=emb_path,
            device_cfg="cpu",
        )

        assert Path(eval_metrics_path).exists()
        assert Path(emb_path).exists()

        metrics = json.loads(Path(eval_metrics_path).read_text())
        assert "accuracy" in metrics
        assert "f1" in metrics

        embs = np.load(emb_path)
        assert embs.ndim == 2
        assert embs.shape[1] == HIDDEN_CHANNELS
