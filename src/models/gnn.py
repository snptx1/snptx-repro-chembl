"""Graph Neural Network architectures for Phase C2 - GNN Pipeline.

Provides PyG-based GNN architectures for biomedical graph tasks:
    - GCNModel  : Graph Convolutional Network (Kipf & Welling 2017)
    - GATModel  : Graph Attention Network (Velickovic et al. 2018)
    - GINModel  : Graph Isomorphism Network — provably matches the
                  1-Weisfeiler-Leman expressiveness upper bound that
                  GCN/GAT do not (Xu et al. 2019). Default backbone for
                  homogeneous graph workloads.
    - GINEModel : GIN with edge-feature conditioning (Hu et al. 2020).
    - MPNNModel : Message Passing Neural Network (Gilmer et al. 2017)
    - RGCNModel : Relational GCN for heterogeneous graphs (Schlichtkrull et al. 2018)

All deep variants (GIN/GINE) include two anti-over-smoothing mechanisms:
    - PairNorm (Zhao & Akoglu 2020): bounds total Dirichlet energy decay,
      preserving node distinguishability across many message-passing layers.
    - DropEdge (Rong et al. 2020): random edge subsampling during training,
      reducing message-aggregation variance and acting as a stochastic
      regularizer on graph topology.

Task types supported:
    - Node classification (e.g., protein function prediction)
    - Graph classification (e.g., molecular toxicity)
    - Link prediction (e.g., drug-target interaction)

All models expose:
    - forward(data) -> logits or embeddings
    - extract_embeddings(data) -> (N, D) node embeddings

References:
    Kipf & Welling. "Semi-Supervised Classification with GCNs" (ICLR 2017)
    Velickovic et al. "Graph Attention Networks" (ICLR 2018)
    Xu et al. "How Powerful are Graph Neural Networks?" (ICLR 2019) — GIN
    Hu et al. "Strategies for Pre-training GNNs" (ICLR 2020) — GINE
    Gilmer et al. "Neural Message Passing for Quantum Chemistry" (ICML 2017)
    Schlichtkrull et al. "Modeling Relational Data with GCNs" (ESWC 2018)
    Zhao & Akoglu. "PairNorm: Tackling Oversmoothing in GNNs" (ICLR 2020)
    Rong et al. "DropEdge: Towards Deep GCNs on Node Classification" (ICLR 2020)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    GINEConv,
    NNConv,
    PairNorm,
    RGCNConv,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.utils import dropout_edge


class GCNModel(nn.Module):
    """Graph Convolutional Network for node or graph classification.

    References:
        Kipf & Welling. "Semi-Supervised Classification with GCNs" (ICLR 2017)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
        task: str = "node",
    ):
        super().__init__()
        self.task = task
        self.dropout = dropout
        self.embed_dim = hidden_channels

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.extract_embeddings(x, edge_index, batch)
        return self.classifier(emb)

    def extract_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph" and batch is not None:
            x = global_mean_pool(x, batch)
        return x


class GATModel(nn.Module):
    """Graph Attention Network with multi-head attention.

    References:
        Velickovic et al. "Graph Attention Networks" (ICLR 2018)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        task: str = "node",
    ):
        super().__init__()
        self.task = task
        self.dropout = dropout
        self.embed_dim = hidden_channels

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        # Final layer: single head
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.extract_embeddings(x, edge_index, batch)
        return self.classifier(emb)

    def extract_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph" and batch is not None:
            x = global_mean_pool(x, batch)
        return x


def _gin_mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    """Standard 2-layer MLP used inside GIN's injective aggregator.

    Per Xu et al. (2019), GIN's expressiveness bound requires the
    aggregation function to be a learnable injective map over multisets;
    a 2-layer MLP with non-linearity is the canonical instantiation.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class GINModel(nn.Module):
    """Graph Isomorphism Network with PairNorm + DropEdge.

    GIN is the maximally expressive message-passing GNN under the
    1-Weisfeiler-Leman (1-WL) hierarchy: it can distinguish any pair of
    graphs that 1-WL can distinguish (Xu et al. 2019). GCN and GAT, by
    contrast, are strictly less expressive than 1-WL because their
    mean/max aggregators are not injective over multisets.

    Sum-pooling at the graph readout matches the WL multiset-hashing
    interpretation; switching to mean pooling collapses the bound.

    Anti-over-smoothing:
        - PairNorm (Zhao & Akoglu 2020) is applied between conv layers.
        - DropEdge (Rong et al. 2020) randomly removes edges during
          training only.

    References:
        Xu et al. "How Powerful are Graph Neural Networks?" (ICLR 2019)
        Zhao & Akoglu. "PairNorm" (ICLR 2020)
        Rong et al. "DropEdge" (ICLR 2020)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
        edge_dropout: float = 0.1,
        pair_norm_scale: float = 1.0,
        train_eps: bool = True,
        task: str = "node",
    ):
        super().__init__()
        self.task = task
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.embed_dim = hidden_channels

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GINConv(_gin_mlp(in_channels, hidden_channels), train_eps=train_eps))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(_gin_mlp(hidden_channels, hidden_channels), train_eps=train_eps)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # PairNorm preserves total pairwise distance across layers,
        # provably bounding Dirichlet-energy decay (over-smoothing).
        self.pair_norm = PairNorm(scale=pair_norm_scale)

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.extract_embeddings(x, edge_index, batch)
        return self.classifier(emb)

    def extract_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # DropEdge: stochastic edge subsampling during training.
        if self.training and self.edge_dropout > 0.0:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=True)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            # PairNorm between layers (skip after final, which feeds readout).
            if i < len(self.convs) - 1:
                x = self.pair_norm(x, batch)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph" and batch is not None:
            # Sum pooling preserves the WL multiset-hashing interpretation
            # (mean/max would collapse the expressiveness bound).
            x = global_add_pool(x, batch)
        return x


class GINEModel(nn.Module):
    """GIN with edge-feature conditioning (Hu et al. 2020).

    GINE extends GIN to incorporate edge attributes in the message
    function: h_v <- MLP((1+eps)*h_v + sum_u (h_u + edge_emb(uv))).
    Suitable for molecular graphs (bond features) and any setting where
    edges carry semantically meaningful features.

    References:
        Hu et al. "Strategies for Pre-training GNNs" (ICLR 2020)
        Xu et al. "How Powerful are Graph Neural Networks?" (ICLR 2019)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        edge_dim: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        edge_dropout: float = 0.1,
        pair_norm_scale: float = 1.0,
        train_eps: bool = True,
        task: str = "graph",
    ):
        super().__init__()
        self.task = task
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.edge_dim = edge_dim
        self.embed_dim = hidden_channels

        self.lin_in = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GINEConv(
                    _gin_mlp(hidden_channels, hidden_channels),
                    train_eps=train_eps,
                    edge_dim=edge_dim,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.pair_norm = PairNorm(scale=pair_norm_scale)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.extract_embeddings(x, edge_index, edge_attr, batch)
        return self.classifier(emb)

    def extract_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.lin_in(x)

        if self.training and self.edge_dropout > 0.0:
            # Co-drop edge_attr with edges to keep tensors aligned.
            edge_index, edge_mask = dropout_edge(
                edge_index, p=self.edge_dropout, training=True
            )
            if edge_attr is not None:
                edge_attr = edge_attr[edge_mask]

        if edge_attr is None:
            # Fall back to zero edge features (still injective on nodes).
            num_edges = edge_index.size(1)
            edge_attr = torch.zeros(num_edges, self.edge_dim, device=x.device)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = self.pair_norm(x, batch)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph" and batch is not None:
            x = global_add_pool(x, batch)
        return x


class MPNNModel(nn.Module):
    """Message Passing Neural Network for molecular property prediction.

    Uses NNConv (edge-conditioned convolution) with a learned edge network,
    suitable for molecular graphs where edges have features (bond type, etc.).

    References:
        Gilmer et al. "Neural Message Passing for Quantum Chemistry" (ICML 2017)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        edge_dim: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        task: str = "graph",
    ):
        super().__init__()
        self.task = task
        self.dropout = dropout
        self.embed_dim = hidden_channels

        self.lin_in = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, hidden_channels * hidden_channels),
            )
            self.convs.append(NNConv(hidden_channels, hidden_channels, edge_nn, aggr="add"))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.extract_embeddings(x, edge_index, edge_attr, batch)
        return self.classifier(emb)

    def extract_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.lin_in(x)
        for conv, bn in zip(self.convs, self.bns):
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                # Fall back to zero edge features
                num_edges = edge_index.size(1)
                dummy_attr = torch.zeros(num_edges, 4, device=x.device)
                x = conv(x, edge_index, dummy_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph" and batch is not None:
            x = global_add_pool(x, batch)
        return x


class RGCNModel(nn.Module):
    """Relational Graph Convolutional Network for heterogeneous graphs.

    Handles multiple edge types (relations), suitable for knowledge graphs
    like Hetionet, Reactome, and biomedical ontologies.

    References:
        Schlichtkrull et al. "Modeling Relational Data with GCNs" (ESWC 2018)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_relations: int = 10,
        num_layers: int = 2,
        dropout: float = 0.3,
        task: str = "node",
    ):
        super().__init__()
        self.task = task
        self.dropout = dropout
        self.embed_dim = hidden_channels

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations=num_relations))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.extract_embeddings(x, edge_index, edge_type, batch)
        return self.classifier(emb)

    def extract_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_type)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph" and batch is not None:
            x = global_mean_pool(x, batch)
        return x


def build_gnn_model(
    model_type: str,
    in_channels: int,
    out_channels: int = 2,
    hidden_channels: int = 64,
    num_layers: int = 3,
    dropout: float = 0.3,
    task: str = "node",
    **kwargs: Any,
) -> nn.Module:
    """Factory function for GNN model dispatch.

    Args:
        model_type: One of 'gcn', 'gat', 'gin', 'gine', 'mpnn', 'rgcn'.
            'gin' is the recommended default for homogeneous graph
            workloads (provably matches 1-WL expressiveness, unlike
            GCN/GAT). 'gine' adds edge-feature conditioning. 'rgcn' is
            recommended for multi-relational knowledge graphs (Hetionet,
            drug-target, ontologies).
        in_channels: Number of input node features.
        out_channels: Number of output classes.
        hidden_channels: Hidden layer dimension.
        num_layers: Number of message-passing layers.
        dropout: Dropout rate.
        task: 'node' for node classification, 'graph' for graph classification.
        **kwargs: Model-specific args (heads for GAT, edge_dim for MPNN/GINE,
                  num_relations for RGCN, edge_dropout/pair_norm_scale/train_eps
                  for GIN/GINE).
    """
    if model_type == "gcn":
        return GCNModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            task=task,
        )
    elif model_type == "gat":
        return GATModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=kwargs.get("heads", 4),
            dropout=dropout,
            task=task,
        )
    elif model_type == "gin":
        return GINModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            edge_dropout=kwargs.get("edge_dropout", 0.1),
            pair_norm_scale=kwargs.get("pair_norm_scale", 1.0),
            train_eps=kwargs.get("train_eps", True),
            task=task,
        )
    elif model_type == "gine":
        return GINEModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=kwargs.get("edge_dim", 4),
            num_layers=num_layers,
            dropout=dropout,
            edge_dropout=kwargs.get("edge_dropout", 0.1),
            pair_norm_scale=kwargs.get("pair_norm_scale", 1.0),
            train_eps=kwargs.get("train_eps", True),
            task=task,
        )
    elif model_type == "mpnn":
        return MPNNModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=kwargs.get("edge_dim", 4),
            num_layers=num_layers,
            dropout=dropout,
            task=task,
        )
    elif model_type == "rgcn":
        return RGCNModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=kwargs.get("num_relations", 10),
            num_layers=num_layers,
            dropout=dropout,
            task=task,
        )
    else:
        raise ValueError(
            f"Unknown GNN model type: {model_type!r}. "
            f"Expected one of: gcn, gat, gin, gine, mpnn, rgcn"
        )
