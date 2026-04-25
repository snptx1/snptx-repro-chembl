"""snptx-repro-chembl model architectures (GNN-only carve-out).

Public reproducibility artifact. Only the GNN factory is shipped here;
fusion / NLP / vision builders live in the private snptx-core package.
"""

from src.models.gnn import build_gnn_model

__all__ = ["build_gnn_model"]
