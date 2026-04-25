"""SNPTX dataset adapters — pluggable data loading for multi-dataset pipelines."""

from src.adapters.base import (
    DEFAULT_MAX_ROWS,
    BaseAdapter,
    DatasetMetadata,
    GraphAdapter,
    ImageAdapter,
    MatrixOmicsAdapter,
    SequenceAdapter,
    SpatialAdapter,
    TabularAdapter,
    TextAdapter,
    available_memory_mb,
    check_memory,
)
from src.adapters.registry import AdapterRegistry

__all__ = [
    "BaseAdapter",
    "DatasetMetadata",
    "AdapterRegistry",
    "TabularAdapter",
    "MatrixOmicsAdapter",
    "SequenceAdapter",
    "ImageAdapter",
    "TextAdapter",
    "GraphAdapter",
    "SpatialAdapter",
    "DEFAULT_MAX_ROWS",
    "available_memory_mb",
    "check_memory",
]
