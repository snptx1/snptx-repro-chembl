"""Base adapter interfaces for all dataset modalities.

Hierarchy:
    BaseAdapter           — abstract root (name, endpoints, build, metadata)
    ├─ TabularAdapter     — row-major DataFrames (EHR, claims, pharmacovigilance, CRISPR)
    ├─ MatrixOmicsAdapter — gene×sample matrices (RNA-seq, variants, scRNA, eQTL)
    ├─ SequenceAdapter    — time-series / physiological signals (ECG, wearables)
    ├─ ImageAdapter       — medical images (radiology, histopathology)
    ├─ TextAdapter        — clinical text / literature (NLP corpora)
    ├─ GraphAdapter       — molecular graphs, PPI networks, knowledge graphs
    └─ SpatialAdapter     — spatially resolved omics (Visium, MERFISH)
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Memory safety defaults — tune for the host machine
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_MAX_ROWS: int = 500_000          # reasonable cap for any single adapter
MEMORY_WARN_THRESHOLD_MB: int = 1_500    # warn when free RAM drops below this


def available_memory_mb() -> float:
    """Return available system memory in MB (Linux only; returns inf elsewhere)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except (OSError, ValueError):
        pass
    return float("inf")


def check_memory(context: str = "") -> None:
    """Log a warning if available memory is below the safety threshold."""
    avail = available_memory_mb()
    if avail < MEMORY_WARN_THRESHOLD_MB:
        logger.warning(
            "Low memory: %.0f MB available (threshold %d MB). %s",
            avail, MEMORY_WARN_THRESHOLD_MB, context,
        )


# ═══════════════════════════════════════════════════════════════════════
# Metadata
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DatasetMetadata:
    """Audit-ready metadata produced alongside every built dataset."""

    dataset_name: str
    endpoint: str
    adapter_family: str
    n_rows: int
    n_features: int
    target_column: str
    class_distribution: dict[str, int]
    features: list[str]
    sha256: str
    built_at: str
    source_dir: str
    adapter_class: str = ""
    modality: str = "tabular"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256_df(df: pd.DataFrame) -> str:
    """Deterministic hash of a DataFrame's CSV representation."""
    h = hashlib.sha256()
    h.update(df.to_csv(index=False).encode("utf-8"))
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ═══════════════════════════════════════════════════════════════════════
# Global target-column convention
# ═══════════════════════════════════════════════════════════════════════

TARGET_COL_MAP: dict[str, str] = {
    # Clinical EHR
    "readmission": "readmitted",
    "mortality": "deceased",
    "los": "los_class",
    # Genomics / functional
    "pathogenicity": "pathogenic",
    "variant_impact": "impact_class",
    "gene_dependency": "dependent",
    "essential_gene": "essential",
    # Transcriptomics / expression
    "tissue_classification": "tissue_label",
    "cancer_subtype": "subtype",
    "differential_expression": "de_label",
    "cell_type": "cell_type_label",
    # Drug / chemical
    "toxicity": "toxic",
    "bioactivity": "active",
    "drug_response": "response_class",
    "drug_indication": "indication",
    # Text / NLP
    "specialty_classification": "specialty",
    "sentiment": "sentiment_label",
    "ner": "entity_label",
    "mesh_classification": "mesh_label",
    # Graph / network
    "link_prediction": "link_exists",
    "node_classification": "node_label",
    "edge_type": "edge_type_label",
    # Signal / timeseries
    "arrhythmia": "arrhythmia_class",
    "beat_classification": "beat_label",
    # Image
    "image_classification": "image_label",
    # Spatial
    "spatial_domain": "spatial_cluster",
    # Regulatory
    "eqtl_significance": "significant",
    # GWAS
    "trait_association": "associated",
    # Adverse events
    "serious_outcome": "serious",
    # Generic fallback
    "classification": "class_label",
}


# ═══════════════════════════════════════════════════════════════════════
# BaseAdapter — abstract root
# ═══════════════════════════════════════════════════════════════════════

class BaseAdapter(ABC):
    """Abstract base class for all dataset adapters.

    Every adapter knows how to:
    1. Load raw data from its source directory
    2. Build supervised datasets for one or more prediction endpoints
    3. Produce audit-ready metadata
    """

    adapter_family: str = "base"

    def __init__(self, raw_dir: str | Path, config: dict | None = None) -> None:
        self.raw_dir = Path(raw_dir)
        self.config = config or {}
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory does not exist: {self.raw_dir}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name matching data_registry.yaml entry."""
        ...

    @property
    @abstractmethod
    def supported_endpoints(self) -> list[str]:
        """Prediction endpoints this adapter can build."""
        ...

    @abstractmethod
    def build(self, endpoint: str, **kwargs) -> pd.DataFrame:
        """Build a supervised dataset for the given endpoint.

        Returns a DataFrame with feature columns + a target column named
        according to the TARGET_COL_MAP convention.
        """
        ...

    def validate_endpoint(self, endpoint: str) -> None:
        if endpoint not in self.supported_endpoints:
            raise ValueError(
                f"Endpoint '{endpoint}' not supported by {self.name}. "
                f"Supported: {self.supported_endpoints}"
            )

    def build_and_save(
        self,
        endpoint: str,
        output_dir: str | Path,
        **kwargs,
    ) -> tuple[pd.DataFrame, DatasetMetadata]:
        """Build dataset, write CSV + metadata JSON, return both."""
        self.validate_endpoint(endpoint)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.build(endpoint, **kwargs)
        target_col = self._target_column(endpoint)
        feature_cols = [c for c in df.columns if c != target_col]

        meta = DatasetMetadata(
            dataset_name=self.name,
            endpoint=endpoint,
            adapter_family=self.adapter_family,
            n_rows=len(df),
            n_features=len(feature_cols),
            target_column=target_col,
            class_distribution={str(k): v for k, v in df[target_col].value_counts().items()},
            features=feature_cols,
            sha256=_sha256_df(df),
            built_at=datetime.now(UTC).isoformat(),
            source_dir=str(self.raw_dir),
            adapter_class=type(self).__name__,
            modality=self.adapter_family,
        )

        csv_path = output_dir / f"{self.name}_{endpoint}.csv"
        meta_path = output_dir / f"{self.name}_{endpoint}_meta.json"

        df.to_csv(csv_path, index=False)
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2, default=str))
        print(f"[✓] {self.name}/{endpoint}: {len(df)} rows × {len(feature_cols)} features → {csv_path}")
        return df, meta

    def _target_column(self, endpoint: str) -> str:
        return TARGET_COL_MAP.get(endpoint, endpoint)

    @staticmethod
    def select_features(
        df: pd.DataFrame,
        target_col: str,
        method: str = "mutual_info",
        top_k: int = 50,
    ) -> list[str]:
        """Select top_k features ranked by information-theoretic criterion.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with feature columns + target column.
        target_col : str
            Name of the target/label column.
        method : str
            'mutual_info' - mutual information (MI) for classification
            'mutual_info_regression' - MI for continuous targets
        top_k : int
            Number of features to select.

        Returns
        -------
        list of selected feature column names, sorted by score descending.

        Reference: Kraskov, Stogbauer & Grassberger, "Estimating Mutual
        Information", Phys. Rev. E 69, 2004.
        """
        from sklearn.feature_selection import (
            mutual_info_classif,
            mutual_info_regression,
        )

        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols]
        y = df[target_col]

        # Handle non-numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return feature_cols[:top_k]

        X_num = X[numeric_cols].fillna(0)

        if method == "mutual_info_regression":
            scores = mutual_info_regression(X_num, y, random_state=42)
        else:
            scores = mutual_info_classif(X_num, y, random_state=42)

        # Rank by score
        ranked = sorted(zip(numeric_cols, scores), key=lambda x: x[1], reverse=True)
        return [col for col, _ in ranked[:top_k]]


# ═══════════════════════════════════════════════════════════════════════
# Modality-specific base classes
# ═══════════════════════════════════════════════════════════════════════

class TabularAdapter(BaseAdapter):
    """Base for row-major tabular datasets (EHR, claims, CRISPR, pharmacovigilance)."""
    adapter_family = "tabular"


class MatrixOmicsAdapter(BaseAdapter):
    """Base for gene×sample or feature×observation matrix datasets.

    Provides helpers for loading expression/count matrices, performing
    variance filtering, and building classification tasks from metadata.
    """
    adapter_family = "matrix_omics"

    @staticmethod
    def variance_filter(df: pd.DataFrame, feature_cols: list[str], top_k: int = 500) -> list[str]:
        """Select top_k features by variance."""
        variances = df[feature_cols].var()
        return variances.nlargest(top_k).index.tolist()

    @staticmethod
    def auto_reduce_dimensions(
        df: pd.DataFrame,
        feature_cols: list[str],
        method: str = "svd",
        n_components: int | None = None,
        target_variance: float = 0.95,
    ) -> tuple[np.ndarray, dict]:
        """Reduce dimensionality of feature columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        feature_cols : list[str]
            Columns to reduce.
        method : str
            'svd' (truncated SVD) or 'randomized_svd' (Halko et al. 2011).
        n_components : int | None
            If None, auto-select to reach target_variance explained.
        target_variance : float
            Cumulative explained variance target (used when n_components is None).

        Returns
        -------
        reduced : np.ndarray of shape (n_samples, n_components)
        info : dict with 'n_components', 'explained_variance_ratio', 'method'
        """
        from sklearn.decomposition import TruncatedSVD
        from sklearn.utils.extmath import randomized_svd as _randomized_svd

        X = df[feature_cols].values.astype(np.float64)
        n_samples, n_features = X.shape

        if n_components is None:
            # Auto-select: start with min(100, n_features-1), increase if needed
            trial_k = min(100, n_features - 1, n_samples - 1)
            svd = TruncatedSVD(n_components=trial_k, random_state=42)
            svd.fit(X)
            cumvar = np.cumsum(svd.explained_variance_ratio_)
            # Find where cumulative exceeds target
            above = np.where(cumvar >= target_variance)[0]
            n_components = int(above[0] + 1) if len(above) > 0 else trial_k

        n_components = min(n_components, n_features - 1, n_samples - 1)

        if method == "randomized_svd":
            U, s, Vt = _randomized_svd(X, n_components=n_components, random_state=42)
            reduced = U * s
            total_var = np.var(X) * n_features
            explained = (s ** 2) / (X.shape[0] - 1) / total_var if total_var > 0 else s * 0
        else:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced = svd.fit_transform(X)
            explained = svd.explained_variance_ratio_

        return reduced, {
            "n_components": n_components,
            "explained_variance_ratio": explained.tolist(),
            "cumulative_variance": float(np.sum(explained)),
            "method": method,
        }


class SequenceAdapter(BaseAdapter):
    """Base for time-series / physiological signal datasets."""
    adapter_family = "sequence_timeseries"


class ImageAdapter(BaseAdapter):
    """Base for medical imaging datasets (radiology, histopathology).

    For deferred datasets, builds a manifest DataFrame listing available
    file paths and labels rather than loading pixel data.
    """
    adapter_family = "image"


class TextAdapter(BaseAdapter):
    """Base for text/NLP datasets (clinical notes, literature).

    Builds DataFrames with text columns + categorical targets.
    Downstream NLP pipelines handle tokenization.
    """
    adapter_family = "text"


class GraphAdapter(BaseAdapter):
    """Base for graph-structured datasets (molecular, PPI, knowledge graphs).

    Builds tabular edge-list/node-feature DataFrames for downstream
    GNN or link-prediction pipelines.
    """
    adapter_family = "graph"


class SpatialAdapter(BaseAdapter):
    """Base for spatially-resolved omics datasets (Visium, MERFISH)."""
    adapter_family = "spatial"
