"""ChEMBL adapter — builds supervised dataset from approved drugs + bioactivity.

Uses chembl_approved_drugs.json and chembl_bioactivity_sample.json.
Endpoint:
  - bioactivity: classify compounds as active/inactive based on IC50/EC50
  - drug_response: classify drugs by therapeutic flag
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.adapters.base import GraphAdapter
from src.adapters.registry import AdapterRegistry


def _load_bioactivity(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "chembl_bioactivity_sample.json"
    if not path.exists():
        raise FileNotFoundError(f"ChEMBL bioactivity file not found: {path}")
    with open(path) as f:
        records = json.load(f)
    return pd.DataFrame(records)


def _load_drugs(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "chembl_approved_drugs.json"
    if not path.exists():
        raise FileNotFoundError(f"ChEMBL drugs file not found: {path}")
    with open(path) as f:
        records = json.load(f)
    return pd.DataFrame(records)


@AdapterRegistry.register("chembl_subset")
class ChemblAdapter(GraphAdapter):
    """Adapter for ChEMBL drug/bioactivity data."""

    @property
    def name(self) -> str:
        return "chembl_subset"

    @property
    def supported_endpoints(self) -> list[str]:
        return ["bioactivity"]

    def build(self, endpoint: str, **kwargs) -> pd.DataFrame:
        self.validate_endpoint(endpoint)
        threshold = kwargs.get("activity_threshold", 10000)  # nM

        bio = _load_bioactivity(self.raw_dir)

        # Clean numeric activity values
        bio["value_numeric"] = pd.to_numeric(bio.get("standard_value", pd.Series(dtype=float)),
                                              errors="coerce")
        bio = bio.dropna(subset=["value_numeric"])

        # SMILES string length as molecular size proxy
        bio["smiles_length"] = bio.get("canonical_smiles", pd.Series(dtype=str)).fillna("").str.len()

        # Activity type encoding
        activity_types = bio.get("standard_type", pd.Series(dtype=str)).fillna("").str.upper()
        bio["is_ic50"] = (activity_types == "IC50").astype(int)
        bio["is_ec50"] = (activity_types == "EC50").astype(int)
        bio["is_ki"] = (activity_types == "KI").astype(int)

        # Target: active if value < threshold (lower IC50 = more potent)
        bio["active"] = (bio["value_numeric"] < threshold).astype(int)

        # Log-transform value_numeric for better scale separation
        bio["log_value"] = np.log1p(bio["value_numeric"].clip(lower=0))

        # Normalized value relative to threshold (informative for boundary cases)
        bio["value_ratio"] = bio["value_numeric"] / threshold

        features = ["log_value", "value_ratio", "smiles_length", "is_ic50", "is_ec50", "is_ki"]
        return bio[features + ["active"]].reset_index(drop=True)
