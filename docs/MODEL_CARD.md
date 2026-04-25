# Model Card — SNPTX Clinical ML Pipeline

*Following [Mitchell et al. 2019](https://arxiv.org/abs/1810.03993) framework.*

---

## Model Details

| Field | Value |
|---|---|
| **Developer** | SNPTX Team |
| **Model date** | 2025 |
| **Model version** | Phase D (Clinical Deployment Readiness) |
| **Model type** | Multi-modal ensemble (tabular, imaging, genomic, NLP, time-series, spatial, fusion) |
| **Framework** | scikit-learn, PyTorch |
| **License** | Proprietary / Research |

### Modality Overview

| Modality | Architecture | Input Format |
|---|---|---|
| Tabular | Logistic Regression, Random Forest, XGBoost | CSV (patient demographics, labs, vitals) |
| Imaging | CNN (ResNet-based) | DICOM / PNG (histopathology, radiology) |
| Genomic | Feed-forward NN / Gradient Boosting | VCF-derived feature vectors |
| NLP | Transformer (BERT-based) | Free-text clinical notes |
| Time-series | LSTM / Temporal CNN | Longitudinal vitals, lab sequences |
| Spatial (ST) | Graph Neural Network | 10x Visium spatial transcriptomics |
| Drug Discovery | GCN/GAT + cross-drug attention (DeepDDS) | Molecular graphs (SMILES → RDKit), cell line features |
| ADMET | GIN multi-task | Molecular graphs (SMILES → RDKit) |
| Fusion | Late-fusion ensemble | Concatenated modality embeddings |

---

## Intended Use

### Primary Use Cases

- **Hospital readmission prediction** — 30-day readmission risk for discharged patients.
- **Clinical decision support** — Augmenting clinician judgment with quantitative risk scores.
- **Drug synergy prediction** — Identifying synergistic anti-cancer drug combinations from molecular structure.
- **ADMET screening** — Pre-filtering candidate compounds for drug-likeness and safety liabilities.
- **Research** — Exploratory multi-modal biomarker discovery and drug repurposing.

### Out-of-Scope Uses

- **Autonomous clinical decisions** — This model is a decision-support tool, NOT a replacement for clinical judgment.
- **Populations not represented in training data** — See Limitations below.
- **Legal or insurance determinations** — Not validated for non-clinical applications.

---

## Training Data

| Dataset | Description | Rows | Features |
|---|---|---|---|
| Synthea | Synthetic patient records (readmission task) | ~10,000 | Demographics, encounters, labs |
| Iris (development) | Fisher's Iris dataset for pipeline testing | 150 | 4 numeric features |
| ChEMBL | Bioactivity database for molecular property prediction | 2.4M+ activities | Molecular graphs, bioactivity labels |
| DrugCombDB / NCI-ALMANAC | Drug combination synergy assays | ~300K combinations | Drug pair SMILES, cell line, synergy score |
| TDC ADMET Group | Therapeutics Data Commons ADMET benchmarks | Varies by endpoint | Molecular graphs, ADMET properties |
| Hetionet | Heterogeneous biomedical knowledge graph | 47K nodes, 2.25M edges | 11 node types, 24 edge types |

Data is versioned via DVC and the `DataVersionManager` module (`src/pipeline/versioning.py`).
All data lineage is tracked in `results/data_registry.json`.

### Preprocessing

- Missing value imputation: median (numeric), mode (categorical).
- Feature normalization via `FeaturePreprocessor` (z-score standardization).
- Automatic refusal when >30% features are missing (refuse-to-predict).

---

## Evaluation Results

### Tabular (Synthea Readmission)

| Metric | Value |
|---|---|
| Accuracy | See `metrics/summary.csv` |
| AUROC | See `metrics/summary.csv` |
| F1 | See `metrics/summary.csv` |

### Fairness Metrics

Fairness audits are run via `FairnessAuditor` (`src/safety/fairness.py`):

- **Demographic parity gap** — Measured per protected attribute.
- **Equalized odds gap** — TPR/FPR disparity across subgroups.
- **Intersectional analysis** — Pairwise attribute combinations.

Reports saved to `results/fairness_report.json`.

---

## Ethical Considerations

- **Bias**: Model performance may vary across demographic subgroups. Fairness audits must be performed before deployment.
- **Privacy**: No PHI is stored in audit logs (SHA-256 hashing). HIPAA compliance enforced via `AuditLogger`.
- **Transparency**: Explainability via SHAP feature attributions and plain-language summaries (`ExplainabilityEngine`).
- **Human oversight**: All predictions include uncertainty quantification and a confidence score. Low-confidence predictions trigger refuse-to-predict.

---

## Limitations

1. **Synthetic training data** — Current models trained on Synthea data, which may not capture real-world distributional complexity.
2. **Single-site validation** — Not yet validated across multiple hospital systems.
3. **Temporal drift** — Model performance may degrade as clinical practices evolve. Drift monitoring is deployed (`DriftDetector`).
4. **Modality gaps** — Not all modalities are available for every patient. Fusion model handles missing modalities gracefully but with reduced confidence.

---

## Uncertainty & Safety

- **Conformal prediction** sets provide finite-sample coverage guarantees.
- **MC-Dropout** estimates epistemic uncertainty for neural models.
- **ECE calibration** measures reliability of predicted probabilities.
- **Refuse-to-predict** when confidence < 0.3 or >30% features missing.

See `src/safety/uncertainty.py` for implementation.

---

## Monitoring

| Component | Module | Description |
|---|---|---|
| Data drift | `src/monitoring/drift_detector.py` | PSI, KS-test, chi-squared |
| Performance | `src/monitoring/performance_tracker.py` | Rolling accuracy, F1, AUROC, latency |
| Alerting | `src/monitoring/alerting.py` | Configurable thresholds with rate limiting |

---

## Drug Synergy Model (DeepDDS — EXP-01)

### Model Details

| Field | Value |
|---|---|
| **Model name** | DeepDDS Drug Synergy Predictor |
| **Architecture** | Dual-branch GCN/GAT encoder + cross-drug attention fusion |
| **Input** | Pair of molecular graphs (SMILES → atom-bond graph via RDKit) + cell line context features |
| **Output** | Synergy score (Loewe additivity / Bliss independence) |
| **Training data** | DrugCombDB, NCI-ALMANAC (synergistic + antagonistic drug pair annotations) |
| **Evaluation** | Scaffold-split holdout; TDC Drug Combination benchmark group |
| **Target module** | `src/models/drug_synergy.py`, `src/adapters/drugcomb.py` |
| **Status** | EXP-01 — Active (E-104 Final Project) |

### Intended Use

- **Primary**: Predict whether a candidate drug pair will exhibit synergistic anti-cancer activity.
- **Research context**: Screen large combinatorial libraries for promising pairs before wet-lab validation.
- **NOT for**: Direct clinical drug combination recommendations. Synergy scores are research signals, not prescribing guidance.

### Limitations

1. **In vitro bias** — Trained on cell-line assay data; synergy in cell lines does not guarantee in vivo efficacy.
2. **Chemical space coverage** — Model generalizes within training scaffold distribution; novel scaffolds may produce unreliable predictions.
3. **Context dependency** — Synergy is cell-line specific; predictions should be contextualized to the assay system.
4. **Score calibration** — Raw synergy scores require calibration against pharmacological standards (Loewe/Bliss/HSA).

### Ethical Considerations

- Drug synergy predictions must be validated experimentally before informing any clinical decision.
- Synergy score alone does not capture toxicity, pharmacokinetics, or patient-specific factors.
- Model should never be used to bypass regulatory drug approval processes.

---

## ADMET Property Prediction Model (EXP-16)

### Model Details

| Field | Value |
|---|---|
| **Model name** | ADMET Multi-Task Predictor |
| **Architecture** | Graph Isomorphism Network (GIN) with multi-task prediction heads |
| **Input** | Molecular graph (SMILES → atom-bond graph via RDKit) |
| **Output** | Multi-endpoint predictions: Caco-2 permeability, CYP450 inhibition (1A2, 2C9, 2C19, 2D6, 3A4), hERG cardiotoxicity, hepatic clearance, BBB penetration |
| **Training data** | TDC ADMET Group benchmarks (scaffold-split) |
| **Target module** | `src/adapters/admet.py` |
| **Status** | EXP-16 — Tier 1 Planned |

### Intended Use

- **Primary**: Pre-screen candidate compounds for drug-likeness and safety liabilities before wet-lab ADMET assays.
- **Research context**: Prioritize compound libraries for experimental testing by filtering high-risk ADMET profiles.
- **NOT for**: Clinical dosing decisions, regulatory ADMET claims, or replacing in vivo pharmacokinetic studies.

### Safety Thresholds

| Endpoint | Safety Threshold | Action |
|---|---|---|
| hERG IC50 | < 10 µM | Flag as cardiotoxicity risk |
| Hepatotoxicity | Positive prediction | Require manual review |
| CYP450 inhibition | Inhibitor for ≥ 2 isoforms | Flag as DDI liability |
| BBB penetration | Context-dependent | Flag for CNS-targeted vs. peripheral drugs |

### Limitations

1. **In silico only** — Predictions are computational estimates; experimental ADMET data is always authoritative.
2. **Scaffold bias** — Performance degrades for scaffolds underrepresented in TDC training data.
3. **Multi-task trade-offs** — Shared GIN backbone may compromise performance on low-data endpoints.
4. **Species gap** — Training data is primarily human-relevant but may include cross-species assay data.

---

## Citation

```
@misc{snptx2025,
  title={SNPTX: Multi-Modal Clinical ML Pipeline},
  author={SNPTX Team},
  year={2025}
}
```
