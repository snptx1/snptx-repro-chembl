.PHONY: help install repro-chembl test clean ui

help:
	@echo "snptx-repro-chembl — public reproducibility artifact for the SNPTX ChEMBL GCN headline result"
	@echo ""
	@echo "Targets:"
	@echo "  make install       Create venv and install requirements"
	@echo "  make repro-chembl  Train + evaluate (≈ 4–6 min on a single GPU; ≈ 30–60 min CPU)"
	@echo "  make test          Run GNN unit tests"
	@echo "  make ui            Open MLflow UI on localhost:5000"
	@echo "  make clean         Remove generated artifacts (results/, mlruns/, *.pt)"

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

repro-chembl:
	PYTHONPATH=$(CURDIR) python repro_chembl.py

test:
	PYTHONPATH=$(CURDIR) pytest tests/ -v

ui:
	mlflow ui --backend-store-uri ./mlruns

clean:
	rm -rf results/ mlruns/ .pytest_cache/ __pycache__/ */__pycache__/ */*/__pycache__/
