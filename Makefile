.PHONY: help install repro-chembl test clean

PY := .venv/bin/python

help:
	@echo "snptx-repro-chembl — public reproducibility artifact for the SNPTX ChEMBL GCN headline result"
	@echo ""
	@echo "Targets:"
	@echo "  make install       Create .venv and install CPU-only deps (~1 min, ~200 MB download / ~1.2 GB on disk)"
	@echo "  make repro-chembl  Train + evaluate the ChEMBL GCN baseline (~15s on CPU)"
	@echo "  make test          Run GNN unit tests"
	@echo "  make clean         Remove generated artifacts (results/, .pytest_cache/, __pycache__/)"

install:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

repro-chembl:
	PYTHONPATH=$(CURDIR) $(PY) repro_chembl.py

test:
	PYTHONPATH=$(CURDIR) $(PY) -m pytest tests/ -v

clean:
	rm -rf results/ .pytest_cache/ __pycache__/ */__pycache__/ */*/__pycache__/
