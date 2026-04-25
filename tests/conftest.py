"""
conftest.py — Shared test configuration.

Adds workflow/scripts and repository root to sys.path so test modules can
import config_manager, adapters, validation, etc. at module level.
"""
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "workflow" / "scripts")

# Add repo root for `src.adapters.*` imports
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Add workflow/scripts for legacy imports (config_manager, etc.)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
