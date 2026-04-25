import importlib, sys, pathlib, importlib.util

mods = [
    "src.adapters.chembl",
    "src.adapters.base",
    "src.adapters.registry",
    "src.models.gnn",
    "workflow.scripts.train_gnn",
    "workflow.scripts.evaluate_gnn",
]
for m in mods:
    try:
        importlib.import_module(m)
        print("OK   ", m)
    except Exception as e:
        print("FAIL ", m, "->", type(e).__name__, str(e)[:300])
        sys.exit(1)

print("--- repro orchestrator ---")
spec = importlib.util.spec_from_file_location("repro_chembl", pathlib.Path("repro_chembl.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("OK    repro_chembl loaded; ROOT=", mod.ROOT)
