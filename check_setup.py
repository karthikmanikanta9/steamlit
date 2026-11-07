"""Phase 1 check: confirms Python and the main libraries are installed."""
import sys
import importlib

print("Python:", sys.version.split()[0])

packages = [
    "streamlit", "numpy", "pandas", "scipy", "sklearn",
    "sentence_transformers", "datasets", "plotly", "psutil", "pytest",
]

failed = []
for name in packages:
    try:
        module = importlib.import_module(name)
        print(f"OK   {name:22} {getattr(module, '__version__', '')}")
    except Exception as error:
        print(f"FAIL {name:22} {error}")
        failed.append(name)

# HDBSCAN is built into scikit-learn 1.3+
try:
    from sklearn.cluster import HDBSCAN  # noqa: F401
    print("OK   sklearn HDBSCAN")
except Exception as error:
    print("FAIL sklearn HDBSCAN", error)
    failed.append("sklearn HDBSCAN")

print("\nAll good." if not failed else f"\nProblems with: {failed}")