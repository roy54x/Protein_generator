import os
from colabfold.download import download_alphafold_params
from pathlib import Path

# Force download
cache_dir = Path.home() / "AppData" / "Local" / "colabfold" / "colabfold" / "Cache"
cache_dir.mkdir(parents=True, exist_ok=True)

print("Downloading AlphaFold model parameters to:", cache_dir)
download_alphafold_params(model_type="alphafold2", data_dir=cache_dir)