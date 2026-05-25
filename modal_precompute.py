"""
Modal function for pre-computing GraphCodeBERT embeddings.
Run this once before training to cache embeddings to disk.
"""
import modal
from pathlib import Path

image = modal.Image.debian_slim(python_version="3.13").uv_pip_install(
    "torch", "polars", "scikit-learn", "tqdm", "numpy", "transformers", "modal", "iterative-stratification"
).env({"PYTHONPATH": "/app"}).add_local_dir(Path(__file__).parent, "/app", ignore=[".git", ".env", ".venv", ".vscode"])

volume = modal.Volume.from_name("smellycode-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("smellycode-cache", create_if_missing=True)
app = modal.App("code-smell-precompute")


@app.function(
    image=image,
    volumes={"/mnt/data": volume, "/app/cache": cache_volume},
    gpu="T4",
    timeout=3600,
)
def precompute_embeddings(max_length: int = 512, batch_size: int = 64):
    """Pre-compute and cache GraphCodeBERT embeddings."""
    import sys
    sys.argv = [
        "precompute_embeddings.py",
        f"--max_length={max_length}",
        f"--batch_size={batch_size}",
    ]
    
    # Import and run the precomputation script
    from precompute_embeddings import main
    main()
    
    # Commit cache volume
    cache_volume.commit()
    print("✅ Cache volume committed!")


@app.local_entrypoint()
def main(
    max_length: int = 512,
    batch_size: int = 64,
):
    """
    Pre-compute GraphCodeBERT embeddings and cache them.
    
    Args:
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for embedding extraction
    """
    print("🚀 Starting pre-computation on Modal...")
    precompute_embeddings.remote(max_length=max_length, batch_size=batch_size)
    print("✅ Pre-computation complete! Cached embeddings are ready for training.")
