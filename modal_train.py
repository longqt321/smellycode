import os
import modal
from pathlib import Path
import shutil

image = modal.Image.debian_slim(python_version="3.13").uv_pip_install(
    "torch", "polars", "scikit-learn", "tqdm", "numpy", "transformers", "modal", "iterative-stratification","wandb","matplotlib",
    "onnx","onnxruntime","seaborn","onnxscript"
).env({"PYTHONPATH": "/app"}).add_local_dir(Path(__file__).parent, "/app", ignore=[".git", ".env", ".venv", ".vscode"])

volume = modal.Volume.from_name("smellycode-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("smellycode-cache", create_if_missing=True)
onnx_volume = modal.Volume.from_name("smellycode-onnx", create_if_missing=True)
app = modal.App("code-smell-detection")

# Local directory for storing ONNX models downloaded from Modal
LOCAL_ONNX_DIR = Path(__file__).parent / "artifacts" / "onnx_models"


@app.function(
    image=image,
    volumes={"/mnt/data": volume,"/app/cache": cache_volume, "/artifacts": onnx_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    gpu="T4",
    timeout=7200,
)
def train_modal(cross_type="standard", deep_type="bottleneck", tiny=False,
                epochs=50, batch_size=2048, lr=1e-3, num_workers=4,
                seeds="1206", loss="bce", focal_gamma=2.0,
                asl_gamma_neg=4.0, asl_gamma_pos=1.0, cb_beta=0.9999,
                threshold_method="grid", gate_analysis=False, export_onnx=False,
                use_semantic=False, fusion_type="gated", embed_dim=128, max_length=512,
                wandb_project="smellycode-dcnv2", wandb_entity=None):
    import sys
    
    sys.argv = [
        "train.py",
        f"--cross_type={cross_type}",
        f"--deep_type={deep_type}",
        f"--epochs={epochs}",
        f"--batch_size={batch_size}",
        f"--lr={lr}",
        f"--num_workers={num_workers}",
        f"--loss={loss}",
        f"--focal_gamma={focal_gamma}",
        f"--asl_gamma_neg={asl_gamma_neg}",
        f"--asl_gamma_pos={asl_gamma_pos}",
        f"--cb_beta={cb_beta}",
        f"--threshold_method={threshold_method}",
        f"--embed_dim={embed_dim}",
        f"--max_length={max_length}",
        f"--wandb_project={wandb_project}",
        "--seed", *seeds.split(","),
    ]
    if tiny:
        sys.argv.append("--tiny")
    if gate_analysis:
        sys.argv.append("--gate_analysis")
    if export_onnx:
        sys.argv.append("--export_onnx")
    if use_semantic:
        sys.argv.append("--use_semantic")
    if fusion_type:
        sys.argv.append(f"--fusion_type={fusion_type}")
    if wandb_entity:
        sys.argv.append(f"--wandb_entity={wandb_entity}")
    from train import main
    result = main()
    
    # Commit volume changes after training completes
    if export_onnx:
        onnx_volume.commit()
        print("ONNX models committed to Modal volume 'smellycode-onnx'")
    
    return result


@app.local_entrypoint()
def train(
    cross_type: str = "gated",   # comma-separated: "standard,gated"
    deep_type: str = "bottleneck",  # comma-separated: "bottleneck,moe"
    tiny: bool = False,
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 1e-4,
    num_workers: int = 4,
    seeds: str = "1206",            # comma-separated: "1206,42,0"
    loss: str = "focal",              # "bce" | "focal" | "asl" | "cb_focal" | "cb"
    focal_gamma: float = 2.0,
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 1.0,
    cb_beta: float = 0.9999,
    threshold_method: str = "grid",  # "grid" | "bayesian" | "roc"
    gate_analysis: bool = False,
    export_onnx: bool = False,
    use_semantic: bool = False,
    fusion_type: str = "gated",     # "gated" | "late_mlp"
    embed_dim: int = 128,
    max_length: int = 512,
    wandb_project: str = "smellycode-dcnv2",
    wandb_entity: str = None,
):
    from itertools import product
    cross_types = [c.strip() for c in cross_type.split(",")]
    deep_types = [d.strip() for d in deep_type.split(",")]
    combos = list(product(cross_types, deep_types))
    print(f"Running {len(combos)} experiment(s): {combos}")
    jobs = [
        train_modal.spawn(
            cross_type=ct, deep_type=dt, tiny=tiny, epochs=epochs,
            batch_size=batch_size, lr=lr, num_workers=num_workers,
            seeds=seeds, loss=loss, focal_gamma=focal_gamma,
            asl_gamma_neg=asl_gamma_neg, asl_gamma_pos=asl_gamma_pos,
            cb_beta=cb_beta, threshold_method=threshold_method,
            gate_analysis=gate_analysis, export_onnx=export_onnx,
            use_semantic=use_semantic, fusion_type=fusion_type,
            embed_dim=embed_dim, max_length=max_length,
            wandb_project=wandb_project, wandb_entity=wandb_entity,
        )
        for ct, dt in combos
    ]
    
    # Wait for all jobs to complete
    for (ct, dt), job in zip(combos, jobs):
        job.get()
        print(f"[{ct}+{dt}] done")
    
    # Download ONNX models from Modal volume to local directory
    if export_onnx:
        print("\nDownloading ONNX models from Modal to local...")
        LOCAL_ONNX_DIR.mkdir(parents=True, exist_ok=True)
        
        # Access the onnx volume to download files
        onnx_volume.reload()
        
        # List all directories in the volume's artifacts folder
        try:
            artifacts_entries = list(onnx_volume.iterdir("/artifacts"))
            for entry in artifacts_entries:
                if entry.name.startswith("onnx_export_seed"):
                    # List files in this export directory
                    export_dir = f"/artifacts/{entry.name}"
                    files = list(onnx_volume.iterdir(export_dir))
                    for file_entry in files:
                        if file_entry.name.endswith('.onnx'):
                            # Read file from volume
                            file_content = b''
                            for chunk in onnx_volume.read_file(f"{export_dir}/{file_entry.name}"):
                                file_content += chunk
                            
                            # Save locally with unique name
                            local_target = LOCAL_ONNX_DIR / f"{entry.name}_{file_entry.name}"
                            with open(local_target, 'wb') as f:
                                f.write(file_content)
                            print(f"  Downloaded: {export_dir}/{file_entry.name} -> {local_target}")
        except Exception as e:
            print(f"Note: Could not auto-download ONNX files: {e}")
            print(f"ONNX models are stored in Modal volume 'smellycode-onnx'")
            print(f"You can access them via: modal volume get smellycode-onnx /artifacts --output {LOCAL_ONNX_DIR}")
        
        print(f"\nONNX models saved locally to: {LOCAL_ONNX_DIR}")


@app.function(image=image, volumes={"/mnt/data": volume}, timeout=600)
def analyze_modal():
    from analyze import main
    main()


@app.local_entrypoint()
def analyze():
    analyze_modal.remote()
