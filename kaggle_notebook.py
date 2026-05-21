# ============================================================
# CELL 1 — Setup: Clone repo & install dependencies
# ============================================================
import os, subprocess, sys

REPO_URL   = "https://github.com/whatsoever025/Multimodal-Cryptocurrency-Market-Sentiment-Forecasting.git"
REPO_DIR   = "/kaggle/working/crypto"
WANDB_KEY  = ""   # ← paste your W&B key here, or leave "" to skip W&B
HF_TOKEN   = os.environ.get("HF_TOKEN", "")  # auto-read from Kaggle Secret "HF_TOKEN"


# Clone / pull latest code
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(f"✓ Working dir: {os.getcwd()}")

# Install ONLY packages genuinely missing from Kaggle's environment.
# DO NOT install or upgrade: torch, transformers, datasets, huggingface_hub,
#   scikit-learn, Pillow — they are pre-installed and compiled for this GPU.
# Upgrading transformers pulls in CUDA ops not in Kaggle's pre-compiled torch
# → causes "no kernel image for device" error.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "wandb",    # not pre-installed on Kaggle
    "timm",     # not always pre-installed
], check=True)
print("✓ Dependencies installed")

# Login W&B (optional)
if WANDB_KEY:
    import wandb
    wandb.login(key=WANDB_KEY)
    print("✓ W&B logged in")
else:
    os.environ["WANDB_MODE"] = "disabled"
    print("⚠ W&B disabled (no key provided)")

# Login HF (optional)
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print("✓ HuggingFace logged in")


# ============================================================
# CELL 2 — Extract ALL features fresh (ensures alignment)
# ============================================================
import torch

# Verify GPU compatibility before starting long extraction
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version    : {torch.version.cuda}")
    # Quick sanity check
    _ = torch.zeros(1).cuda()
    print("✅ GPU tensor test passed")
else:
    print("⚠ No GPU — extraction will use CPU (slower)")

FEATURES_DIR = "/kaggle/working/features_v5"
os.makedirs(FEATURES_DIR, exist_ok=True)

print("=" * 60)
print("Extracting features (text + image + tabular + target)...")
print("Estimated time with GPU: ~25-30 minutes")
print("=" * 60)

result = subprocess.run([
    sys.executable, "-m", "src.training.extract_features",
    "--output_dir", FEATURES_DIR,
    "--force",          # ← re-extract ALL files, ignore cached
], capture_output=False)

if result.returncode != 0:
    raise RuntimeError("Feature extraction failed! Check logs above.")

# Verify shapes
import torch, json
print("\n" + "=" * 60)
print("Feature shapes verification:")
print("=" * 60)
for fname in ["text_embeddings.pt", "image_embeddings.pt",
              "tabular_features.pt", "target_scores.pt"]:
    fpath = os.path.join(FEATURES_DIR, fname)
    t = torch.load(fpath, map_location="cpu")
    print(f"  {fname}: {tuple(t.shape)}")

with open(os.path.join(FEATURES_DIR, "split_metadata.json")) as f:
    meta = json.load(f)
print(f"  split_metadata: {meta}")

# Sanity check: all must have same N
shapes = {}
for fname in ["text_embeddings.pt", "image_embeddings.pt",
              "tabular_features.pt", "target_scores.pt"]:
    t = torch.load(os.path.join(FEATURES_DIR, fname), map_location="cpu")
    shapes[fname] = t.shape[0]

if len(set(shapes.values())) == 1:
    N = list(shapes.values())[0]
    print(f"\n✅ All {N} rows aligned across 4 feature files.")
else:
    print(f"\n❌ MISALIGNMENT DETECTED: {shapes}")
    raise RuntimeError("Feature files have different row counts!")


# ============================================================
# CELL 3 — Train (3 independent models: y_baseline / y_heuristic / y_pca)
# ============================================================
print("\n" + "=" * 60)
print("Starting training pipeline...")
print("  Targets: y_baseline, y_heuristic, y_pca")
print("  Walk-forward folds: 5")
print("  Max epochs per fold: 60 (early stopping patience=7)")
print("=" * 60)

result = subprocess.run([
    sys.executable, "-m", "src.training.train",
    "--features-dir", FEATURES_DIR,
    "--num-folds", "5",
], capture_output=False)

if result.returncode != 0:
    print("\n⚠ Training exited with non-zero code. Check logs above.")
else:
    print("\n✅ Training complete!")


# ============================================================
# CELL 4 (OPTIONAL) — Push fresh features to Kaggle dataset
#   Run this only AFTER training is verified to work correctly.
#   This updates the Kaggle dataset so future sessions skip extraction.
# ============================================================
PUSH_TO_KAGGLE = False   # ← set True to upload

if PUSH_TO_KAGGLE:
    KAGGLE_USERNAME = "namkhanhng"
    KAGGLE_KEY      = "c8d35b256cd9417262f02d4970eb7a82"

    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"]      = KAGGLE_KEY

    # Copy dataset-metadata.json into features dir
    import shutil, json as _json
    meta_src = os.path.join(REPO_DIR, "data", "features", "dataset-metadata.json")
    meta_dst = os.path.join(FEATURES_DIR, "dataset-metadata.json")
    if os.path.exists(meta_src):
        shutil.copy(meta_src, meta_dst)

    subprocess.run(["pip", "install", "-q", "kaggle"], check=True)

    os.chdir(FEATURES_DIR)
    subprocess.run([
        "kaggle", "datasets", "version",
        "-p", ".",
        "-m", "v5-aligned: all 4 files extracted in single pass from same dataset (89000 rows)",
    ], check=True)
    os.chdir(REPO_DIR)
    print("✅ Pushed fresh features to Kaggle dataset!")
