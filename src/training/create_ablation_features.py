"""
Create tabular feature variants for ablation experiments.

Currently supports:
    no_funding — drops funding_rate (col 2) from the 7-feature base tensor
                 → output is a 6-feature tensor (is_post_ETF shifts to idx 5).

Kaggle usage:
    python src/training/create_ablation_features.py \\
        --input-dir /kaggle/input/<dataset-name>/features \\
        --output-dir /kaggle/working \\
        --asset MULTI --variant no_funding

Feature order in tabular_features.pt (7 cols):
    0  return_1h
    1  volume
    2  funding_rate        ← dropped in no_funding variant
    3  gdelt_econ_volume
    4  gdelt_econ_tone
    5  gdelt_conflict_volume
    6  is_post_ETF
"""

import argparse
import sys
from pathlib import Path

import torch

FUNDING_RATE_IDX = 2  # column index of funding_rate in the 7-feature base tensor


def create_no_funding(input_dir: Path, output_dir: Path, asset: str, force: bool = False) -> None:
    """Drop the funding_rate column (idx 2) from the base 7-feature tensor and save as a new file."""
    coins = ["BTC", "ETH"] if asset == "MULTI" else [asset]

    for coin in coins:
        src = input_dir / coin / "tabular_features.pt"
        out_coin_dir = output_dir / coin
        out_coin_dir.mkdir(parents=True, exist_ok=True)
        dst = out_coin_dir / "tabular_features_no_funding.pt"

        if not src.exists():
            print(f"[ERROR] Source not found: {src}", file=sys.stderr)
            sys.exit(1)

        if dst.exists() and not force:
            print(f"[SKIP] {dst} already exists — use --force to overwrite")
            continue

        tensor = torch.load(src, map_location="cpu")
        if tensor.shape[1] != 7:
            print(f"[ERROR] Expected 7 features, got {tensor.shape[1]} in {src}", file=sys.stderr)
            sys.exit(1)

        no_funding = torch.cat([
            tensor[:, :FUNDING_RATE_IDX],
            tensor[:, FUNDING_RATE_IDX + 1:],
        ], dim=1)

        torch.save(no_funding, dst)
        print(f"[OK] {coin}: {list(tensor.shape)} → {list(no_funding.shape)} → {dst}")


VARIANTS = {
    "no_funding": create_no_funding,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create tabular feature variants for ablation experiments"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing BTC/ and ETH/ subdirs with tabular_features.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save output files (default: same as --input-dir). "
             "Set to /kaggle/working when input-dir is read-only.",
    )
    parser.add_argument(
        "--asset",
        choices=["BTC", "ETH", "MULTI"],
        default="MULTI",
        help="Asset(s) to process (MULTI = BTC + ETH)",
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        required=True,
        help="Which variant to create",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    VARIANTS[args.variant](input_dir, output_dir, args.asset, args.force)
