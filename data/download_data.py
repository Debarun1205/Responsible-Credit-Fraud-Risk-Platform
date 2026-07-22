"""
Downloads the full datasets used by this project into data/full/.

Usage:
    pip install kagglehub
    python data/download_data.py

Requires a Kaggle account. On first run, kagglehub will prompt you to
authenticate (or read credentials from ~/.kaggle/kaggle.json / the
KAGGLE_USERNAME + KAGGLE_KEY environment variables).

This script does NOT commit the downloaded files to git — data/full/
is covered by .gitignore. Only the small samples in data/samples/ are
checked into version control.
"""

import shutil
from pathlib import Path

import kagglehub

OUTPUT_DIR = Path(__file__).parent / "full"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = {
    # Credit risk model input.
    # Full LendingClub loan data. If this exact slug is unavailable,
    # search "lending club loan data" on Kaggle and update the slug below.
    "credit_risk": "wordsforthewise/lending-club",
    # Fraud detection model input.
    "fraud": "mlg-ulb/creditcardfraud",
}


def download(name: str, slug: str) -> None:
    print(f"Downloading '{slug}' for {name}...")
    path = kagglehub.dataset_download(slug)
    dest = OUTPUT_DIR / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(path, dest)
    print(f"  -> saved to {dest}")


if __name__ == "__main__":
    for name, slug in DATASETS.items():
        try:
            download(name, slug)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed to download {name} ({slug}): {exc}")
            print("    Check the dataset slug on kaggle.com and update DATASETS above.")

    print("\nDone. Full datasets are in data/full/ (gitignored).")
    print("Small runnable samples are already in data/samples/.")
