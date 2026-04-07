"""
Download the SFT checkpoint from the Kaggle notebook output.

Usage:
  python scripts/download_checkpoint.py

What it does:
  1. Calls `kaggle kernels output wuyue22/rl-sft-research`
  2. Extracts the zip to checkpoints/qwen-sft/final/
  3. Verifies adapter_config.json is present
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

KAGGLE_USERNAME = "wuyue22"
KERNEL_SLUG     = f"{KAGGLE_USERNAME}/rl-sft-research"
DOWNLOAD_DIR    = Path("checkpoints/kaggle_output")
FINAL_DIR       = Path("checkpoints/qwen-sft/final")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def check_kaggle_auth() -> bool:
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print("[ERROR] ~/.kaggle/kaggle.json not found.")
        print("  Get your API key from https://www.kaggle.com/settings → API → Create New Token")
        return False
    with open(cred) as f:
        d = json.load(f)
    print(f"[OK] Kaggle credentials found for user: {d['username']}")
    return True


def download_kernel_output() -> Path | None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading output from kernel: {KERNEL_SLUG}")
    result = run(
        ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(DOWNLOAD_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[ERROR] kaggle kernels output failed:\n{result.stderr}")
        print("\nPossible reasons:")
        print("  1. The notebook was run interactively — it must be 'committed' (saved as a version).")
        print("     On Kaggle: click Save Version → Save & Run All, then wait for it to finish.")
        print(f"  2. Kernel slug may differ. Check your kernels at:")
        print(f"     https://www.kaggle.com/{KAGGLE_USERNAME}/code")
        print(f"     Then re-run: python scripts/download_checkpoint.py --slug <username>/<kernel-name>")
        return None
    print(result.stdout)

    # Find the downloaded zip
    zips = list(DOWNLOAD_DIR.glob("*.zip"))
    if zips:
        return zips[0]

    # Or a bare directory output
    files = list(DOWNLOAD_DIR.iterdir())
    print(f"Downloaded files: {[f.name for f in files]}")
    return DOWNLOAD_DIR


def extract_checkpoint(output_path: Path) -> bool:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.is_file() and output_path.suffix == ".zip":
        print(f"\nExtracting {output_path} → {FINAL_DIR}")
        with zipfile.ZipFile(output_path) as zf:
            # Find the adapter files inside the zip
            names = zf.namelist()
            adapter_files = [n for n in names if "adapter" in n or "tokenizer" in n or "special_tokens" in n]
            if not adapter_files:
                print("[WARN] No adapter files found in zip. Extracting everything.")
                zf.extractall(FINAL_DIR)
            else:
                for name in names:
                    zf.extract(name, FINAL_DIR)
    else:
        print(f"\nCopying files from {output_path} → {FINAL_DIR}")
        for f in output_path.rglob("*"):
            if f.is_file():
                dest = FINAL_DIR / f.relative_to(output_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)

    return verify_checkpoint()


def verify_checkpoint() -> bool:
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required if not (FINAL_DIR / f).exists()]

    # safetensors might be split into shards
    shards = list(FINAL_DIR.glob("adapter_model*.safetensors"))
    if missing and shards:
        missing = [f for f in missing if "safetensors" not in f]

    if missing:
        print(f"\n[WARN] Missing files in {FINAL_DIR}: {missing}")
        print("Contents found:")
        for f in sorted(FINAL_DIR.rglob("*")):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  {size_kb:8.1f} KB  {f.relative_to(FINAL_DIR)}")
        return False

    print(f"\n[OK] Checkpoint verified at: {FINAL_DIR}")
    for f in sorted(FINAL_DIR.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {size_kb:8.1f} KB  {f.relative_to(FINAL_DIR)}")
    return True


def main():
    # Allow overriding the kernel slug via CLI
    global KERNEL_SLUG
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--slug" and i + 1 < len(sys.argv) - 1:
            KERNEL_SLUG = sys.argv[i + 2]

    print("=" * 55)
    print("  SFT Checkpoint Downloader")
    print("=" * 55)

    if not check_kaggle_auth():
        sys.exit(1)

    output_path = download_kernel_output()
    if output_path is None:
        sys.exit(1)

    success = extract_checkpoint(output_path)
    if not success:
        sys.exit(1)

    print(f"\nCheckpoint ready for Week 2 at: {FINAL_DIR.resolve()}")


if __name__ == "__main__":
    main()
