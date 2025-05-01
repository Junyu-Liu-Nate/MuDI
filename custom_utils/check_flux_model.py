#!/usr/bin/env python3
"""Sanity‑check a local FLUX.1‑dev clone against the official file list."""
import argparse, json, sys
from pathlib import Path

REQUIRED = {
    # top‑level
    "flux1-dev.safetensors",
    "ae.safetensors",
    "model_index.json",
    # sub‑directories
    "scheduler", "text_encoder", "text_encoder_2",
    "tokenizer", "tokenizer_2", "transformer", "vae",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", help="Path to local FLUX.1‑dev clone")
    args = p.parse_args()
    root = Path(args.model_dir).expanduser().resolve()

    if not root.exists():
        sys.exit(f"[ERROR] {root} does not exist")

    present = {item.name for item in root.iterdir()}
    missing = REQUIRED - present
    extras  = present - REQUIRED

    if missing:
        print("[FAIL] Missing entries:")
        for m in sorted(missing):
            print("   └─", m)
        sys.exit(1)

    # Quick LFS‑pointer check (any *.safetensors smaller than 1 MB → LFS stub)
    bad_weights = [
        f for f in root.rglob("*.safetensors")
        if f.stat().st_size < 1_000_000          # < 1 MB means stub
    ]
    if bad_weights:
        print("[FAIL] Found Git‑LFS pointer files:")
        for f in bad_weights:
            print("   └─", f.relative_to(root))
        sys.exit(1)

    print("[OK ] All required files present & real (not LFS stubs).")
    if extras:
        print("      (Extra files ignored):", ", ".join(sorted(extras)))

if __name__ == "__main__":
    main()