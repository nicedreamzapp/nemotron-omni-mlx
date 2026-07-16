#!/usr/bin/env python3
"""Fetch the Nemotron Omni weights and prepare the two directories the runtime needs.

  model-4bit/  the mlx-community 4-bit checkpoint as published (~19 GB).
               Towers read their weights straight out of here.
  text-only/   a filtered copy holding only `backbone.*` + `lm_head.*` (~17.8 GB),
               with `llm_config` promoted to config.json.

Why text-only/ has to exist: mlx_lm's `load()` globs every .safetensors in a directory and
rejects the checkpoint over the unexpected `vision_model.*` / `sound_*` keys, so the LLM can't
be loaded from the full snapshot as-is. Symlinking doesn't help (same glob). A filtered copy is
the least-bad fix and costs disk, not time.

Set HF_HUB_ENABLE_HF_TRANSFER=1 before running — it's roughly 30x faster.

    python scripts/setup_weights.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ID = "mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit"
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model-4bit"
TEXT_DIR = ROOT / "text-only"
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
]


def download() -> None:
    if (MODEL_DIR / "config.json").exists():
        print(f"✓ {MODEL_DIR.name}/ already present — skipping download")
        return
    if not os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"):
        print("! HF_HUB_ENABLE_HF_TRANSFER is unset — this will be ~30x slower.")
        print("  Ctrl-C and re-run with:  export HF_HUB_ENABLE_HF_TRANSFER=1\n")
    from huggingface_hub import snapshot_download

    print(f"→ downloading {REPO_ID} (~19 GB)")
    snapshot_download(REPO_ID, local_dir=str(MODEL_DIR))
    print(f"✓ {MODEL_DIR}")


def build_text_only() -> None:
    if (TEXT_DIR / "model.safetensors").exists():
        print(f"✓ {TEXT_DIR.name}/ already built — skipping")
        return
    import mlx.core as mx

    TEXT_DIR.mkdir(exist_ok=True)
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    llm = dict(cfg["llm_config"])
    llm["quantization"] = cfg["quantization"]
    llm.setdefault("model_type", "nemotron_h")
    (TEXT_DIR / "config.json").write_text(json.dumps(llm, indent=1))

    print("→ filtering backbone weights out of the checkpoint")
    weights = {}
    for shard in sorted(MODEL_DIR.glob("*.safetensors")):
        for key, value in mx.load(str(shard)).items():
            if key.startswith(("backbone.", "lm_head.")):
                weights[key] = value
    if not weights:
        sys.exit("no backbone.* tensors found — is model-4bit/ complete?")
    mx.save_safetensors(str(TEXT_DIR / "model.safetensors"), weights)

    for name in TOKENIZER_FILES:
        src = MODEL_DIR / name
        if src.exists():
            shutil.copy(src, TEXT_DIR / name)

    size_gb = (TEXT_DIR / "model.safetensors").stat().st_size / 1e9
    print(f"✓ {TEXT_DIR} — {len(weights)} tensors, {size_gb:.1f} GB")


if __name__ == "__main__":
    download()
    build_text_only()
    print("\nReady. Try:\n  python -m src.omni --image photo.jpg --prompt 'What is this?'")
