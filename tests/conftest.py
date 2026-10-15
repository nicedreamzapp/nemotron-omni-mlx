"""Shared test setup.

Two kinds of tests live here:

* Unit tests (tests/test_*.py without "parity" in the name). No weights, no network,
  no GPU. They run anywhere MLX runs, including the Linux CPU build used in CI.
* Parity tests (tests/test_*_parity.py). They compare against NVIDIA's PyTorch reference
  and need local assets that are never committed: `reference/` (NVIDIA's HF custom code +
  tokenizer), `weights-bf16/` (the BF16 shard holding the towers), torch and transformers.
  When those are missing the tests are skipped with the reason, instead of erroring.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REF_DIR = ROOT / "reference"
BF16_SHARD = ROOT / "weights-bf16" / "model-00001-of-00017.safetensors"


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _parity_skip_reason(filename: str) -> str | None:
    """Why a parity file can't run on this machine, or None if it can."""
    missing = []
    if not REF_DIR.is_dir():
        missing.append("reference/ (NVIDIA's HF custom code + tokenizer)")
    if filename in ("test_audio_parity.py", "test_vision_parity.py") and not BF16_SHARD.exists():
        missing.append(f"weights-bf16/{BF16_SHARD.name}")
    for mod in ("torch", "transformers"):
        if not _have(mod):
            missing.append(mod)
    if not missing:
        return None
    return "parity test needs local assets: " + ", ".join(missing)


def pytest_collection_modifyitems(config, items):
    for item in items:
        name = Path(str(item.fspath)).name
        if not name.endswith("_parity.py"):
            continue
        item.add_marker(pytest.mark.parity)
        reason = _parity_skip_reason(name)
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))


# --------------------------------------------------------------------------------------
# A tiny stand-in for the HF tokenizer. Enough surface for OmniProcessor and src.omni:
# special tokens map to fixed ids, every other character maps to 1000 + ord(ch).
# --------------------------------------------------------------------------------------
class FakeTokenizer:
    SPECIAL = {
        "<image>": 18,
        "<img>": 19,
        "</img>": 20,
        "<so_embedding>": 27,
        "<so_start>": 28,
        "<so_end>": 29,
        "<|im_start|>": 10,
        "<|im_end|>": 11,
    }
    eos_token_id = 11

    def __init__(self):
        self._re = re.compile("|".join(re.escape(t) for t in sorted(self.SPECIAL, key=len, reverse=True)))
        self._inv = {v: k for k, v in self.SPECIAL.items()}
        self.chat_calls = []

    def convert_tokens_to_ids(self, token):
        # HF returns the unk id for unknown tokens; `<video>` is not a real token.
        return self.SPECIAL.get(token, 0)

    def _encode(self, text):
        ids, pos = [], 0
        for m in self._re.finditer(text):
            ids.extend(1000 + ord(c) for c in text[pos:m.start()])
            ids.append(self.SPECIAL[m.group(0)])
            pos = m.end()
        ids.extend(1000 + ord(c) for c in text[pos:])
        return ids

    def __call__(self, texts):
        ids = [self._encode(t) for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        self.chat_calls.append(messages)
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out

    def decode(self, ids, **_):
        return "".join(self._inv.get(i, chr(i - 1000) if i >= 1000 else "") for i in ids)

    def batch_decode(self, batch, **k):
        return [self.decode(x, **k) for x in batch]


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def processor(fake_tokenizer):
    from src.processing import OmniImageProcessor, OmniProcessor

    return OmniProcessor(OmniImageProcessor(), fake_tokenizer)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "parity: compares against NVIDIA's PyTorch reference; needs local weights"
    )
    config.addinivalue_line("markers", "metal: needs an Apple Silicon GPU (Metal)")
