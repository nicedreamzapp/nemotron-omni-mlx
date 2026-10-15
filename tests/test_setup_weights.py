"""Unit tests for scripts/setup_weights.py on a fake, tiny checkpoint. No download."""
from __future__ import annotations

import importlib.util
import json
import sys
import types

import pytest

from conftest import ROOT

mx = pytest.importorskip("mlx.core")


@pytest.fixture
def sw(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("setup_weights", ROOT / "scripts" / "setup_weights.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "MODEL_DIR", tmp_path / "model-4bit")
    monkeypatch.setattr(mod, "TEXT_DIR", tmp_path / "text-only")
    return mod


def _fake_checkpoint(model_dir, with_backbone=True):
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "llm_config": {"hidden_size": 8, "num_hidden_layers": 1},
                "quantization": {"group_size": 64, "bits": 4},
                "vision_config": {},
            }
        )
    )
    shard1 = {"vision_model.a": mx.zeros((2,)), "sound_encoder.b": mx.zeros((2,)), "mlp1.0.weight": mx.zeros((2,))}
    shard2 = {"sound_projection.c": mx.zeros((2,))}
    if with_backbone:
        shard1["backbone.layers.0.w"] = mx.ones((2, 2))
        shard2["backbone.embeddings.weight"] = mx.ones((3, 2))
        shard2["lm_head.weight"] = mx.ones((3, 2))
    mx.save_safetensors(str(model_dir / "model-00001-of-00002.safetensors"), shard1)
    mx.save_safetensors(str(model_dir / "model-00002-of-00002.safetensors"), shard2)
    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "generation_config.json").write_text('{"eos_token_id": 11}')


def test_build_text_only_filters_and_promotes_config(sw, capsys):
    _fake_checkpoint(sw.MODEL_DIR)
    sw.build_text_only()

    cfg = json.loads((sw.TEXT_DIR / "config.json").read_text())
    assert cfg["hidden_size"] == 8
    assert cfg["quantization"] == {"group_size": 64, "bits": 4}
    assert cfg["model_type"] == "nemotron_h"

    weights = mx.load(str(sw.TEXT_DIR / "model.safetensors"))
    assert sorted(weights) == ["backbone.embeddings.weight", "backbone.layers.0.w", "lm_head.weight"]

    assert (sw.TEXT_DIR / "tokenizer.json").exists()
    assert (sw.TEXT_DIR / "generation_config.json").exists()
    assert not (sw.TEXT_DIR / "special_tokens_map.json").exists()  # absent upstream -> skipped
    assert "3 tensors" in capsys.readouterr().out


def test_build_text_only_keeps_existing_model_type(sw):
    _fake_checkpoint(sw.MODEL_DIR)
    cfg = json.loads((sw.MODEL_DIR / "config.json").read_text())
    cfg["llm_config"]["model_type"] = "custom"
    (sw.MODEL_DIR / "config.json").write_text(json.dumps(cfg))
    sw.build_text_only()
    assert json.loads((sw.TEXT_DIR / "config.json").read_text())["model_type"] == "custom"


def test_build_text_only_is_idempotent(sw, capsys):
    _fake_checkpoint(sw.MODEL_DIR)
    sw.build_text_only()
    capsys.readouterr()
    sw.build_text_only()
    assert "already built" in capsys.readouterr().out


def test_build_text_only_without_backbone_exits(sw):
    _fake_checkpoint(sw.MODEL_DIR, with_backbone=False)
    with pytest.raises(SystemExit, match="no backbone"):
        sw.build_text_only()


def test_download_skips_when_present(sw, capsys, monkeypatch):
    sw.MODEL_DIR.mkdir()
    (sw.MODEL_DIR / "config.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # would fail if imported
    sw.download()
    assert "skipping download" in capsys.readouterr().out


def test_download_calls_snapshot_and_warns_without_hf_transfer(sw, capsys, monkeypatch):
    calls = []
    hub = types.SimpleNamespace(snapshot_download=lambda repo, local_dir: calls.append((repo, local_dir)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    sw.download()
    assert calls == [(sw.REPO_ID, str(sw.MODEL_DIR))]
    assert "HF_HUB_ENABLE_HF_TRANSFER is unset" in capsys.readouterr().out
