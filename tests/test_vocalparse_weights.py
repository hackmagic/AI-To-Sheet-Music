"""VocalParse 真实权重加载的兼容性测试。

覆盖两件事：
1. 键名映射规则（纯函数，不需要 torch/模型）；
2. 真实 checkpoint 的维度事实（有本地权重时才跑，绝不联网）。

实测背景（2026-10）：VocalParse 由 **Qwen3-ASR 的 Thinker 结构**导出，
audio ``output_dim=2048``、词表 ``152101``；标准 ``transformers`` 期望
3584 / 151936，因此必须走 ``qwen_asr`` 包的建模代码 + 键重映射。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aitsm.transcribe.vocal import (
    VOCALPARSE_VOCAB_SIZE,
    _VOCALPARSE_KEY_PREFIXES,
    _remap_vocalparse_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]
VOCALPARSE = ROOT / "models" / "VocalParse"


# ---------------------------------------------------------------- 纯映射逻辑


def test_remap_audio_tower_drops_model_prefix():
    """audio 部分映射到 Thinker 的顶层属性 ``audio_tower.*``（不带 model.）。"""
    remapped = _remap_vocalparse_state_dict(
        {"thinker.audio_tower.conv2d1.weight": "W", "thinker.audio_tower.proj2.bias": "B"}
    )
    assert remapped == {"audio_tower.conv2d1.weight": "W", "audio_tower.proj2.bias": "B"}


def test_remap_language_model_keeps_model_prefix():
    """语言模型部分保留 ``model.*`` 前缀（Thinker 内部就这么命名）。"""
    remapped = _remap_vocalparse_state_dict(
        {"thinker.model.embed_tokens.weight": "E", "thinker.model.layers.0.mlp.down_proj.weight": "D"}
    )
    assert remapped["model.embed_tokens.weight"] == "E"
    assert remapped["model.layers.0.mlp.down_proj.weight"] == "D"
    # 词嵌入存在时会顺带补上 tied 的 lm_head
    assert remapped["lm_head.weight"] == "E"


def test_remap_adds_tied_lm_head():
    """checkpoint 不存 ``lm_head.weight``，必须与词嵌入共享（tied weights）。"""
    remapped = _remap_vocalparse_state_dict({"thinker.model.embed_tokens.weight": "EMB"})
    assert remapped["lm_head.weight"] == "EMB"
    assert remapped["model.embed_tokens.weight"] == "EMB"


def test_remap_does_not_override_explicit_lm_head():
    remapped = _remap_vocalparse_state_dict(
        {"thinker.model.embed_tokens.weight": "EMB", "thinker.lm_head.weight": "HEAD"}
    )
    assert remapped["lm_head.weight"] == "HEAD"


def test_remap_passes_through_unrelated_keys():
    assert _remap_vocalparse_state_dict({"something.else": 1}) == {"something.else": 1}


def test_remap_is_idempotent():
    once = _remap_vocalparse_state_dict({"thinker.model.x": 1})
    twice = _remap_vocalparse_state_dict(once)
    assert once["model.x"] == twice["model.x"] == 1


def test_prefix_table_is_documented_and_nonempty():
    assert len(_VOCALPARSE_KEY_PREFIXES) >= 2
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in _VOCALPARSE_KEY_PREFIXES)


def test_vocab_size_matches_ast_extension():
    """151936（基础） + ~165 个 AST token = 152101。"""
    assert VOCALPARSE_VOCAB_SIZE == 152101
    assert VOCALPARSE_VOCAB_SIZE > 151936


# ---------------------------------------------------------------- 真实权重


pytestmark_real = pytest.mark.skipif(
    not (VOCALPARSE / "model.safetensors").is_file(),
    reason="本地没有 VocalParse 权重（离线环境正常）",
)


@pytestmark_real
def test_real_checkpoint_keys_are_all_remapped():
    from safetensors.torch import load_file

    raw = load_file(str(VOCALPARSE / "model.safetensors"))
    remapped = _remap_vocalparse_state_dict(raw)

    # 每个原始键都被映射，且只落在两个命名空间里
    assert len(remapped) >= len(raw)
    assert all(key.startswith(("audio_tower.", "model.", "lm_head.")) for key in remapped)
    assert any(key.startswith("audio_tower.") for key in remapped)
    assert any(key.startswith("model.layers.") for key in remapped)


@pytestmark_real
def test_real_checkpoint_dimensions():
    """把实测维度钉成断言，避免以后把「权重版本差异」误当成加载 bug。"""
    from safetensors.torch import load_file

    raw = load_file(str(VOCALPARSE / "model.safetensors"))
    assert tuple(raw["thinker.audio_tower.proj2.weight"].shape) == (2048, 1024)
    assert tuple(raw["thinker.model.embed_tokens.weight"].shape)[0] == VOCALPARSE_VOCAB_SIZE
    assert tuple(raw["thinker.audio_tower.conv_out.weight"].shape)[1] == 7680


@pytestmark_real
def test_real_config_declares_thinker_architecture():
    config = json.loads((VOCALPARSE / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "qwen3_asr"
    assert "thinker_config" in config
    assert config["thinker_config"]["audio_config"]["output_dim"] == 2048


@pytestmark_real
def test_real_tokenizer_has_ast_tokens():
    """AST token（音高/时值/BPM）必须真的在扩展词表里，否则解码会得到乱码。"""
    added = json.loads((VOCALPARSE / "added_tokens.json").read_text(encoding="utf-8"))
    values = set(added) if isinstance(added, list) else set(added)
    assert any(str(item).startswith("<BPM_") for item in values)
    assert len(values) > 100
