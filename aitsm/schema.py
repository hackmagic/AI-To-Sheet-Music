"""JSON Schema 校验：加载 schemas/*.json 并提供轻量校验函数。

schema 校验是“契约守门人”：分析产物与谱面产物在落盘前必须通过，
否则下游（前端/导出器）会在更难排查的地方炸掉。
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .config import SCHEMA_DIR

try:  # jsonschema 属于核心依赖，但允许缺失时降级为结构检查
    import jsonschema
except ImportError:  # pragma: no cover - 只有依赖缺失时走到
    jsonschema = None  # type: ignore[assignment]


class SchemaError(ValueError):
    """数据不符合契约。"""

    def __init__(self, name: str, errors: list[str]) -> None:
        self.name = name
        self.errors = errors
        super().__init__(f"{name} 校验失败: " + "; ".join(errors[:5]))


@lru_cache(maxsize=8)
def load_schema(name: str) -> dict[str, Any]:
    path = SCHEMA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"找不到 schema: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _format_errors(validator_errors: Any) -> list[str]:
    messages: list[str] = []
    for error in validator_errors:
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        messages.append(f"{location}: {error.message}")
    return messages


def validate_analysis(payload: Any) -> list[str]:
    """校验 analysis.json，返回错误列表（空表示通过）。"""
    return _validate("analysis.schema.json", payload)


def validate_score(payload: Any) -> list[str]:
    """校验 score.json，返回错误列表（空表示通过）。"""
    return _validate("score.schema.json", payload)


def _validate(schema_name: str, payload: Any) -> list[str]:
    if jsonschema is None:
        return _fallback_validate(schema_name, payload)
    schema = load_schema(schema_name)
    validator = jsonschema.Draft202012Validator(schema)
    return _format_errors(sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path)))


def _fallback_validate(schema_name: str, payload: Any) -> list[str]:
    """jsonschema 缺失时的最低限度检查，保证主流程不中断。"""
    if not isinstance(payload, dict):
        return [f"{schema_name}: 顶层必须是对象"]
    expected = "aitsm.analysis/v1" if schema_name.startswith("analysis") else "aitsm.score/v1"
    if payload.get("schema") != expected:
        return [f"schema 字段必须为 {expected}"]
    required = (
        ("source", "audio", "beats", "key", "chords", "sections")
        if schema_name.startswith("analysis")
        else ("meta", "sections", "bars")
    )
    return [f"缺少必填字段 {key}" for key in required if key not in payload]


def require_valid(schema_name: str, payload: Any) -> None:
    """不符合契约时抛 SchemaError。"""
    errors = _validate(schema_name, payload)
    if errors:
        raise SchemaError(schema_name, errors)
