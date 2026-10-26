"""编配层：大模型顾问（结构化 JSON、可注入 transport、离线规则兜底）。

设计目标
--------
- **绝不联网失败崩掉**：本模块的公开函数永不抛异常。默认只有显式配置
  ``AITSM_LLM_URL`` 时才可能发起网络请求；未配置、超时、非法 JSON、字段
  不合法都会退化为确定性规则建议，并通过 ``AdvisorResult.engine="rules"``
  显式标注；
- **可测试**：构造时可注入 ``transport(prompt) -> str``（同步函数），测试用
  假 transport 模拟超时/坏 JSON，不需要任何网络；
- **只给候选池与评审意见**：返回值全部是 ``advisory=True`` 的建议，不会直接
  写入最终谱面；候选 id 必须能在 :mod:`aitsm.score.voicings` 中找到，否则丢弃。

环境变量
--------
``AITSM_LLM_URL``
    形如 ``https://host/v1/chat`` 的 HTTP 端点；未设置则不发起任何请求。
``AITSM_LLM_API_KEY`` / ``AITSM_LLM_MODEL``
    Bearer token 与模型名（可选）。请求体为 ``{"model","prompt","max_tokens"}``，
    响应优先取 ``completion`` / ``text`` / ``content`` / OpenAI choices。

结构化输出（模型必须返回 JSON）
--------------------------------
指法建议::

    {"voicings": [{"symbol": "G", "voicing_id": "G_320003", "reason": "开放把位"}]}

谱面评审::

    {"review": {"score": 0.82, "issues": ["D 到 G 有把位跳进"],
                "suggestions": ["副歌改用 drive8"]}}

节奏型建议::

    {"sections": [{"start": 0.0, "pattern_id": "ballad8", "reason": "前奏能量低"}]}

综合编配建议（:func:`advise`，jsonschema 校验）::

    {"section": "整体",
     "suggested_strumming": [{"start": 0.0, "kind": "intro",
                              "pattern_id": "ballad8", "reason": "..."}],
     "chord_substitutions": [{"original": "H13", "suggested": "G", "reason": "..."}],
     "notes": "..."}
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from aitsm.contracts import ChordEvent, Section, SymbolicScore
from aitsm.score.model import STANDARD_TUNING
from aitsm.score.strum import pattern_ids
from aitsm.score.voicings import voicings_for

from .fingering_dp import VoicingChoice, candidates_for

__all__ = [
    "ADVICE_SCHEMA",
    "AdvisorResult",
    "LLMAdvisor",
    "LLM_URL_ENV",
    "advise",
    "rule_advice",
    "rule_voicings",
    "rule_review",
    "rule_strum",
]

#: 端点环境变量名
LLM_URL_ENV = "AITSM_LLM_URL"

Transport = Callable[[str], str]

DEFAULT_TIMEOUT = 3.0
MAX_REASON = 200

#: :func:`advise` 返回结构的 jsonschema（模型输出必须通过它才算成功）
ADVICE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["section", "suggested_strumming", "chord_substitutions", "notes"],
    "properties": {
        "section": {"type": "string"},
        "suggested_strumming": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["pattern_id"],
                "properties": {
                    "pattern_id": {"type": "string", "enum": pattern_ids()},
                    "start": {"type": "number"},
                    "kind": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "chord_substitutions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["original", "suggested"],
                "properties": {
                    "original": {"type": "string"},
                    "suggested": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "notes": {"type": "string"},
    },
    "additionalProperties": True,
}


# --------------------------------------------------------------------------- #
# 结果类型
# --------------------------------------------------------------------------- #


@dataclass
class AdvisorResult:
    """一次顾问调用的结果（永远是可用的建议，不是最终决策）。"""

    engine: str
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    advisory: bool = True

    @property
    def used_llm(self) -> bool:
        return self.engine == "llm"

    def to_dict(self) -> dict[str, Any]:
        # ``data`` 理论上总是 dict，但顾问层承诺「绝不抛异常」，
        # 这里也必须兜住：非 Mapping 的资料一律降级成 {"raw": ...}，
        # 否则 to_dict 会把异常抛到编配主流程。
        payload = self.data if isinstance(self.data, Mapping) else {"raw": self.data}
        return {
            "engine": self.engine,
            "advisory": self.advisory,
            "data": dict(payload),
            "warnings": list(self.warnings),
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# JSON 抽取与校验
# --------------------------------------------------------------------------- #


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_json(raw: Any) -> dict[str, Any]:
    """从模型输出中抽出 JSON 对象；失败抛 ValueError。"""
    if isinstance(raw, Mapping):
        return dict(raw)
    text = str(raw or "").strip()
    if not text:
        raise ValueError("模型返回为空")
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型输出不含 JSON 对象")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("模型 JSON 顶层必须是对象")
    return data


def _short_text(value: Any, *, default: str = "") -> str:
    text = str(value or default).strip()
    return text[:MAX_REASON]


def _parse_voicings(
    data: Mapping[str, Any],
    symbols: Sequence[str],
    *,
    tuning: Sequence[int],
    max_candidates: int,
    warnings: list[str],
) -> dict[str, Any]:
    """校验 LLM 指法建议：候选必须真实存在且指法合法。"""
    raw_items = data.get("voicings")
    if not isinstance(raw_items, list):
        raise ValueError("voicings 必须是数组")
    pools = {
        str(symbol): [str(v.get("id") or "") for v in voicings_for(str(symbol), tuning=tuning, limit=max_candidates)]
        for symbol in symbols
    }
    out: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            warnings.append("忽略非对象的指法建议")
            continue
        symbol = str(item.get("symbol") or "").strip()
        voicing_id = str(item.get("voicing_id") or item.get("id") or "").strip()
        if symbol not in pools or voicing_id not in pools[symbol]:
            warnings.append(f"忽略未知指法建议 {symbol}/{voicing_id}")
            continue
        if not any(existing["symbol"] == symbol for existing in out):
            out.append(
                {
                    "symbol": symbol,
                    "voicing_id": voicing_id,
                    "reason": _short_text(item.get("reason"), default="模型建议"),
                }
            )
    if not out:
        raise ValueError("没有一条可用的指法建议")
    return {"voicings": out[: len(symbols)]}


def _parse_review(data: Mapping[str, Any]) -> dict[str, Any]:
    raw = data.get("review", data)
    if not isinstance(raw, Mapping):
        raise ValueError("review 必须是对象")
    try:
        score = float(raw.get("score", 0.5))
    except (TypeError, ValueError):
        score = 0.5
    issues = [str(x) for x in raw.get("issues", [])] if isinstance(raw.get("issues", []), list) else []
    suggestions = (
        [str(x) for x in raw.get("suggestions", [])]
        if isinstance(raw.get("suggestions", []), list)
        else []
    )
    result: dict[str, Any] = {
        "review": {
            "score": round(max(0.0, min(1.0, score)), 4),
            "issues": issues[:10],
            "suggestions": suggestions[:10],
        }
    }
    # ``cost_adjustments`` 是「LLM → 调权重 → 重跑 DP」闭环的输入端
    # （见 :mod:`aitsm.arrangement.replan`）。它不是常规评审字段，但**必须原样
    # 透传**，否则闭环永远拿不到建议。合法性校验留给 replan.parse_adjustment。
    adjustments = data.get("cost_adjustments")
    if not isinstance(adjustments, Mapping):
        adjustments = raw.get("cost_adjustments")
    if isinstance(adjustments, Mapping):
        result["cost_adjustments"] = dict(adjustments)
    return result


def _parse_sections(
    data: Mapping[str, Any],
    sections: Sequence[Section | Mapping[str, Any]],
    *,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    raw_items = data.get("sections")
    if not isinstance(raw_items, list):
        raise ValueError("sections 必须是数组")
    snapshots: list[dict[str, Any]] = []
    for raw in sections:
        if isinstance(raw, Section):
            snapshot = raw.to_dict()
        elif isinstance(raw, Mapping):
            snapshot = dict(raw)
        else:
            snapshot = {}
        try:
            start = float(snapshot.get("start", 0.0))
        except (TypeError, ValueError):
            start = 0.0
        snapshots.append({"start": start, "kind": str(snapshot.get("kind") or "other"), "raw": snapshot})

    valid_ids = set(pattern_ids())
    out: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        pattern_id = str(item.get("pattern_id") or "").strip()
        if pattern_id not in valid_ids:
            continue
        try:
            want_start = float(item.get("start"))
        except (TypeError, ValueError):
            continue
        match = min(snapshots, key=lambda snap: abs(snap["start"] - want_start), default=None)
        if match is None or abs(match["start"] - want_start) > 0.51:
            continue
        if any(existing["start"] == match["start"] for existing in out):
            continue
        out.append(
            {
                "start": match["start"],
                "kind": match["kind"],
                "pattern_id": pattern_id,
                "reason": _short_text(item.get("reason"), default="模型建议"),
            }
        )
    if not out:
        raise ValueError("没有一条可用的节奏型建议")
    return {"sections": out}


# --------------------------------------------------------------------------- #
# 规则兜底
# --------------------------------------------------------------------------- #


def rule_voicings(
    symbols: Sequence[str],
    *,
    tuning: Sequence[int] = STANDARD_TUNING,
    max_candidates: int = 8,
) -> dict[str, Any]:
    """确定性兜底：为每个和弦挑一个开放把位/低难度候选。"""
    out: list[dict[str, Any]] = []
    for raw in symbols or ():
        symbol = str(raw or "").strip()
        if not symbol:
            continue
        pool = candidates_for(ChordEvent(0.0, 0.0, symbol), tuning=tuning, max_candidates=max_candidates)
        if not pool:
            continue
        ranked = sorted(
            pool,
            key=lambda voicing: (
                int(voicing.get("difficulty") or 5),
                int(voicing.get("base_fret") or 0),
                -len([f for f in voicing.get("frets") or [] if f == 0]),
                str(voicing.get("id") or ""),
            ),
        )
        best = ranked[0]
        reason_bits = [f"难度 {int(best.get('difficulty') or 0)}"]
        if int(best.get("base_fret") or 0) > 0:
            reason_bits.append(f"第 {int(best.get('base_fret') or 0)} 把位")
        else:
            reason_bits.append("开放把位")
        out.append(
            {
                "symbol": symbol,
                "voicing_id": str(best.get("id") or ""),
                "reason": "规则兜底：" + "、".join(reason_bits),
            }
        )
    return {"voicings": out}


def rule_review(
    choices: Sequence[VoicingChoice | Mapping[str, Any]],
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """确定性兜底评审：用成本模型给出分数与问题清单。"""
    from .fingering_dp import COST_WEIGHTS, plan_cost

    weight_map = dict(COST_WEIGHTS)
    weight_map.update({k: float(v) for k, v in dict(weights or {}).items() if k in weight_map})
    normalized: list[VoicingChoice] = []
    for item in choices or ():
        if isinstance(item, VoicingChoice):
            normalized.append(item)
        elif isinstance(item, Mapping):
            frets = item.get("frets") or []
            base_fret = item.get("base_fret")
            if base_fret is None:
                played = [int(f) for f in frets if isinstance(f, int) and f > 0]
                base_fret = min(played) if played else 0
            chord = ChordEvent(0.0, 0.0, str(item.get("symbol") or ""))
            normalized.append(
                VoicingChoice(
                    chord=chord,
                    voicing_id=str(item.get("voicing_id") or item.get("id") or ""),
                    frets=[int(f) for f in frets],
                    base_fret=int(base_fret or 0),
                )
            )
    if not normalized:
        return {"review": {"score": 0.0, "issues": ["缺少可评审的指法"], "suggestions": []}}

    total = plan_cost(normalized, weights=weight_map)
    issues: list[str] = []
    unplayable = [c for c in normalized if not c.frets]
    if unplayable:
        issues.append(f"{len(unplayable)} 个和弦没有合法指法（已按占位处理）")
    shifts = [
        abs(b.base_fret - a.base_fret)
        for a, b in zip(normalized, normalized[1:])
        if a.base_fret or b.base_fret
    ]
    big_shifts = sum(1 for delta in shifts if delta >= 4)
    if big_shifts:
        issues.append(f"{big_shifts} 处把位跳进 ≥ 4 品")
    average = total / max(1, len(normalized))
    score = 1.0 / (1.0 + average / 3.0)
    if unplayable:
        score *= 0.6
    suggestions: list[str] = []
    if big_shifts:
        suggestions.append("考虑换用低把位或横按形状减少跳进")
    if unplayable:
        suggestions.append("检查这些和弦的拼写，或人工指定可用指法")
    return {
        "review": {
            "score": round(max(0.0, min(1.0, score)), 4),
            "issues": issues,
            "suggestions": suggestions,
        }
    }


def rule_strum(
    sections: Sequence[Section | Mapping[str, Any]],
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """确定性兜底：复用 :func:`aitsm.arrangement.strumming_rules.recommend`。"""
    from .strumming_rules import recommend

    decisions = recommend(sections, meta)
    return {
        "sections": [
            {
                "start": item.section.get("start", 0.0),
                "kind": item.section.get("kind", "other"),
                "pattern_id": item.pattern_id,
                "reason": item.reason,
            }
            for item in decisions
        ]
    }


# --------------------------------------------------------------------------- #
# 综合建议（advise）：SymbolicScore 摘要 -> 结构化建议
# --------------------------------------------------------------------------- #


def _advice_view(subject: SymbolicScore | Mapping[str, Any]) -> dict[str, Any]:
    """从 SymbolicScore 或纯数据摘要里提取顾问需要的只读视图。"""
    if isinstance(subject, SymbolicScore):
        return {
            "chords": [chord.symbol for chord in subject.chords],
            "sections": [section.to_dict() for section in subject.sections],
            "key": subject.key.to_dict(),
            "meta": dict(subject.meta or {}),
            "time_signature": subject.beats.time_signature,
        }
    if not isinstance(subject, Mapping):
        raise TypeError("advise 只接受 SymbolicScore 或其纯数据摘要")
    data = dict(subject)
    chords: list[str] = []
    for item in data.get("chords") or []:
        if isinstance(item, ChordEvent):
            symbol = item.symbol
        elif isinstance(item, Mapping):
            symbol = str(item.get("symbol") or "")
        else:
            symbol = str(item or "")
        if symbol.strip():
            chords.append(symbol.strip())
    sections: list[dict[str, Any]] = []
    for item in data.get("sections") or []:
        if isinstance(item, Section):
            sections.append(item.to_dict())
        elif isinstance(item, Mapping):
            sections.append(dict(item))
    meta = dict(data.get("meta") or {})
    key = data.get("key") if isinstance(data.get("key"), Mapping) else {}
    beats = data.get("beats") if isinstance(data.get("beats"), Mapping) else {}
    return {
        "chords": chords,
        "sections": sections,
        "key": dict(key),
        "meta": meta,
        "time_signature": str(meta.get("time_signature") or beats.get("time_signature") or "4/4"),
    }


def _parse_advice(data: Mapping[str, Any]) -> dict[str, Any]:
    """jsonschema 校验并规整 :func:`advise` 的模型输出。"""
    try:
        import jsonschema
    except ImportError as exc:  # 依赖缺失也走规则兜底，绝不让编配崩掉
        raise RuntimeError("jsonschema 未安装，无法校验顾问输出") from exc

    payload = dict(data)
    jsonschema.validate(instance=payload, schema=ADVICE_SCHEMA)

    substitutions: list[dict[str, str]] = []
    for item in payload.get("chord_substitutions") or []:
        original = str(item.get("original") or "").strip()
        suggested = str(item.get("suggested") or "").strip()
        if not original or not suggested:
            continue
        substitutions.append(
            {
                "original": original,
                "suggested": suggested,
                "reason": _short_text(item.get("reason"), default="模型建议"),
            }
        )

    strumming: list[dict[str, Any]] = []
    for item in payload.get("suggested_strumming") or []:
        pattern_id = str(item.get("pattern_id") or "").strip()
        if pattern_id not in set(pattern_ids()):
            continue
        try:
            start = float(item.get("start", 0.0))
        except (TypeError, ValueError):
            start = 0.0
        strumming.append(
            {
                "start": round(start, 6),
                "kind": str(item.get("kind") or "other"),
                "pattern_id": pattern_id,
                "reason": _short_text(item.get("reason"), default="模型建议"),
            }
        )

    section = str(payload.get("section") or "").strip()[:MAX_REASON]
    notes = str(payload.get("notes") or "").strip()[:1000]
    return {
        "section": section,
        "suggested_strumming": strumming[:20],
        "chord_substitutions": substitutions[:20],
        "notes": notes,
    }


def _root_name(symbol: str) -> str:
    """``"c#m7" -> "C#"``；无法识别时返回空串。"""
    match = re.match(r"^\s*([A-Ga-g][#b]?)", str(symbol or ""))
    if not match:
        return ""
    token = match.group(1)
    return token[0].upper() + token[1:]


def rule_advice(view: Mapping[str, Any], meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """确定性兜底建议：节奏型用规则选，无法弹的和弦退化为大三和弦。"""
    from .strumming_rules import recommend

    meta_map = dict(meta or {})
    sections = list(view.get("sections") or [])
    decisions = recommend(sections, meta_map)

    substitutions: list[dict[str, str]] = []
    for symbol in view.get("chords") or []:
        if candidates_for(ChordEvent(0.0, 0.0, str(symbol))):
            continue
        root = _root_name(str(symbol))
        if root and root != symbol:
            substitutions.append(
                {
                    "original": str(symbol),
                    "suggested": root,
                    "reason": "库中没有该和弦的合法指法，建议退化为大三和弦",
                }
            )

    kinds: list[str] = []
    for item in decisions:
        kind = str(item.section.get("kind") or "other")
        if kind not in kinds:
            kinds.append(kind)
    notes = (
        f"规则兜底：{len(view.get('chords') or [])} 个和弦、{len(sections)} 个段落；"
        "建议仅作候选，不影响确定性谱面。"
    )
    return {
        "section": " / ".join(kinds) if kinds else "整体编配",
        "suggested_strumming": [
            {
                "start": item.section.get("start", 0.0),
                "kind": item.section.get("kind", "other"),
                "pattern_id": item.pattern_id,
                "reason": item.reason,
            }
            for item in decisions
        ],
        "chord_substitutions": substitutions[:20],
        "notes": notes,
    }


def _advice_prompt(view: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    chords = ", ".join(view.get("chords") or []) or "（无）"
    section_lines = []
    for section in view.get("sections") or []:
        section_lines.append(
            f"- start={section.get('start',0)} kind={section.get('kind','other')} "
            f"energy={section.get('energy',0.5)}"
        )
    return (
        "你是吉他编配顾问。请基于符号谱摘要给出建议，只返回一个 JSON 对象，不要解释。\n"
        "字段必须齐全：\n"
        '{"section":"整体或段落说明",'
        '"suggested_strumming":[{"start":0.0,"kind":"intro","pattern_id":"ballad8","reason":"..."}],'
        '"chord_substitutions":[{"original":"H13","suggested":"G","reason":"..."}],'
        '"notes":"..."}\n'
        f"可用 pattern_id: {', '.join(pattern_ids())}\n"
        f"调性: {view.get('key', {}).get('tonic', '未指定')}；拍号: {view.get('time_signature', '4/4')}；"
        f"风格: {meta.get('style', 'pop')}；难度: {meta.get('difficulty', 'intermediate')}\n"
        f"和弦序列: {chords}\n"
        "段落:\n" + ("\n".join(section_lines) if section_lines else "- （无段落信息）")
    )


# --------------------------------------------------------------------------- #
# 默认 HTTP transport（仅显式配置时才会用）
# --------------------------------------------------------------------------- #


def _default_transport(prompt: str) -> str:
    """POST 到 ``AITSM_LLM_URL``；未配置时直接报错（由上层降级）。"""
    url = (os.environ.get(LLM_URL_ENV) or "").strip()
    if not url:
        raise RuntimeError(f"未配置 {LLM_URL_ENV}，使用规则建议")
    import requests  # 延迟导入：离线环境不触发

    headers = {"Content-Type": "application/json"}
    api_key = (os.environ.get("AITSM_LLM_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": (os.environ.get("AITSM_LLM_MODEL") or "claude-3-5-sonnet-latest").strip(),
        "prompt": prompt,
        "max_tokens": 600,
    }
    response = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    body = response.json()
    if isinstance(body, Mapping):
        for key in ("completion", "text", "content", "output"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(first.get("text"), str):
                    return first["text"]
        raise RuntimeError("LLM 响应中没有可识别的文本字段")
    raise RuntimeError("LLM 响应不是 JSON 对象")


# --------------------------------------------------------------------------- #
# 顾问
# --------------------------------------------------------------------------- #


class LLMAdvisor:
    """大模型顾问。

    参数
    ----
    transport:
        ``transport(prompt) -> str``。``None`` 表示使用默认 HTTP（仅当设置了
        ``AITSM_LLM_URL`` 时才联网），否则直接规则兜底。
    timeout:
        调用 transport 的硬超时（秒）；``None`` 表示不限制。
    """

    def __init__(
        self,
        transport: Transport | None = None,
        *,
        timeout: float | None = DEFAULT_TIMEOUT,
        max_candidates: int = 8,
    ) -> None:
        self._transport = transport
        self._timeout = timeout
        self._max_candidates = max(1, int(max_candidates))

    # -- 基础设施 -------------------------------------------------------- #

    @property
    def configured(self) -> bool:
        return self._resolve_transport() is not None

    def _resolve_transport(self) -> Transport | None:
        if self._transport is not None:
            return self._transport
        if (os.environ.get(LLM_URL_ENV) or "").strip():
            return _default_transport
        return None

    def _invoke(self, prompt: str) -> Any:
        transport = self._resolve_transport()
        if transport is None:
            raise RuntimeError(f"未配置 LLM transport（{LLM_URL_ENV} 为空）")
        timeout = self._timeout
        if timeout is None:
            return transport(prompt)

        box: dict[str, Any] = {}

        def _run() -> None:
            try:
                box["value"] = transport(prompt)
            except BaseException as exc:  # noqa: BLE001 - 需要把异常带回主线程
                box["error"] = exc

        worker = threading.Thread(target=_run, name="aitsm-llm-advisor", daemon=True)
        worker.start()
        worker.join(max(0.05, float(timeout)))
        if worker.is_alive():
            raise TimeoutError(f"LLM 调用超过 {timeout:g}s")
        if "error" in box:
            error = box["error"]
            if isinstance(error, BaseException):
                raise error
            raise RuntimeError(str(error))
        value = box.get("value")
        return "" if value is None else value

    def _ask(
        self,
        prompt: str,
        parser: Callable[[Mapping[str, Any], list[str]], dict[str, Any]],
        fallback: Callable[[], dict[str, Any]],
        *,
        kind: str,
    ) -> AdvisorResult:
        warnings: list[str] = []
        if self._resolve_transport() is None:
            data = self._fallback(fallback)
            warnings.append(f"未配置 LLM（{LLM_URL_ENV} 为空），已使用规则建议")
            return AdvisorResult(engine="rules", data=data, warnings=warnings)
        try:
            raw = self._invoke(prompt)
            parsed = parser(_extract_json(raw), warnings)
            if not parsed:
                raise ValueError("解析结果为空")
            return AdvisorResult(engine="llm", data=parsed, warnings=warnings)
        except BaseException as exc:  # noqa: BLE001 - 公开函数绝不抛异常
            if isinstance(exc, KeyboardInterrupt):
                raise
            data = self._fallback(fallback)
            warnings.append(f"{kind} LLM 不可用（{type(exc).__name__}: {exc}），已降级为规则建议")
            return AdvisorResult(
                engine="rules", data=data, warnings=warnings, error=f"{type(exc).__name__}: {exc}"
            )

    @staticmethod
    def _fallback(fallback: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return dict(fallback() or {})
        except Exception:  # pragma: no cover - 兜底自身失败也不抛
            return {}

    # -- 公开能力 -------------------------------------------------------- #

    def suggest_voicings(
        self,
        chords: Sequence[ChordEvent | str],
        *,
        key: Any = None,
        style: str = "pop",
        tuning: Sequence[int] = STANDARD_TUNING,
    ) -> AdvisorResult:
        """指法建议（候选池）；失败时退回规则建议。永不抛异常。"""
        symbol_list: list[str] = []
        for item in chords or ():
            if isinstance(item, ChordEvent):
                symbol = item.symbol
            elif isinstance(item, Mapping):
                symbol = str(item.get("symbol") or "")
            else:
                symbol = str(item or "")
            symbol = symbol.strip()
            if symbol and symbol not in symbol_list:
                symbol_list.append(symbol)

        def fallback() -> dict[str, Any]:
            return rule_voicings(symbol_list, tuning=tuning, max_candidates=self._max_candidates)

        def parser(data: Mapping[str, Any], warnings: list[str]) -> dict[str, Any]:
            return _parse_voicings(
                data,
                symbol_list,
                tuning=tuning,
                max_candidates=self._max_candidates,
                warnings=warnings,
            )

        key_name = getattr(key, "tonic", key) if key is not None else ""
        prompt = (
            "你是吉他编配顾问。下面是一段和弦进行，请为每个和弦选择一个真实可弹的指法。\n"
            f"调性: {key_name or '未指定'}；风格: {style}。\n"
            "只返回 JSON，不要解释："
            '{"voicings":[{"symbol":"G","voicing_id":"G_320003","reason":"..."}]}\n'
            "voicing_id 只能从给定候选中选择：\n"
        )
        for symbol in symbol_list:
            ids = [str(v.get("id") or "") for v in voicings_for(symbol, tuning=tuning, limit=self._max_candidates)]
            prompt += f"- {symbol}: {', '.join(ids) if ids else '（无候选）'}\n"
        return self._ask(prompt, parser, fallback, kind="指法建议")

    def review_plan(
        self,
        choices: Sequence[VoicingChoice | Mapping[str, Any]],
        *,
        key: Any = None,
        meta: Mapping[str, Any] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> AdvisorResult:
        """谱面评审意见；失败时用成本模型兜底。永不抛异常。"""
        serialized = []
        for item in choices or ():
            if isinstance(item, VoicingChoice):
                serialized.append(item.to_dict())
            elif isinstance(item, Mapping):
                serialized.append(dict(item))
        key_name = getattr(key, "tonic", key) if key is not None else ""
        meta_map = dict(meta or {})

        def fallback() -> dict[str, Any]:
            return rule_review(choices, weights=weights)

        def parser(data: Mapping[str, Any], _warnings: list[str]) -> dict[str, Any]:
            return _parse_review(data)

        lines = [
            f"- {item.get('symbol','')}: {item.get('voicing_id','')} frets={item.get('frets',[])} base={item.get('base_fret',0)}"
            for item in serialized
        ]
        prompt = (
            "你是吉他编配评审。请评审下面的指法序列（把位是否连贯、是否好弹），"
            "给出 0~1 分数、问题与建议。只返回 JSON：\n"
            '{"review":{"score":0.8,"issues":[],"suggestions":[]}}\n'
            f"调性: {key_name or '未指定'}；拍号: {meta_map.get('time_signature','4/4')}；风格: {meta_map.get('style','pop')}\n"
            + "\n".join(lines)
        )
        return self._ask(prompt, parser, fallback, kind="谱面评审")

    def suggest_strumming(
        self,
        sections: Sequence[Section | Mapping[str, Any]],
        meta: Mapping[str, Any] | None = None,
    ) -> AdvisorResult:
        """按段落建议节奏型；失败时用确定性规则兜底。永不抛异常。"""
        meta_map = dict(meta or {})
        section_list = list(sections or ())

        def fallback() -> dict[str, Any]:
            return rule_strum(section_list, meta_map)

        def parser(data: Mapping[str, Any], _warnings: list[str]) -> dict[str, Any]:
            return _parse_sections(data, section_list, meta=meta_map)

        described = []
        for raw in section_list:
            snapshot = raw.to_dict() if isinstance(raw, Section) else dict(raw or {})
            described.append(
                f"- start={snapshot.get('start',0)} kind={snapshot.get('kind','other')} "
                f"energy={snapshot.get('energy',0.5)}"
            )
        prompt = (
            "你是吉他编配顾问。请为每个段落选择节奏型。只返回 JSON：\n"
            '{"sections":[{"start":0.0,"pattern_id":"ballad8","reason":"..."}]}\n'
            f"可用 pattern_id: {', '.join(pattern_ids())}\n"
            f"拍号: {meta_map.get('time_signature','4/4')}；风格: {meta_map.get('style','pop')}；"
            f"难度上限: {meta_map.get('difficulty','intermediate')}\n"
            + "\n".join(described)
        )
        return self._ask(prompt, parser, fallback, kind="节奏型建议")

    def advise(
        self,
        symbolic_or_summary: SymbolicScore | Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """综合编配建议（任务约定的公开契约）。

        输入 :class:`SymbolicScore` 或其纯数据摘要；返回 JSON 化 dict：:

            {"engine": "llm"|"rules", "advisory": True,
             "section": str,
             "suggested_strumming": [{"start","kind","pattern_id","reason"}],
             "chord_substitutions": [{"original","suggested","reason"}],
             "notes": str, "warnings": [...]}

        LLM 输出经 :data:`ADVICE_SCHEMA` 的 jsonschema 校验；任何失败（未配置、
        超时、坏 JSON、字段不合法）都降级为规则建议。**永不抛异常**，且建议
        不会写入最终谱面。
        """
        try:
            view = _advice_view(symbolic_or_summary)
        except BaseException as exc:  # noqa: BLE001 - 公开函数绝不抛异常
            if isinstance(exc, KeyboardInterrupt):
                raise
            return {
                "engine": "rules",
                "advisory": True,
                "section": "",
                "suggested_strumming": [],
                "chord_substitutions": [],
                "notes": "输入无法解析为符号谱摘要，未生成建议",
                "warnings": [f"输入无法解析（{type(exc).__name__}: {exc}）"],
            }

        meta_map = dict(view.get("meta") or {})
        meta_map.update({str(key): value for key, value in dict(config or {}).items()})
        meta_map.setdefault("time_signature", view.get("time_signature", "4/4"))

        def fallback() -> dict[str, Any]:
            return rule_advice(view, meta_map)

        def parser(data: Mapping[str, Any], _warnings: list[str]) -> dict[str, Any]:
            return _parse_advice(data)

        result = self._ask(
            _advice_prompt(view, meta_map), parser, fallback, kind="编配建议"
        )
        payload: dict[str, Any] = {
            "engine": result.engine,
            "advisory": True,
            "section": "",
            "suggested_strumming": [],
            "chord_substitutions": [],
            "notes": "",
        }
        payload.update(result.data or {})
        payload["warnings"] = list(result.warnings)
        if result.error:
            payload["error"] = result.error
        return payload


def advise(
    symbolic_or_summary: SymbolicScore | Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    transport: Transport | None = None,
) -> dict[str, Any]:
    """模块级入口：给定符号谱（或摘要）返回结构化建议 dict。

    ``transport`` 可注入假实现用于测试；为 ``None`` 时只在设置了
    ``AITSM_LLM_URL`` 的情况下才可能发起网络请求。任何异常都转为
    ``engine="rules"`` 的兜底建议，**绝不抛出**。
    """
    try:
        return LLMAdvisor(transport=transport).advise(symbolic_or_summary, config=config)
    except BaseException as exc:  # noqa: BLE001 - 公开函数绝不抛异常
        if isinstance(exc, KeyboardInterrupt):
            raise
        return {
            "engine": "rules",
            "advisory": True,
            "section": "",
            "suggested_strumming": [],
            "chord_substitutions": [],
            "notes": "顾问初始化失败，未生成建议",
            "warnings": [f"顾问初始化失败（{type(exc).__name__}: {exc}）"],
        }
