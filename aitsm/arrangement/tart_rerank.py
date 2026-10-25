"""编配层：TART 指法重排（可选）与确定性兜底微调。

背景
----
TART 是「Tablature-Aware Regression Transformer」类的指法重排模型：给和弦进行
与候选形状，输出更符合真实演奏习惯的排序。本机 **transformers 未安装、无网络**，
因此本模块的职责是：

1. :func:`is_available` 只在「transformers 可导入」且 ``AITSM_TART_DIR`` 指向
   存在的本地权重目录时才返回 True —— **绝不联网下载**；
2. 可用时走本地 T5 适配器（``local_files_only=True``），把模型输出的候选 id
   当作重排先验；
3. 不可用、加载失败或输出异常时，退回**确定性启发式**：在候选池里做两轮坐标
   下降，按 DP 的成本模型（手跨 / 空弦 / 把位 / 切换）局部改选，只接受让总
   成本下降的改动；
4. 无论走哪条路，都用 ``VoicingChoice.engine`` 标注真实生效的实现
   （``tart-t5`` / ``tart-heuristic``）。

重排只调整**候选选择**，不会凭空创造指法：候选始终来自
:func:`aitsm.score.voicings.voicings_for`。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Sequence

from aitsm.contracts import ChordEvent
from aitsm.score.model import STANDARD_TUNING

from .fingering_dp import (
    COST_WEIGHTS,
    VoicingChoice,
    assign_path_costs,
    candidates_for,
    choice_from_voicing,
    plan_cost,
)

__all__ = ["TART_ENV", "is_available", "rerank", "model_dir", "active_engine"]

#: 本地 TART 权重目录的环境变量名
TART_ENV = "AITSM_TART_DIR"

#: 重排引擎名
ENGINE_T5 = "tart-t5"
ENGINE_HEURISTIC = "tart-heuristic"


def model_dir() -> Path | None:
    """``AITSM_TART_DIR`` 指向的目录；未设置或不存在时返回 None。"""
    raw = (os.environ.get(TART_ENV) or "").strip()
    if not raw:
        return None
    directory = Path(raw).expanduser()
    try:
        if directory.is_dir():
            return directory
    except OSError:  # pragma: no cover - 权限异常
        return None
    return None


def _transformers_available() -> bool:
    """transformers 是否已安装（只查 spec，不 import、不联网）。"""
    try:
        return importlib.util.find_spec("transformers") is not None
    except (ImportError, ValueError):  # pragma: no cover - 依赖损坏
        return False


def is_available() -> bool:
    """TART 是否可用：transformers 已安装 + 本地权重目录存在且非空。"""
    if not _transformers_available():
        return False
    directory = model_dir()
    if directory is None:
        return False
    try:
        return any(directory.iterdir())
    except OSError:  # pragma: no cover
        return False


def active_engine() -> str:
    """当前机器上 rerank 实际会用的引擎名（不加载模型）。"""
    return ENGINE_T5 if is_available() else ENGINE_HEURISTIC


# --------------------------------------------------------------------------- #
# 确定性启发式
# --------------------------------------------------------------------------- #


def _rerank_heuristic(
    chords: Sequence[ChordEvent],
    choices: Sequence[VoicingChoice],
    *,
    tuning: Sequence[int],
    max_candidates: int,
    melody_notes: Sequence[Any],
    weights: dict[str, float],
    pool_order: Sequence[Sequence[str]] | None = None,
) -> list[VoicingChoice]:
    """两轮坐标下降：只接受降低总成本的候选替换。

    ``pool_order`` 可选，用于把 T5 模型给出的候选 id 顺序前置（作为平局先验）。
    """
    n = min(len(chords), len(choices))
    if n == 0:
        return [c for c in choices]

    pools = [
        candidates_for(chords[index], tuning=tuning, max_candidates=max_candidates)
        for index in range(n)
    ]
    if pool_order:
        reordered: list[list[dict]] = []
        for index, pool in enumerate(pools):
            priority = list(pool_order[index]) if index < len(pool_order) else []
            rank = {voicing_id: pos for pos, voicing_id in enumerate(priority)}

            def _key(item: dict, rank=rank) -> tuple[int, str]:
                return (rank.get(str(item.get("id") or ""), len(rank)), str(item.get("id") or ""))

            reordered.append(sorted(pool, key=_key))
        pools = reordered

    current = [choice for choice in choices[:n]]
    total = plan_cost(current, weights=weights)

    for _ in range(2):
        improved = False
        for index in range(n):
            best_choice: VoicingChoice | None = None
            best_total = total
            for voicing in pools[index]:
                candidate = choice_from_voicing(
                    chords[index],
                    voicing,
                    candidates_considered=len(pools[index]),
                    melody_notes=melody_notes,
                    weights=weights,
                    tuning=tuning,
                    engine=ENGINE_HEURISTIC,
                )
                trial = list(current)
                trial[index] = candidate
                trial_total = plan_cost(trial, weights=weights)
                if trial_total < best_total - 1e-9:
                    best_choice, best_total = candidate, trial_total
            if best_choice is not None:
                current[index] = best_choice
                total = best_total
                improved = True
        if not improved:
            break

    remaining = [c for c in choices[n:]]
    return assign_path_costs(
        list(current) + remaining, weights=weights, engine=ENGINE_HEURISTIC
    )


# --------------------------------------------------------------------------- #
# 可选 T5 适配器（离线）
# --------------------------------------------------------------------------- #


def _t5_prompt(chords: Sequence[ChordEvent], choices: Sequence[VoicingChoice],
               pools: Sequence[Sequence[dict]]) -> str:
    """构造给 T5 的紧凑输入：和弦序列 + 候选 id。"""
    lines: list[str] = ["Rank the guitar voicings for this chord progression."]
    for index, chord in enumerate(chords):
        ids = [
            str(item.get("id") or "")
            for item in (pools[index] if index < len(pools) else [])
        ]
        chosen = choices[index].voicing_id if index < len(choices) else ""
        lines.append(
            f"{chord.symbol} | chosen={chosen} | candidates={','.join(ids)}"
        )
    lines.append("Return the chosen ids in order, comma separated.")
    return "\n".join(lines)


def _parse_t5_order(text: str) -> list[str]:
    """从模型输出里抽出形如 ``C_x32010`` 的候选 id。"""
    import re

    return re.findall(r"[A-Za-z0-9#b]+_[0-9x]+", str(text or ""))


def _rerank_with_t5(
    chords: Sequence[ChordEvent],
    choices: Sequence[VoicingChoice],
    *,
    tuning: Sequence[int],
    max_candidates: int,
    melody_notes: Sequence[Any],
    weights: dict[str, float],
) -> list[VoicingChoice]:
    """本地 T5 重排；任何异常由调用方捕获并退回启发式。"""
    directory = model_dir()
    if directory is None:
        raise RuntimeError(f"{TART_ENV} 未指向本地权重目录")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # 惰性导入

    n = min(len(chords), len(choices))
    pools = [candidates_for(chord, tuning=tuning, max_candidates=max_candidates) for chord in chords[:n]]
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(directory), local_files_only=True)
    prompt = _t5_prompt(chords[:n], choices[:n], pools)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output = model.generate(**inputs, max_new_tokens=64, num_beams=1, do_sample=False)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    ranked = _parse_t5_order(text)
    if not ranked:  # 模型没给出可用排序：不算成功，交回启发式
        raise ValueError("TART 输出不含可识别的候选 id")

    order: list[list[str]] = [[] for _ in range(n)]
    for index, pool in enumerate(pools):
        known = {str(item.get("id") or "") for item in pool}
        order[index] = [voicing_id for voicing_id in ranked if voicing_id in known]

    result = _rerank_heuristic(
        chords[:n],
        choices[:n],
        tuning=tuning,
        max_candidates=max_candidates,
        melody_notes=melody_notes,
        weights=weights,
        pool_order=order,
    )
    return [choice for choice in result]


# --------------------------------------------------------------------------- #
# 公开入口
# --------------------------------------------------------------------------- #


def rerank(
    chords: Sequence[ChordEvent],
    choices: Sequence[VoicingChoice],
    *,
    tuning: Sequence[int] = STANDARD_TUNING,
    max_candidates: int = 8,
    melody_notes: Sequence[Any] = (),
    weights: dict[str, float] | None = None,
) -> list[VoicingChoice]:
    """重排 DP 给出的指法序列；永不抛异常。

    - 可用时尝试本地 TART（T5），成功则 ``engine="tart-t5"``；
    - 否则（或失败）走确定性启发式，``engine="tart-heuristic"``；
    - 入参长度不一致、空序列等异常输入都原样返回副本，不做猜测。
    """
    weight_map = dict(COST_WEIGHTS)
    weight_map.update({k: float(v) for k, v in dict(weights or {}).items() if k in weight_map})

    chord_list = list(chords or ())
    choice_list = list(choices or ())
    if not chord_list or not choice_list:
        return assign_path_costs(choice_list, weights=weight_map, engine=ENGINE_HEURISTIC)
    if len(chord_list) != len(choice_list):
        # 长度不一致说明上游已经出问题：不猜测，原样返回并标注引擎
        return assign_path_costs(choice_list, weights=weight_map, engine=ENGINE_HEURISTIC)

    if is_available():
        try:
            return _rerank_with_t5(
                chord_list,
                choice_list,
                tuning=tuning,
                max_candidates=max_candidates,
                melody_notes=melody_notes,
                weights=weight_map,
            )
        except Exception:  # 模型/权重/显存任何问题都不能让编配失败
            pass

    try:
        return _rerank_heuristic(
            chord_list,
            choice_list,
            tuning=tuning,
            max_candidates=max_candidates,
            melody_notes=melody_notes,
            weights=weight_map,
        )
    except Exception:  # pragma: no cover - 理论不可达
        return assign_path_costs(choice_list, weights=weight_map, engine=ENGINE_HEURISTIC)
