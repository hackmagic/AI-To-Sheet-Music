"""编配层：和弦进行上的全局最优指法选择（Viterbi / 动态规划）。

为什么需要 DP
-------------
和弦进行里逐和弦挑「最容易按的形状」会掉进局部最优：一个横按把位可能本身
最简单，却让前后两个开放和弦都要来回换把。真实谱面追求的是**整段最低的
演奏成本**，所以用动态规划一次性求全局最优：

.. code-block:: text

    cost(路径) = Σ 局部成本(第 i 个和弦的形状)
               + Σ 切换成本(第 i-1 → 第 i 个和弦)

局部成本与切换成本严格按任务约定的五项权重计算（见 :data:`COST_WEIGHTS`）：

============  ==========================================================
权重            含义
============  ==========================================================
w1 position_shift  相邻和弦的换把距离（以把位 ``base_fret`` 计，单位品）
w2 hand_span       手跨惩罚（按弦跨度超过 2 品的部分 + 横按额外负担）
w3 open_string     空弦奖励（**负成本**：开放弦越多，成本越低）
w4 switch          和弦切换成本（6 根弦上手指/闷弦的变化比例）
w5 melody_cover    旋律音是否落在指法内的惩罚（音级不匹配的旋律音）
============  ==========================================================

本模块只读 :class:`aitsm.contracts.ChordEvent` / :class:`MelodyNote`，
候选形状来自 :func:`aitsm.score.voicings.voicings_for`，不碰音频，也不
import :mod:`aitsm.pipeline`。

公开函数
--------
- :func:`candidates_for`：取某个和弦的候选指法（带缓存，纯查询）；
- :func:`optimize_progression`：DP 求全局最优指法序列；
- :func:`local_cost` / :func:`transition_cost` / :func:`plan_cost`：成本模型，
  供 TART 微调与测试复用；
- :func:`assign_path_costs`：把局部成本改写成累计路径成本。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from aitsm.contracts import ChordEvent, MelodyNote
from aitsm.score.model import STANDARD_TUNING
from aitsm.score.voicings import voicings_for

__all__ = [
    "COST_WEIGHTS",
    "UNPLAYABLE_PENALTY",
    "VoicingChoice",
    "candidates_for",
    "choice_from_voicing",
    "local_cost",
    "transition_cost",
    "plan_cost",
    "assign_path_costs",
    "optimize_progression",
]

#: 五项成本权重。测试与上层（TART 微调、decisions 报告）都从这里取值，
#: 不要在别处重复写死魔法数字。
COST_WEIGHTS: dict[str, float] = {
    # w1：换把距离（每品）
    "position_shift": 1.2,
    # w2：手跨惩罚（每超 1 品 / 每个横按）
    "hand_span": 0.5,
    # w3：空弦奖励（成本里是负项）
    "open_string": 0.15,
    # w4：和弦切换成本（0..1 的变化比例）
    "switch": 0.8,
    # w5：旋律音未被指法覆盖的惩罚（每个音）
    "melody_cover": 2.0,
}

#: 找不到任何合法指法时的兜底成本（远高于正常路径，DP 会尽量避开，
#: 但仍保证每个和弦都有输出，交给上层告警）。
UNPLAYABLE_PENALTY = 25.0


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass
class VoicingChoice:
    """DP 为一个 :class:`ChordEvent` 选中的指法。

    ``cost`` 是从进行开头累加到当前和弦的**路径成本**（包含局部成本与所有
    切换成本），因此最后一个元素的 ``cost`` 就是整段总成本。``local_cost``
    只记录该形状自身的成本，便于重新计算全局成本。
    """

    chord: ChordEvent
    voicing_id: str
    frets: list[int]
    cost: float = 0.0
    candidates_considered: int = 0
    local_cost: float = 0.0
    base_fret: int = 0
    difficulty: int = 0
    barres: list[dict] = field(default_factory=list)
    engine: str = "dp"

    @property
    def symbol(self) -> str:
        """和弦符号（来自 ``ChordEvent.symbol``）。"""
        return self.chord.symbol

    @property
    def position(self) -> float:
        """把位：以 ``base_fret`` 计，0 表示开放把位。"""
        return float(self.base_fret)

    @property
    def sounding_count(self) -> int:
        """发声弦数（``fret >= 0``）。"""
        return sum(1 for fret in self.frets if int(fret) >= 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.chord.symbol,
            "start": round(float(self.chord.start), 6),
            "end": round(float(self.chord.end), 6),
            "voicing_id": self.voicing_id,
            "frets": [int(fret) for fret in self.frets],
            "base_fret": int(self.base_fret),
            "cost": round(float(self.cost), 6),
            "local_cost": round(float(self.local_cost), 6),
            "candidates_considered": int(self.candidates_considered),
            "difficulty": int(self.difficulty),
            "barres": [dict(barre) for barre in self.barres],
            "engine": self.engine,
        }


# --------------------------------------------------------------------------- #
# 候选与成本模型
# --------------------------------------------------------------------------- #


def _weights(overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    """合并权重覆盖值；未知键直接忽略，保证成本模型稳定。"""
    weights = dict(COST_WEIGHTS)
    for name, value in dict(overrides or {}).items():
        if name in weights:
            try:
                weights[name] = float(value)
            except (TypeError, ValueError):
                continue
    return weights


def _normalize_frets(frets: Any) -> list[int]:
    """把形状规整成 6 个 int；``None`` / 非数字按闷弦 ``-1`` 处理。"""
    values = list(frets or [])
    out: list[int] = []
    for value in values[:6]:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            out.append(-1)
    while len(out) < 6:
        out.append(-1)
    return out


def _sounding_pcs(frets: Sequence[int], tuning: Sequence[int]) -> set[int]:
    """指法实际发声的音级集合（与调弦对齐）。"""
    pcs: set[int] = set()
    for index, fret in enumerate(frets):
        if index >= len(tuning) or fret < 0:
            continue
        pcs.add((int(tuning[index]) + int(fret)) % 12)
    return pcs


def _note_fields(note: Any) -> tuple[float, float, int | None]:
    """从 :class:`MelodyNote` / dict / int 里取出 ``(start, end, midi)``。"""
    if isinstance(note, MelodyNote):
        return float(note.start), float(note.end), int(note.midi)
    if isinstance(note, Mapping):
        try:
            start = float(note.get("start", 0.0) or 0.0)
        except (TypeError, ValueError):
            start = 0.0
        try:
            end = float(note.get("end", start) or start)
        except (TypeError, ValueError):
            end = start
        midi = note.get("midi")
        try:
            midi_value = int(midi) if midi is not None else None
        except (TypeError, ValueError):
            midi_value = None
        return start, end, midi_value
    if isinstance(note, bool):
        return float("-inf"), float("inf"), None
    if isinstance(note, (int, float)):
        return float("-inf"), float("inf"), int(note)
    return float("-inf"), float("inf"), None


def _overlaps(note: Any, chord: ChordEvent) -> bool:
    """旋律音是否与和弦区间相交（零时值音符按「落在区间内」算）。"""
    start, end, _ = _note_fields(note)
    if start == float("-inf") and end == float("inf"):
        return True
    if end <= start:
        return float(chord.start) <= start <= float(chord.end)
    return start <= float(chord.end) and end >= float(chord.start)


def local_cost(
    chord: ChordEvent,
    voicing: Mapping[str, Any],
    *,
    melody_notes: Sequence[Any] = (),
    weights: Mapping[str, float] | None = None,
    tuning: Sequence[int] = STANDARD_TUNING,
) -> float:
    """单个指法的局部成本（w2 手跨 + w3 空弦奖励 + w5 旋律覆盖）。

    找不到合法指法（``frets`` 为空）时返回 :data:`UNPLAYABLE_PENALTY`。
    """
    w = _weights(weights)
    frets = _normalize_frets(voicing.get("frets"))
    if not any(fret >= 0 for fret in frets):
        return UNPLAYABLE_PENALTY

    fretted = [fret for fret in frets if fret > 0]
    span = (max(fretted) - min(fretted)) if fretted else 0
    barres = voicing.get("barres") or []
    span_penalty = max(0, span - 2) + 0.5 * len(barres)
    open_count = sum(1 for fret in frets if fret == 0)

    covered_pcs = _sounding_pcs(frets, tuning)
    missing = 0
    for note in melody_notes or ():
        if not _overlaps(note, chord):
            continue
        _, _, midi = _note_fields(note)
        if midi is None:
            continue
        if int(midi) % 12 not in covered_pcs:
            missing += 1

    return (
        w["hand_span"] * span_penalty
        - w["open_string"] * open_count
        + w["melody_cover"] * missing
    )


def transition_cost(
    previous: VoicingChoice,
    current: VoicingChoice,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """相邻两个指法的切换成本（w1 换把距离 + w4 手型变化比例）。"""
    w = _weights(weights)
    shift = abs(float(current.position) - float(previous.position))

    if previous.symbol == current.symbol and previous.voicing_id == current.voicing_id:
        return w["position_shift"] * shift

    left = _normalize_frets(previous.frets)
    right = _normalize_frets(current.frets)
    changed = sum(1 for a, b in zip(left, right) if a != b) / 6.0
    if previous.symbol == current.symbol:
        # 同一个和弦换把位/换形状，比换一个和弦稍便宜
        changed *= 0.5
    return w["position_shift"] * shift + w["switch"] * changed


def plan_cost(
    choices: Sequence[VoicingChoice],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """一条指法路径的总成本（用 ``local_cost`` + 切换成本重算）。"""
    total = 0.0
    previous: VoicingChoice | None = None
    for choice in choices or ():
        total += float(choice.local_cost)
        if previous is not None:
            total += transition_cost(previous, choice, weights=weights)
        previous = choice
    return total


def candidates_for(
    chord: ChordEvent,
    *,
    tuning: Sequence[int] = STANDARD_TUNING,
    max_candidates: int = 8,
) -> list[dict]:
    """某和弦的候选指法（``voicings_for`` 的容错包装）。"""
    try:
        limit = max(1, int(max_candidates))
    except (TypeError, ValueError):
        limit = 8
    try:
        return list(voicings_for(chord.symbol, tuning=tuning, limit=limit))
    except Exception:  # pragma: no cover - voicings_for 内部已容错，双保险
        return []


def choice_from_voicing(
    chord: ChordEvent,
    voicing: Mapping[str, Any],
    *,
    candidates_considered: int = 1,
    melody_notes: Sequence[Any] = (),
    weights: Mapping[str, float] | None = None,
    tuning: Sequence[int] = STANDARD_TUNING,
    engine: str = "dp",
) -> VoicingChoice:
    """把 ``voicings_for`` 的 dict 包装成 :class:`VoicingChoice`。"""
    frets = _normalize_frets(voicing.get("frets"))
    try:
        base_fret = int(voicing.get("base_fret") or 0)
    except (TypeError, ValueError):
        base_fret = 0
    try:
        difficulty = int(voicing.get("difficulty") or 0)
    except (TypeError, ValueError):
        difficulty = 0
    return VoicingChoice(
        chord=chord,
        voicing_id=str(voicing.get("id") or voicing.get("voicing_id") or ""),
        frets=frets,
        cost=0.0,
        candidates_considered=max(0, int(candidates_considered)),
        local_cost=local_cost(
            chord, voicing, melody_notes=melody_notes, weights=weights, tuning=tuning
        ),
        base_fret=base_fret,
        difficulty=difficulty,
        barres=[dict(barre) for barre in (voicing.get("barres") or [])],
        engine=engine,
    )


def assign_path_costs(
    choices: Sequence[VoicingChoice],
    *,
    weights: Mapping[str, float] | None = None,
    engine: str | None = None,
) -> list[VoicingChoice]:
    """按顺序把 ``local_cost`` 累加成路径成本，返回新的列表（不改入参）。"""
    total = 0.0
    previous: VoicingChoice | None = None
    out: list[VoicingChoice] = []
    for choice in choices or ():
        total += float(choice.local_cost)
        if previous is not None:
            total += transition_cost(previous, choice, weights=weights)
        out.append(
            replace(
                choice,
                frets=list(choice.frets),
                barres=[dict(barre) for barre in choice.barres],
                cost=round(total, 6),
                engine=engine or choice.engine,
            )
        )
        previous = out[-1]
    return out


# --------------------------------------------------------------------------- #
# DP 主体
# --------------------------------------------------------------------------- #


def _unplayable_choice(chord: ChordEvent) -> VoicingChoice:
    """没有候选时的占位：保留和弦符号，交给上层告警。"""
    return VoicingChoice(
        chord=chord,
        voicing_id="",
        frets=[],
        cost=UNPLAYABLE_PENALTY,
        candidates_considered=0,
        local_cost=UNPLAYABLE_PENALTY,
        base_fret=0,
        difficulty=0,
        engine="dp",
    )


def optimize_progression(
    chords: Sequence[ChordEvent],
    *,
    tuning: Sequence[int] = STANDARD_TUNING,
    melody_notes: Sequence[Any] = (),
    key: Any = None,
    max_candidates: int = 8,
    weights: Mapping[str, float] | None = None,
) -> list[VoicingChoice]:
    """对整个和弦进行做 Viterbi，返回全局最优的指法序列。

    参数
    ----
    chords:
        按时间排序的 :class:`aitsm.contracts.ChordEvent` 序列。
    tuning:
        6 个空弦 MIDI 音高（低到高），默认标准调弦。
    melody_notes:
        旋律音（:class:`MelodyNote` / dict / MIDI int），用于 w5 覆盖惩罚。
    key:
        预留参数（调性上下文）。当前版本不改变成本，保持签名稳定。
    max_candidates:
        每个和弦参与 DP 的候选数上限。
    weights:
        覆盖 :data:`COST_WEIGHTS` 的权重（未知键忽略）。

    返回
    ----
    list[VoicingChoice]：与 ``chords`` 一一对应；``cost`` 为累计路径成本。
    """
    chord_list = list(chords or ())
    if not chord_list:
        return []

    weight_map = _weights(weights)
    states: list[list[VoicingChoice]] = []
    best_costs: list[list[float]] = []
    back_pointers: list[list[int]] = []

    for index, chord in enumerate(chord_list):
        pool = candidates_for(chord, tuning=tuning, max_candidates=max_candidates)
        if not pool:
            states.append([_unplayable_choice(chord)])
            best_costs.append([UNPLAYABLE_PENALTY])
            back_pointers.append([-1])
            continue

        row = [
            choice_from_voicing(
                chord,
                voicing,
                candidates_considered=len(pool),
                melody_notes=melody_notes,
                weights=weight_map,
                tuning=tuning,
            )
            for voicing in pool
        ]
        states.append(row)

        if index == 0:
            best_costs.append([round(choice.local_cost, 9) for choice in row])
            back_pointers.append([-1 for _ in row])
            continue

        previous_row = states[index - 1]
        previous_costs = best_costs[index - 1]
        row_costs: list[float] = []
        pointers: list[int] = []
        for current in row:
            best_previous: int | None = None
            best_value = float("inf")
            for prev_index, previous in enumerate(previous_row):
                value = previous_costs[prev_index] + transition_cost(
                    previous, current, weights=weight_map
                )
                if best_previous is None or value < best_value - 1e-12:
                    best_previous, best_value = prev_index, value
                elif (
                    abs(value - best_value) <= 1e-12
                    and previous.voicing_id < previous_row[best_previous].voicing_id
                ):
                    best_previous = prev_index
            pointers.append(int(best_previous if best_previous is not None else 0))
            row_costs.append(best_value + current.local_cost)
        best_costs.append(row_costs)
        back_pointers.append(pointers)

    # 回溯
    last_row = best_costs[-1]
    best_last = min(range(len(last_row)), key=lambda idx: (last_row[idx], states[-1][idx].voicing_id))
    path_indices = [best_last]
    for index in range(len(chord_list) - 1, 0, -1):
        path_indices.append(back_pointers[index][path_indices[-1]])
    path_indices.reverse()

    path = [states[index][choice_index] for index, choice_index in enumerate(path_indices)]
    return assign_path_costs(path, weights=weight_map)
