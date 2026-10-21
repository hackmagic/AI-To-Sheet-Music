"""和弦指法库（F12）：算法生成 + 手工校验的常用把位清单。

设计思路
--------
1. **手工校验的常用清单**（``CURATED``）：C/G/D/A/E/Am/Em/Dm/F/Bm 等开放与横按
   指型，全部经过音级核对，排在最前，保证成品像真谱而不是「算法猜出来的形状」。
2. **算法候选生成**（``_generate_candidates``）：对给定和弦符号，在标准调弦
   （可选 Drop D）上枚举 6 根弦的品位组合，约束：
   - 发声弦必须是连续的一段（低音弦可闷掉，中间不允许空洞）；
   - 按弦跨度 ≤ 4 品；至少 3 根发声弦；
   - 最低音必须是和弦根音（斜杠和弦则是指定低音）；
   - 必须覆盖和弦的关键音级（根音/三音/五音；挂留用二度或四度替代三音）。
3. 每个形状推导 ``fingers[]`` / ``base_fret`` / ``barres[]`` / ``difficulty``(1-5) /
   中文 ``label``，并以确定性 id 写入 ``score.json`` 的 ``voicings`` 表。

``build_voicing_db()`` 会生成 ``web/voicings.json``（前端移调后查表用）；本模块
还提供 ``voicing_for()`` 供编配器挑选把位。全部纯标准库实现。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from .model import (
    STANDARD_TUNING,
    TUNINGS,
    ChordSymbol,
    format_chord_symbol,
    parse_chord_symbol,
    pc_to_name,
    tuning_to_midis,
)

__all__ = [
    "CURATED",
    "VOICINGS_PATH",
    "build_voicing_db",
    "db_voicing",
    "voicing_for",
    "voicings_for",
    "voicing_id",
    "validate_voicing",
    "difficulty_label",
    "enharmonic_variants",
    "spell_as",
    "respell_voicing",
    "uses_flats",
    "clear_cache",
    "TUNINGS",
    "STANDARD_TUNING",
]

#: 生成数据的目标路径（前端静态查表用）
VOICINGS_PATH = Path(__file__).resolve().parents[2] / "web" / "voicings.json"

#: 单个和弦最多保存的候选数（数据库体积与可选空间折中）
_MAX_PER_SYMBOL = 4
#: 单次生成搜索时的候选上限
_GEN_LIMIT = 24
#: 品位上限
_MAX_FRET = 15

# --------------------------------------------------------------------------
# 手工校验的常用把位（低到高 6 弦；-1 = 不弹）
# --------------------------------------------------------------------------

CURATED: dict[str, list[list[int]]] = {
    # ---- 大三和弦（开放把位 + 必选横按）----
    "C": [[-1, 3, 2, 0, 1, 0]],
    "G": [[3, 2, 0, 0, 0, 3]],
    "D": [[-1, -1, 0, 2, 3, 2]],
    "A": [[-1, 0, 2, 2, 2, 0]],
    "E": [[0, 2, 2, 1, 0, 0]],
    "F": [[1, 3, 3, 2, 1, 1], [-1, -1, 3, 2, 1, 1], [-1, 3, 3, 2, 1, 1]],
    "B": [[-1, 2, 4, 4, 4, 2]],
    "Bb": [[-1, 1, 3, 3, 3, 1]],
    "Eb": [[-1, 6, 5, 3, 4, 3]],
    "Ab": [[4, 6, 6, 5, 4, 4]],
    "Db": [[-1, 4, 6, 6, 6, 4]],
    "F#": [[2, 4, 4, 3, 2, 2]],
    # ---- 小三和弦 ----
    "Am": [[-1, 0, 2, 2, 1, 0]],
    "Em": [[0, 2, 2, 0, 0, 0]],
    "Dm": [[-1, -1, 0, 2, 3, 1]],
    "Bm": [[-1, 2, 4, 4, 3, 2]],
    "F#m": [[2, 4, 4, 2, 2, 2]],
    "Cm": [[-1, 3, 5, 5, 4, 3]],
    "Gm": [[3, 5, 5, 3, 3, 3]],
    "C#m": [[-1, 4, 6, 6, 5, 4]],
    "G#m": [[4, 6, 6, 4, 4, 4]],
    "D#m": [[-1, 6, 8, 8, 7, 6]],
    "Bbm": [[-1, 1, 3, 3, 2, 1]],
    "Fm": [[1, 3, 3, 1, 1, 1]],
    # ---- 属七 ----
    "C7": [[-1, 3, 2, 3, 1, 0]],
    "D7": [[-1, -1, 0, 2, 1, 2]],
    "E7": [[0, 2, 0, 1, 0, 0]],
    "F7": [[1, 3, 1, 2, 1, 1]],
    "G7": [[3, 2, 0, 0, 0, 1]],
    "A7": [[-1, 0, 2, 0, 2, 0]],
    "B7": [[-1, 2, 1, 2, 0, 2]],
    "Bb7": [[-1, 1, 3, 1, 4, 1]],
    # ---- 大七 ----
    "Cmaj7": [[-1, 3, 2, 0, 0, 0]],
    "Dmaj7": [[-1, -1, 0, 2, 2, 2]],
    "Emaj7": [[0, 2, 1, 1, 0, 0]],
    "Fmaj7": [[-1, -1, 3, 2, 1, 0]],
    "Gmaj7": [[3, 2, 0, 0, 0, 2]],
    "Amaj7": [[-1, 0, 2, 1, 2, 0]],
    "Bmaj7": [[-1, 2, 4, 3, 4, 2]],
    # ---- 小七 ----
    "Am7": [[-1, 0, 2, 0, 1, 0]],
    "Em7": [[0, 2, 2, 0, 3, 0]],
    "Dm7": [[-1, -1, 0, 2, 1, 1]],
    "Bm7": [[-1, 2, 4, 2, 3, 2]],
    "Cm7": [[-1, 3, 5, 3, 4, 3]],
    "Gm7": [[3, 5, 3, 3, 3, 3]],
    "F#m7": [[2, 4, 2, 2, 2, 2]],
    # ---- 六和弦 / 挂留 / 加音 ----
    "C6": [[-1, 3, 2, 2, 1, 0]],
    "G6": [[3, 2, 0, 0, 0, 0]],
    "D6": [[-1, -1, 0, 2, 0, 2]],
    "A6": [[-1, 0, 2, 2, 2, 2]],
    "E6": [[0, 2, 2, 1, 2, 0]],
    "Cm6": [[-1, 3, 1, 2, 1, 3]],
    "Am6": [[-1, 0, 2, 2, 1, 2]],
    "Dm6": [[-1, -1, 0, 2, 0, 1]],
    "Csus2": [[-1, 3, 0, 0, 1, 3]],
    "Dsus2": [[-1, -1, 0, 2, 3, 0]],
    "Asus2": [[-1, 0, 2, 2, 0, 0]],
    "Esus2": [[0, 2, 4, 4, 0, 0]],
    "Gsus2": [[3, 0, 0, 0, 3, 3]],
    "Csus4": [[-1, 3, 3, 0, 1, 1]],
    "Dsus4": [[-1, -1, 0, 2, 3, 3]],
    "Asus4": [[-1, 0, 2, 2, 3, 0]],
    "Esus4": [[0, 2, 2, 2, 0, 0]],
    "Gsus4": [[3, 3, 0, 0, 1, 3]],
    "Cadd9": [[-1, 3, 2, 0, 3, 3]],
    "Gadd9": [[3, 2, 0, 2, 0, 3]],
    "Dadd9": [[-1, -1, 0, 2, 3, 5]],
    "C9": [[-1, 3, 2, 3, 3, 3]],
    "Cmaj9": [[-1, 3, 2, 4, 3, 0]],
    # ---- 减 / 增 ----
    "Cdim": [[-1, 3, 4, 2, 4, -1]],
    "Bdim": [[-1, 2, 3, 4, 3, -1]],
    "Bdim7": [[-1, 2, 3, 1, 3, -1]],
    "Caug": [[-1, 3, 2, 1, 1, 0]],
    "Eaug": [[0, 3, 2, 1, 1, 0]],
    # ---- 强力和弦（5）----
    "C5": [[-1, 3, 5, 5, -1, -1]],
    "G5": [[3, 5, 5, -1, -1, -1]],
    "D5": [[-1, -1, 0, 2, 3, -1]],
    "A5": [[-1, 0, 2, 2, -1, -1]],
    "E5": [[0, 2, 2, -1, -1, -1]],
    "F5": [[1, 3, 3, -1, -1, -1]],
    "B5": [[-1, 2, 4, 4, -1, -1]],
    "Bb5": [[-1, 1, 3, 3, -1, -1]],
    "F#5": [[2, 4, 4, -1, -1, -1]],
    # ---- 斜杠和弦（转位 / 低音设计）----
    "C/E": [[0, 3, 2, 0, 1, 0]],
    "C/G": [[3, 3, 2, 0, 1, 0]],
    "G/B": [[-1, 2, 0, 0, 0, 3]],
    "G/D": [[-1, -1, 0, 0, 0, 3]],
    "D/F#": [[2, -1, 0, 2, 3, 2]],
    "D/A": [[-1, 0, 0, 2, 3, 2]],
    "A/C#": [[-1, 4, 2, 2, 2, 0]],
    "A/E": [[0, 0, 2, 2, 2, 0]],
    "E/B": [[-1, 2, 2, 1, 0, 0]],
    "E/G#": [[4, 2, 2, 1, 0, 0]],
    "F/A": [[-1, 0, 3, 2, 1, 1]],
    "F/C": [[-1, 3, 3, 2, 1, 1]],
    "Bm/A": [[-1, 0, 4, 4, 3, 2]],
    "Am/G": [[3, -1, 2, 2, 1, 0]],
    "Am/E": [[0, 0, 2, 2, 1, 0]],
    "Dm/F": [[1, -1, 0, 2, 3, 1]],
    "Em/B": [[-1, 2, 2, 0, 0, 0]],
    "G/F#": [[2, 0, 0, 0, 0, 3]],
    "Cmaj7/E": [[0, 3, 2, 0, 0, 0]],
    "Fmaj7/A": [[-1, 0, 3, 2, 1, 0]],
    "C7/Bb": [[-1, 1, 2, 3, 1, 0]],
}

#: 常用开放把位的符号集合（排序时优先）
_OPEN_FRIENDLY = frozenset(CURATED)


# --------------------------------------------------------------------------
# 形状推导：手指 / 横按 / 把位 / 难度
# --------------------------------------------------------------------------

def _normalize_frets(frets: Sequence[Any]) -> list[int]:
    """规整 frets 数组：-1/null 统一成 -1，其余转 int。"""
    out: list[int] = []
    for fret in frets:
        if fret is None:
            out.append(-1)
        else:
            out.append(int(fret))
    if len(out) != 6:
        raise ValueError(f"frets 必须为 6 个元素: {frets!r}")
    return out


def _sounding_indices(frets: Sequence[int]) -> list[int]:
    return [index for index, fret in enumerate(frets) if fret >= 0]


def _infer_fingers(frets: Sequence[int]) -> tuple[list[int | None], list[dict[str, int]]]:
    """从品位推导手指编号与横按。返回 ``(fingers, barres)``。

    ``fingers`` 与 ``frets`` 同长度，空弦 / 不弹为 None。``barres`` 的 ``from``/``to``
    使用 frets 数组下标（0 = 6 弦，5 = 1 弦），与 ``samples/demo-score.json`` 一致。
    """
    frets = list(frets)
    fingers: list[int | None] = [None] * 6
    barres: list[dict[str, int]] = []
    sounding = _sounding_indices(frets)
    if not sounding:
        return fingers, barres
    lowest = sounding[0]
    fretted = [(index, fret) for index, fret in enumerate(frets) if fret > 0]
    if not fretted:
        return fingers, barres

    # ---- 横按推断 ------------------------------------------------------
    # 条件（缺一不可）：
    #   a) 最低发声弦本身按在 f 品（横按必须盖住最低音弦）；
    #   b) f 品在更高的弦上还有别的按弦音（跨度 ≥ 2 根弦）；
    #   c) 两根横按弦之间的所有弦要么也按在 f 品，要么按在更高品位
    #      （即“手指压在横按手掌上方”）。G 320003 因为中间有 2 品与空弦，
    #      不满足 c，所以不会被误判成 3 品横按。
    barre_fret = frets[lowest]
    at_barre = [index for index in range(lowest, 6) if frets[index] == barre_fret and barre_fret > 0]
    if barre_fret > 0 and len(at_barre) >= 2:
        span_end = max(at_barre)
        inner = frets[lowest : span_end + 1]
        # 横按范围内的每一根弦都必须按着（>0）：有开弦或闷弦说明手指压不住，
        # 也说明这个形状不是横按（例如 Bm 的打工变体 x2402x）。
        if all(fret > 0 for fret in inner) and min(inner) >= barre_fret:
            barres.append({"fret": barre_fret, "from": lowest, "to": span_end})
            for index in range(lowest, span_end + 1):
                if frets[index] == barre_fret:
                    fingers[index] = 1

    # ---- 其余手指：按品位从低到高分派，指号约等于“品位 - 手位 + 1” ----
    used = {1} if barres else set()
    hand_base = min(fret for _index, fret in fretted)

    def _natural(fret: int) -> int:
        return max(1, min(4, fret - hand_base + 1))

    def _take(fret: int) -> int:
        finger = _natural(fret)
        for candidate in list(range(finger, 5)) + list(range(finger - 1, 0, -1)):
            if candidate not in used:
                used.add(candidate)
                return candidate
        return min(4, finger)

    # 按 (品位, 弦序) 升序；同品且相邻弦共用一个手指（等效小横按，如 xx3211）
    assigned_fret: dict[int, int] = {}
    prev_index_at_fret: dict[int, int] = {}
    for index, fret in sorted(fretted, key=lambda item: (item[1], item[0])):
        if fingers[index] is not None:
            continue
        if fret in assigned_fret and prev_index_at_fret.get(fret) == index - 1:
            fingers[index] = assigned_fret[fret]
        else:
            finger = _take(fret)
            if fret not in assigned_fret:
                assigned_fret[fret] = finger
            fingers[index] = finger
        prev_index_at_fret[fret] = index
    return fingers, barres


def _base_fret(frets: Sequence[int]) -> int:
    """把位：有开放弦为 0，否则取最低按弦品。"""
    if any(fret == 0 for fret in frets):
        return 0
    fretted = [fret for fret in frets if fret > 0]
    return min(fretted) if fretted else 0


def _difficulty(frets: Sequence[int], barres: Sequence[dict[str, int]], base_fret: int) -> int:
    """难度 1-5：开放和弦 1-2，横按 3-4，高把位 4-5。"""
    fretted = [fret for fret in frets if fret > 0]
    span = (max(fretted) - min(fretted)) if fretted else 0
    if not fretted:
        return 1
    difficulty = 1
    if barres:
        difficulty += 2
    elif span >= 4:
        difficulty += 1
    # 非相邻弦上的同品按弦（两个手指并排跨弦）比普通开放和弦难。
    # 横按覆盖的弦不算（那是同一根手指压的，不额外加难度）。
    barre_frets = {int(barre.get("fret", -1)) for barre in barres}
    split_same_fret = False
    for fret in sorted(set(fretted)):
        if fret in barre_frets:
            continue
        indices = [index for index, value in enumerate(frets) if value == fret]
        if len(indices) >= 2 and indices[-1] - indices[0] > len(indices) - 1:
            split_same_fret = True
            break
    if split_same_fret:
        difficulty += 1
    if base_fret >= 1:
        difficulty += 1
    if base_fret >= 5:
        difficulty += 1
    if base_fret >= 9:
        difficulty += 1
    return max(1, min(5, difficulty))


def difficulty_label(difficulty: int) -> str:
    """难度 -> 中文说明。"""
    return {
        1: "入门",
        2: "简单",
        3: "中等",
        4: "较难",
        5: "进阶",
    }.get(int(difficulty), "中等")


def voicing_id(symbol: str, frets: Sequence[int | None]) -> str:
    """确定性指法 id：``C_x32010`` / ``G_over_B_x20003``。"""
    safe = str(symbol).replace("/", "_over_").replace("#", "s")
    tag = "".join("x" if fret is None or int(fret) < 0 else str(int(fret)) for fret in frets)
    return f"{safe}_{tag}"


# --------------------------------------------------------------------------
# 等音（enharmonic）书写
# --------------------------------------------------------------------------

_FLAT_SPELLING_RE = re.compile(r"[A-G]b")


def uses_flats(symbol: str) -> bool:
    """符号是否用降号书写（``Bb`` / ``Ebm7`` / ``C/Eb``）。"""
    return bool(_FLAT_SPELLING_RE.search(str(symbol)))


def spell_as(symbol: str, *, prefer_flats: bool | None = None) -> str:
    """把和弦符号按指定书写习惯规范化（``prefer_flats=None`` 时沿用输入习惯）。"""
    chord = parse_chord_symbol(symbol)
    if prefer_flats is None:
        prefer_flats = uses_flats(symbol)
    return format_chord_symbol(
        chord.root_pc, chord.quality, chord.bass_pc, prefer_flats=prefer_flats
    )


def enharmonic_variants(symbol: str) -> list[str]:
    """返回该符号的等音写法（含自身，去重，长度 1-2）。

    ``Bb`` -> ``["Bb", "A#"]``；``C#m`` -> ``["C#m", "Dbm"]``；
    不带变化音或降号拼写不存在时只返回自身。
    """
    chord = parse_chord_symbol(symbol)
    sharp = format_chord_symbol(chord.root_pc, chord.quality, chord.bass_pc)
    flat = format_chord_symbol(chord.root_pc, chord.quality, chord.bass_pc, prefer_flats=True)
    result: list[str] = []
    for text in (spell_as(symbol), sharp, flat):
        if text not in result:
            result.append(text)
    return result


def respell_voicing(voicing: dict, symbol: str) -> dict:
    """把指法的 ``symbol`` / ``id`` / ``label`` 改写成另一种等音写法。"""
    new = dict(voicing)
    new["symbol"] = symbol
    new["id"] = voicing_id(symbol, new.get("frets") or [])
    new["label"] = _shape_label(
        symbol, int(new.get("base_fret", 0)), new.get("barres") or [], str(new.get("source") or "generated")
    )
    return new


def _shape_label(symbol: str, base_fret: int, barres: Sequence[dict[str, int]], source: str) -> str:
    """中文标签，如 ``C 开放把位`` / ``F 1品横按``。"""
    if source == "curated" and base_fret == 0:
        return f"{symbol} 常用开放把位"
    if barres:
        return f"{symbol} {base_fret}品横按"
    if base_fret == 0:
        return f"{symbol} 开放把位"
    return f"{symbol} 第{base_fret}把位"


def _voicing_dict(
    symbol: str,
    frets: Sequence[int | None],
    *,
    source: str = "generated",
    extra_tags: Iterable[str] = (),
) -> dict:
    """构造一个完整的 voicing dict（含 id，可直接进 ``score.voicings``）。"""
    normalized = _normalize_frets(frets)
    fingers, barres = _infer_fingers(normalized)
    base_fret = _base_fret(normalized)
    difficulty = _difficulty(normalized, barres, base_fret)
    tags = ["open" if base_fret == 0 else "movable"]
    if barres:
        tags.append("barre")
    if source == "curated":
        tags.append("curated")
    tags.extend(extra_tags)
    return {
        "id": voicing_id(symbol, normalized),
        "symbol": symbol,
        "frets": normalized,
        "fingers": fingers,
        "base_fret": base_fret,
        "barres": barres,
        "difficulty": difficulty,
        "difficulty_label": difficulty_label(difficulty),
        "label": _shape_label(symbol, base_fret, barres, source),
        "tags": tags,
        "source": source,
    }


# --------------------------------------------------------------------------
# 合法性校验
# --------------------------------------------------------------------------

def validate_voicing(frets: Sequence[int | None], symbol: str, *, tuning: Sequence[int] = STANDARD_TUNING) -> list[str]:
    """校验一个指法形状与和弦符号是否自洽，返回问题列表（空表示合法）。

    检查项：音级集合 ⊆ 和弦音、覆盖关键音级、最低音为根音/指定低音、
    按弦跨度 ≤ 4、发声弦 ≥ 3、发声弦连续（允许低音弦闷弦）、品位范围。
    """
    problems: list[str] = []
    try:
        chord: ChordSymbol = parse_chord_symbol(symbol)
    except ValueError as exc:
        return [f"和弦符号无法解析: {exc}"]
    normalized = _normalize_frets(frets)
    if any(fret > _MAX_FRET for fret in normalized):
        problems.append(f"品位超过 {_MAX_FRET}")
    sounding = _sounding_indices(normalized)
    if len(sounding) < 3:
        problems.append("发声弦少于 3 根")
        return problems
    # 发声弦内部不能有空洞（低音弦与高音弦两端允许闷弦，强力和弦就是如此）
    if sounding != list(range(sounding[0], sounding[-1] + 1)):
        problems.append("发声弦不连续（内弦空洞）")
    pcs: set[int] = set()
    lowest_pc: int | None = None
    for index, fret in enumerate(normalized):
        if fret < 0:
            continue
        pitch = int(tuning[index]) + fret
        pc = pitch % 12
        if lowest_pc is None:
            lowest_pc = pc
        pcs.add(pc)
    if not pcs:
        problems.append("没有发声弦")
        return problems
    allowed = chord.pitch_classes
    extra = pcs - allowed
    if extra:
        names = ", ".join(pc_to_name(pc) for pc in sorted(extra))
        problems.append(f"含非和弦音: {names}")
    missing = chord.required_pcs - pcs
    if missing:
        names = ", ".join(pc_to_name(pc) for pc in sorted(missing))
        problems.append(f"缺少关键音级: {names}")
    expected_bass = chord.bass_pc if chord.bass_pc is not None else chord.root_pc
    if lowest_pc is not None and lowest_pc % 12 != expected_bass % 12:
        problems.append(f"最低音不是根音/低音（应为 {pc_to_name(expected_bass)}）")
    fretted = [fret for fret in normalized if fret > 0]
    if fretted and max(fretted) - min(fretted) > 4:
        problems.append("按弦跨度超过 4 品")
    if not _open_mix_ok(normalized):
        problems.append("开放弦与高把位按弦混用（跨度过大）")
    return problems


def _open_mix_ok(frets: Sequence[int]) -> bool:
    """开放弦 + 按弦混用时的可演奏性判断。

    真实吉他手型里，只要还有空弦在响，按弦手就必须**留在琴枕附近**：

    1. 最低按弦品 ≤ 2，且按弦跨度 ≤ 2 品（C x32010 / A x02220 / Fmaj7 xx3210 满足）；
    2. 同一品位不允许跨越 ≥3 根弦、且中间夹着空弦去按（那等于要求横按压住空弦）。
       例如 Bb 的 ``[-1,1,0,3,3,1]``（5 弦 1 品 + 1 弦 1 品，中间 4 弦空弦）不可演奏；
       而 Cadd9 ``[-1,3,2,0,3,3]`` 只在 3 品上隔 2 根弦，属于正常手型，保留。
    像 Am 的 ``[5,0,2,5,5,0]`` 这种「低音弦 5 品 + 5 弦空弦 + 2 品按弦」组合，
    虽然音级合法，但需要一只手同时按住 5 品与 2 品、还得让中间的弦空响，
    实际不可演奏，直接剔除。
    """
    fretted = [fret for fret in frets if fret > 0]
    if not fretted:
        return True
    if not any(fret == 0 for fret in frets):
        return True
    if min(fretted) > 2 or max(fretted) - min(fretted) > 2:
        return False
    # 跨弦同品：同品两端相隔很远、中间夹着空弦，且中间还有**更高品位**的按弦
    # （手指要跨过空弦去按更高的品）→ 手型不可能。
    # G 320003 的 3 品在 6 弦与 1 弦、中间 A 弦按 2 品（比 3 低），属于正常手型，保留；
    # Bb 的 [-1,1,0,3,3,1] 中间按 3 品高于 1 品，剔除。
    for fret in sorted(set(fretted)):
        indices = [index for index, value in enumerate(frets) if value == fret]
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                if right - left < 3:
                    continue
                middle = [frets[index] for index in range(left + 1, right)]
                if 0 in middle and any(0 < value > fret for value in middle):
                    return False
    return True


# --------------------------------------------------------------------------
# 候选生成
# --------------------------------------------------------------------------

def _block_ranges() -> list[tuple[int, int]]:
    """发声弦块：一段连续弦，两端都可闷（强力和弦会闷掉高音弦）。

    优先给出“覆盖高音弦”的块（扫弦更饱满），最后才是 3 弦小块。
    """
    blocks: list[tuple[int, int]] = []
    for end in range(6, 2, -1):
        for start in range(0, end - 2):
            blocks.append((start, end))
    return blocks


def _candidate_frets(
    tuning: Sequence[int],
    index: int,
    allowed: frozenset[int],
    base: int,
) -> list[int]:
    """某根弦在窗口 ``[base, base+4]`` 内、且音级属于和弦的可行品位。"""
    options: list[int] = []
    if int(tuning[index]) % 12 in allowed:
        options.append(0)
    low = max(1, base)
    high = min(_MAX_FRET, base + 4)
    for fret in range(low, high + 1):
        if (int(tuning[index]) + fret) % 12 in allowed:
            options.append(fret)
    return options


def _generate_candidates(
    chord: ChordSymbol,
    tuning: Sequence[int] = STANDARD_TUNING,
    *,
    max_windows: int = 12,
) -> list[list[int]]:
    """枚举满足全部约束的候选指法（未含手工清单）。"""
    allowed = chord.pitch_classes
    required = chord.required_pcs
    bass_pc = (chord.bass_pc if chord.bass_pc is not None else chord.root_pc) % 12
    results: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    min_sounding = max(3, len(required) + 1)

    for start, end in _block_ranges():
        if end - start < min_sounding:
            continue
        strings = list(range(start, end))
        for base in range(0, max_windows + 1):
            options = [_candidate_frets(tuning, index, allowed, base) for index in strings]
            if any(not opts for opts in options):
                continue
            if len(results) >= _GEN_LIMIT and base >= 6:
                break

            frets = [-1] * 6
            covered: set[int] = set()

            def _dfs(position: int, low_pc: int | None, min_f: int, max_f: int, covered: set[int]) -> None:
                if len(results) >= _GEN_LIMIT:
                    return
                if position == len(strings):
                    if low_pc is None or low_pc % 12 != bass_pc:
                        return
                    if not required <= covered:
                        return
                    key = tuple(frets)
                    if key in seen:
                        return
                    fretted = [fret for fret in frets if fret > 0]
                    if fretted and max(fretted) - min(fretted) > 4:
                        return
                    if not _open_mix_ok(frets):
                        return
                    seen.add(key)
                    results.append(list(frets))
                    return
                index = strings[position]
                remaining = len(strings) - position
                missing = required - covered
                if len(missing) > remaining:
                    return
                for fret in options[position]:
                    pc = (int(tuning[index]) + fret) % 12
                    if position == 0 and pc % 12 != bass_pc:
                        continue
                    new_min = min_f
                    new_max = max_f
                    if fret > 0:
                        new_min = min(min_f, fret)
                        new_max = max(max_f, fret)
                        if new_max - new_min > 4:
                            continue
                    frets[index] = fret
                    added = pc not in covered
                    if added:
                        covered.add(pc)
                    _dfs(position + 1, pc if low_pc is None else low_pc, new_min, new_max, covered)
                    if added:
                        covered.discard(pc)
                    frets[index] = -1

            _dfs(0, None, 999, -1, covered)
            if len(results) >= _GEN_LIMIT and base >= 6:
                break
    return results


def _sort_key(voicing: dict) -> tuple:
    """候选排序（真实谱面手感优先，而不是纯难度）：

    1. 手工校验的常用形状永远第一梯队（F 的简易版 ``xx3211`` 因此排在完整横按前）；
    2. 开放把位（含空弦）优先于任何横按；
    3. 可移动模板（正规 E 型/A 型横按）优先于 DFS 生成的"理论合法"形状；
    4. 然后才是难度、把位、发声弦数。

    这样 ``Bb`` 会给出 ``x13331`` 而不是 DFS 造出来的奇怪组合，``B7`` 仍然优先
    给出开放把位 ``x21202``（开放把位规则在模板规则之前）。
    """
    tags = voicing.get("tags") or []
    curated = 0 if "curated" in tags else 1
    open_shape = 0 if voicing.get("base_fret", 0) == 0 else 1
    template = 0 if ("template" in tags or "curated" in tags) else 1
    frets = [fret for fret in voicing.get("frets", []) if fret is not None and fret >= 0]
    sounding = len(frets)
    max_fret = max(frets) if frets else 0
    return (
        curated,
        open_shape,
        template,
        voicing.get("difficulty", 5),
        voicing.get("base_fret", 0),
        -sounding,
        max_fret,
        voicing.get("id", ""),
    )


_CURATED_INDEX: dict[str, list[list[int]]] | None = None


def _curated_index() -> dict[str, list[list[int]]]:
    """把手工清单按**规范化符号**建索引（Eb → D#，C/E 保持）。

    这样 ``voicing_for("Db")`` 与 ``voicing_for("C#")`` 都能命中同一份指法。
    """
    global _CURATED_INDEX
    if _CURATED_INDEX is None:
        index: dict[str, list[list[int]]] = {}
        for symbol, shapes in CURATED.items():
            try:
                chord = parse_chord_symbol(symbol)
            except ValueError:  # pragma: no cover - 手工清单自身保证可解析
                continue
            key = format_chord_symbol(chord.root_pc, chord.quality, chord.bass_pc)
            index.setdefault(key, []).extend(shapes)
        _CURATED_INDEX = index
    return _CURATED_INDEX


def _curated_for(symbol: str) -> list[dict]:
    """取手工清单里的形状（再跑一次合法性校验，防止手误）。"""
    result: list[dict] = []
    for frets in _curated_index().get(symbol, []):
        if validate_voicing(frets, symbol):
            continue
        result.append(_voicing_dict(symbol, frets, source="curated"))
    return result


_GENERATED_CACHE: dict[tuple[str, tuple[int, ...]], list[dict]] = {}


# 标准「可移动把位」模板：相对根音品位的半音偏移（-1 = 不弹）。
# 这是所有正规和弦手册里那套 E 型 / A 型横按指法，DFS 生成器很少能把它们排到
# 前面（它们的难度分不占优），但真实谱子里这些才是最常用的按法。
_E_MOVABLE: dict[str, tuple[int, ...]] = {
    "maj": (0, 2, 2, 1, 0, 0),
    "min": (0, 2, 2, 0, 0, 0),
    "7": (0, 2, 0, 1, 0, 0),
    "m7": (0, 2, 0, 0, 0, 0),
    "maj7": (0, 2, 1, 1, 0, 0),
    "sus4": (0, 2, 2, 2, 0, 0),
    "5": (0, 2, 2, -1, -1, -1),
}
_A_MOVABLE: dict[str, tuple[int, ...]] = {
    "maj": (-1, 0, 2, 2, 2, 0),
    "min": (-1, 0, 2, 2, 1, 0),
    "7": (-1, 0, 2, 0, 2, 0),
    "m7": (-1, 0, 2, 0, 1, 0),
    "maj7": (-1, 0, 2, 1, 2, 0),
    "sus4": (-1, 0, 2, 2, 3, 0),
    "5": (-1, 0, 2, 2, -1, -1),
}


def _movable_shapes(chord: ChordSymbol, tuning: Sequence[int]) -> list[list[int]]:
    """按 E 型 / A 型模板生成标准横按要求（只用于根音即低音的普通和弦）。"""
    if chord.bass_pc is not None and chord.bass_pc != chord.root_pc:
        return []
    shapes: list[list[int]] = []
    low_e_pc = int(tuning[0]) % 12
    low_a_pc = int(tuning[1]) % 12
    for table, open_pc in ((_E_MOVABLE, low_e_pc), (_A_MOVABLE, low_a_pc)):
        base = table.get(chord.quality)
        if base is None:
            continue
        root_fret = (chord.root_pc - open_pc) % 12
        frets = [-1 if fret < 0 else fret + root_fret for fret in base]
        if max(frets) > _MAX_FRET:
            continue
        shapes.append(frets)
    return shapes


def voicings_for(
    symbol: str,
    *,
    tuning: Sequence[int] = STANDARD_TUNING,
    limit: int = 6,
) -> list[dict]:
    """生成某和弦符号的全部候选指法（手工清单 + 算法候选，已排序）。

    返回值沿用**调用方输入的书写习惯**：``voicings_for("Bb")`` 返回的 symbol 是
    ``Bb``，``voicings_for("A#")`` 返回 ``A#``（内部统一按升号名缓存）。
    """
    try:
        chord = parse_chord_symbol(symbol)
    except ValueError:
        return []
    canonical = format_chord_symbol(chord.root_pc, chord.quality, chord.bass_pc)
    requested = spell_as(symbol)
    tuning_tuple = tuple(int(pitch) for pitch in tuning)
    cache_key = (canonical, tuning_tuple)
    if cache_key in _GENERATED_CACHE:
        cached = _GENERATED_CACHE[cache_key]
        items = [dict(item) for item in cached[:limit]]
        if requested != canonical:
            items = [respell_voicing(item, requested) for item in items]
        return items

    voicings: list[dict] = []
    seen_ids: set[str] = set()
    for item in _curated_for(canonical):
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        voicings.append(item)
    for frets in _movable_shapes(chord, tuning_tuple):
        if validate_voicing(frets, canonical, tuning=tuning_tuple):
            continue
        item = _voicing_dict(canonical, frets, source="template", extra_tags=("template",))
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        voicings.append(item)
    for frets in _generate_candidates(chord, tuning_tuple):
        item = _voicing_dict(canonical, frets, source="generated")
        if item["id"] in seen_ids:
            continue
        if validate_voicing(frets, canonical, tuning=tuning_tuple):
            continue
        seen_ids.add(item["id"])
        voicings.append(item)
    voicings.sort(key=_sort_key)
    _GENERATED_CACHE[cache_key] = voicings
    items = [dict(item) for item in voicings[:limit]]
    if requested != canonical:
        items = [respell_voicing(item, requested) for item in items]
    return items


def voicing_for(symbol: str, *, prefer: dict | None = None) -> dict | None:
    """为和弦符号挑一个最合适的指法。

    ``prefer`` 可选项：

    - ``max_difficulty``: int，硬上限（超过的候选直接剔除）；
    - ``difficulty`` / ``min_difficulty``: int(1-5)，**期望难度**；给了之后按
      ``|候选难度 - 期望|`` 加权，而不是一味挑最简单的。中级/高级谱面因此会优先
      给出完整横按版（如 F 用 ``[1,3,3,2,1,1]``），入门谱面仍优先简易版；
    - ``avoid_barre``: bool，尽量不用横按；
    - ``open_only``: bool，只用开放把位；
    - ``position``: int，期望的品位中心（按与它的距离加权）；
    - ``previous``: dict，上一个指法（用于减少换把）；
    - ``exclude_ids``: list[str]，排除指定 id；
    - ``tuning``: str|list[int]，调弦。
    """
    prefer = prefer or {}
    tuning = tuning_to_midis(prefer.get("tuning") or "standard")
    candidates = voicings_for(symbol, tuning=tuning, limit=12)
    if not candidates:
        return None
    max_difficulty = prefer.get("max_difficulty")
    exclude = set(prefer.get("exclude_ids") or ())
    filtered = [item for item in candidates if item.get("id") not in exclude]
    if max_difficulty is not None:
        capped = [item for item in filtered if item.get("difficulty", 5) <= int(max_difficulty)]
        filtered = capped or filtered
    if prefer.get("avoid_barre"):
        no_barre = [item for item in filtered if not item.get("barres")]
        filtered = no_barre or filtered
    if prefer.get("open_only"):
        open_only = [item for item in filtered if item.get("base_fret", 0) == 0]
        filtered = open_only or filtered
    min_sounding = prefer.get("min_sounding")
    if isinstance(min_sounding, (int, float)) and int(min_sounding) > 0:
        full = [
            item
            for item in filtered
            if sum(1 for fret in item.get("frets") or [] if fret is not None and int(fret) >= 0)
            >= int(min_sounding)
        ]
        filtered = full or filtered

    # 期望难度：difficulty 优先，其次 min_difficulty
    target_difficulty: float | None = None
    if isinstance(prefer.get("difficulty"), (int, float)):
        target_difficulty = float(prefer["difficulty"])
    elif isinstance(prefer.get("min_difficulty"), (int, float)):
        target_difficulty = float(prefer["min_difficulty"])
    difficulty_weight = float(prefer.get("difficulty_weight", 4.0))

    position = prefer.get("position")
    previous = prefer.get("previous") or {}
    previous_center = None
    if isinstance(previous, dict) and previous:
        previous_frets = [fret for fret in previous.get("frets") or [] if fret and fret > 0]
        previous_center = (
            (min(previous_frets) + max(previous_frets)) / 2 if previous_frets else 0
        )

    # 候选顺序本身就是“真实谱面手感”的先验（手工形状 > 开放把位 > 标准横按 > 生成），
    # 这里用排名当基础分，再叠加期望难度、把位、换把平滑度等偏好。
    rank_of = {id(item): index for index, item in enumerate(filtered)}

    def _score(item: dict) -> tuple:
        fret_center = 0
        fretted = [fret for fret in item.get("frets") or [] if fret and fret > 0]
        if fretted:
            fret_center = (min(fretted) + max(fretted)) / 2
        item_difficulty = float(item.get("difficulty", 5))
        penalty = float(rank_of.get(id(item), 0)) * 1.0
        if target_difficulty is not None:
            penalty += abs(item_difficulty - target_difficulty) * difficulty_weight
        penalty += item.get("base_fret", 0) * 0.25
        if isinstance(position, (int, float)):
            penalty += abs(fret_center - float(position)) * 3
        if previous_center is not None:
            penalty += abs(fret_center - previous_center) * 2
        return (round(penalty, 4), item.get("id", ""))

    filtered.sort(key=_score)
    return dict(filtered[0]) if filtered else None


# --------------------------------------------------------------------------
# 数据库生成
# --------------------------------------------------------------------------

#: 数据库覆盖的和弦性质 -> 每个符号保存的候选数。
#: 三和弦与最常用的七和弦最多存 4 个；色彩和弦存 2-3 个，控制 json 体积。
_DB_QUALITIES: dict[str, int] = {
    "maj": 4,
    "min": 4,
    "7": 4,
    "maj7": 3,
    "m7": 3,
    "6": 3,
    "m6": 2,
    "sus2": 2,
    "sus4": 2,
    "dim": 2,
    "aug": 2,
    "add9": 2,
    "9": 2,
    "madd9": 2,
    "5": 2,
    "m7b5": 2,
    "dim7": 2,
    "7sus4": 2,
    "69": 2,
    "maj9": 2,
    "m9": 2,
}


def _db_symbols() -> list[str]:
    """需要写入数据库的全部和弦符号。"""
    symbols: list[str] = []
    for pc in range(12):
        for quality in _DB_QUALITIES:
            symbols.append(format_chord_symbol(pc, quality))
    # 斜杠和弦：大三/小三的转位（三音、五音在低音）
    intervals = {"maj": (0, 4, 7), "min": (0, 3, 7)}
    for pc in range(12):
        for quality, tones in intervals.items():
            for index in (1, 2):
                bass = (pc + tones[index]) % 12
                symbols.append(format_chord_symbol(pc, quality, bass))
    # 手工清单里出现、但不在上面的符号（如 C/G 之外的 voicing 变体）
    for symbol in CURATED:
        if symbol not in symbols:
            symbols.append(symbol)
    # 去重并保持稳定顺序
    unique: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        try:
            canonical = parse_chord_symbol(symbol).symbol
        except ValueError:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append(canonical)
    return unique


def build_voicing_db(path: str | Path | None = None) -> dict:
    """生成和弦指法库并写出 ``web/voicings.json``。

    结构::

        {"version": 1, "generated_by": "aitsm.score.voicings",
         "standard": {"C": [{voicing...}], ...},
         "voicings": {"C_x32010": {voicing...}, ...},   # 按 id 的扁平索引
         "tunings": {"standard": [40,45,50,55,59,64], "drop_d": [38,...]}}

    对 24 个大小调的三和弦与常用七和弦（7 / maj7 / m7 / 6 / m6 / sus / add9 ...）
    保证至少提供一个把位；斜杠和弦覆盖大三与小三的转位。
    """
    target = Path(path) if path is not None else VOICINGS_PATH
    standard: dict[str, list[dict]] = {}
    flat: dict[str, dict] = {}
    missing: list[str] = []
    for symbol in _db_symbols():
        quality = parse_chord_symbol(symbol).quality
        limit = _DB_QUALITIES.get(quality, _MAX_PER_SYMBOL)
        voicings = voicings_for(symbol, limit=limit)
        if voicings:
            standard[symbol] = voicings
            for voicing in voicings:
                flat[voicing["id"]] = voicing
        else:
            missing.append(symbol)

    # 等音别名：库里按升号名生成，这里为每个带变化音的符号补一份降号写法，
    # 使 Bb/A#、Eb/D#、Ab/G#、Db/C#、Gb/F# 双向都能直接查表（前端不需要自己换算）。
    aliases: dict[str, str] = {}
    for symbol in list(standard):
        for variant in enharmonic_variants(symbol):
            if variant == symbol or variant in standard:
                continue
            standard[variant] = [respell_voicing(voicing, variant) for voicing in standard[symbol]]
            for voicing in standard[variant]:
                flat[voicing["id"]] = voicing
            aliases[variant] = symbol

    payload = {
        "version": 1,
        "generated_by": "aitsm.score.voicings",
        "generated_at": _now_iso(),
        "max_fret": _MAX_FRET,
        "standard": standard,
        "voicings": flat,
        "aliases": aliases,
        "tunings": {name: list(midis) for name, midis in TUNINGS.items()},
        "missing": missing,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return payload


_DB_CACHE: dict[str, Any] = {"path": None, "data": None}


def _load_db() -> dict | None:
    """读取已经生成的 ``web/voicings.json``（不存在则返回 None）。"""
    path = VOICINGS_PATH
    if _DB_CACHE["path"] == str(path) and _DB_CACHE["data"] is not None:
        return _DB_CACHE["data"]
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    _DB_CACHE["path"] = str(path)
    _DB_CACHE["data"] = data
    return data


def db_voicing(symbol: str) -> list[dict] | None:
    """从静态数据库取指法（按规范化符号）；没有则返回 None。"""
    data = _load_db()
    if not data:
        return None
    try:
        canonical = parse_chord_symbol(symbol).symbol
    except ValueError:
        return None
    standard = data.get("standard") or {}
    items = standard.get(canonical)
    if not items:
        return None
    return [dict(item) for item in items]


def clear_cache() -> None:
    """清空内存缓存（测试用）。"""
    _GENERATED_CACHE.clear()
    _DB_CACHE["path"] = None
    _DB_CACHE["data"] = None


def _now_iso() -> str:
    """生成时间戳（本地时区，秒级）。"""
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")
