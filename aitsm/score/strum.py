"""节奏型库（F13）：4/4 与 3/4 的扫弦 / 分解型，以及按段落选型的规则。

数据形态与 ``samples/demo-score.json`` 的 ``strum_patterns`` 完全一致（可直接写进
``score["strum_patterns"]``），额外带有 ``kind`` / ``energy`` / ``styles`` /
``difficulty`` 供 :func:`pick_pattern` 打分使用（schema 允许扩展字段）。

事件约定
--------
- ``beat``：小节内以四分音符为一拍的 0 基偏移（``0`` = 正拍第一拍）；
- ``dur``：时值，四分音符为 1 拍（八分为 0.5，三连八分为 ``"1/3"``）；
- ``stroke``：``down`` 下扫 / ``up`` 上扫 / ``bass`` 低音（拇指拨根音）/
  ``pluck`` 单音分解 / ``mute`` 闷扫；
- ``strings``：可选的 1 基弦号（1 = 最高音弦 e，6 = 最低音弦 E），仅 ``bass`` /
  ``pluck`` 需要；扫弦留空表示「按指法自动决定弦组」。

全部为纯数据 + 纯函数，无随机数：同样的输入永远选同一个型（便于回归测试）。
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = [
    "PATTERNS",
    "get_pattern",
    "pick_pattern",
    "pattern_ids",
    "patterns_for_meter",
    "expand_pattern",
    "pattern_note_count",
]

# --------------------------------------------------------------------------
# 节奏型定义
# --------------------------------------------------------------------------

def _e(beat: float, dur: float, stroke: str, strings: list[int] | None = None, accent: bool = False) -> dict:
    """构造一个节奏事件（内部辅助）。"""
    event: dict[str, Any] = {"beat": beat, "dur": dur, "stroke": stroke}
    if strings is not None:
        event["strings"] = strings
    if accent:
        event["accent"] = True
    return event


#: 民谣八分：最通用的 4/4 扫弦
_POP8 = {
    "id": "pop8",
    "name": "民谣八分",
    "kind": "strum",
    "beats_per_bar": 4,
    "subdivision": "eighth",
    "display": "↓ ↓↑ ↑↓↑",
    "energy": 0.5,
    "styles": ["pop", "folk", "rock", "general"],
    "difficulty": 2,
    "events": [
        _e(0, 0.5, "down", accent=True),
        _e(0.5, 0.5, "up"),
        _e(1, 0.5, "down"),
        _e(1.5, 0.5, "up"),
        _e(2, 0.5, "down", accent=True),
        _e(2.5, 0.5, "up"),
        _e(3, 0.5, "down"),
        _e(3.5, 0.5, "up"),
    ],
}

#: 四分扫弦：慢歌 / 入门
_FOLK4 = {
    "id": "folk4",
    "name": "四分扫弦",
    "kind": "strum",
    "beats_per_bar": 4,
    "subdivision": "quarter",
    "display": "↓ ↓↑ ↓",
    "energy": 0.3,
    "styles": ["folk", "pop", "ballad"],
    "difficulty": 1,
    "events": [
        _e(0, 1, "down", accent=True),
        _e(1, 0.5, "down"),
        _e(1.5, 0.5, "up"),
        _e(2, 1, "down", accent=True),
        _e(3, 1, "down"),
    ],
}

#: 慢歌分解 53231323：钢琴式分解，抒情段落
_BALLAD8 = {
    "id": "ballad8",
    "name": "分解八分 53231323",
    "kind": "arpeggio",
    "beats_per_bar": 4,
    "subdivision": "eighth",
    "display": "T 3 2 3 1 3 2 3",
    "energy": 0.2,
    "styles": ["ballad", "folk", "pop"],
    "difficulty": 2,
    "events": [
        _e(0, 0.5, "bass", [5]),
        _e(0.5, 0.5, "pluck", [3]),
        _e(1, 0.5, "pluck", [2]),
        _e(1.5, 0.5, "pluck", [3]),
        _e(2, 0.5, "pluck", [1]),
        _e(2.5, 0.5, "pluck", [3]),
        _e(3, 0.5, "pluck", [2]),
        _e(3.5, 0.5, "pluck", [3]),
    ],
}

#: 四分分解：极简慢歌（T 3 2 1）
_BALLAD4 = {
    "id": "ballad4",
    "name": "四分分解",
    "kind": "arpeggio",
    "beats_per_bar": 4,
    "subdivision": "quarter",
    "display": "T 3 2 1",
    "energy": 0.15,
    "styles": ["ballad", "pop"],
    "difficulty": 1,
    "events": [
        _e(0, 1, "bass", [5]),
        _e(1, 1, "pluck", [3]),
        _e(2, 1, "pluck", [2]),
        _e(3, 1, "pluck", [1]),
    ],
}

#: 十六分扫弦：副歌高潮
_POP16 = {
    "id": "pop16",
    "name": "流行十六分",
    "kind": "strum",
    "beats_per_bar": 4,
    "subdivision": "sixteenth",
    "display": "↓ ↓↑↓ ↑↓↑↓↑",
    "energy": 0.8,
    "styles": ["pop", "rock"],
    "difficulty": 4,
    "events": [
        _e(0, 0.25, "down", accent=True),
        _e(0.25, 0.25, "up"),
        _e(0.5, 0.25, "down"),
        _e(0.75, 0.25, "up"),
        _e(1, 0.25, "down"),
        _e(1.25, 0.25, "up"),
        _e(1.5, 0.25, "down"),
        _e(1.75, 0.25, "up"),
        _e(2, 0.25, "down", accent=True),
        _e(2.25, 0.25, "up"),
        _e(2.5, 0.25, "down"),
        _e(2.75, 0.25, "up"),
        _e(3, 0.25, "down"),
        _e(3.25, 0.25, "up"),
        _e(3.5, 0.25, "down"),
        _e(3.75, 0.25, "up"),
    ],
}

#: 摇滚八分：全部下扫，重音在 1、3 拍
_DRIVE8 = {
    "id": "drive8",
    "name": "摇滚八分",
    "kind": "strum",
    "beats_per_bar": 4,
    "subdivision": "eighth",
    "display": "↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓",
    "energy": 0.75,
    "styles": ["rock", "pop"],
    "difficulty": 2,
    "events": [
        _e(0, 0.5, "down", accent=True),
        _e(0.5, 0.5, "down"),
        _e(1, 0.5, "down"),
        _e(1.5, 0.5, "down"),
        _e(2, 0.5, "down", accent=True),
        _e(2.5, 0.5, "down"),
        _e(3, 0.5, "down"),
        _e(3.5, 0.5, "down"),
    ],
}

#: 乡村 boom-chick：低音 + 切分和弦
_COUNTRY = {
    "id": "country",
    "name": "乡村 boom-chick",
    "kind": "strum",
    "beats_per_bar": 4,
    "subdivision": "eighth",
    "display": "T ↓ T ↓↑",
    "energy": 0.45,
    "styles": ["country", "folk"],
    "difficulty": 3,
    "events": [
        _e(0, 0.5, "bass", [5], accent=True),
        _e(0.5, 0.5, "down"),
        _e(1, 0.5, "up"),
        _e(1.5, 0.5, "down"),
        _e(2, 0.5, "bass", [6], accent=True),
        _e(2.5, 0.5, "up"),
        _e(3, 0.5, "down"),
        _e(3.5, 0.5, "up"),
    ],
}

#: 切分八分：副歌推进
_SYNC8 = {
    "id": "sync8",
    "name": "切分八分",
    "kind": "strum",
    "beats_per_bar": 4,
    "subdivision": "eighth",
    "display": "↓ ·↓↑ ·↓↑",
    "energy": 0.6,
    "styles": ["pop", "rock", "folk"],
    "difficulty": 3,
    "events": [
        _e(0, 0.5, "down", accent=True),
        _e(0.75, 0.25, "up"),
        _e(1, 0.5, "down"),
        _e(1.5, 0.5, "up"),
        _e(2, 0.5, "down", accent=True),
        _e(2.75, 0.25, "up"),
        _e(3, 0.5, "down"),
        _e(3.5, 0.5, "up"),
    ],
}

#: 闷音八分：低能量摇滚 / 说唱段落
_MUTE8 = {
    "id": "mute8",
    "name": "闷音八分",
    "kind": "strum",
    "beats_per_bar": 4,
    "subdivision": "eighth",
    "display": "× × × × × × × ×",
    "energy": 0.65,
    "styles": ["rock", "funk"],
    "difficulty": 3,
    "events": [
        _e(0, 0.5, "mute", accent=True),
        _e(0.5, 0.5, "mute"),
        _e(1, 0.5, "mute"),
        _e(1.5, 0.5, "mute"),
        _e(2, 0.5, "mute", accent=True),
        _e(2.5, 0.5, "mute"),
        _e(3, 0.5, "mute"),
        _e(3.5, 0.5, "mute"),
    ],
}

#: 分解 63231323：更宽广的分解（低音换弦）
_ARP6 = {
    "id": "arp6",
    "name": "分解 63231323",
    "kind": "arpeggio",
    "beats_per_bar": 4,
    "subdivision": "eighth",
    "display": "T 3 2 3 1 3 2 3",
    "energy": 0.25,
    "styles": ["ballad", "folk"],
    "difficulty": 2,
    "events": [
        _e(0, 0.5, "bass", [6]),
        _e(0.5, 0.5, "pluck", [3]),
        _e(1, 0.5, "pluck", [2]),
        _e(1.5, 0.5, "pluck", [3]),
        _e(2, 0.5, "pluck", [1]),
        _e(2.5, 0.5, "pluck", [3]),
        _e(3, 0.5, "pluck", [2]),
        _e(3.5, 0.5, "pluck", [3]),
    ],
}

#: 3/4 民谣华尔兹
_WALTZ3 = {
    "id": "waltz3",
    "name": "三拍子扫弦",
    "kind": "strum",
    "beats_per_bar": 3,
    "subdivision": "quarter",
    "display": "↓ ↓↑ ↓↑",
    "energy": 0.4,
    "styles": ["pop", "folk", "ballad"],
    "difficulty": 2,
    "events": [
        _e(0, 1, "down", accent=True),
        _e(1, 0.5, "down"),
        _e(1.5, 0.5, "up"),
        _e(2, 0.5, "down"),
        _e(2.5, 0.5, "up"),
    ],
}

#: 3/4 分解
_WALTZ_ARP = {
    "id": "waltz_arp",
    "name": "三拍子分解",
    "kind": "arpeggio",
    "beats_per_bar": 3,
    "subdivision": "eighth",
    "display": "T 3 2 3 1 3",
    "energy": 0.2,
    "styles": ["ballad", "folk", "pop"],
    "difficulty": 1,
    "events": [
        _e(0, 0.5, "bass", [5]),
        _e(0.5, 0.5, "pluck", [3]),
        _e(1, 0.5, "pluck", [2]),
        _e(1.5, 0.5, "pluck", [3]),
        _e(2, 0.5, "pluck", [1]),
        _e(2.5, 0.5, "pluck", [3]),
    ],
}

#: 3/4 强力华尔兹（每拍下扫）
_WALTZ_DRIVE = {
    "id": "waltz_drive",
    "name": "三拍子强扫",
    "kind": "strum",
    "beats_per_bar": 3,
    "subdivision": "quarter",
    "display": "↓ ↓ ↓",
    "energy": 0.7,
    "styles": ["rock", "pop"],
    "difficulty": 2,
    "events": [
        _e(0, 1, "down", accent=True),
        _e(1, 1, "down"),
        _e(2, 1, "down"),
    ],
}


#: 6/8 民谣摇摆（六个八分音符，一拍半一组）
_SIX8 = {
    "id": "six8",
    "name": "6/8 民谣摇摆",
    "kind": "strum",
    "beats_per_bar": 6,
    "subdivision": "eighth",
    "display": "↓ ↓↑ ↓↑",
    "energy": 0.5,
    "styles": ["pop", "folk", "ballad", "country"],
    "difficulty": 2,
    "events": [
        _e(0, 0.5, "down", accent=True),
        _e(0.5, 0.5, "up"),
        _e(1, 0.5, "down"),
        _e(1.5, 0.5, "up"),
        _e(2, 0.5, "down"),
        _e(2.5, 0.5, "up"),
    ],
}

#: 2/4 进行曲 / 波尔卡
_TWO4 = {
    "id": "two4",
    "name": "2/4 进行曲",
    "kind": "strum",
    "beats_per_bar": 2,
    "subdivision": "eighth",
    "display": "↓ ↑ ↓ ↑",
    "energy": 0.6,
    "styles": ["pop", "folk", "rock", "country"],
    "difficulty": 1,
    "events": [
        _e(0, 0.5, "down", accent=True),
        _e(0.5, 0.5, "up"),
        _e(1, 0.5, "down"),
        _e(1.5, 0.5, "up"),
    ],
}

#: 12/8 慢板（六个八分音符，一小节 6 个四分音符位置）
_TWELVE8 = {
    "id": "twelve8",
    "name": "12/8 慢板",
    "kind": "strum",
    "beats_per_bar": 12,
    "subdivision": "eighth",
    "display": "↓ ↓↑ ↓↑ × 2",
    "energy": 0.45,
    "styles": ["ballad", "pop", "folk"],
    "difficulty": 2,
    "events": [
        _e(0, 0.5, "down", accent=True),
        _e(0.5, 0.5, "up"),
        _e(1, 0.5, "down"),
        _e(1.5, 0.5, "up"),
        _e(2, 0.5, "down"),
        _e(2.5, 0.5, "up"),
        _e(3, 0.5, "down"),
        _e(3.5, 0.5, "up"),
        _e(4, 0.5, "down"),
        _e(4.5, 0.5, "up"),
        _e(5, 0.5, "down"),
        _e(5.5, 0.5, "up"),
    ],
}


#: 全部节奏型（id -> 定义）
PATTERNS: dict[str, dict] = {
    pattern["id"]: pattern
    for pattern in (
        _POP8,
        _FOLK4,
        _BALLAD8,
        _BALLAD4,
        _POP16,
        _DRIVE8,
        _COUNTRY,
        _SYNC8,
        _MUTE8,
        _ARP6,
        _WALTZ3,
        _WALTZ_ARP,
        _WALTZ_DRIVE,
        _SIX8,
        _TWO4,
        _TWELVE8,
    )
}

#: 段落类型 -> 目标能量（section 未给 energy 时的兜底）
_KIND_ENERGY: dict[str, float] = {
    "intro": 0.2,
    "verse": 0.42,
    "prechorus": 0.6,
    "chorus": 0.8,
    "bridge": 0.5,
    "interlude": 0.35,
    "instrumental": 0.4,
    "outro": 0.3,
    "solo": 0.5,
}

#: 中文段落名 -> 目标能量（analysis 只给中文 label 时使用）
_LABEL_ENERGY: dict[str, float] = {
    "前奏": 0.2,
    "主歌": 0.42,
    "预副歌": 0.6,
    "副歌": 0.8,
    "间奏": 0.35,
    "桥段": 0.5,
    "尾奏": 0.3,
    "结束": 0.25,
    "导歌": 0.6,
}

#: 难度名 -> 允许的最高和弦/节奏难度
_DIFFICULTY_CAP: dict[str, int] = {
    "beginner": 2,
    "elementary": 2,
    "easy": 2,
    "simple": 2,
    "入门": 2,
    "初级": 2,
    "simple-1": 2,
    "intermediate": 3,
    "medium": 3,
    "中等": 3,
    "中级": 3,
    "advanced": 5,
    "hard": 5,
    "进阶": 5,
    "高级": 5,
}

#: 风格 -> 加成（(pattern_id, 权重)）
_STYLE_BONUS: dict[str, dict[str, float]] = {
    "folk": {"folk4": 0.12, "ballad8": 0.08, "country": 0.05},
    "ballad": {"ballad8": 0.12, "ballad4": 0.1, "waltz_arp": 0.08},
    "rock": {"drive8": 0.12, "mute8": 0.08, "pop16": 0.06},
    "pop": {"pop8": 0.08, "pop16": 0.05},
    "country": {"country": 0.15},
    "funk": {"mute8": 0.12},
}


#: 风格 -> 目标能量缩放（抒情段落更稀疏、摇滚段落更密）
_STYLE_ENERGY_SCALE: dict[str, float] = {
    "ballad": 0.65,
    "folk": 0.85,
    "pop": 1.0,
    "country": 1.0,
    "rock": 1.15,
    "funk": 1.1,
}


def _section_energy(section: dict) -> float:
    """段落能量：显式 energy > kind > 中文 label > 默认 0.5。"""
    if isinstance(section.get("energy"), (int, float)):
        return float(section["energy"])
    kind = str(section.get("kind") or "").strip().lower()
    if kind in _KIND_ENERGY:
        return _KIND_ENERGY[kind]
    label = str(section.get("label") or section.get("name") or "").strip()
    if label in _LABEL_ENERGY:
        return _LABEL_ENERGY[label]
    for key, value in _LABEL_ENERGY.items():
        if key and key in label:
            return value
    return 0.5


def _difficulty_cap(meta: dict) -> int:
    """根据 meta.difficulty 得到节奏难度上限。"""
    raw = meta.get("difficulty")
    if isinstance(raw, (int, float)):
        return max(1, min(5, int(raw)))
    text = str(raw or "intermediate").strip().lower()
    return _DIFFICULTY_CAP.get(text, 3)


def _meter(meta: dict) -> int:
    """每小节拍数（用于选节奏型）。

    支持的拍号族：4/4 族（含 4/8 等复拍）、3/4、2/4、6/8、12/8；
    其余拍号（5/4、7/8 等）退回按 4 拍处理，由校验器给告警。
    """
    ts = meta.get("time_signature") or [4, 4]
    try:
        beats = int(ts[0])
    except (TypeError, ValueError, IndexError):
        beats = 4
    if beats in (2, 3, 6, 12):
        return beats
    return 4


def patterns_for_meter(beats_per_bar: int) -> list[dict]:
    """取某拍号的可用节奏型（严格匹配拍号分子）。"""
    return [
        pattern
        for pattern in PATTERNS.values()
        if int(pattern["beats_per_bar"]) == int(beats_per_bar)
    ]


def pattern_ids() -> list[str]:
    """全部节奏型 id（稳定顺序）。"""
    return list(PATTERNS)


def _copy_event(event: dict) -> dict:
    """复制事件（strings 深拷贝，不额外塞 null 字段）。"""
    copy = dict(event)
    if copy.get("strings") is not None:
        copy["strings"] = list(copy["strings"])
    return copy


def get_pattern(pattern_id: str | None) -> dict | None:
    """按 id 取节奏型（返回副本，避免调用方污染库）。"""
    if not pattern_id:
        return None
    pattern = PATTERNS.get(str(pattern_id))
    if pattern is None:
        return None
    return {**pattern, "events": [_copy_event(event) for event in pattern["events"]]}


def pick_pattern(section: dict, meta: dict) -> str:
    """按段落特征挑节奏型，返回 id。

    - 副歌比主歌密（能量高 → 选密度更大的型）；
    - 前奏 / 主歌允许分解型；
    - 3/4 拍只用三拍子的型；
    - ``meta["difficulty"]`` 限制最高难度；
    - 同分时按 id 字典序，保证确定性。
    """
    beats = _meter(meta)
    candidates = patterns_for_meter(beats)
    if not candidates:  # 5/4、7/8 等特殊拍号：退回 4 拍族
        candidates = patterns_for_meter(4)
    if not candidates:  # 理论上不会发生（库中必有 4 拍族）
        return "pop8"

    style = str(meta.get("style") or "pop").strip().lower()
    target = _section_energy(section) * _STYLE_ENERGY_SCALE.get(style, 1.0)
    cap = _difficulty_cap(meta)
    tempo = meta.get("tempo")
    try:
        tempo_value = float(tempo) if tempo is not None else 90.0
    except (TypeError, ValueError):
        tempo_value = 90.0

    allowed = [pattern for pattern in candidates if int(pattern["difficulty"]) <= cap]
    pool = allowed or candidates  # 都超难度时退回全部，校验器会提示

    bonus_map = _STYLE_BONUS.get(style, {})

    def _score(pattern: dict) -> tuple:
        diff = abs(float(pattern["energy"]) - target)
        # 速度修正：极慢（<70）偏向细分更粗的型，极快（>140）偏向八分
        if tempo_value < 70 and pattern["subdivision"] in ("sixteenth",):
            diff += 0.15
        if tempo_value > 140 and pattern["subdivision"] == "quarter":
            diff += 0.1
        diff -= bonus_map.get(pattern["id"], 0.0)
        # 段落类型微调：主歌略偏向稀疏、副歌略偏向密集
        kind = str(section.get("kind") or "").lower()
        if kind == "verse" and pattern["subdivision"] in ("quarter", "eighth"):
            diff -= 0.03
        if kind == "chorus" and pattern["subdivision"] in ("eighth", "sixteenth") and float(pattern["energy"]) >= 0.55:
            diff -= 0.05
        if kind == "intro" and pattern["kind"] == "arpeggio":
            diff -= 0.05
        if kind == "outro" and pattern["kind"] == "arpeggio":
            diff -= 0.03
        return (round(diff, 6), pattern["id"])

    pool = sorted(pool, key=_score)
    return pool[0]["id"]


# --------------------------------------------------------------------------
# 节奏型展开（供 arrange 生成 hits、供 MIDI 导出用）
# --------------------------------------------------------------------------

def _sounding_strings(frets: Iterable[int | None]) -> list[int]:
    """指法 -> 发声弦的 1 基弦号，低音弦在前（6 → 1）。"""
    result: list[int] = []
    values = list(frets)
    for index, fret in enumerate(values):
        if fret is None or int(fret) < 0:
            continue
        result.append(6 - index)
    return result


def _nearest_string(number: int, sounding: list[int]) -> int:
    """在发声弦里找离 ``number`` 最近的弦号（分解音遇到闷弦时换弦）。"""
    if not sounding:
        return number
    if number in sounding:
        return number
    return min(sounding, key=lambda candidate: (abs(candidate - number), -candidate))


def expand_pattern(
    pattern_id: str,
    frets: list[int | None] | None = None,
    *,
    include_strings: bool = True,
) -> list[dict]:
    """把节奏型展开成一小节的 ``guitar.hits``。

    参数 ``frets`` 是「低到高 6 弦」的品位数组（与 ``voicing.frets`` 同序）。
    给出 ``frets`` 时，扫弦会自动决定实际发声的弦组：下扫从低音弦扫到高音弦，
    上扫只扫高音 3-4 根；分解音若该弦闷掉会自动找最近的发声弦替换。
    """
    pattern = PATTERNS.get(str(pattern_id))
    if pattern is None:
        return []
    sounding = _sounding_strings(frets) if frets else [6, 5, 4, 3, 2, 1]
    low_to_high = list(sounding)  # 6 → 1
    high_to_low = list(reversed(sounding))  # 1 → 6

    hits: list[dict] = []
    for event in pattern["events"]:
        hit: dict[str, Any] = {
            "beat": event["beat"],
            "dur": event["dur"],
            "stroke": event["stroke"],
        }
        if event.get("accent"):
            hit["accent"] = True
        explicit = event.get("strings")
        if explicit:
            strings = [_nearest_string(int(s), sounding) for s in explicit]
        elif event["stroke"] == "down":
            strings = low_to_high
        elif event["stroke"] == "up":
            strings = high_to_low[: min(4, len(high_to_low))]
        elif event["stroke"] == "mute":
            strings = low_to_high
        else:  # bass / pluck 且未指定弦：取最低发声音
            strings = low_to_high[:1]
        if include_strings and frets:
            indexed: list[list[int | None]] = []
            for number in strings:
                index = 6 - number
                fret = frets[index] if 0 <= index < len(frets) else None
                indexed.append([number, None if fret is None or int(fret) < 0 else int(fret)])
            hit["strings"] = indexed
        elif include_strings:
            hit["strings"] = [[number, None] for number in strings]
        hits.append(hit)
    return hits


def pattern_note_count(pattern_id: str) -> int:
    """节奏型的事件数（用于信息展示与测试）。"""
    pattern = PATTERNS.get(str(pattern_id))
    return len(pattern["events"]) if pattern else 0
