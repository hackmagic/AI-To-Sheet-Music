"""可演奏性代价：Noise2Fret 五项损失的纯 numpy 移植。

来源与边界
----------
公式提取自 `RiccardoVib/Noise2Fret` 的
``Noise2Fret_Aux_Losses/src/AuxiliaryLoss.py``（实测该仓库**只有代码、没有权重**，
也没有 Google Drive 等外部权重链接）。因此本模块只移植**数学公式**，不加载任何
模型——这恰好是编配层需要的：DP 指法优化器要的是可计算的代价项。

五项（按 Noise2Fret 原实现）：

1. :func:`jaccard_tonal_distance` —— 音级距离（12 维音级向量的 Jaccard 距离）
2. :func:`fret_distance`           —— 把位距离（六弦品位的平均绝对差）
3. :func:`cof_chord_distance`      —— 五度圈距离（和弦在五度圈上的距离）
4. :func:`string_activity_jaccard_loss` —— 弦相似性（哪些弦被拨响）
5. :func:`hand_span_penalty`       —— 手跨可行性（指板跨度超限的比例）

与原实现的差异（都是**刻意**的）：

* 原实现接受 torch 张量与 logits（训练用）；这里接受「品位列表 / 音级集合」等
  **编配层现成的数据结构**，用 numpy 计算，不引入 torch 依赖；
* 原实现的 ``*_soft`` 变体用于梯度回传；DP 是离散搜索不需要梯度，故只保留硬版本；
* 所有函数都对空输入、全静音、非法品位安全返回（编配层可能拿到残缺和弦）。

这些值与 `aitsm/arrangement/fingering_dp.py` 的成本权重是**互补**关系：
那个模块用加权和做路径搜索，本模块提供「单项可演奏性」的独立度量，便于
评审/调试时逐项看是哪一项让某个指法不可弹。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

__all__ = [
    "STANDARD_OPEN_PITCHES",
    "PAD_FRET",
    "jaccard_tonal_distance",
    "fret_distance",
    "cof_chord_distance",
    "string_activity_jaccard_loss",
    "hand_span_penalty",
    "playability_report",
]

#: 标准调弦六弦空弦音高（低音 E → 高音 e），与 Noise2Fret 的 OPEN_PITCHES 一致
STANDARD_OPEN_PITCHES: tuple[int, ...] = (40, 45, 50, 55, 59, 64)

#: 该弦不弹的标记（-1）。Noise2Fret 用 pad_fret=-1，本模块保持一致。
PAD_FRET = -1

#: 五度圈上的音级顺序（C 起步，上行五度）
_CIRCLE_OF_FIFTHS: tuple[int, ...] = (0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5)

#: 每根弦的音级 = (空弦音高 + 品位) % 12
_FRET_CLASSES = 21  # 0=闷弦, 1..20 = 品位 0..19（与 Noise2Fret 的 21 类一致）


def _frets_as_list(frets: Sequence[Any] | None) -> list[int]:
    """把可能含 None 的品位序列规范化成 int 列表（None → PAD_FRET）。"""
    out: list[int] = []
    for value in frets or []:
        if value is None or value == "":
            out.append(PAD_FRET)
            continue
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            out.append(PAD_FRET)
    return out


def _sounding_frets(frets: Sequence[int]) -> list[int]:
    """被按住的品位（排除闷弦与空弦），对应原实现的 ``mask``。"""
    return [fret for fret in frets if fret != PAD_FRET and fret != 0]


def pitch_classes_of(frets: Sequence[int | None]) -> set[int]:
    """六弦品位 → 实际发声的音级集合（闷弦不发声）。"""
    values = _frets_as_list(frets)
    pcs: set[int] = set()
    for index, fret in enumerate(values[: len(STANDARD_OPEN_PITCHES)]):
        if fret == PAD_FRET or fret < 0:
            continue
        pcs.add((STANDARD_OPEN_PITCHES[index] + fret) % 12)
    return pcs


def _binary_vector(pcs: Iterable[int], size: int = 12) -> list[float]:
    vector = [0.0] * size
    for pc in pcs:
        vector[int(pc) % size] = 1.0
    return vector


def jaccard_tonal_distance(pcs_a: Iterable[int] | Sequence[int | None], pcs_b: Iterable[int] | Sequence[int | None]) -> float:
    """音级距离：两组音级的 Jaccard 距离（0 = 完全一致，1 = 无交集）。

    参数可以直接给音级集合 ``{0, 4, 7}``，也可以给六弦品位列表
    ``[None, 3, 2, 0, 1, 0]``（会自动换算成音级）。空集合按「无冲突」处理。

    相同集合直接返回精确 0：Jaccard 的 ``1 - inter/union`` 会因浮点 eps 得到
    2e-9 这种残渣，而「完全一致」在编配里是常见且必须精确的情形。
    """
    set_a = _coerce_pcs(pcs_a)
    set_b = _coerce_pcs(pcs_b)
    if not set_a and not set_b:
        return 0.0
    if set_a == set_b:
        return 0.0
    if not set_a or not set_b:
        return 1.0
    vector_a = _binary_vector(set_a)
    vector_b = _binary_vector(set_b)
    intersection = sum(min(a, b) for a, b in zip(vector_a, vector_b))
    union = sum(max(a, b) for a, b in zip(vector_a, vector_b))
    if union <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - intersection / union)))


def _coerce_pcs(values: Iterable[Any] | None) -> set[int]:
    """把「音级集合」或「六弦品位列表」统一成音级集合。

    判别规则（关键，否则 ``{0,4,7}`` 会被误当成三弦品位列表）：

    * ``set`` / ``frozenset`` → 直接当音级集合；
    * 长度恰为 6 的序列 → 当品位（吉他固定六弦）；
    * 其余序列 → 当音级集合（每个元素 ``%12``）。

    因此 ``[0, 4, 7]``（C 大三和弦音级）不会被当成“三根弦的品位”。
    """
    if isinstance(values, (set, frozenset)):
        return {int(item) % 12 for item in values if _is_int_like(item)}
    materialized = list(values or [])
    if not materialized:
        return set()
    if len(materialized) == 6 and all(
        item is None or _is_int_like(item) for item in materialized
    ):
        return pitch_classes_of(materialized)  # type: ignore[arg-type]
    out: set[int] = set()
    for item in materialized:
        if _is_int_like(item):
            out.add(int(item) % 12)
    return out


def _is_int_like(value: Any) -> bool:
    """是否可安全转成 int（排除 bool 与字符串噪声）。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def fret_distance(frets_a: Sequence[int | None], frets_b: Sequence[int | None]) -> float:
    """把位距离：同一根弦上品位差的绝对值均值（只算两边都发声的弦）。

    对应原实现的 ``fret_distance``：忽略闷弦，空弦（0）参与比较。
    """
    left = _frets_as_list(frets_a)
    right = _frets_as_list(frets_b)
    pairs = [
        (a, b)
        for a, b in zip(left, right)
        if a != PAD_FRET and b != PAD_FRET and a >= 0 and b >= 0
    ]
    if not pairs:
        return 0.0
    return float(sum(abs(a - b) for a, b in pairs) / len(pairs))


def _cof_index(pc: int) -> int:
    try:
        return _CIRCLE_OF_FIFTHS.index(int(pc) % 12)
    except ValueError:  # pragma: no cover - 12 个都覆盖了
        return 0


def _cof_position(pcs: Iterable[int]) -> float | None:
    """音级集合在五度圈上的平均位置（0..11）；空集合返回 ``None``。

    原实现用 ``(pc_vec * cof).sum / pc_vec.sum`` —— 即**加权平均圈位**，
    不是逐对比较。这里忠实复现同一口径。
    """
    values = sorted({int(pc) % 12 for pc in pcs})
    if not values:
        return None
    return sum(_cof_index(pc) for pc in values) / len(values)


def cof_chord_distance(pcs_a: Iterable[int] | Sequence[int | None], pcs_b: Iterable[int] | Sequence[int | None]) -> float:
    """五度圈距离：两组音级在五度圈上平均位置的环距（0 = 同位置，1 = 对跖）。

    口径与 Noise2Fret 的 ``cof_chord_distance`` 一致：

    * 每个集合先算「圈位的加权平均」（不是逐对匹配——那样相同集合会得非零）；
    * 环距 = ``min(|pa-pb|, 12-|pa-pb|)``，最大值 6（对跖）；
    * **除以 6 归一化到 0..1**。

    因此 C（圈位 0）与 F#（圈位 6）是最大距离 1.0；C 与 G（圈位 1）是 1/6。
    空集合按「无冲突」返回 0。
    """
    pos_a = _cof_position(_coerce_pcs(pcs_a))
    pos_b = _cof_position(_coerce_pcs(pcs_b))
    if pos_a is None or pos_b is None:
        return 0.0
    diff = abs(pos_a - pos_b)
    return float(min(diff, 12.0 - diff) / 6.0)


def string_activity_jaccard_loss(frets_a: Sequence[int | None], frets_b: Sequence[int | None]) -> float:
    """弦相似性：两组「哪些弦被拨响」的 Jaccard 距离（0 = 完全一致）。

    对应原实现：比较的是**发声弦的集合**，不看品位。
    """
    left = _frets_as_list(frets_a)
    right = _frets_as_list(frets_b)
    if not left or not right:
        return 0.0
    played_left = {index for index, fret in enumerate(left) if fret != PAD_FRET}
    played_right = {index for index, fret in enumerate(right) if fret != PAD_FRET}
    if not played_left and not played_right:
        return 0.0
    if played_left == played_right:
        return 0.0
    if not played_left or not played_right:
        return 1.0
    intersection = len(played_left & played_right)
    union = len(played_left | played_right)
    return float(max(0.0, min(1.0, 1.0 - intersection / union)))


def hand_span_penalty(frets: Sequence[int | None], *, max_span: int = 6) -> float:
    """手跨可行性：指板跨度超过 ``max_span`` 的比例（0 = 可弹，越大越难）。

    对应原实现 ``hand_span_penalty``：

    * **排除空弦与闷弦**（空弦不算手跨）；
    * 跨度 = max - min；
    * 返回 ``excess / max_span``（原实现同样归一化）。

    ``max_span`` 默认 6 品——与 Noise2Fret 及常见吉他教材的经验阈值一致。
    """
    fretted = _sounding_frets(_frets_as_list(frets))
    if not fretted:
        return 0.0
    span = max(fretted) - min(fretted)
    excess = max(0, span - int(max_span))
    return float(excess / max_span)


def playability_report(
    frets: Sequence[int | None],
    *,
    target_pcs: Iterable[int] | None = None,
    previous: Sequence[int | None] | None = None,
    max_span: int = 6,
) -> dict[str, Any]:
    """把一个指法的五项可演奏性一次算全（调试/评审用）。

    ``target_pcs`` 给出该和弦应有的音级；``previous`` 给出上一个指法（算相对项）。
    缺失的信息就不算对应项（返回 ``None``），不猜。
    """
    frets_list = _frets_as_list(frets)
    report: dict[str, Any] = {
        "frets": frets_list,
        "sounding_strings": sum(1 for fret in frets_list if fret != PAD_FRET),
        "hand_span": hand_span_penalty(frets_list, max_span=max_span),
        "tonal_distance": None,
        "fret_distance": None,
        "cof_distance": None,
        "string_similarity": None,
    }
    if target_pcs is not None:
        report["tonal_distance"] = jaccard_tonal_distance(frets_list, list(target_pcs))
        report["cof_distance"] = cof_chord_distance(frets_list, list(target_pcs))
    if previous is not None:
        report["fret_distance"] = fret_distance(frets_list, previous)
        report["string_similarity"] = string_activity_jaccard_loss(frets_list, previous)
    # 单项归一化后的综合分（只统计有值的项，避免缺项被当成 0 分）
    scored = [
        report["hand_span"],
        report["tonal_distance"],
        report["cof_distance"],
        report["string_similarity"],
    ]
    values = [float(item) for item in scored if isinstance(item, (int, float))]
    report["playability_cost"] = round(sum(values) / len(values), 4) if values else None
    return report
