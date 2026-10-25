"""fusion 层（v2 架构）：把两路联合转录融合成 :class:`aitsm.contracts.SymbolicScore`。

对外接口
--------
============  =================================================================
``fuse``      VocalResult + AccompanimentResult -> SymbolicScore
``snap_notes_to_beats``  只吸音头、保留时值的网格吸附
``write_performance_midi``  总谱 -> 合法 performance SMF（format 1 / PPQ 480）
``quantize_symbolic``  节奏量化（DuET 优先，缺依赖时显式降级到 metrical HMM）
``is_available`` / ``probe``  DuET 可用性探测（不联网）
``write_musicxml``  总谱 -> MusicXML 4.0 文本
``read_musicxml``   MusicXML 文本 -> 总谱（可带 sidecar 覆盖绝对时间）
============  =================================================================

``aitsm.contracts.SymbolicScore.to_musicxml`` / ``from_musicxml`` 惰性委托给
:mod:`aitsm.fusion.musicxml_io`，所以本模块在 ``import aitsm.fusion`` 时建立
同名导出即可，不会形成循环导入。
"""

from __future__ import annotations

from .musicxml_io import (
    DIVISIONS,
    MUSICXML_VERSION,
    format_chord_symbol,
    parse_chord_symbol,
    read_musicxml,
    write_musicxml,
)
from .quantize import (
    DUET_ENV,
    ENGINE_DUET,
    ENGINE_HMM,
    is_available,
    probe,
    quantize_symbolic,
)
from .skeleton import (
    FUSION_ENGINE,
    fuse,
    snap_notes_to_beats,
    write_performance_midi,
)

__all__ = [
    "DIVISIONS",
    "DUET_ENV",
    "ENGINE_DUET",
    "ENGINE_HMM",
    "FUSION_ENGINE",
    "MUSICXML_VERSION",
    "format_chord_symbol",
    "fuse",
    "is_available",
    "parse_chord_symbol",
    "probe",
    "quantize_symbolic",
    "read_musicxml",
    "snap_notes_to_beats",
    "write_musicxml",
    "write_performance_midi",
]
