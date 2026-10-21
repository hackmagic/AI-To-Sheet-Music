"""乐谱导出子包：MusicXML 与 MIDI。

两个导出器都遵循同一条约定：

* 只读取 ``aitsm.score/v1`` 的 score dict，不做任何音乐决策；
* 失败时抛 :class:`ExportError`（含中文说明），并且**绝不留下半截文件**；
* 返回一份可记录的统计报告（字节数、小节数、事件数……），供 service 层写日志。

对外函数使用惰性导入，避免 ``import aitsm.score.export`` 时就把两个重量级模块拉起来。
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ExportError", "export_musicxml", "export_midi"]


class ExportError(RuntimeError):
    """导出失败（输入谱面不合法、写盘失败、结构自检不通过）。"""


def export_musicxml(score: dict, out_path: Path) -> dict:
    """导出 MusicXML 4.0，返回统计报告。"""
    from .musicxml import export_musicxml as _export_musicxml

    return _export_musicxml(score, out_path)


def export_midi(score: dict, out_path: Path) -> dict:
    """导出标准 MIDI 文件（SMF format 1，PPQ=480），返回统计报告。"""
    from .midi import export_midi as _export_midi

    return _export_midi(score, out_path)
