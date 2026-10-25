"""SheetSage2 真实产物解析的离线测试（不加载模型）。

用**真实** lab 文件内容做样本，验证：
和弦符号转换（含 no-chord）、lab 时间轴、key/structure 解析。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aitsm.transcribe import accompaniment as A

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 和弦符号


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("B:sus4(b7)", "B7sus4"),
        ("G:maj7", "Gmaj7"),
        ("D:maj/5", "D/A"),
        ("B:min", "Bm"),
        ("A:maj", "A"),
        ("C:min7", "Cm7"),
        ("E:dim", "Edim"),
        ("F:aug", "Faug"),
    ],
)
def test_chord_symbol_conversion(raw: str, expected: str) -> None:
    assert A._chord_symbol_from_lab(raw) == expected


@pytest.mark.parametrize("raw", ["N", "N:", "NC", "X", "", "  "])
def test_no_chord_markers_are_dropped(raw: str) -> None:
    """SheetSage2 用 N 表示无和弦，它绝不能变成一个和弦符号。"""
    assert A._chord_symbol_from_lab(raw) == ""


# ---------------------------------------------------------------- lab 解析


def _write_lab(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _fake_sheetsage_output(tmp_path: Path) -> Path:
    """按实测产物结构造一份最小输出目录。"""
    _write_lab(
        tmp_path / "beat.lab",
        [f"{index * 0.32:.2f}\t{index + 1}\t4\t4" for index in range(9)],
    )
    _write_lab(
        tmp_path / "downbeat.lab",
        [f"{index * 1.28:.2f}\t{index + 1}\t4\t4" for index in range(3)],
    )
    _write_lab(
        tmp_path / "chord.lab",
        [
            "0.00\t1.28\tB:sus4(b7)",
            "1.28\t2.57\tG:maj7",
            "2.57\t3.86\tN",
            "3.86\t5.15\tD:maj/5",
        ],
    )
    _write_lab(tmp_path / "key.lab", ["0.00\t14.00\tB:minor"])
    _write_lab(
        tmp_path / "structure.lab",
        ["0.00\t1.28\tchorus", "1.28\t11.60\tchorus", "11.60\t14.00\tinterlude"],
    )
    (tmp_path / "score.abc").write_text("X:1\nT:\nM:4/4\nK:Bm\n", encoding="utf-8")
    return tmp_path


def test_events_from_lab_parses_all_fields(tmp_path: Path) -> None:
    folder = _fake_sheetsage_output(tmp_path)
    warnings: list[str] = []

    payload = A._events_from_sheetsage2_dir(folder, warnings)

    assert payload is not None
    assert warnings == []
    beats = payload["beats"]
    assert beats["time_signature"] == "4/4"
    assert beats["beats_per_bar"] == 4
    assert len(beats["beats"]) == 9
    assert beats["bpm"] == pytest.approx(187.5, abs=0.5)
    assert payload["key"] == {"tonic": "B", "mode": "minor", "confidence": 0.9}
    # N 被跳过，只剩 3 个真和弦
    assert [item["symbol"] for item in payload["chords"]] == ["B7sus4", "Gmaj7", "D/A"]
    assert [item["kind"] for item in payload["sections"]] == ["chorus", "chorus", "interlude"]
    assert payload["abc"].startswith("X:1")


def test_events_from_lab_returns_none_for_empty_dir(tmp_path: Path) -> None:
    assert A._events_from_sheetsage2_dir(tmp_path, []) is None


def test_events_from_lab_survives_missing_optional_files(tmp_path: Path) -> None:
    """缺 downbeat/structure 也要能出结果（不伪造，只留空）。"""
    _write_lab(tmp_path / "beat.lab", ["0.00\t1\t4\t4", "0.50\t2\t4\t4"])
    _write_lab(tmp_path / "chord.lab", ["0.00\t0.50\tC:maj"])

    payload = A._events_from_sheetsage2_dir(tmp_path, [])

    assert payload is not None
    assert payload["sections"] == []
    assert len(payload["chords"]) == 1
    # 没有 downbeat 时按拍号推导，而不是留空
    assert payload["beats"]["downbeats"]


# ---------------------------------------------------------------- 真实产物回归


_REAL = sorted((ROOT / "work").glob("sheetsage-smoke/run-*"))


@pytest.mark.skipif(not _REAL, reason="本地没有 SheetSage2 真实运行产物")
def test_real_sheetsage_output_parses() -> None:
    folder = _REAL[-1]
    warnings: list[str] = []
    payload = A._events_from_sheetsage2_dir(folder, warnings)

    assert payload is not None
    assert payload["beats"]["beats"], "真实产物必须有拍点"
    assert payload["chords"], "真实产物必须有和弦"
    assert payload["key"]["tonic"]
    assert all(chord["symbol"] for chord in payload["chords"]), "no-chord 必须已被过滤"


# ---------------------------------------------------------------- 引擎名传递


def test_accompaniment_stage_analysis_dicts_shapes():
    from aitsm.contracts import AccompanimentResult, BeatGrid, Section
    from aitsm.pipeline import accompaniment_stage

    result = AccompanimentResult(
        beats=BeatGrid(bpm=120.0, beats=[0.0, 0.5], beats_per_bar=4, time_signature="4/4"),
        key=_key(),
        sections=[Section(start=0.0, end=2.0, kind="chorus")],
    )

    data = accompaniment_stage.analysis_dicts(result)

    assert data["beats"]["bpm"] == 120.0
    assert data["beats"]["source"] == "sheetsage2"
    assert data["key"]["name"] == "C major"
    assert data["sections"][0]["label"] == "chorus"
    # 段落没有 energy 时按 kind 给保守默认，不编造具体数值
    assert 0.0 < data["sections"][0]["energy"] <= 1.0


def _key():
    from aitsm.contracts import KeySignature

    return KeySignature(tonic="C", mode="major", confidence=0.9, fifths=0)


def test_symbolic_stage_reports_sheetsage2_engine(tmp_path: Path):
    """tools.sheetsage2 存在时，伴奏层不能再被标成降级。"""
    from aitsm.pipeline import symbolic_stage

    analysis = json.loads((ROOT / "artifacts" / "analysis.json").read_text(encoding="utf-8"))
    analysis.setdefault("tools", {})["sheetsage2"] = {"engine": "sheetsage2", "available": True}

    result = symbolic_stage.accompaniment_from_analysis(analysis)

    assert result.engine == "sheetsage2"
    assert result.warnings == [] or all("本机实际使用" not in item for item in result.warnings)


def test_symbolic_stage_flags_degradation_without_sheetsage2(tmp_path: Path):
    from aitsm.pipeline import symbolic_stage

    analysis = json.loads((ROOT / "artifacts" / "analysis.json").read_text(encoding="utf-8"))
    analysis.setdefault("tools", {}).pop("sheetsage2", None)

    result = symbolic_stage.accompaniment_from_analysis(analysis)

    assert result.engine != "sheetsage2"
    assert any("本机实际使用" in item for item in result.warnings)
