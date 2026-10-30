"""指法库单测：等音覆盖 / 手型合法性 / 排序 / 难度偏好 / 斜杠和弦。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aitsm.score import voicings as V
from aitsm.score.model import format_chord_symbol, parse_chord_symbol

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICINGS_JSON = REPO_ROOT / "web" / "voicings.json"

QUALITIES = ["maj", "min", "7", "m7", "maj7"]
ROOT_PCS = list(range(12))


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """生成一次指法库，写到临时文件。

    ``build_voicing_db`` 会**真的写盘**。传仓库路径会让每次 ``pytest`` 都改写被
    git 跟踪的 ``web/voicings.json``（只有 ``generated_at`` 变），留下一份脏 diff，
    历史上已经因此提交过一次「还原时间戳噪声」。这里必须用 tmp 路径。
    """
    V.clear_cache()
    target = tmp_path_factory.mktemp("voicings") / "voicings.json"
    return V.build_voicing_db(target)


def _frets(voicing: dict) -> list[int]:
    return [int(fret) for fret in voicing["frets"]]


# --------------------------------------------------------------------------
# 契约与结构
# --------------------------------------------------------------------------


def test_db_structure(db: dict) -> None:
    assert db["version"] == 1
    assert isinstance(db["tunings"], dict) and db["tunings"]
    assert isinstance(db["standard"], dict) and db["standard"]
    assert db["missing"] == [], f"覆盖缺口: {db['missing']}"
    for symbol, items in db["standard"].items():
        assert items, f"{symbol} 没有指法"
        ids = [item["id"] for item in items]
        assert len(ids) == len(set(ids)), f"{symbol} 指法 id 重复"
        for item in items:
            assert item["symbol"] == symbol
            assert set(item) >= {
                "id",
                "symbol",
                "frets",
                "fingers",
                "base_fret",
                "barres",
                "difficulty",
                "tags",
                "source",
            }
            assert len(item["frets"]) == 6
            assert 1 <= int(item["difficulty"]) <= 5


def test_voicings_json_matches_built_db(db: dict) -> None:
    if not VOICINGS_JSON.exists():  # pragma: no cover - CI 首次运行
        pytest.skip("web/voicings.json 尚未生成")
    on_disk = json.loads(VOICINGS_JSON.read_text(encoding="utf-8"))
    assert set(on_disk["standard"]) == set(db["standard"])


def test_24_keys_covered(db: dict) -> None:
    """24 个大小调的主和弦（maj/min）都要有把位。"""
    for pc in ROOT_PCS:
        for quality in ("maj", "min"):
            symbol = format_chord_symbol(pc, quality)
            assert db["standard"].get(symbol), f"{symbol} 缺少把位"


# --------------------------------------------------------------------------
# 等音（enharmonic）覆盖：12 半音 × 5 性质 × 升/降两种写法 = 120
# --------------------------------------------------------------------------


@pytest.mark.parametrize("prefer_flats", [False, True], ids=["sharp", "flat"])
@pytest.mark.parametrize("quality", QUALITIES)
@pytest.mark.parametrize("pc", ROOT_PCS)
def test_enharmonic_lookup(db: dict, pc: int, quality: str, prefer_flats: bool) -> None:
    symbol = format_chord_symbol(pc, quality, prefer_flats=prefer_flats)
    items = db["standard"].get(symbol)
    assert items, f"{symbol} 在指法库里查不到"
    voicing = V.voicing_for(symbol)
    assert voicing is not None, f"voicing_for({symbol!r}) 返回 None"
    assert not V.validate_voicing(_frets(voicing), symbol), (symbol, V.validate_voicing(_frets(voicing), symbol))
    assert voicing["symbol"] == symbol or parse_chord_symbol(voicing["symbol"]).root_pc == pc


def test_enharmonic_pairs_share_shapes() -> None:
    """升/降号两种写法必须能查到同一组把位。"""
    pairs = [
        ("Bb", "A#"),
        ("Eb", "D#"),
        ("Ab", "G#"),
        ("Db", "C#"),
        ("Gb", "F#"),
        ("Bbm7", "A#m7"),
        ("Ebmaj7", "D#maj7"),
    ]
    for flat_name, sharp_name in pairs:
        flat = V.voicing_for(flat_name)
        sharp = V.voicing_for(sharp_name)
        assert flat and sharp
        assert _frets(flat) == _frets(sharp), f"{flat_name}/{sharp_name} 指法不一致"
        assert flat["symbol"] == flat_name
        assert sharp["symbol"] == sharp_name


def test_enharmonic_variants() -> None:
    assert set(V.enharmonic_variants("Bb")) == {"Bb", "A#"}
    assert set(V.enharmonic_variants("C#m")) == {"C#m", "Dbm"}
    assert V.enharmonic_variants("Am") == ["Am"]
    assert V.uses_flats("Ebm7") and not V.uses_flats("D#m7")


# --------------------------------------------------------------------------
# 手型合法性：这是“像不像真谱”的关键
# --------------------------------------------------------------------------


def test_curated_first_shapes() -> None:
    """手工校验的常用形状必须是首推。"""
    expected = {
        "C": [-1, 3, 2, 0, 1, 0],
        "G": [3, 2, 0, 0, 0, 3],
        "D": [-1, -1, 0, 2, 3, 2],
        "A": [-1, 0, 2, 2, 2, 0],
        "E": [0, 2, 2, 1, 0, 0],
        "Am": [-1, 0, 2, 2, 1, 0],
        "Em": [0, 2, 2, 0, 0, 0],
        "Dm": [-1, -1, 0, 2, 3, 1],
        "Bm": [-1, 2, 4, 4, 3, 2],
        "C7": [-1, 3, 2, 3, 1, 0],
        "G7": [3, 2, 0, 0, 0, 1],
        "Am7": [-1, 0, 2, 0, 1, 0],
        "Cmaj7": [-1, 3, 2, 0, 0, 0],
        "Dm7": [-1, -1, 0, 2, 1, 1],
        "Fmaj7": [-1, -1, 3, 2, 1, 0],
        "Bm": [-1, 2, 4, 4, 3, 2],
        "Bb": [-1, 1, 3, 3, 3, 1],
    }
    for symbol, frets in expected.items():
        voicing = V.voicing_for(symbol)
        assert voicing is not None, symbol
        assert _frets(voicing) == frets, f"{symbol} 首推 {_frets(voicing)}，期望 {frets}"


def test_f_beginner_before_barre() -> None:
    """F 的简易版先给初学者，完整横按版也必须存在。"""
    items = V.voicings_for("F", limit=8)
    frets = [_frets(item) for item in items]
    assert frets[0] == [-1, -1, 3, 2, 1, 1]
    assert [1, 3, 3, 2, 1, 1] in frets


def test_voicing_for_difficulty_preference() -> None:
    """中高级谱面要拿到完整横按版，入门谱面用简易版。"""
    beginner = V.voicing_for("F", prefer={"max_difficulty": 2})
    full = V.voicing_for("F", prefer={"min_sounding": 6})
    assert beginner and full
    assert _frets(beginner) == [-1, -1, 3, 2, 1, 1]
    assert _frets(full) == [1, 3, 3, 2, 1, 1]
    assert full["difficulty"] <= 4
    # 期望难度权重：难度 4 应命中横按版
    weighted = V.voicing_for("F", prefer={"difficulty": 4})
    assert weighted is not None


def test_no_impossible_open_mix() -> None:
    """“低音弦按 5 品 + 中间空弦”这类理论合法但手型不可能的指法必须剔除。"""
    assert V.validate_voicing([5, 0, 2, 5, 5, 0], "Am")
    assert V.validate_voicing([-1, 1, 0, 3, 3, 1], "Bb")
    # 标准开放和弦不能被误杀
    for symbol, frets in (
        ("C", [-1, 3, 2, 0, 1, 0]),
        ("G", [3, 2, 0, 0, 0, 3]),
        ("A", [-1, 0, 2, 2, 2, 0]),
        ("E", [0, 2, 2, 1, 0, 0]),
        ("Em", [0, 2, 2, 0, 0, 0]),
        ("D", [-1, -1, 0, 2, 3, 2]),
        ("Fmaj7", [-1, -1, 3, 2, 1, 0]),
        ("G/B", [-1, 2, 0, 0, 0, 3]),
    ):
        assert V.validate_voicing(frets, symbol) == [], (symbol, V.validate_voicing(frets, symbol))


def test_am_candidates_have_no_open_fretted_mix() -> None:
    for item in V.voicings_for("Am", limit=8):
        frets = _frets(item)
        if 0 in frets:
            fretted = [fret for fret in frets if fret > 0]
            assert max(fretted) <= 2, f"Am 候选 {frets} 混用了高把位与空弦"


def test_all_generated_voicings_legal(db: dict) -> None:
    """库里每个把位都必须通过音级/手型校验。"""
    problems: list[str] = []
    for symbol, items in db["standard"].items():
        for item in items:
            issue = V.validate_voicing(_frets(item), symbol)
            if issue:
                problems.append(f"{symbol} {_frets(item)}: {issue}")
    assert not problems, problems[:10]


def test_voicing_shape_constraints(db: dict) -> None:
    for symbol, items in db["standard"].items():
        for item in items:
            frets = _frets(item)
            assert all(fret == -1 or 0 <= fret <= db["max_fret"] for fret in frets), (symbol, frets)
            fretted = [fret for fret in frets if fret > 0]
            assert len([fret for fret in frets if fret >= 0]) >= 3, (symbol, frets)
            if fretted:
                assert max(fretted) - min(fretted) <= 4, (symbol, frets)
            sounding = [index for index, fret in enumerate(frets) if fret >= 0]
            assert sounding == list(range(sounding[0], sounding[-1] + 1)), (symbol, frets)


# --------------------------------------------------------------------------
# 斜杠和弦 / 性质覆盖 / 调弦
# --------------------------------------------------------------------------


def test_slash_chords_bass() -> None:
    cases = {
        "C/E": 4,
        "G/B": 11,
        "D/F#": 6,
        "Am/G": 7,
        "F/A": 9,
    }
    for symbol, bass_pc in cases.items():
        voicing = V.voicing_for(symbol)
        assert voicing is not None, symbol
        frets = _frets(voicing)
        index = next(i for i, fret in enumerate(frets) if fret >= 0)
        lowest_midi = V.STANDARD_TUNING[index] + frets[index]
        assert lowest_midi % 12 == bass_pc, f"{symbol} 最低音 {lowest_midi % 12} != {bass_pc}"


@pytest.mark.parametrize(
    "quality",
    ["maj", "min", "7", "maj7", "m7", "sus2", "sus4", "dim", "aug", "6", "m6", "add9", "5"],
)
@pytest.mark.parametrize("pc", [0, 2, 4, 5, 7, 9, 11])
def test_quality_coverage(pc: int, quality: str) -> None:
    symbol = format_chord_symbol(pc, quality)
    voicing = V.voicing_for(symbol)
    assert voicing is not None, f"{symbol} 没有把位"
    assert V.validate_voicing(_frets(voicing), symbol) == []


def test_drop_d_tuning(tmp_path: Path) -> None:
    # 同样不能写仓库路径：build_voicing_db 会落盘。
    db = V.build_voicing_db(tmp_path / "voicings.json")
    tunings = db["tunings"]
    assert "drop_d" in tunings
    item = V.voicing_for("D", prefer={"tuning": "drop_d"})
    assert item is not None


def test_voicing_id_stable() -> None:
    assert V.voicing_id("C", [-1, 3, 2, 0, 1, 0]) == "C_x32010"
    assert V.voicing_id("G/B", [-1, 2, 0, 0, 0, 3]) == "G_over_B_x20003"
    assert V.voicing_id("C#m", [-1, 4, 6, 6, 5, 4]) == "Csm_x46654"
