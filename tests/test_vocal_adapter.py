"""人声联合转录层的离线单测（纯解析 + 降级路径，不碰真音频/不下载模型）。

覆盖：

* token 解析（正常串、无 BPM、附点、多字符歌词、噪声/畸形/未闭合标签、空输入）；
* 时值换算（1/2/4/8/16、附点、非法值退回四分音符）；
* 事件 → :class:`aitsm.contracts.MelodyNote` 的时间轴（有/无 beatgrid、逐事件 bpm）；
* 可用性判定（环境变量缺失、目录不存在/为空、transformers 不可导入）；
* 降级路径（monkeypatch 掉 lyrics/melody/align 三个 pipeline 函数，跑的是内存数据）；
* 首选 VocalParse 路径被优先使用、失败/空输出时回落到降级路径；
* 进度回调两种协议（dict / (fraction, message)）；
* 结果能沿着 contracts 的 ``to_dict`` / ``from_dict`` 往返。

运行方式（pytest 或直接 python 均可）：
    .venv/Scripts/python.exe -m pytest tests/test_vocal_adapter.py -q
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from aitsm.contracts import (  # noqa: E402
    BeatGrid,
    LyricLine,
    MelodyNote,
    VocalResult,
)
from aitsm.pipeline import align as align_mod  # noqa: E402
from aitsm.pipeline import lyrics as lyrics_mod  # noqa: E402
from aitsm.pipeline import melody as melody_mod  # noqa: E402
from aitsm.transcribe import vocal as V  # noqa: E402

#: 120 BPM、每拍 0.5 秒；锚点 1.0 秒（第一小节第一拍不在 0，测试锚点是否真被用上）
BEATGRID = BeatGrid(
    bpm=120.0,
    beats=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
    beats_per_bar=4,
    first_downbeat_sec=1.0,
    source="test",
)


def _approx(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _audio_file(tmp_path: Path, name: str = "vocals.wav") -> str:
    """造一个占位文件：只让 ``Path.is_file()`` 成立，不写任何真音频。"""
    path = tmp_path / name
    path.write_bytes(b"RIFF-ish placeholder")
    return str(path)


# ---------------------------------------------------------------------------
# 时值换算
# ---------------------------------------------------------------------------


def test_note_value_to_quarters_standard_values() -> None:
    assert V.note_value_to_quarters("1") == 4.0
    assert V.note_value_to_quarters(2) == 2.0
    assert V.note_value_to_quarters("4") == 1.0
    assert V.note_value_to_quarters(8) == 0.5
    assert V.note_value_to_quarters("16") == 0.25
    assert V.note_value_to_quarters(8.0) == 0.5


def test_note_value_to_quarters_dotted() -> None:
    # 附点两种写法：`<NOTE_4.>` 与 `<NOTE_4d>`
    assert V.note_value_to_quarters("4.") == 1.5
    assert V.note_value_to_quarters("4d") == 1.5
    assert V.note_value_to_quarters("8.") == 0.75
    assert _approx(V.note_value_to_quarters("2."), 3.0)


def test_note_value_to_quarters_falls_back_to_quarter_note() -> None:
    for bad in ("", None, "abc", "0", "-4", "4.5.5", "999", True):
        assert V.note_value_to_quarters(bad) == 1.0, bad


# ---------------------------------------------------------------------------
# token 解析
# ---------------------------------------------------------------------------


def test_parse_basic_token_stream() -> None:
    events = V.parse_vocalparse_tokens(
        "<BPM_89><P_68><NOTE_4>你<P_69><NOTE_8>好<P_70><NOTE_2>爱"
    )
    assert [(item.midi_pitch, item.note_value, item.lyric) for item in events] == [
        (68, "4", "你"),
        (69, "8", "好"),
        (70, "2", "爱"),
    ]
    assert [item.bpm for item in events] == [89.0, 89.0, 89.0]


def test_parse_without_bpm_leaves_bpm_none() -> None:
    events = V.parse_vocalparse_tokens("<P_60><NOTE_4>la")
    assert len(events) == 1
    assert events[0].bpm is None  # 没有速度证据就不编造
    assert events[0].lyric == "la"


def test_parse_bpm_applies_only_after_it_appears() -> None:
    events = V.parse_vocalparse_tokens("<P_60><NOTE_4>a<BPM_100><P_62><NOTE_4>b")
    assert [item.bpm for item in events] == [None, 100.0]


def test_parse_dotted_note_values_in_tokens() -> None:
    events = V.parse_vocalparse_tokens("<P_60><NOTE_4.>啊<P_62><NOTE_8d>哦")
    assert [item.note_value for item in events] == ["4.", "8d"]
    quarters = [V.note_value_to_quarters(item.note_value) for item in events]
    assert quarters == [1.5, 0.75]


def test_parse_multi_character_lyric_is_kept() -> None:
    events = V.parse_vocalparse_tokens("<P_68>你好<NOTE_4>")
    assert len(events) == 1
    assert events[0].lyric == "你好"


def test_parse_lyric_before_pitch_is_held_for_next_event() -> None:
    # 倒序写法：歌词在音高之前出现
    events = V.parse_vocalparse_tokens("你<P_68><NOTE_4>")
    assert [item.lyric for item in events] == ["你"]


def test_parse_ascii_words_are_joined_with_space() -> None:
    events = V.parse_vocalparse_tokens("<P_60>hel<NOTE_4>lo")
    assert events[0].lyric == "hel lo"


def test_parse_missing_note_token_leaves_empty_value() -> None:
    events = V.parse_vocalparse_tokens("<P_70>啊")
    assert len(events) == 1
    assert events[0].note_value == ""
    # 缺时值时按四分音符处理（丢一个时值不该让整首歌消失）
    assert V.note_value_to_quarters(events[0].note_value) == 1.0


def test_parse_tolerates_noise_and_malformed_tokens() -> None:
    raw = (
        "<BPM_-> 前奏 "  # 非法 BPM 值
        "<P_><P_abc><P_999><NOTE_x><NOTE_2> "  # 空/非数字/越界音高；无主时值被丢弃
        "你 <P_70><NOTE_4> <FOO_bar> 好 <P_300>"  # 未知标签、越界音高
        "<P_71 未闭合"  # 少了右尖括号的半个标签
    )
    events = V.parse_vocalparse_tokens(raw)
    # 只有 <P_70> 是合法的；越界／非法音高、未知标签、未闭合标签都不产生事件
    assert len(events) == 1
    assert events[0].midi_pitch == 70
    assert events[0].lyric == "前奏你好"
    assert events[0].note_value == "4"
    assert events[0].bpm is None


def test_parse_ignores_punctuation_only_text() -> None:
    events = V.parse_vocalparse_tokens("<P_60><NOTE_4> | , 。 <BPM_120>")
    assert len(events) == 1
    assert events[0].lyric == ""


def test_parse_empty_and_none_input() -> None:
    assert V.parse_vocalparse_tokens("") == []
    assert V.parse_vocalparse_tokens(None) == []  # type: ignore[arg-type]
    assert V.parse_vocalparse_tokens("<P_abc><BPM_>") == []


def test_vocal_event_normalizes_fields() -> None:
    event = V.VocalEvent(lyric="  你\n好  ", midi_pitch=200, note_value=" 4 ", bpm=-3)
    assert event.lyric == "你 好"  # 空白折叠，内容保留
    assert event.midi_pitch == 127  # 夹到合法 MIDI 区间
    assert event.note_value == "4"
    assert event.bpm is None  # 非正速度视为没有证据

    # 只有标点/分隔符的「歌词」不是歌词
    assert V.VocalEvent(lyric=" | , 。 ").lyric == ""
    assert V.VocalEvent(lyric=None).lyric == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 事件 -> 音符时间轴
# ---------------------------------------------------------------------------


def test_vocal_events_to_notes_uses_beatgrid_anchor_and_tempo() -> None:
    events = [
        V.VocalEvent(lyric="你", midi_pitch=67, note_value="4", bpm=120.0),
        V.VocalEvent(lyric="好", midi_pitch=69, note_value="8", bpm=120.0),
    ]
    notes = V.vocal_events_to_notes(events, BEATGRID)
    assert len(notes) == 2
    assert all(isinstance(item, MelodyNote) for item in notes)
    assert _approx(notes[0].start, 1.0) and _approx(notes[0].end, 1.5)
    assert _approx(notes[1].start, 1.5) and _approx(notes[1].end, 1.75)
    assert [item.midi for item in notes] == [67, 69]
    assert [item.lyric for item in notes] == ["你", "好"]
    assert [item.syllable_index for item in notes] == [0, 1]
    assert all(item.source == "vocalparse" for item in notes)


def test_vocal_events_to_notes_without_beatgrid_still_returns_notes() -> None:
    events = [V.VocalEvent(lyric="la", midi_pitch=60, note_value="4")]
    notes = V.vocal_events_to_notes(events, None)
    assert len(notes) == 1
    assert _approx(notes[0].start, 0.0)
    assert _approx(notes[0].end, 0.5)  # 120 BPM 默认值


def test_vocal_events_to_notes_honours_per_event_bpm_override() -> None:
    events = [
        V.VocalEvent(midi_pitch=60, note_value="4", bpm=60.0),
        V.VocalEvent(midi_pitch=62, note_value="4", bpm=60.0),
    ]
    notes = V.vocal_events_to_notes(events, BeatGrid(bpm=120.0, first_downbeat_sec=0.0))
    assert _approx(notes[0].start, 0.0) and _approx(notes[0].end, 1.0)
    assert _approx(notes[1].start, 1.0) and _approx(notes[1].end, 2.0)


def test_vocal_events_to_notes_accepts_mapping_beatgrid() -> None:
    events = [V.VocalEvent(midi_pitch=60, note_value="4", bpm=100.0)]
    notes = V.vocal_events_to_notes(events, {"bpm": 100.0, "first_downbeat_sec": 2.0})
    assert _approx(notes[0].start, 2.0)
    assert _approx(notes[0].end, 2.6)  # 60 / 100 = 0.6 s


def test_vocal_events_to_notes_skips_no_lyric_syllable_index() -> None:
    notes = V.vocal_events_to_notes(
        [V.VocalEvent(lyric="", midi_pitch=60, note_value="4")], BEATGRID
    )
    assert notes[0].lyric == ""
    assert notes[0].syllable_index is None


def test_vocal_events_to_notes_empty_events() -> None:
    assert V.vocal_events_to_notes([], BEATGRID) == []
    assert V.vocal_events_to_notes(None, None) == []


def test_vocal_events_to_notes_accepts_mapping_events() -> None:
    notes = V.vocal_events_to_notes(
        [{"lyric": "你", "midi_pitch": 64, "note_value": "4", "bpm": 120.0}], BEATGRID
    )
    assert notes[0].midi == 64 and notes[0].lyric == "你"


# ---------------------------------------------------------------------------
# 可用性
# ---------------------------------------------------------------------------


def test_is_available_false_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(V.VOCALPARSE_DIR_ENV, raising=False)
    assert V.vocalparse_dir() is None
    assert V.is_available() is False


def test_is_available_false_for_missing_or_empty_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(V.VOCALPARSE_DIR_ENV, str(tmp_path / "not-there"))
    assert V.is_available() is False

    monkeypatch.setenv(V.VOCALPARSE_DIR_ENV, str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    assert V.is_available() is False

    # 路径指向普通文件也不算权重目录
    plain = tmp_path / "weights.bin"
    plain.write_bytes(b"x")
    monkeypatch.setenv(V.VOCALPARSE_DIR_ENV, str(plain))
    assert V.is_available() is False


def test_is_available_requires_qwen_asr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """VocalParse 依赖 ``qwen_asr`` 的建模代码，不是标准 transformers。

    实测：只装 transformers 时 ``thinker.*`` 键全部对不上（315 个 MISSING），
    因为 Qwen3-ASR 的建模代码由 ``qwen-asr`` 包分发。

    注意：``is_available()`` 最终走 ``qwen_backend.probe()``，而后者用**文件系统
    查找**而不是 ``importlib.util.find_spec``（后者会被空壳父包污染而误判，
    见 tests/test_qwen_backend_reload.py）。所以这里 monkeypatch ``_package_root``。
    """
    from aitsm.transcribe import qwen_backend

    model_dir = tmp_path / "vocalparse"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(V.VOCALPARSE_DIR_ENV, str(model_dir))

    real_root = qwen_backend._package_root

    monkeypatch.setattr(qwen_backend, "_package_root", lambda: None)
    assert V.is_available() is False

    monkeypatch.setattr(qwen_backend, "_package_root", real_root)
    assert V.is_available() is True
    assert V.vocalparse_dir() == model_dir


# ---------------------------------------------------------------------------
# 降级路径（monkeypatch 三个 pipeline 函数，不跑真音频）
# ---------------------------------------------------------------------------


class _Recorder:
    """记录三个 pipeline 函数的入参，并返回内存里的假结果。"""

    def __init__(self) -> None:
        self.calls: dict[str, list[dict]] = {"lyrics": [], "melody": [], "align": []}

    def lyrics(self, vocals_wav, *, duration=None, language="auto", progress=None):
        self.calls["lyrics"].append(
            {
                "path": str(vocals_wav),
                "duration": duration,
                "language": language,
                "whisper_model_env": os.environ.get(V.WHISPER_MODEL_ENV),
            }
        )
        return [
            {
                "start": 1.0,
                "end": 2.0,
                "text": "你好",
                "words": [
                    {"start": 1.0, "end": 1.5, "text": "你"},
                    {"start": 1.5, "end": 2.0, "text": "好"},
                ],
                "confidence": 0.8,
            }
        ]

    def melody(self, vocals_wav, beats, *, duration=None, harmony_onsets=None):
        self.calls["melody"].append({"beats": dict(beats), "duration": duration})
        return {
            "notes": [
                {"start": 1.0, "end": 1.5, "midi": 67, "confidence": 0.7},
                {"start": 1.5, "end": 2.0, "midi": 69, "confidence": 0.7},
            ],
            "range": {"low_midi": 67, "high_midi": 69},
            "engine": "pyin",
            "warnings": ["输入疑似非人声分轨"],
        }

    def align(self, melody, lyrics, beats):
        self.calls["align"].append({"notes": len(melody.get("notes") or []), "lines": len(lyrics)})
        result = copy.deepcopy(melody)
        result["notes"][0]["lyric"] = "你"
        result["notes"][1]["lyric"] = "好"
        result["alignment"] = {"matched_syllables": 2, "notes_with_lyric": 2, "coverage": 1.0}
        return result


@pytest.fixture
def degraded(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """把 VocalParse 判为不可用，并把三段式全部换成内存实现。"""
    recorder = _Recorder()
    monkeypatch.setattr(V, "is_available", lambda: False)
    monkeypatch.setattr(lyrics_mod, "transcribe_lyrics", recorder.lyrics)
    monkeypatch.setattr(lyrics_mod, "last_run_info", lambda: {"engine": "whisper", "model": "medium"})
    monkeypatch.setattr(melody_mod, "transcribe_melody", recorder.melody)
    monkeypatch.setattr(align_mod, "align_lyrics_to_melody", recorder.align)
    return recorder


def test_transcribe_vocal_falls_back_to_whisper_pyin(
    degraded: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(V.WHISPER_MODEL_ENV, "small")  # 调用方预置的型号，调用后必须恢复
    source = _audio_file(tmp_path)

    result = V.transcribe_vocal(source, duration=12.5, beatgrid=BEATGRID, whisper_model="medium")

    assert isinstance(result, VocalResult)
    assert result.engine == "whisper+pyin"
    assert result.requested == "VocalParse"  # 首选永远是被降级的那个
    assert result.bpm == 120.0
    assert result.elapsed_seconds is not None

    # 三个阶段都被调用，且参数透传正确
    assert degraded.calls["lyrics"][0]["duration"] == 12.5
    assert degraded.calls["lyrics"][0]["language"] == "auto"
    assert degraded.calls["lyrics"][0]["whisper_model_env"] == "medium"
    assert degraded.calls["melody"][0]["duration"] == 12.5
    assert degraded.calls["melody"][0]["beats"]["bpm"] == 120.0
    assert degraded.calls["melody"][0]["beats"]["beats_per_bar"] == 4
    assert degraded.calls["align"][0] == {"notes": 2, "lines": 1}
    # 临时覆盖的 whisper 型号在调用结束后恢复成调用方预置值
    assert os.environ.get(V.WHISPER_MODEL_ENV) == "small"

    # 产物是 contracts 类型，歌词挂到了音符上
    assert [item.midi for item in result.notes] == [67, 69]
    assert [item.lyric for item in result.notes] == ["你", "好"]
    assert all(item.source == "pyin" for item in result.notes)
    assert len(result.lyrics) == 1
    assert isinstance(result.lyrics[0], LyricLine)
    assert result.lyrics[0].text == "你好"
    assert [word.text for word in result.lyrics[0].words] == ["你", "好"]

    # 降级原因与子引擎在 warnings / alignment 里说清楚
    assert any("VocalParse 不可用" in item for item in result.warnings)
    assert any(item.startswith("旋律：") for item in result.warnings)
    assert result.alignment["engine"] == "whisper+pyin"
    assert result.alignment["lyrics_engine"] == "whisper"
    assert result.alignment["melody_engine"] == "pyin"
    assert result.alignment["whisper_model"] == "medium"
    assert result.alignment["beatgrid"] == "provided"
    assert result.alignment["notes_with_lyric"] == 2


def test_transcribe_vocal_fallback_without_beatgrid(
    degraded: _Recorder, tmp_path: Path
) -> None:
    result = V.transcribe_vocal(_audio_file(tmp_path))
    assert result.engine == "whisper+pyin"
    assert result.bpm is None  # 没有节拍证据就不写 BPM
    assert "bpm" not in degraded.calls["melody"][0]["beats"]
    assert result.alignment["beatgrid"] == "absent"


def test_transcribe_vocal_fallback_survives_stage_failures(
    degraded: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("模型炸了")

    monkeypatch.setattr(lyrics_mod, "transcribe_lyrics", broken)
    monkeypatch.setattr(melody_mod, "transcribe_melody", broken)
    monkeypatch.setattr(align_mod, "align_lyrics_to_melody", broken)

    result = V.transcribe_vocal(_audio_file(tmp_path), beatgrid=BEATGRID)

    assert result.engine == "whisper+pyin"
    assert result.notes == [] and result.lyrics == []
    assert any("歌词识别失败" in item for item in result.warnings)
    assert any("旋律提取失败" in item for item in result.warnings)
    assert any("旋律为空" in item for item in result.warnings)
    assert any("歌词为空" in item for item in result.warnings)


def test_transcribe_vocal_fallback_handles_align_failure(
    degraded: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("对齐炸了")

    monkeypatch.setattr(align_mod, "align_lyrics_to_melody", broken)

    result = V.transcribe_vocal(_audio_file(tmp_path), beatgrid=BEATGRID)

    # 对齐失败只丢歌词归属，音符与歌词本身都还在
    assert [item.midi for item in result.notes] == [67, 69]
    assert all(item.lyric == "" for item in result.notes)
    assert len(result.lyrics) == 1
    assert any("歌词-旋律对齐失败" in item for item in result.warnings)


# ---------------------------------------------------------------------------
# 首选路径与回落
# ---------------------------------------------------------------------------


def test_transcribe_vocal_prefers_vocalparse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _audio_file(tmp_path)
    calls: list[dict] = []

    def fake_primary(path, *, duration, language, beatgrid, device, progress):
        calls.append({"path": path, "duration": duration, "device": device})
        return VocalResult(
            notes=[MelodyNote(start=0.0, end=0.5, midi=64, lyric="啊", source="vocalparse")],
            lyrics=[LyricLine(start=0.0, end=0.5, text="啊")],
            bpm=100.0,
            engine="vocalparse",
            requested="VocalParse",
        )

    def boom(*args, **kwargs):  # 降级路径一旦被调用就说明选错了
        raise AssertionError("VocalParse 可用时不应走降级路径")

    monkeypatch.setattr(V, "is_available", lambda: True)
    monkeypatch.setattr(V, "_transcribe_with_vocalparse", fake_primary)
    monkeypatch.setattr(lyrics_mod, "transcribe_lyrics", boom)
    monkeypatch.setattr(melody_mod, "transcribe_melody", boom)

    result = V.transcribe_vocal(source, duration=3.0, device="cpu")

    assert result.engine == "vocalparse"
    assert result.requested == "VocalParse"
    assert result.bpm == 100.0
    assert result.elapsed_seconds is not None
    assert result.warnings == []
    assert calls == [{"path": source, "duration": 3.0, "device": "cpu"}]


def test_transcribe_vocal_falls_back_when_vocalparse_raises(
    degraded: _Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(V, "is_available", lambda: True)

    def broken(*args, **kwargs):
        raise V.VocalParseError("本地权重加载失败：不支持的模型结构")

    monkeypatch.setattr(V, "_transcribe_with_vocalparse", broken)

    result = V.transcribe_vocal(_audio_file(tmp_path), beatgrid=BEATGRID)

    assert result.engine == "whisper+pyin"
    assert len(result.notes) == 2
    assert any("VocalParse 推理失败" in item for item in result.warnings)
    assert any("不支持的模型结构" in item for item in result.warnings)


def test_transcribe_vocal_falls_back_when_vocalparse_returns_nothing(
    degraded: _Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(V, "is_available", lambda: True)
    monkeypatch.setattr(
        V,
        "_transcribe_with_vocalparse",
        lambda *args, **kwargs: VocalResult(engine="vocalparse", requested="VocalParse"),
    )

    result = V.transcribe_vocal(_audio_file(tmp_path), beatgrid=BEATGRID)

    assert result.engine == "whisper+pyin"
    assert any("未解出任何音符" in item for item in result.warnings)


def test_transcribe_vocal_missing_input_is_empty_not_raising(tmp_path: Path) -> None:
    result = V.transcribe_vocal(str(tmp_path / "nope.wav"), beatgrid=BEATGRID)
    assert result.engine == "none"
    assert result.requested == "VocalParse"
    assert result.notes == [] and result.lyrics == []
    assert result.bpm is None
    assert any("人声音频不存在" in item for item in result.warnings)
    assert result.elapsed_seconds is not None

    empty = V.transcribe_vocal(None)
    assert empty.engine == "none" and empty.warnings


# ---------------------------------------------------------------------------
# 进度回调
# ---------------------------------------------------------------------------


def test_progress_callback_accepts_dict_protocol(
    degraded: _Recorder, tmp_path: Path
) -> None:
    seen: list[dict] = []
    V.transcribe_vocal(_audio_file(tmp_path), beatgrid=BEATGRID, progress=seen.append)

    assert seen, "至少要有一次进度上报"
    assert all(item["stage"] == "vocal" for item in seen)
    assert seen[-1]["percent"] == 100.0
    assert any("歌词 ASR" in item["message"] for item in seen)


def test_progress_callback_accepts_fraction_protocol(
    degraded: _Recorder, tmp_path: Path
) -> None:
    seen: list[tuple[float, str]] = []

    def callback(fraction, message):
        seen.append((fraction, message))

    V.transcribe_vocal(_audio_file(tmp_path), beatgrid=BEATGRID, progress=callback)

    assert seen
    assert all(isinstance(item[0], float) for item in seen)
    assert seen[-1][0] == 1.0


def test_progress_callback_exceptions_are_swallowed(
    degraded: _Recorder, tmp_path: Path
) -> None:
    def broken_progress(payload):
        raise ValueError("回调自己坏了")

    result = V.transcribe_vocal(
        _audio_file(tmp_path), beatgrid=BEATGRID, progress=broken_progress
    )
    assert result.engine == "whisper+pyin"


# ---------------------------------------------------------------------------
# contracts 往返
# ---------------------------------------------------------------------------


def test_vocal_result_round_trips_through_contracts(
    degraded: _Recorder, tmp_path: Path
) -> None:
    result = V.transcribe_vocal(_audio_file(tmp_path), beatgrid=BEATGRID)
    payload = result.to_dict()

    assert payload["engine"] == "whisper+pyin"
    assert payload["requested"] == "VocalParse"
    assert len(payload["notes"]) == 2

    restored = VocalResult.from_dict(payload)
    assert [item.to_dict() for item in restored.notes] == [item.to_dict() for item in result.notes]
    assert [item.to_dict() for item in restored.lyrics] == [
        item.to_dict() for item in result.lyrics
    ]
    assert restored.bpm == result.bpm
