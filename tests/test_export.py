"""导出器单测：MusicXML 结构 / MuseScore 可读性要点 / MIDI 往返。

测试策略：先用 ``arrange()`` 造一份真实结构的谱（而不是手搓半份 score），再导出、
解析回来逐项核对。这样导出器与编配器的契约一旦漂移，这里会立刻报错。
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aitsm.score.arrange import arrange  # noqa: E402
from aitsm.score.export import ExportError, export_midi, export_musicxml  # noqa: E402
from aitsm.score.export.midi import PPQ, parse_midi  # noqa: E402
from aitsm.score.voicings import voicing_for  # noqa: E402

BEAT_SEC = 60.0 / 84.0


def _t(beat: float) -> float:
    return round(beat * BEAT_SEC, 4)


def make_analysis(bars: int = 4, *, sections: list[dict] | None = None) -> dict:
    """紧凑的合成 analysis：C-G-Am-F，每小节一个和弦，逐字歌词。"""
    progression = ["C", "G", "Am", "F"]
    chords = []
    for index in range(bars):
        symbol = progression[index % len(progression)]
        root = {"C": 0, "G": 7, "Am": 9, "F": 5}[symbol]
        chords.append(
            {
                "start": _t(index * 4),
                "end": _t((index + 1) * 4),
                "symbol": symbol,
                "confidence": 0.9,
                "root_pc": root,
                "bass_pc": root,
            }
        )
    lyric_chars = list("月亮代表我的心呀真呀好呀")[: bars * 2]
    words = [{"start": _t(i * 2), "end": _t(i * 2 + 2), "text": char} for i, char in enumerate(lyric_chars)]
    melody = [
        {
            "start": words[i]["start"],
            "end": words[i]["end"],
            "midi": [64, 67, 69, 67, 65, 64, 62, 60, 62, 64, 65, 67][i % 12],
            "confidence": 0.8,
            "lyric": words[i]["text"],
        }
        for i in range(len(words))
    ]
    section_list = sections or [
        {"start": _t(0), "end": _t(bars * 2), "kind": "verse", "label": "主歌", "energy": 0.4, "confidence": 0.8},
        {"start": _t(bars * 2), "end": _t(bars * 4), "kind": "chorus", "label": "副歌", "energy": 0.8, "confidence": 0.8},
    ]
    return {
        "schema": "aitsm.analysis/v1",
        "source": {"original_name": "export-test.mp3", "sha256": "0" * 64},
        "audio": {"analysis_wav": "work/export-test.wav", "duration": _t(bars * 4), "sample_rate": 22050},
        "stems": {"vocals": None, "drums": None, "bass": None, "other": None},
        "beats": {
            "bpm": 84.0,
            "beats": [_t(i) for i in range(bars * 4 + 1)],
            "downbeats": [_t(i * 4) for i in range(bars)],
            "beats_per_bar": 4,
            "time_signature": [4, 4],
            "confidence": 0.9,
        },
        "key": {"tonic": "C", "mode": "major", "confidence": 0.85, "num_sharps_flats": 0},
        "chords": chords,
        "sections": section_list,
        "lyrics": [{"start": words[0]["start"], "end": words[-1]["end"], "text": "".join(lyric_chars), "words": words}],
        "melody": {"notes": melody},
        "tools": {"beats": "synthetic", "chords": "synthetic", "melody": "synthetic"},
        "warnings": [],
    }


def make_score(**options) -> dict:
    """arrange 出来的完整谱面。"""
    merged = {"include_tab": True, "difficulty": "beginner", "style": "pop"}
    merged.update(options)
    return arrange(make_analysis(), merged)


def _parse(path: Path) -> ET.Element:
    return ET.fromstring(path.read_bytes())


def _notes(measure: ET.Element) -> list[ET.Element]:
    return measure.findall("note")


def _system_sizes(part: ET.Element) -> list[int]:
    """按 <print new-system/new-page> 把小节切成每行的有效小节数。"""
    sizes: list[int] = []
    count = 0
    for measure in part.findall("measure"):
        if measure.find("print") is not None and count:
            sizes.append(count)
            count = 0
        count += 1
    if count:
        sizes.append(count)
    return sizes


# ----------------------------------------------------------------- MusicXML


def test_musicxml_structure_matches_score(tmp_path: Path) -> None:
    """导出的 XML 必须能被解析回来，且小节/音符/和弦/歌词数与 score 一致。"""
    score = make_score()
    path = tmp_path / "song.musicxml"
    report = export_musicxml(score, path)
    assert path.is_file() and path.stat().st_size > 2000
    raw = path.read_bytes()
    assert raw.startswith(b'<?xml version="1.0" encoding="UTF-8"?>')
    assert b"score-partwise" in raw

    root = _parse(path)
    assert root.tag == "score-partwise"
    assert root.get("version") == "4.0"
    parts = root.findall("part")
    assert [part.get("id") for part in parts] == ["P1", "P2"]

    bar_count = len(score["bars"])
    for part in parts:
        assert len(part.findall("measure")) == bar_count == report["measures"]

    vocal_part, tab_part = parts
    # 第一小节属性齐全：divisions / key / time / clef
    attributes = vocal_part.find("measure/attributes")
    assert attributes is not None
    assert attributes.findtext("divisions") == "480"
    assert int(attributes.findtext("key/fifths")) == score["meta"]["key"]["num_sharps_flats"]
    assert attributes.findtext("time/beats") == "4"
    assert attributes.findtext("time/beat-type") == "4"
    assert attributes.findtext("clef/sign") == "G"

    # TAB 谱表：TAB 谱号 + 6 线 + 六根弦调弦
    tab_attributes = tab_part.find("measure/attributes")
    assert tab_attributes is not None
    assert tab_attributes.findtext("clef/sign") == "TAB"
    details = tab_attributes.find("staff-details")
    assert details is not None
    assert details.findtext("staff-lines") == "6"
    assert len(details.findall("staff-tuning")) == 6
    assert details.findall("staff-tuning")[0].findtext("tuning-octave") == "4"

    # 每个 note 都必须有 duration / type / voice，休止符必须有 <rest/>
    for measure in vocal_part.findall("measure"):
        for note in _notes(measure):
            assert note.findtext("duration"), "note 缺少 duration"
            assert note.findtext("type"), "note 缺少 type"
            assert note.findtext("voice"), "note 缺少 voice"
    for measure in tab_part.findall("measure"):
        for note in _notes(measure):
            assert note.findtext("duration") and note.findtext("type")

    # 和弦符号：数量与 score 中实际能解析的和弦一致
    harmonies = vocal_part.findall(".//harmony")
    assert len(harmonies) == report["harmonies"] == sum(len(bar["chords"]) for bar in score["bars"])
    kinds = {harmony.findtext("kind") for harmony in harmonies}
    assert kinds <= {
        "major",
        "minor",
        "dominant",
        "major-seventh",
        "minor-seventh",
        "suspended-fourth",
        "suspended-second",
        "diminished",
        "augmented",
        "major-sixth",
        "minor-sixth",
        "dominant-ninth",
        "major-ninth",
        "minor-ninth",
        "power",
        "half-diminished",
        "diminished-seventh",
    }

    # 歌词：一个字一个 <lyric>
    lyrics = [element for element in vocal_part.findall(".//lyric") if element.findtext("text")]
    expected_lyrics = sum(
        1 for bar in score["bars"] for note in bar.get("vocal") or [] if str(note.get("lyric") or "").strip()
    )
    assert len(lyrics) == expected_lyrics
    assert lyrics[0].findtext("syllabic") == "single"

    # TAB 指法：[弦号, 品位] 成对出现
    technicals = tab_part.findall(".//technical")
    assert len(technicals) == report["tab_notes"]
    for technical in technicals:
        string_number = int(technical.findtext("string"))
        fret = int(technical.findtext("fret"))
        assert 1 <= string_number <= 6
        assert 0 <= fret <= 24


def test_musicxml_tempo_and_section_marks(tmp_path: Path) -> None:
    """速度写在 <sound tempo>，段落用 <rehearsal> 标注且段落开头强制换系统。"""
    score = make_score()
    path = tmp_path / "song.musicxml"
    export_musicxml(score, path)
    root = _parse(path)
    vocal_part = root.findall("part")[0]
    first_measure = vocal_part.findall("measure")[0]
    sound = first_measure.find("direction/sound")
    assert sound is not None and abs(float(sound.get("tempo")) - 84.0) < 1e-6
    assert first_measure.findtext("direction/direction-type/metronome/per-minute") == "84"

    rehearsals = [element.text for element in vocal_part.findall(".//rehearsal")]
    assert rehearsals == ["主歌", "副歌"]
    # 副歌第一小节（第 3 小节，因为每段 2 小节）必须有 print new-system
    third = vocal_part.findall("measure")[2]
    assert third.find("print") is not None and third.find("print").get("new-system") == "yes"
    # 第一小节不应有 print（开头本来就在第一系统）
    assert first_measure.find("print") is None


def test_musicxml_capo_and_pickup(tmp_path: Path) -> None:
    """变调夹写进 <capo> 且调号随记谱调变化；弱起小节标 implicit。"""
    score = arrange(make_analysis(), {"capo": 2, "include_tab": True})
    path = tmp_path / "capo.musicxml"
    export_musicxml(score, path)
    root = _parse(path)
    tab_attributes = root.findall("part")[1].find("measure/attributes")
    assert tab_attributes is not None
    assert tab_attributes.findtext("staff-details/capo") == "2"
    assert root.findall("part")[0].findtext("measure/attributes/key/fifths") == str(
        score["meta"]["key"]["num_sharps_flats"]
    )

    # 手工构造弱起：第一小节只有 1 拍内容（一个四分音符 + 一个跨满小节的和弦时值）
    pickup_score = json.loads(json.dumps(score))
    first_bar = pickup_score["bars"][0]
    first_bar["pickup"] = True
    first_bar["vocal"] = [{"beat": 0.0, "dur": 1.0, "midi": 64, "lyric": "月", "confidence": 0.9}]
    # 上游可能把弱起小节的和弦写成 dur=4（按完整小节算），不能因此把小节撑满
    first_bar["chords"] = [{"beat": 0.0, "dur": 4.0, "symbol": first_bar["chords"][0]["symbol"]}]
    pickup_path = tmp_path / "pickup.musicxml"
    export_musicxml(pickup_score, pickup_path)
    pickup_root = _parse(pickup_path)
    pickup_first = pickup_root.findall("part")[0].findall("measure")[0]
    assert pickup_first.get("implicit") == "yes"
    # 弱起小节内容长度 = 1 拍 = 480 divisions，不能补满 4 拍
    durations = [int(note.findtext("duration")) for note in pickup_first.findall("note")]
    assert sum(durations) == 480, f"弱起小节被补成了 {sum(durations)} divisions"
    # 后面正常小节仍然是 4 拍
    second = pickup_root.findall("part")[0].findall("measure")[1]
    assert sum(int(note.findtext("duration")) for note in second.findall("note")) == 1920
    # TAB 声部同样是 1 拍：每根弦一个音，时值都等于 1 拍（不是 4 拍），也不多出补满的休止
    tab_first = pickup_root.findall("part")[1].findall("measure")[0]
    tab_durations = [int(note.findtext("duration")) for note in tab_first.findall("note")]
    assert tab_durations and set(tab_durations) == {480}, f"TAB 弱起小节时值异常：{tab_durations}"


def test_musicxml_slash_chord_bass(tmp_path: Path) -> None:
    """斜杠和弦必须写出 <bass>。"""
    score = make_score()
    score["bars"][0]["chords"] = [
        {"beat": 0.0, "dur": 4.0, "symbol": "G/B", "voicing_id": None},
    ]
    # 从指法库取一个合法 voicing，保证 TAB 也能写出来
    voicing = voicing_for("G/B")
    assert voicing is not None
    score["voicings"]["test-G/B"] = dict(voicing, symbol="G/B")
    score["bars"][0]["chords"][0]["voicing_id"] = "test-G/B"
    path = tmp_path / "slash.musicxml"
    export_musicxml(score, path)
    harmony = _parse(path).findall("part")[0].find(".//harmony")
    assert harmony is not None
    assert harmony.findtext("root/root-step") == "G"
    assert harmony.findtext("bass/bass-step") == "B"


def test_musicxml_four_bars_per_system_no_orphan(tmp_path: Path) -> None:
    """长曲排版：每行最多 4 小节，余数不能变成单小节孤行（9 小节 -> 4+3+2）。"""
    analysis = make_analysis(
        bars=9,
        sections=[{"start": 0.0, "end": _t(36), "kind": "verse", "label": "主歌", "energy": 0.5, "confidence": 0.8}],
    )
    score = arrange(analysis, {"include_tab": True, "difficulty": "beginner"})
    path = tmp_path / "long.musicxml"
    report = export_musicxml(score, path)
    vocal_part = _parse(path).findall("part")[0]
    sizes = _system_sizes(vocal_part)
    assert sum(sizes) == 9
    assert max(sizes) <= 4, f"某行超过 4 小节：{sizes}"
    assert 1 not in sizes, f"出现单小节孤行：{sizes}"
    assert sizes == [4, 3, 2], sizes
    assert report["systems"] == len(sizes)
    measures = vocal_part.findall("measure")
    assert measures[0].find("print") is None  # 第一行从开头开始
    assert measures[4].find("print") is not None  # 第 5 小节换行
    assert measures[7].find("print") is not None  # 第 8 小节换行


def test_musicxml_sections_always_break(tmp_path: Path) -> None:
    """段落开头必须换行，段内继续按 4 小节一行。"""
    analysis = make_analysis(
        bars=12,
        sections=[
            {"start": 0.0, "end": _t(16), "kind": "verse", "label": "主歌", "energy": 0.4, "confidence": 0.8},
            {"start": _t(16), "end": _t(48), "kind": "chorus", "label": "副歌", "energy": 0.9, "confidence": 0.8},
        ],
    )
    score = arrange(analysis, {"include_tab": True, "difficulty": "beginner"})
    path = tmp_path / "sections.musicxml"
    export_musicxml(score, path)
    vocal_part = _parse(path).findall("part")[0]
    sizes = _system_sizes(vocal_part)
    assert sum(sizes) == 12
    assert max(sizes) <= 4 and 1 not in sizes, sizes
    measures = vocal_part.findall("measure")
    assert measures[4].find("print") is not None, "副歌开头没有换行"
    assert measures[4].findtext("direction/direction-type/rehearsal") == "副歌"
    assert measures[8].find("print") is not None


def test_musicxml_bars_per_system_override(tmp_path: Path) -> None:
    """meta.bars_per_system 可覆盖每行小节数，且同样避免孤行（4 小节/3 -> 2+2）。"""
    analysis = make_analysis(
        bars=4,
        sections=[{"start": 0.0, "end": _t(16), "kind": "verse", "label": "主歌", "energy": 0.4, "confidence": 0.8}],
    )
    score = arrange(analysis, {"include_tab": True, "difficulty": "beginner"})
    score["meta"]["bars_per_system"] = 3
    path = tmp_path / "three.musicxml"
    export_musicxml(score, path)
    assert _system_sizes(_parse(path).findall("part")[0]) == [2, 2]


def test_musicxml_rejects_bad_input(tmp_path: Path) -> None:
    """输入不合法时抛 ExportError，且不留下半截文件。"""
    for bad in ({"schema": "wrong", "bars": []}, {"schema": "aitsm.score/v1", "bars": []}):
        path = tmp_path / "bad.musicxml"
        try:
            export_musicxml(bad, path)
        except ExportError:
            pass
        else:  # pragma: no cover - 必须抛异常
            raise AssertionError(f"应当抛 ExportError：{bad}")
        assert not path.exists()
        assert not (tmp_path / "bad.musicxml.tmp").exists()


def test_musicxml_write_failure_cleans_temp(tmp_path: Path) -> None:
    """写盘失败（目标是目录）要抛 ExportError 且清掉临时文件。"""
    score = make_score()
    target = tmp_path / "dir.musicxml"
    target.mkdir()
    try:
        export_musicxml(score, target)
    except ExportError:
        pass
    else:  # pragma: no cover
        raise AssertionError("写盘失败应当抛 ExportError")
    assert not (tmp_path / "dir.musicxml.tmp").exists()


# ----------------------------------------------------------------- MIDI


def test_midi_header_tracks_and_roundtrip(tmp_path: Path) -> None:
    """SMF 头信息、轨道数量与音符数必须和写入时一致（往返自检）。"""
    score = make_score()
    path = tmp_path / "song.mid"
    report = export_midi(score, path)
    assert path.is_file() and path.stat().st_size > 200

    parsed = parse_midi(path.read_bytes())
    assert parsed["format"] == 1
    assert parsed["division"] == PPQ == 480
    assert parsed["track_count"] == report["smf_format"] * 0 + len(report["tracks"])
    assert [track["name"] for track in parsed["tracks"]] == ["Conductor", "Vocal", "Guitar"]

    parsed_total = sum(len(track["notes"]) for track in parsed["tracks"])
    assert parsed_total == report["notes"]["total"]
    assert len(parsed["tracks"][1]["notes"]) == report["notes"]["vocal"]
    assert len(parsed["tracks"][2]["notes"]) == report["notes"]["guitar"]

    # 人声轨音高集合 == score 里的旋律音高集合
    score_pitches = {
        int(note["midi"]) for bar in score["bars"] for note in bar.get("vocal") or [] if note.get("midi") is not None
    }
    parsed_pitches = {note["pitch"] for note in parsed["tracks"][1]["notes"]}
    assert parsed_pitches == score_pitches


def test_midi_tempo_timesig_keysig(tmp_path: Path) -> None:
    """指挥轨必须带速度（84 BPM -> 714285 µs/拍）、拍号与调号。"""
    score = make_score()
    path = tmp_path / "song.mid"
    export_midi(score, path)
    conductor = parse_midi(path.read_bytes())["tracks"][0]
    tempos = [item for item in conductor["meta"] if item["type"] == 0x51]
    assert tempos, "缺少 tempo meta"
    microseconds = int.from_bytes(tempos[0]["data"], "big")
    assert abs(microseconds - round(60_000_000 / 84.0)) <= 1

    time_signatures = [item for item in conductor["meta"] if item["type"] == 0x58]
    assert time_signatures and time_signatures[0]["data"][0] == 4 and time_signatures[0]["data"][1] == 2

    key_signatures = [item for item in conductor["meta"] if item["type"] == 0x59]
    assert key_signatures and key_signatures[0]["data"][0] == score["meta"]["key"]["num_sharps_flats"]


def test_midi_guitar_has_rhythm_not_all_on_beat_zero(tmp_path: Path) -> None:
    """扫弦要按 pattern 铺开，不能所有音都落在 tick 0。"""
    score = make_score()
    path = tmp_path / "song.mid"
    export_midi(score, path)
    guitar_notes = parse_midi(path.read_bytes())["tracks"][2]["notes"]
    assert len(guitar_notes) > len(score["bars"])
    onsets = {note["tick"] for note in guitar_notes}
    assert len(onsets) >= len(score["bars"]) * 2, "扫弦起音点太少，节奏型没有展开"
    # 每个小节都要有吉他音（第一小节起点 0，之后逐小节推进）
    bar_ticks = 4 * PPQ
    for index in range(len(score["bars"])):
        start = index * bar_ticks
        assert any(start <= note["tick"] < start + bar_ticks for note in guitar_notes), f"第 {index + 1} 小节没有吉他音"
    # 时值必须为正
    assert all(note["duration"] > 0 for note in guitar_notes)


def test_midi_metronome_track(tmp_path: Path) -> None:
    """meta.include_metronome 为真时才附加节拍器轨（ch9）。"""
    score = make_score()
    quiet = tmp_path / "quiet.mid"
    report = export_midi(score, quiet)
    assert len(report["tracks"]) == 3
    assert report["notes"]["metronome"] == 0

    score.setdefault("meta", {})["include_metronome"] = True
    loud = tmp_path / "loud.mid"
    report_with_click = export_midi(score, loud)
    assert report_with_click["tracks"][-1] == "Click"
    parsed = parse_midi(loud.read_bytes())
    clicks = parsed["tracks"][3]["notes"]
    assert len(clicks) == len(score["bars"]) * 4
    assert {note["channel"] for note in clicks} == {9}
    assert report_with_click["notes"]["metronome"] == len(clicks)


def test_midi_no_duplicate_or_overlapping_same_pitch(tmp_path: Path) -> None:
    """同一轨同一音高不得重复 Note On（打嗝）：同 tick 去重、重叠截短。"""
    score = make_score()
    path = tmp_path / "clean.mid"
    export_midi(score, path)
    for track in parse_midi(path.read_bytes())["tracks"]:
        seen: set[tuple[int, int]] = set()
        by_pitch: dict[int, list[tuple[int, int]]] = {}
        for note in track["notes"]:
            key = (note["tick"], note["pitch"])
            assert key not in seen, f"{track['name']} 在同一 tick 重复发同音高 {note['pitch']}"
            seen.add(key)
            by_pitch.setdefault(note["pitch"], []).append((note["tick"], note["duration"]))
        for spans in by_pitch.values():
            spans.sort()
            for (prev_tick, prev_dur), (tick, _dur) in zip(spans, spans[1:]):
                assert tick >= prev_tick + prev_dur, f"{track['name']} 音高 {note['pitch']} 有重叠音符"


def test_midi_rejects_bad_input(tmp_path: Path) -> None:
    """非法输入抛 ExportError 且不写文件。"""
    for bad in (None, {"schema": "wrong"}, {"schema": "aitsm.score/v1", "bars": []}):
        path = tmp_path / "bad.mid"
        try:
            export_midi(bad, path)  # type: ignore[arg-type]
        except ExportError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"应当抛 ExportError：{bad!r}")
        assert not path.exists()


def test_export_package_reexports(tmp_path: Path) -> None:
    """子包顶层函数可用（service.py 就是这么调的）。"""
    score = make_score()
    xml_report = export_musicxml(score, tmp_path / "a.musicxml")
    midi_report = export_midi(score, tmp_path / "a.mid")
    assert xml_report["format"] == "musicxml" and xml_report["measures"] == len(score["bars"])
    assert midi_report["format"] == "midi" and midi_report["ppq"] == 480


def main() -> int:
    import tempfile

    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    func(Path(tmp))
                except AssertionError as exc:
                    failures += 1
                    print(f"[FAIL] {name}: {exc}")
                except Exception as exc:  # noqa: BLE001 - 手工跑的时候都打出来
                    failures += 1
                    print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
                else:
                    print(f"[ok] {name}")
    print(f"\n{'全部通过' if not failures else f'{failures} 个失败'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


def _music_note_order_ok(note: object) -> bool:
    """按 MusicXML 4.0 XSD 的 note 子元素顺序检查（顺序错 MuseScore 可能拒读）。"""
    order = {
        "grace": 0,
        "cue": 1,
        "chord": 2,
        "pitch": 3,
        "rest": 3,
        "duration": 4,
        "tie": 5,
        "instrument": 6,
        "footnote": 7,
        "level": 8,
        "voice": 9,
        "type": 10,
        "dot": 11,
        "accidental": 12,
        "time-modification": 13,
        "stem": 14,
        "notehead": 15,
        "notehead-text": 16,
        "staff": 17,
        "beam": 18,
        "notations": 19,
        "lyric": 20,
        "play": 21,
        "listen": 22,
    }
    ranks = [order[child.tag] for child in note if child.tag in order]
    return ranks == sorted(ranks)


def test_musicxml_note_child_order_is_schema_valid() -> None:
    """note 子元素必须符合 XSD 顺序：duration → tie → voice → type → notations → lyric。"""
    import xml.etree.ElementTree as ET

    score = make_score(bars=4)
    out = Path("work/test-export-order.musicxml")
    export_musicxml(score, out)
    root = ET.parse(out).getroot()
    bad = [note for note in root.iter("note") if not _music_note_order_ok(note)]
    assert not bad, [ET.tostring(item, encoding="unicode")[:200] for item in bad[:3]]
