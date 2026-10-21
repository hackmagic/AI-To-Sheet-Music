"""谱面校验（F24）：结构契约 + 音乐合理性。

两层校验
--------
1. **结构层**：直接调用 :func:`aitsm.schema.validate_score`（jsonschema）。
   结构错误一律判为 error，因为下游（前端渲染 / MusicXML / MIDI）会直接崩。
2. **音乐层**：小节拍数、和弦重叠、旋律越界、歌词覆盖、段落连续性、
   capo 与调性一致性、指法和弦匹配。

返回 ``{"ok": bool, "errors": [...], "warnings": [...], "stats": {...}}``，
其中 ``errors`` / ``warnings`` 的元素是 ``{level, code, message, path?}`` 字典，
便于 service 层直接并入 ``score.quality.warnings``。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .model import (
    bar_length_quarters,
    chord_intervals,
    chord_pitch_classes,
    duration_value,
    midi_to_name,
    parse_chord_symbol,
    parse_time_signature,
    tuning_to_midis,
)

try:  # 结构校验依赖 jsonschema；缺失时降级为内置检查
    from ..schema import validate_score as _schema_errors
except Exception:  # pragma: no cover - 循环导入 / 缺依赖时的降级
    _schema_errors = None  # type: ignore[assignment]

__all__ = ["validate_score", "ValidationReport"]

_TOL = 1e-3
#: 旋律建议音域（记谱音高，MIDI 编号）
_MELODY_LOW = 48
_MELODY_HIGH = 88


def _issue(level: str, code: str, message: str, path: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"level": level, "code": code, "message": message}
    if path:
        item["path"] = path
    return item


def _num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


class ValidationReport:
    """累积校验结果的小容器。"""

    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {}

    def error(self, code: str, message: str, path: str | None = None) -> None:
        self.errors.append(_issue("error", code, message, path))

    def warn(self, code: str, message: str, path: str | None = None) -> None:
        self.warnings.append(_issue("warn", code, message, path))

    def info(self, code: str, message: str, path: str | None = None) -> None:
        self.warnings.append(_issue("info", code, message, path))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": not self.errors,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


# --------------------------------------------------------------------------
# 结构层
# --------------------------------------------------------------------------


def _structural(score: dict, report: ValidationReport) -> None:
    if _schema_errors is None:  # pragma: no cover - 只在 schema 模块不可用时
        if score.get("schema") != "aitsm.score/v1":
            report.error("schema.mismatch", "score.schema 必须是 aitsm.score/v1", "schema")
        for key in ("meta", "sections", "bars"):
            if key not in score:
                report.error("schema.missing", f"缺少必填字段 {key}", key)
        return
    messages = _schema_errors(score) or []
    for message in messages:
        report.error("schema.invalid", str(message), "score")


# --------------------------------------------------------------------------
# 音乐层：小节
# --------------------------------------------------------------------------


def _check_bars(score: dict, report: ValidationReport) -> list[dict]:
    bars = score.get("bars") or []
    if not isinstance(bars, list) or not bars:
        report.error("bars.empty", "score.bars 为空，无法弹唱", "bars")
        return []

    previous_end: float | None = None
    for position, bar in enumerate(bars):
        if not isinstance(bar, dict):
            report.error("bars.type", f"第 {position + 1} 个 bar 不是对象", f"bars/{position}")
            continue
        expected_index = position + 1
        index = bar.get("index")
        if index != expected_index:
            report.error(
                "bars.index",
                f"小节序号不连续：位置 {expected_index} 上是 index={index}",
                f"bars/{position}/index",
            )
        start_sec = _num(bar.get("start_sec"))
        end_sec = _num(bar.get("end_sec"), start_sec)
        if end_sec <= start_sec - _TOL:
            report.error(
                "bars.span",
                f"第 {index} 小节 start_sec({start_sec}) >= end_sec({end_sec})",
                f"bars/{position}",
            )
        if previous_end is not None and abs(start_sec - previous_end) > 0.05:
            report.error(
                "bars.contiguous",
                f"第 {index} 小节与上一小节不连续（{previous_end} -> {start_sec}）",
                f"bars/{position}/start_sec",
            )
        previous_end = end_sec

        ts = parse_time_signature(bar.get("time_signature") or score.get("meta", {}).get("time_signature"), (4, 4))
        bar_len = bar_length_quarters(ts)
        _check_bar_chords(bar, bar_len, report, position)
        _check_bar_vocals(bar, bar_len, report, position)
        _check_bar_guitar(bar, bar_len, score, report, position)

    # 末尾：最后小节必须被段落覆盖（见 _check_sections）
    report.stats["bars"] = len(bars)
    report.stats["pickup"] = bool(bars and bars[0].get("pickup"))
    return bars


def _check_bar_chords(bar: dict, bar_len: float, report: ValidationReport, position: int) -> None:
    chords = bar.get("chords") or []
    index = bar.get("index")
    previous_end = 0.0
    for order, chord in enumerate(chords):
        path = f"bars/{position}/chords/{order}"
        beat = _num(chord.get("beat"))
        if beat < -_TOL or beat >= bar_len - _TOL:
            report.error(
                "chords.beat_range",
                f"第 {index} 小节第 {order + 1} 个和弦 beat={beat} 超出小节范围 0~{bar_len}",
                path,
            )
        try:
            dur = duration_value(chord.get("dur"))
        except ValueError:
            report.error("chords.dur", f"和弦时值无法解析：{chord.get('dur')!r}", path)
            dur = 0.0
        if dur <= 0:
            report.warn("chords.dur_zero", f"第 {index} 小节和弦时值为 0", path)
        if abs(beat - previous_end) > 0.05 and beat > previous_end + 0.05 and order > 0:
            report.warn(
                "chords.gap",
                f"第 {index} 小节第 {order} 与第 {order + 1} 个和弦之间有空档"
                f"（{previous_end:.2f} -> {beat:.2f} 拍）",
                path,
            )
        if beat < previous_end - 0.05:
            report.error(
                "chords.overlap",
                f"第 {index} 小节和弦重叠：新和弦 beat={beat} 早于前一个结束 {previous_end:.2f}",
                path,
            )
        if beat + dur > bar_len + 0.05:
            report.warn(
                "chords.overflow",
                f"第 {index} 小节和弦越过小节线（{beat}+{dur}>{bar_len}）",
                path,
            )
        previous_end = max(previous_end, beat + dur)
        if not chord.get("symbol"):
            report.error("chords.symbol", f"第 {index} 小节和弦缺少 symbol", path)
        else:
            try:
                parse_chord_symbol(str(chord["symbol"]))
            except ValueError as exc:
                report.error("chords.symbol_invalid", f"和弦符号无法解析：{chord['symbol']!r}（{exc}）", path)


def _check_bar_vocals(bar: dict, bar_len: float, report: ValidationReport, position: int) -> None:
    notes = bar.get("vocal") or []
    index = bar.get("index")
    previous_end = 0.0
    for order, note in enumerate(notes):
        path = f"bars/{position}/vocal/{order}"
        beat = _num(note.get("beat"))
        if beat < -_TOL or beat >= bar_len - _TOL:
            report.error(
                "vocal.beat_range",
                f"第 {index} 小节第 {order + 1} 个旋律音 beat={beat} 超出小节范围 0~{bar_len}",
                path,
            )
        try:
            dur = duration_value(note.get("dur")) if note.get("dur") is not None else 0.0
        except ValueError:
            report.error("vocal.dur", f"旋律时值无法解析：{note.get('dur')!r}", path)
            dur = 0.0
        if dur < 0:
            report.error("vocal.dur_negative", f"旋律时值为负：{dur}", path)
        tie = note.get("tie")
        if beat < previous_end - 0.05 and tie not in ("start", "continue"):
            report.warn(
                "vocal.overlap",
                f"第 {index} 小节旋律音重叠（beat={beat} < 前一个结束 {previous_end:.2f}）",
                path,
            )
        if beat + dur > bar_len + 0.05 and tie not in ("start", "continue"):
            report.warn(
                "vocal.overflow",
                f"第 {index} 小节旋律越过小节线（{beat}+{dur}>{bar_len}）",
                path,
            )
        previous_end = max(previous_end, beat + dur)


def _tuning_midis(score: dict) -> list[int]:
    """从 meta.guitar.tuning 解析六弦空弦音高（缺省/非法时用标准调弦）。"""
    guitar = (score.get("meta") or {}).get("guitar") or {}
    tuning = guitar.get("tuning") or guitar.get("tuning_name")
    try:
        result = list(tuning_to_midis(tuning))
    except Exception:  # noqa: BLE001 - 调弦写错时退回标准调弦，避免校验整体崩掉
        return [40, 45, 50, 55, 59, 64]
    if len(result) != 6:
        return [40, 45, 50, 55, 59, 64]
    return result


def _chord_at_beat(chords: Sequence[dict], beat: float) -> dict | None:
    """取某个拍位置上「正在响」的和弦（早于该拍且最靠后的那个）。"""
    active: dict | None = None
    for chord in chords:
        if _num(chord.get("beat")) <= beat + _TOL:
            active = chord
    if active is None and chords:
        active = chords[0]
    return active


def _check_hit_strings(
    bar: dict,
    hits: Sequence[dict],
    score: dict,
    report: ValidationReport,
    position: int,
) -> None:
    """校验 ``hits[].strings`` 的 ``[弦号, 品位]`` 与和弦指法音级一致。

    这是 TAB 与和弦符号之间的「跨字段一致性」：如果弦品换算出的音高落在和弦的
    音级集合之外，说明扫弦位置写错或指法选错，必须报错而不是悄悄导出。
    """
    chords = sorted(
        (chord for chord in (bar.get("chords") or []) if isinstance(chord, dict)),
        key=lambda chord: _num(chord.get("beat")),
    )
    if not chords:
        return
    tuning = _tuning_midis(score)
    voicings = score.get("voicings") or {}
    checked = 0
    for order, hit in enumerate(hits):
        pairs = hit.get("strings") or []
        if not pairs:
            continue
        chord = _chord_at_beat(chords, _num(hit.get("beat")))
        if not isinstance(chord, dict):
            continue
        allowed: set[int] = set()
        voicing = voicings.get(str(chord.get("voicing_id") or ""))
        if isinstance(voicing, dict):
            allowed |= _voicing_pcs(voicing.get("frets") or [], tuning)
        symbol = str(chord.get("symbol") or "")
        if symbol:
            try:
                allowed |= set(chord_pitch_classes(symbol))
            except ValueError:
                pass
        if not allowed:
            continue
        for pair_order, pair in enumerate(pairs):
            path = f"bars/{position}/guitar/hits/{order}/strings/{pair_order}"
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                report.error(
                    "guitar.hit_strings",
                    f"第 {bar.get('index')} 小节扫弦第 {order + 1} 击的 strings 元素必须是 [弦号, 品位]",
                    path,
                )
                continue
            string_number = int(_num(pair[0], 0))
            fret = int(_num(pair[1], 0))
            if not 1 <= string_number <= 6:
                report.error(
                    "guitar.hit_string",
                    f"第 {bar.get('index')} 小节扫弦第 {order + 1} 击的弦号 {string_number} 不在 1..6",
                    path,
                )
                continue
            if fret < 0:
                continue  # 闷音/不弹不参与音级判断
            midi = tuning[6 - string_number] + fret
            checked += 1
            if midi % 12 not in allowed:
                report.error(
                    "guitar.hit_pitch",
                    f"第 {bar.get('index')} 小节扫弦第 {order + 1} 击 [弦号, 品位]=[{string_number}, {fret}] "
                    f"换算为 {midi_to_name(midi)}，其音级不在和弦 {symbol} 的音级集合内",
                    path,
                )
    if checked:
        report.stats["tab_pairs_checked"] = report.stats.get("tab_pairs_checked", 0) + checked


def _check_bar_guitar(
    bar: dict, bar_len: float, score: dict, report: ValidationReport, position: int
) -> None:
    guitar = bar.get("guitar")
    if not isinstance(guitar, dict):
        report.error("guitar.missing", f"第 {bar.get('index')} 小节缺少 guitar", f"bars/{position}/guitar")
        return
    pattern_id = guitar.get("pattern_id")
    patterns = score.get("strum_patterns") or {}
    if not pattern_id:
        report.warn("guitar.pattern", f"第 {bar.get('index')} 小节没有节奏型", f"bars/{position}/guitar")
    elif pattern_id not in patterns:
        report.error(
            "guitar.pattern_unknown",
            f"节奏型 {pattern_id!r} 不在 score.strum_patterns 中",
            f"bars/{position}/guitar/pattern_id",
        )
    else:
        pattern = patterns[pattern_id]
        expected = pattern.get("beats_per_bar")
        ts = parse_time_signature(bar.get("time_signature") or score.get("meta", {}).get("time_signature"), (4, 4))
        if expected is not None and int(expected) != int(ts[0]):
            report.warn(
                "guitar.pattern_meter",
                f"节奏型 {pattern_id} 的 beats_per_bar={expected} 与拍号 {ts[0]}/{ts[1]} 不一致",
                f"bars/{position}/guitar/pattern_id",
            )
    hits = [hit for hit in (guitar.get("hits") or []) if isinstance(hit, dict)]
    for order, hit in enumerate(hits):
        beat = _num(hit.get("beat"))
        if beat < -_TOL or beat >= bar_len + 0.2:
            report.error(
                "guitar.hit_range",
                f"第 {bar.get('index')} 小节扫弦第 {order + 1} 击 beat={beat} 越界",
                f"bars/{position}/guitar/hits/{order}",
            )
    if hits:
        _check_hit_strings(bar, hits, score, report, position)


# --------------------------------------------------------------------------
# 音乐层：段落 / 调性 / 指法 / 歌词
# --------------------------------------------------------------------------


def _check_sections(score: dict, bars: Sequence[dict], report: ValidationReport) -> None:
    sections = score.get("sections")
    if not isinstance(sections, list) or not sections:
        report.error("sections.empty", "score.sections 为空", "sections")
        return
    bar_numbers = [bar.get("index") for bar in bars if isinstance(bar, dict)]
    if not bar_numbers:
        return
    covered: dict[int, int] = {}
    last_start = 0
    for order, section in enumerate(sections):
        path = f"sections/{order}"
        start_bar = section.get("start_bar")
        end_bar = section.get("end_bar")
        if not isinstance(start_bar, int) or not isinstance(end_bar, int):
            report.error("sections.range", f"段落 {section.get('id')} 的 start_bar/end_bar 不是整数", path)
            continue
        if end_bar < start_bar:
            report.error("sections.range", f"段落 {section.get('id')} end_bar < start_bar", path)
        if start_bar < last_start:
            report.error("sections.overlap", f"段落 {section.get('id')} 与上一段小节范围重叠", path)
        last_start = end_bar + 1
        if start_bar < 1 or end_bar > max(bar_numbers):
            report.error(
                "sections.out_of_range",
                f"段落 {section.get('id')} 小节范围 {start_bar}-{end_bar} 越界（共 {max(bar_numbers)} 小节）",
                path,
            )
        for number in range(start_bar, end_bar + 1):
            covered[number] = covered.get(number, 0) + 1
    holes = [number for number in bar_numbers if number not in covered]
    if holes:
        report.error("sections.hole", f"小节 {holes[:8]} 没有任何段落覆盖", "sections")
    duplicated = [number for number, count in covered.items() if count > 1]
    if duplicated:
        report.error("sections.duplicate", f"小节 {duplicated[:8]} 被多个段落覆盖", "sections")

    # 小节上的 section 引用必须存在
    ids = {section.get("id") for section in sections}
    for position, bar in enumerate(bars):
        ref = bar.get("section")
        if ref is not None and ref not in ids:
            report.error(
                "bars.section_ref",
                f"第 {bar.get('index')} 小节引用了不存在的段落 {ref!r}",
                f"bars/{position}/section",
            )
    report.stats["sections"] = len(sections)


def _check_key_and_capo(score: dict, report: ValidationReport) -> None:
    meta = score.get("meta") or {}
    key = meta.get("key") or {}
    capo = meta.get("capo")
    if not isinstance(capo, int) or not (0 <= capo <= 12):
        report.error("meta.capo", f"meta.capo 应为 0~12 的整数，实际 {capo!r}", "meta/capo")
        capo = 0
    for field in ("detected", "sounding", "play", "mode", "num_sharps_flats"):
        if field not in key:
            report.error("meta.key", f"meta.key 缺少 {field}", f"meta/key/{field}")
    sounding = str(key.get("sounding") or "")
    play = str(key.get("play") or "")
    if sounding and play:
        from .model import note_to_pc

        try:
            diff = (note_to_pc(sounding) - note_to_pc(play)) % 12
        except ValueError:
            diff = None
        if diff is not None and diff != capo % 12:
            report.error(
                "meta.capo_mismatch",
                f"capo={capo} 与 sounding({sounding})/play({play}) 不一致：应为 {diff}",
                "meta/capo",
            )
    guitar = meta.get("guitar") or {}
    tuning = guitar.get("tuning") or []
    if tuning and len(tuning) != 6:
        report.error("meta.guitar", f"吉他调弦应是 6 根弦，实际 {len(tuning)}", "meta/guitar/tuning")
    report.stats["capo"] = capo
    report.stats["play_key"] = play
    report.stats["sounding_key"] = sounding


def _voicing_pcs(frets: Sequence[Any], tuning: Sequence[int]) -> set[int]:
    """指法 -> 实际发声的音级集合。"""
    pcs: set[int] = set()
    for string_index, fret in enumerate(frets):
        if string_index >= len(tuning):
            continue
        if fret is None or not isinstance(fret, int) or fret < 0:
            continue
        pcs.add((tuning[string_index] + fret) % 12)
    return pcs


def _check_voicings(score: dict, bars: Sequence[dict], report: ValidationReport) -> None:
    voicings = score.get("voicings") or {}
    meta = score.get("meta") or {}
    guitar = meta.get("guitar") or {}
    tuning = guitar.get("tuning") or [40, 45, 50, 55, 59, 64]
    if len(tuning) != 6:
        tuning = [40, 45, 50, 55, 59, 64]
    used = 0
    matched = 0
    for voicing_id, voicing in voicings.items():
        if not isinstance(voicing, dict):
            report.error("voicings.type", f"voicing {voicing_id} 不是对象", f"voicings/{voicing_id}")
            continue
        frets = voicing.get("frets")
        if not isinstance(frets, list) or len(frets) != 6:
            report.error(
                "voicings.frets",
                f"voicing {voicing_id} 的 frets 必须是 6 元素数组",
                f"voicings/{voicing_id}/frets",
            )
            continue
        sounding_count = sum(1 for fret in frets if isinstance(fret, int) and fret >= 0)
        if sounding_count < 3:
            report.warn(
                "voicings.thin",
                f"voicing {voicing_id} 只响 {sounding_count} 根弦",
                f"voicings/{voicing_id}",
            )
        symbol = voicing.get("symbol") or voicing_id.split("_", 1)[0]
        try:
            chord = parse_chord_symbol(str(symbol))
        except ValueError:
            report.warn(
                "voicings.symbol",
                f"voicing {voicing_id} 的 symbol {symbol!r} 无法解析，跳过音级匹配",
                f"voicings/{voicing_id}/symbol",
            )
            continue
        pcs = _voicing_pcs(frets, tuning)
        intervals = chord_intervals(chord.quality)
        required = {chord.root_pc}
        for interval in intervals:
            if interval in (3, 4, 7):
                required.add((chord.root_pc + interval) % 12)
        if chord.bass_pc is not None:
            required.add(chord.bass_pc)
        missing = required - pcs
        if missing:
            report.warn(
                "voicings.missing_tones",
                f"voicing {voicing_id} 缺少和弦音 {sorted(missing)}（实际音级 {sorted(pcs)}）",
                f"voicings/{voicing_id}",
            )

    for position, bar in enumerate(bars):
        for order, chord in enumerate(bar.get("chords") or []):
            used += 1
            voicing_id = chord.get("voicing_id")
            if not voicing_id:
                continue
            voicing = voicings.get(voicing_id)
            if voicing is None:
                report.error(
                    "chords.voicing_unknown",
                    f"和弦 {chord.get('symbol')} 引用了不存在的 voicing {voicing_id!r}",
                    f"bars/{position}/chords/{order}/voicing_id",
                )
                continue
            matched += 1
            # concert 与记谱符号之间的音级差应等于 capo
            symbol = chord.get("symbol")
            concert = chord.get("concert")
            if symbol and concert:
                try:
                    expected = (parse_chord_symbol(str(concert)).root_pc - parse_chord_symbol(str(symbol)).root_pc) % 12
                except ValueError:
                    expected = None
                if expected is not None and expected != int(meta.get("capo") or 0) % 12:
                    report.error(
                        "chords.concert_capo",
                        f"和弦 {symbol} 与 concert {concert} 的音级差 {expected} 与 capo 不一致",
                        f"bars/{position}/chords/{order}",
                    )
    report.stats["chords"] = used
    report.stats["chords_with_voicing"] = matched
    report.stats["voicing_coverage"] = round(matched / used, 3) if used else 0.0


def _check_lyrics_and_melody(score: dict, bars: Sequence[dict], report: ValidationReport) -> None:
    notes: list[dict] = []
    for bar in bars:
        notes.extend(bar.get("vocal") or [])
    if not notes:
        report.warn("melody.empty", "整曲没有旋律音符，只能作为纯和弦谱使用", "bars")
        report.stats["vocal_notes"] = 0
        report.stats["lyric_coverage"] = 0.0
        return
    with_lyric = sum(1 for note in notes if note.get("lyric"))
    coverage = with_lyric / len(notes)
    report.stats["vocal_notes"] = len(notes)
    report.stats["lyric_coverage"] = round(coverage, 3)
    if coverage < 0.5:
        report.warn(
            "lyrics.coverage",
            f"歌词覆盖率仅 {coverage:.0%}（{with_lyric}/{len(notes)}），可能无法逐字演唱",
            "bars",
        )
    out_of_range = [
        note.get("midi")
        for note in notes
        if isinstance(note.get("midi"), int) and not (_MELODY_LOW <= int(note["midi"]) <= _MELODY_HIGH)
    ]
    if out_of_range:
        report.warn(
            "melody.range",
            f"{len(out_of_range)} 个旋律音超出建议音域 {_MELODY_LOW}-{_MELODY_HIGH}："
            f"{sorted(set(out_of_range))[:8]}",
            "bars",
        )
    invalid = [note for note in notes if not isinstance(note.get("midi"), int)]
    if invalid:
        report.error("melody.midi", f"{len(invalid)} 个旋律音缺少整数 midi", "bars")

    # 旋律音域跨度（超过两个八度通常意味着八度跳变错误，但对女声主旋律属正常 → info）
    midis = [int(note["midi"]) for note in notes if isinstance(note.get("midi"), int)]
    if midis and max(midis) - min(midis) > 24:
        report.info(
            "melody.span",
            f"旋律音域跨 {max(midis) - min(midis)} 个半音（{min(midis)}-{max(midis)}），建议检查八度跳变",
            "bars",
        )
    report.stats["melody_low"] = min(midis) if midis else None
    report.stats["melody_high"] = max(midis) if midis else None


def _check_meta(score: dict, report: ValidationReport) -> None:
    meta = score.get("meta") or {}
    tempo = _num(meta.get("tempo"), 0.0)
    if not (20 <= tempo <= 320):
        report.error("meta.tempo", f"meta.tempo={tempo} 不合理（应在 20~320 BPM）", "meta/tempo")
    ts = parse_time_signature(meta.get("time_signature"), (4, 4))
    if ts[0] not in (2, 3, 4, 6, 9, 12):
        report.warn("meta.time_signature", f"拍号 {ts[0]}/{ts[1]} 不常见", "meta/time_signature")
    quality = score.get("quality") or {}
    stages = quality.get("stages") or {}
    report.stats["quality_overall"] = quality.get("overall")
    report.stats["stages"] = {
        key: value for key, value in stages.items() if isinstance(value, (int, float))
    }


def validate_score(score: dict) -> dict:
    """校验 score.json（结构 + 音乐合理性）。

    返回 ``{"ok": bool, "errors": [...], "warnings": [...], "stats": {...}}``。
    """
    report = ValidationReport()
    if not isinstance(score, dict):
        report.error("score.type", "score 必须是对象")
        return report.as_dict()

    _structural(score, report)
    bars = _check_bars(score, report)
    _check_sections(score, bars, report)
    _check_key_and_capo(score, report)
    _check_voicings(score, bars, report)
    _check_lyrics_and_melody(score, bars, report)
    _check_meta(score, report)

    # 统计：时长 / 和弦种类 / 节奏型
    meta = score.get("meta") or {}
    report.stats["duration_sec"] = round(_num(meta.get("source", {}).get("duration_sec")), 3)
    symbols: list[str] = []
    for bar in bars:
        for chord in bar.get("chords") or []:
            symbol = chord.get("symbol")
            if isinstance(symbol, str) and symbol:
                symbols.append(symbol)
    report.stats["chord_symbols"] = sorted(set(symbols))
    report.stats["unique_chords"] = len(set(symbols))
    report.stats["strum_patterns"] = sorted((score.get("strum_patterns") or {}).keys())
    report.stats["voicings"] = len(score.get("voicings") or {})
    return report.as_dict()
