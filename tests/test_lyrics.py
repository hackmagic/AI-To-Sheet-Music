"""歌词 ASR 清洗层的离线单测（不依赖 whisper / numpy / 网络）。

覆盖：文本清洗、噪声行过滤、音节切分、词级时间戳规整、段落断句合并、去重、
行 -> 音节展开、以及 whisper 不可用时的安全降级。

运行方式（pytest 或直接 python 均可）：
    .venv/Scripts/python.exe tests/test_lyrics.py
    .venv/Scripts/python.exe -m pytest tests/test_lyrics.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aitsm.pipeline import lyrics as L  # noqa: E402


def _approx(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= tolerance


# ---------------------------------------------------------------------------
# 文本清洗 / 噪声行
# ---------------------------------------------------------------------------


def test_clean_lyric_text_strips_annotations_and_music_symbols() -> None:
    assert L.clean_lyric_text("[00:12.34] 一 个 人 走") == "一个人走"
    assert L.clean_lyric_text("♪ 我 爱 你 ♪") == "我爱你"
    assert L.clean_lyric_text("（合）风 吹 过") == "风 吹 过".replace(" ", "")
    assert L.clean_lyric_text("Hello   world") == "Hello world"
    assert L.clean_lyric_text(None) == ""
    assert L.clean_lyric_text("   ") == ""


def test_is_noise_line_flags_metadata_but_keeps_real_lyrics() -> None:
    assert L.is_noise_line("[Music]") is True
    assert L.is_noise_line("[间奏]") is True
    assert L.is_noise_line("作词: 张三") is True
    assert L.is_noise_line("♪♪♪") is True
    assert L.is_noise_line("字幕由 XX 制作") is True
    assert L.is_noise_line("") is True
    # 真实歌词不能被误杀
    assert L.is_noise_line("音乐响起") is False
    assert L.is_noise_line("We are the champions") is False
    assert L.is_noise_line("一个人走") is False


# ---------------------------------------------------------------------------
# 音节切分
# ---------------------------------------------------------------------------


def test_split_syllables_chinese_one_char_per_syllable() -> None:
    assert L.split_syllables("一个人走") == ["一", "个", "人", "走"]
    assert L.split_syllables("一个人，走。") == ["一", "个", "人", "走"]


def test_split_syllables_mixed_latin_and_cjk() -> None:
    syllables = L.split_syllables("Hello 世界")
    assert syllables == ["Hello", "世", "界"]


def test_split_syllables_english_words() -> None:
    assert L.split_syllables("I love you", "en") == ["I", "love", "you"]
    pieces = L.split_syllables("beautiful", "en")
    assert len(pieces) >= 2
    assert "".join(pieces) == "beautiful"


# ---------------------------------------------------------------------------
# 时间戳规整
# ---------------------------------------------------------------------------


def test_build_words_distributes_evenly_and_monotonically() -> None:
    words = L.build_words("一二三四", 1.0, 3.0)
    assert len(words) == 4
    assert _approx(words[0]["start"], 1.0)
    assert _approx(words[-1]["end"], 3.0)
    starts = [word["start"] for word in words]
    assert starts == sorted(starts)
    for left, right in zip(words, words[1:]):
        assert right["start"] >= left["end"] - 1e-6


def test_normalize_segments_cleans_and_keeps_word_timestamps() -> None:
    raw = [
        {
            "start": 0.5,
            "end": 2.0,
            "text": "一个人走",
            "words": [
                {"word": "一个人", "start": 0.5, "end": 1.3},
                {"word": "走", "start": 1.4, "end": 2.0},
            ],
            "avg_logprob": -0.2,
            "no_speech_prob": 0.01,
        },
        {"start": 2.5, "end": 3.0, "text": "[Music]", "words": []},
        {"start": 3.0, "end": 3.0, "text": "时间无效", "words": []},
    ]
    segments = L.normalize_segments(raw, language="zh")
    assert len(segments) == 1
    segment = segments[0]
    assert segment["text"] == "一个人走"
    assert [word["text"] for word in segment["words"]] == ["一", "个", "人", "走"]
    starts = [word["start"] for word in segment["words"]]
    assert starts == sorted(starts)
    assert 0.0 <= segment["confidence"] <= 1.0


def test_normalize_segments_without_words_falls_back_to_even_split() -> None:
    segments = L.normalize_segments(
        [{"start": 0.0, "end": 2.0, "text": "一个人走"}], language="zh"
    )
    assert len(segments) == 1
    assert len(segments[0]["words"]) == 4
    assert _approx(segments[0]["words"][0]["start"], 0.0, 1e-3)
    assert _approx(segments[0]["words"][-1]["end"], 2.0, 1e-3)


# ---------------------------------------------------------------------------
# 断句 / 合并 / 去重
# ---------------------------------------------------------------------------


def _segment(text: str, start: float, end: float, syllables: list[str] | None = None) -> dict:
    syllables = syllables or list(text.replace("，", "").replace("。", ""))
    step = (end - start) / max(len(syllables), 1)
    words = [
        {"text": syllable, "start": round(start + index * step, 3), "end": round(start + (index + 1) * step, 3)}
        for index, syllable in enumerate(syllables)
    ]
    return {"start": start, "end": end, "text": text, "words": words, "confidence": 0.8}


def test_group_segments_merges_close_segments() -> None:
    segments = [
        _segment("第一句", 0.0, 1.0),
        _segment("第二句", 1.1, 2.0),
    ]
    lines = L.group_segments_into_lines(segments, language="zh")
    assert len(lines) == 1
    assert lines[0]["text"] == "第一句第二句"
    assert len(lines[0]["words"]) == 6


def test_group_segments_splits_on_large_gap() -> None:
    segments = [
        _segment("第一句", 0.0, 1.0),
        _segment("第二句", 2.0, 3.0),
    ]
    lines = L.group_segments_into_lines(segments, language="zh")
    assert len(lines) == 2


def test_group_segments_splits_after_terminal_punctuation() -> None:
    segments = [
        _segment("第一句。", 0.0, 1.0),
        _segment("第二句", 1.0, 2.0),
    ]
    lines = L.group_segments_into_lines(segments, language="zh")
    assert len(lines) == 2
    assert lines[0]["text"] == "第一句。"


def test_group_segments_splits_long_pause_inside_segment() -> None:
    segment = {
        "start": 0.0,
        "end": 4.0,
        "text": "一个人走两个人走",
        "confidence": 0.7,
        "words": [
            {"text": "一", "start": 0.0, "end": 0.3},
            {"text": "个", "start": 0.3, "end": 0.6},
            {"text": "人", "start": 0.6, "end": 0.9},
            {"text": "走", "start": 0.9, "end": 1.2},
            {"text": "两", "start": 2.5, "end": 2.8},
            {"text": "个", "start": 2.8, "end": 3.1},
            {"text": "人", "start": 3.1, "end": 3.4},
            {"text": "走", "start": 3.4, "end": 3.7},
        ],
    }
    lines = L.group_segments_into_lines([segment], language="zh")
    assert len(lines) == 2
    assert lines[0]["text"] == "一个人走"
    assert lines[1]["text"] == "两个人走"


def test_dedupe_lines_removes_exact_and_looped_duplicates() -> None:
    lines = [
        _segment("爱情", 0.0, 2.0),
        _segment("爱情", 0.0, 2.0),  # 完全相同
        _segment("爱情", 2.1, 4.0),  # whisper 循环幻觉（紧邻重复）
        _segment("爱情", 10.0, 12.0),  # 真实副歌重复必须保留
        _segment("[Music]", 20.0, 22.0),  # 噪声
    ]
    result = L.dedupe_lines(lines)
    assert [line["text"] for line in result] == ["爱情", "爱情"]
    assert [round(line["start"], 2) for line in result] == [0.0, 10.0]


# ---------------------------------------------------------------------------
# 行 -> 音节（对齐层入口）
# ---------------------------------------------------------------------------


def test_line_syllables_uses_word_timestamps() -> None:
    line = {
        "start": 0.0,
        "end": 2.0,
        "text": "一个人走",
        "words": [
            {"start": 0.0, "end": 0.5, "text": "一"},
            {"start": 0.5, "end": 1.0, "text": "个"},
            {"start": 1.0, "end": 1.5, "text": "人"},
            {"start": 1.5, "end": 2.0, "text": "走"},
        ],
    }
    syllables = L.line_syllables(line)
    assert [item["text"] for item in syllables] == ["一", "个", "人", "走"]
    for item, expected in zip(syllables, [0.0, 0.5, 1.0, 1.5]):
        assert _approx(item["start"], expected, 1e-3)


def test_line_syllables_without_words_splits_evenly() -> None:
    syllables = L.line_syllables({"start": 0.0, "end": 2.0, "text": "一个人走"})
    assert [item["text"] for item in syllables] == ["一", "个", "人", "走"]
    starts = [item["start"] for item in syllables]
    assert _approx(starts[0], 0.0, 1e-3)
    assert _approx(starts[-1], 1.5, 1e-3)


def test_line_syllables_empty_line() -> None:
    assert L.line_syllables({"start": 0.0, "end": 1.0, "text": ""}) == []
    assert L.line_syllables({"start": 0.0, "end": 1.0, "text": "[Music]"}) == []


# ---------------------------------------------------------------------------
# 降级与契约
# ---------------------------------------------------------------------------


def test_transcribe_lyrics_missing_file_returns_empty() -> None:
    assert L.transcribe_lyrics(str(ROOT / "no-such-file.wav")) == []


def test_transcribe_lyrics_tolerates_progress_callback_errors() -> None:
    def broken_callback(payload):  # noqa: ANN001 - 故意抛异常
        raise RuntimeError("callback boom")

    assert L.transcribe_lyrics(str(ROOT / "no-such-file.wav"), progress=broken_callback) == []


def test_lyrics_lines_pass_schema() -> None:
    try:
        import jsonschema
    except Exception:
        print("  (skip) jsonschema 未安装，跳过 schema 校验")
        return
    schema = json.loads((ROOT / "schemas" / "analysis.schema.json").read_text(encoding="utf-8"))
    lyric_schema = schema["properties"]["lyrics"]
    segments = L.normalize_segments(
        [
            {"start": 0.0, "end": 2.0, "text": "一个人走"},
            {"start": 2.5, "end": 4.0, "text": "在街头"},
        ],
        language="zh",
    )
    lines = L.dedupe_lines(L.group_segments_into_lines(segments, language="zh"))
    jsonschema.validate(lines, lyric_schema)
    for line in lines:
        assert {"start", "end", "text"} <= set(line)
        assert isinstance(line["words"], list)


# ---------------------------------------------------------------------------
# 真实音频歌词准确率评估（LRC 对照）
# ---------------------------------------------------------------------------

REAL_LRC = Path(
    "K:/workspaces/hackmagic/hackmagic-music/assets/小林未郁、HOYO-MiX/"
    "小林未郁、HOYO-MiX - 下一站 Beyond This Station.lrc"
)

_LRC_TAG_RE = __import__("re").compile(r"^\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$")
_LRC_META_KEYWORDS = (
    "作词", "作曲", "编曲", "制作人", "人声", "和声", "电吉他", "制谱", "母带",
    "混音", "录音", "监制", "演唱", "Lyricist", "Composer", "Arranger", "Producer",
    "Vocal", "Guitar", "Copyist", "Mix", "Master",
)


def _lrc_lines(path: Path) -> list[tuple[float, str]]:
    """读 LRC：只保留真正被唱出来的歌词行，跳过 [ti:]/[ar:] 与制作人员名单行。"""
    if not path.is_file():
        return []
    entries: list[tuple[float, str]] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        match = _LRC_TAG_RE.match(raw.strip())
        if not match:
            continue
        minutes, seconds, text = match.groups()
        text = text.strip()
        if not text:
            continue
        if any(keyword in text for keyword in _LRC_META_KEYWORDS):
            continue
        # 标题行：「曲名 - 歌手」形式，不是歌词
        if " - " in text and ("/" in text or "&" in text):
            continue
        entries.append((int(minutes) * 60 + float(seconds), text))
    return entries


def _levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    """经典编辑距离（列表元素级），用于 CER/WER。"""
    if not reference:
        return len(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """字符错误率：编辑距离 / 参考字符数（去掉空白后比较）。"""
    ref_chars = [char for char in reference if not char.isspace()]
    hyp_chars = [char for char in hypothesis if not char.isspace()]
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """词错误率：按空白切词后比较。"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def test_error_rate_helpers_on_known_pairs() -> None:
    assert _approx(character_error_rate("hello world", "hello world"), 0.0)
    # "helo world" 少一个 l -> 1 个字符编辑 / 10 个非空白字符
    assert _approx(character_error_rate("hello world", "helo world"), 0.1)
    assert _approx(word_error_rate("hello world", "helo world"), 0.5)
    assert _approx(word_error_rate("", ""), 0.0)


def test_lrc_loader_returns_sung_lines_only() -> None:
    entries = _lrc_lines(REAL_LRC)
    if not entries:
        print(f"  (skip) 找不到参考 LRC：{REAL_LRC}")
        return
    assert len(entries) > 20
    assert entries[0][0] >= 0.0
    texts = [text for _, text in entries]
    assert all(text for text in texts)
    # 元数据行（作词/作曲/编曲…）不应出现在正文里
    assert not any(text.startswith("作词") or text.startswith("作曲") for text in texts)


def test_whisper_transcribes_real_song_against_lrc() -> None:
    import os

    audio = os.environ.get("AITSM_LYRIC_AUDIO", "").strip()
    if not audio:
        print("  (skip) 未设置 AITSM_LYRIC_AUDIO，跳过真实音频歌词准确率测试")
        return
    reference_lines = _lrc_lines(REAL_LRC)
    lines = L.transcribe_lyrics(audio, language="auto")
    assert lines, "whisper 应至少识别出一行歌词"
    hypothesis = " ".join(line["text"] for line in lines)
    reference = " ".join(text for _, text in reference_lines) if reference_lines else ""
    print(
        f"  歌词行数={len(lines)} 参考行数={len(reference_lines)} "
        f"CER={character_error_rate(reference, hypothesis):.3f} "
        f"WER={word_error_rate(reference, hypothesis):.3f}"
    )
    if reference:
        assert character_error_rate(reference, hypothesis) < 0.8, hypothesis[:200]


def test_faster_whisper_segments_conversion() -> None:
    from types import SimpleNamespace

    words = [
        SimpleNamespace(word=" Hello", start=0.1, end=0.4, probability=0.9),
        SimpleNamespace(word=" world", start=0.4, end=0.9, probability=0.8),
    ]
    segment = SimpleNamespace(
        start=0.1, end=0.9, text=" Hello world", words=words, avg_logprob=-0.2, no_speech_prob=0.05
    )
    raw = L.faster_whisper_segments_to_raw([segment])
    assert raw == [
        {
            "start": 0.1,
            "end": 0.9,
            "text": " Hello world",
            "words": [
                {"word": " Hello", "start": 0.1, "end": 0.4, "probability": 0.9},
                {"word": " world", "start": 0.4, "end": 0.9, "probability": 0.8},
            ],
            "avg_logprob": -0.2,
            "no_speech_prob": 0.05,
        }
    ]
    # 转出的 dict 必须能被清洗层直接消费
    segments = L.normalize_segments(raw, language="en")
    assert segments and segments[0]["text"] == "Hello world"
    assert 0.0 < segments[0]["confidence"] <= 1.0
    assert [word["text"] for word in segments[0]["words"]] == ["Hel", "lo", "world"]


def test_faster_whisper_conversion_handles_missing_fields() -> None:
    from types import SimpleNamespace

    segment = SimpleNamespace(start=1.0, end=2.0)
    raw = L.faster_whisper_segments_to_raw([segment])
    assert raw[0]["text"] == "" and raw[0]["words"] == []
    assert L.faster_whisper_segments_to_raw(None) == []


def test_initial_prompt_disabled_for_auto_language() -> None:
    # 自动检测：不加 prompt，避免任何语言被 prompt 带偏
    assert L._initial_prompt(None) is None
    assert L._initial_prompt("auto") is None
    # 明确指定中日文时才加 prompt，且必须是祈使句元语言，不能是“歌词内容”式句子
    zh_prompt = L._initial_prompt("zh")
    assert zh_prompt is not None and "歌词行" in zh_prompt
    # 红线：英文 prompt 会被 whisper 当歌词复读，必须禁用或至少不含该短语
    en_prompt = L._initial_prompt("en")
    assert en_prompt is None or "lyrics of a song" not in en_prompt.lower()
    assert L._initial_prompt("zh") != L._initial_prompt(None)


def test_prompt_echo_and_meta_lyric_hallucination_are_rejected() -> None:
    fake_prompt = (
        "The following are the lyrics of a song, transcribed line by line with punctuation."
    )
    # whisper 把 prompt 当歌词复读（可能改标点/断行），必须能识别
    assert L.is_prompt_echo(
        "The following are the lyrics of a song transcribed", fake_prompt
    )
    assert L.is_prompt_echo(
        "The lyrics of this song are the lyrics of a song, transcribed line by line with",
        fake_prompt,
    )
    # 子词被空格拆开的复读（"lyr ics"）由元描述正则兜住
    assert L.is_meta_lyric_hallucination("The lyr ics of this song are the")
    # 真歌词不能误伤
    assert not L.is_prompt_echo("Let's pave it out", fake_prompt)
    assert not L.is_prompt_echo("What's stretching so far along", fake_prompt)
    # 元描述（自述）也要丢
    assert L.is_meta_lyric_hallucination("The lyrics of this song are the")
    assert L.is_meta_lyric_hallucination(
        "The lyrics of this song are the lyrics of a song, transcribed line by line with"
    )
    assert not L.is_meta_lyric_hallucination("Let's pave it out")


def test_low_quality_segments_are_dropped() -> None:
    raw = [
        {"start": 0.0, "end": 2.0, "text": "real line", "no_speech_prob": 0.1, "avg_logprob": -0.2},
        {"start": 2.0, "end": 4.0, "text": "invented line", "no_speech_prob": 0.9, "avg_logprob": -1.6},
        # 只满足一个条件时不丢（避免误伤轻声演唱）
        {"start": 4.0, "end": 6.0, "text": "soft line", "no_speech_prob": 0.9, "avg_logprob": -0.4},
        {"start": 6.0, "end": 8.0, "text": "quiet line", "no_speech_prob": 0.2, "avg_logprob": -1.8},
    ]
    texts = [segment["text"] for segment in L.normalize_segments(raw)]
    assert "invented line" not in texts
    assert "real line" in texts and "soft line" in texts and "quiet line" in texts


def test_words_to_text_keeps_word_boundaries() -> None:
    words = [
        {"text": "We", "start": 0.0, "end": 0.1, "space_before": True},
        {"text": "de", "start": 0.1, "end": 0.2, "space_before": True},
        {"text": "ploy", "start": 0.2, "end": 0.3, "space_before": False},
        {"text": "an", "start": 0.3, "end": 0.4, "space_before": True},
        {"text": "ti", "start": 0.4, "end": 0.5, "space_before": False},
    ]
    # 同一词的音节不插空格，词之间才插
    assert L.words_to_text(words, "en") == "We deploy anti"
    # 中文直接相连
    assert L.words_to_text(words, "zh") == "Wedeployanti"


def test_split_words_are_merged_before_syllable_split() -> None:
    # whisper 用前导空格标记词边界：" lyr" + "ics" 必须并回 "lyrics"
    merged = L.merge_split_words(
        [
            {"word": " The", "start": 0.0, "end": 0.2},
            {"word": " lyr", "start": 0.2, "end": 0.4},
            {"word": "ics", "start": 0.4, "end": 0.6},
            {"word": " of", "start": 0.6, "end": 0.8},
        ]
    )
    assert [item["word"] for item in merged] == [" The", " lyrics", " of"]
    # 中文 token 不带空格，不能被并成一个词
    merged_cn = L.merge_split_words(
        [{"word": "我", "start": 0.0, "end": 0.2}, {"word": "们", "start": 0.2, "end": 0.4}]
    )
    assert [item["word"] for item in merged_cn] == ["我", "们"]


def test_strip_repeated_edge_interjections() -> None:
    assert (
        L.strip_repeated_edge_interjections("oh oh oh I stand in promise with my heart and")
        == "I stand in promise with my heart and"
    )
    assert (
        L.strip_repeated_edge_interjections("I stand in promise oh oh oh oh")
        == "I stand in promise"
    )
    # 少于 3 个的语气词是真歌词，保留
    assert L.strip_repeated_edge_interjections("Oh~ I miss you") == "Oh~ I miss you"
    assert L.strip_repeated_edge_interjections("oh oh") == "oh oh"
    assert L.strip_repeated_edge_interjections("oh oh oh") == ""


def test_split_long_lines_uses_breath_gaps() -> None:
    line = {
        "start": 0.0,
        "end": 4.1,
        "text": "hold on tight to me my friend",
        "confidence": 0.8,
        "words": [
            {"start": 0.0, "end": 0.4, "text": "hold"},
            {"start": 0.4, "end": 0.8, "text": "on"},
            {"start": 0.8, "end": 1.1, "text": "tight"},
            {"start": 2.0, "end": 2.4, "text": "to"},  # 0.9s 换气
            {"start": 2.4, "end": 2.8, "text": "me"},
            {"start": 2.8, "end": 3.2, "text": "my"},
            {"start": 3.2, "end": 3.6, "text": "friend"},
        ],
    }
    split = L.split_long_lines([line])
    assert [item["text"] for item in split] == ["hold on tight", "to me my friend"]
    assert split[0]["start"] == 0.0 and split[0]["end"] == 1.1
    assert split[1]["start"] == 2.0 and split[1]["end"] == 3.6
    # 没有停顿、也不算长的行保持原样（不拆）
    short = {"start": 0.0, "end": 1.0, "text": "a b", "confidence": 0.8,
             "words": [{"start": 0.0, "end": 0.5, "text": "a"}, {"start": 0.5, "end": 1.0, "text": "b"}]}
    assert L.split_long_lines([short]) == [short]


def test_repetition_hallucination_detection() -> None:
    assert L.is_repetition_hallucination("oh oh oh oh oh oh")
    assert L.is_repetition_hallucination("Oh oh oh oh")
    assert not L.is_repetition_hallucination("Oh~")  # 真歌词里的短语气词要保留
    assert not L.is_repetition_hallucination("oh oh")
    assert not L.is_repetition_hallucination("I storm the silence with our hearts awake")
    lines = [
        {"start": 0.0, "end": 1.0, "text": "real line one", "words": [], "confidence": 0.5},
        {"start": 1.0, "end": 9.0, "text": "oh oh oh oh oh oh oh oh", "words": [], "confidence": 0.5},
        {"start": 9.0, "end": 12.0, "text": "real line two", "words": [], "confidence": 0.5},
    ]
    kept = L.drop_repetition_hallucinations(lines)
    assert [line["text"] for line in kept] == ["real line one", "real line two"]


def test_resolve_line_overlaps_trims_and_drops_fragments() -> None:
    lines = [
        {"start": 0.0, "end": 3.0, "text": "a", "words": [], "confidence": 0.5},
        {"start": 2.0, "end": 5.0, "text": "b", "words": [], "confidence": 0.5},
        {"start": 4.9, "end": 5.0, "text": "c", "words": [], "confidence": 0.5},  # 处理后成碎片
    ]
    fixed = L.resolve_line_overlaps(lines)
    assert [line["text"] for line in fixed] == ["a", "b"]
    assert fixed[0]["end"] == 2.0  # 前一行被裁到后一行开始处
    assert fixed[1]["start"] == 2.0
    assert all(line["end"] > line["start"] for line in fixed)


def test_interjection_line_marking() -> None:
    assert L.is_interjection_line("Oh~")
    assert L.is_interjection_line("Woah oh oh")
    assert L.is_interjection_line("La la la, yeah!")
    assert not L.is_interjection_line("Oh, I stormed the silence")
    assert not L.is_interjection_line("Hello world")
    assert not L.is_interjection_line("")


def test_compute_type_defaults_per_device() -> None:
    import os

    saved = os.environ.pop("AITSM_WHISPER_COMPUTE", None)
    try:
        assert L._resolve_compute_type("cpu") == "int8"
        assert L._resolve_compute_type("cuda") == "float16"
        os.environ["AITSM_WHISPER_COMPUTE"] = "int8_float16"
        assert L._resolve_compute_type("cuda") == "int8_float16"
    finally:
        os.environ.pop("AITSM_WHISPER_COMPUTE", None)
        if saved is not None:
            os.environ["AITSM_WHISPER_COMPUTE"] = saved


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
            except Exception:  # noqa: BLE001 - 简易测试运行器
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
            else:
                print(f"PASS {name}")
    print(f"\n{'OK' if failures == 0 else f'{failures} FAILED'}")
    raise SystemExit(1 if failures else 0)


# ---------------------------------------------------------------------------
# 中文歌词清洗（实机跑《遥远的你》抓到的三类缺陷）
# ---------------------------------------------------------------------------
def test_zh_credit_lines_are_dropped_but_real_lyrics_survive() -> None:
    """幻觉署名行（「詞曲李宗盛」）要丢，但正常歌词不能被误杀。"""
    assert L.is_zh_credit_line("詞曲李宗盛")
    assert L.is_zh_credit_line("作曲：王明明")
    assert L.is_zh_credit_line("和声编写 : 廖国钺")
    assert L.is_zh_credit_line("制作公司：Hikoon Music")
    # 正片歌词即便含「作曲」这类字眼也不能丢
    assert not L.is_zh_credit_line("七月的风八月的雨")
    assert not L.is_zh_credit_line("我曾经和制作人聊过这件事")
    assert not L.is_zh_credit_line("")


def test_merge_tiny_lines_absorbs_single_char_fragments() -> None:
    """whisper 在换气处会切出单字行（去/息/底），要并回相邻行。"""
    lines = [
        {"start": 0.0, "end": 2.0, "text": "七月的风"},
        {"start": 2.0, "end": 4.0, "text": "八月的雨"},
        {"start": 4.0, "end": 5.0, "text": "去"},
        {"start": 5.0, "end": 9.0, "text": "卑微的我喜欢遥远的你"},
    ]
    merged = L.merge_tiny_lines(lines, language="zh")
    assert all(len(line["text"]) >= 3 for line in merged), merged
    assert merged[0]["start"] == 0.0
    assert merged[-1]["end"] == 9.0
    # 碎行并入前一行的尾部（这里是「八月的雨」+「去」）
    assert any(text.endswith("去") for text in (line["text"] for line in merged)), merged
    # 首行是碎行时向后合并
    head = L.merge_tiny_lines(
        [{"start": 0.0, "end": 1.0, "text": "去"}, {"start": 1.0, "end": 4.0, "text": "卑微的我喜欢你"}],
        language="zh",
    )
    assert len(head) == 1 and head[0]["start"] == 0.0


def test_merge_tiny_lines_latin_uses_word_threshold() -> None:
    """拉丁语言按「词」计数：单字行（Oh）并掉，两词以上的行保留。"""
    lines = [
        {"start": 0.0, "end": 1.0, "text": "Oh"},
        {"start": 1.0, "end": 3.0, "text": "I stormed the silence"},
        {"start": 3.0, "end": 5.0, "text": "we drive through"},
    ]
    merged = L.merge_tiny_lines(lines, language="en")
    assert len(merged) == 2, merged
    assert merged[0]["text"] == "OhI stormed the silence"
    # 中文模式阈值是 3 字，同样的输入行为不同
    assert len(L.merge_tiny_lines(lines, language="zh")) == 2


def test_to_simplified_lines_converts_traditional_and_keeps_content() -> None:
    """整段繁体要统一成简体，字符不能丢。"""
    lines = [
        {"start": 0.0, "end": 2.0, "text": "眼角的淚光"},
        {"start": 2.0, "end": 4.0, "text": "無邊無際的遊蕩"},
    ]
    converted, did = L.to_simplified_lines(lines, language="zh")
    import importlib.util

    if importlib.util.find_spec("zhconv") is not None:  # 装了 zhconv 才断言具体字形
        assert did is True
        assert converted[0]["text"] == "眼角的泪光"
        assert converted[1]["text"] == "无边无际的游荡"
    assert len(converted) == len(lines)


def test_to_simplified_lines_is_noop_for_non_chinese() -> None:
    english = [{"start": 0.0, "end": 1.0, "text": "we drive through the quiet land"}]
    assert L.to_simplified_lines(english, language="en")[1] is False
    japanese = [{"start": 0.0, "end": 1.0, "text": "君の涙が"}]
    assert L.to_simplified_lines(japanese, language="ja")[1] is False
    # 未知语言但文本里没有中文，也不该动
    assert L.to_simplified_lines([{"start": 0.0, "end": 1.0, "text": "abc"}], language=None)[1] is False


def test_drop_silent_lines_removes_silent_intro_hallucination(tmp_path: Path) -> None:
    """静音段编造的署名行要丢掉，正常唱歌的行要留下。"""
    import numpy as np

    # 0-4s 静音，4-8s「唱歌」
    sr = 8000
    quiet = np.zeros(sr * 4, dtype="float32")
    loud = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr * 4) / sr)).astype("float32")
    audio = np.concatenate([quiet, loud])
    path = tmp_path / "vocals.wav"
    import soundfile as sf

    sf.write(path, audio, sr)
    lines = [
        {"start": 0.5, "end": 3.5, "text": "詞曲李宗盛"},
        {"start": 4.2, "end": 7.5, "text": "七月的风八月的雨"},
    ]
    kept = L.drop_silent_lines(lines, path)
    assert [line["text"] for line in kept] == ["七月的风八月的雨"]


def test_drop_silent_lines_without_audio_keeps_everything() -> None:
    lines = [{"start": 0.0, "end": 1.0, "text": "x"}]
    assert L.drop_silent_lines(lines, None) == lines
    assert L.drop_silent_lines(lines, "no/such/file.wav") == lines
