"""歌词 ASR（F9）：faster-whisper 转写 + 纯算法清洗/断句/去重。

分层设计（便于离线单测、便于降级）：
- 纯算法层：文本清洗、噪声行过滤、按停顿/标点/长度断句、去重、音节切分、词级时间戳规整。
  这一层不依赖 numpy / whisper / 网络，可独立测试。
- 模型层：优先 faster-whisper（CTranslate2，py312/Windows 有 wheel，支持 VAD 与词级时间戳），
  退回 openai-whisper；都不可用时返回空列表并记录 warning，绝不编造歌词。

公共契约::

    transcribe_lyrics(vocals_wav, *, duration=None, language="auto", progress=None)
        -> [{"start": 1.2, "end": 4.5, "text": "…",
             "words": [{"start": .., "end": .., "text": ".."}], "confidence": 0.8}]

设计参考 AI-ChordCraft/src/lyric_asr.py 的 coerce / dedupe / offset 思路：
保留“按时间排序、空文本丢弃、精确重复丢弃”的纪律，但 ASR 引擎换成本地 whisper，
因此不需要 LLM 文本解析和窗口 offset 猜测，改为直接用 whisper 的绝对时间戳。

模型缓存由 ``aitsm/__init__.py`` 设置的 ``HF_HOME``（配合 ``HF_HUB_DISABLE_XET=1``）
决定，因此本模块必须作为 ``aitsm`` 包的一部分被导入，不要裸跑本文件。
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

LOGGER = logging.getLogger(__name__)

__all__ = [
    "transcribe_lyrics",
    "clean_lyric_text",
    "is_noise_line",
    "is_interjection_line",
    "is_prompt_echo",
    "is_meta_lyric_hallucination",
    "merge_split_words",
    "words_to_text",
    "decode_audio_16k",
    "split_syllables",
    "line_syllables",
    "build_words",
    "normalize_segments",
    "group_segments_into_lines",
    "dedupe_lines",
    "is_repetition_hallucination",
    "drop_repetition_hallucinations",
    "resolve_line_overlaps",
    "split_long_lines",
    "strip_repeated_edge_interjections",
    "faster_whisper_segments_to_raw",
    "LyricsEngineUnavailable",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 默认 medium：实测同一首英文歌（229s，vocals stem，CPU int8）
#   small  CER 0.655  语言置信度 0.25  20 行  149s
#   medium CER 0.498  语言置信度 0.56  26 行  348s
# 准确率提升显著，代价是推理时间约 2.3 倍；需要更快时设 AITSM_WHISPER_MODEL=small。
DEFAULT_WHISPER_MODEL = "medium"
_WHISPER_SAMPLE_RATE = 16000

# faster-whisper 模型单例缓存：(model, device, compute_type) -> WhisperModel
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}

# 最近一次 ASR 的引擎信息。transcribe_lyrics() 的返回值是纯歌词行列表（渲染层契约），
# 型号/设备必须靠这里带出去写进 manifest —— 否则同一首歌用 small 还是 medium
# 跑出来的谱面差异（CER 0.64 vs 0.50）在产物里无从解释。
_LAST_RUN_INFO: dict[str, Any] = {}


def last_run_info() -> dict[str, Any]:
    """返回最近一次 transcribe_lyrics() 的引擎信息（engine / model / device）。"""
    return dict(_LAST_RUN_INFO)

# 纯语气词（"Oh~" / "Whoa~" / "Oh oh oh"）：它们是真人声内容，不能当噪声删掉，
# 但要在输出里标记，方便渲染层用弱样式展示、也避免它们参与歌词准确率统计。
_INTERJECTION_TOKENS = frozenset(
    {"oh", "ooh", "ah", "aah", "eh", "hey", "whoa", "woah", "woo", "la", "na", "da", "mm", "hmm", "uh", "ya", "yeah", "huh"}
)

# 整行被括号包住时视为标注（[Music]、[间奏]、（合）、【副歌】…）
_ANNOTATION_ONLY_RE = re.compile(
    r"^\s*[\[\(（【〔<《][^\]\)）】〕>》]{0,40}[\]\)）】〕>》]\s*$"
)
# 行内出现的乐器/音效标注与音乐符号
_ANNOTATION_INLINE_RE = re.compile(r"[\[\(（【〔<《][^\]\)）】〕>》]{0,40}[\]\)）】〕>》]")
_MUSIC_SYMBOL_RE = re.compile(r"[♪♫♬♩♭♯©®™]")
# 制作人员/版权行（必须带冒号，避免误杀“音乐响起”这类真实歌词）
_CREDIT_RE = re.compile(
    r"^\s*[（(\[]?\s*"
    r"(作词|作曲|编曲|制作人|监制|出品|录音|混音|母带|发行|词|曲|OP|SP|"
    r"演唱|和声|吉他|贝斯|鼓|键盘|弦乐|LRC|翻译|校对|策划|统筹|混音师|录音师)"
    r"\s*[:：]",
    re.IGNORECASE,
)
# 字幕组/平台水印类噪声
_SPAM_RE = re.compile(
    r"(字幕由|字幕提供|自动生成字幕|Amara\.org|网易云音乐|QQ音乐|酷狗音乐|"
    r"未经许可|版权所有|follow the subtitles|subtitles by)",
    re.IGNORECASE,
)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_LATIN_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
_PUNCT_ONLY_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)
_WS_RE = re.compile(r"\s+")

_TERMINAL_PUNCT = "。！？?!；;…"
_CLAUSE_PUNCT = "，,、：:—–~～"

_EN_VOWELS = "aeiouy"


class LyricsEngineUnavailable(RuntimeError):
    """whisper 不可用（未安装 / 权重下载失败 / 推理异常）。"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    """把任意值转成有限 float；非法值返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _detect_language(text: str) -> str:
    """按字符集合猜语言：zh / ja / en。"""
    if _KANA_RE.search(text):
        return "ja"
    if _CJK_RE.search(text):
        return "zh"
    return "en"


def _join_text(parts: Sequence[str], language: str | None = None) -> str:
    """按语言拼接文本：中日文直接相连，英文用空格。"""
    pieces = [str(part).strip() for part in parts if str(part).strip()]
    if not pieces:
        return ""
    lang = (language or _detect_language("".join(pieces))).lower()
    if lang.startswith("zh") or lang.startswith("ja"):
        return "".join(pieces)
    return " ".join(pieces)


def _emit_progress(progress: Callable[..., Any] | None, fraction: float, message: str) -> None:
    """以宽松协议上报进度：优先 dict，其次 (fraction, message)，任何异常都吞掉。"""
    if not callable(progress):
        return
    payload = {
        "stage": "lyrics",
        "percent": round(_clamp(float(fraction), 0.0, 1.0) * 100.0, 1),
        "message": message,
    }
    try:
        progress(payload)
        return
    except TypeError:
        pass
    except Exception:  # pragma: no cover - 回调本身不应影响主流程
        LOGGER.debug("歌词进度回调异常", exc_info=True)
        return
    try:
        progress(_clamp(float(fraction), 0.0, 1.0), message)
    except Exception:  # pragma: no cover
        LOGGER.debug("歌词进度回调异常", exc_info=True)


# ---------------------------------------------------------------------------
# 文本清洗
# ---------------------------------------------------------------------------


def clean_lyric_text(text: Any) -> str:
    """清洗一行 ASR 文本：去标注/音乐符号、压缩空白、去掉中文之间的空格。"""
    if text is None:
        return ""
    raw = str(text)
    raw = _MUSIC_SYMBOL_RE.sub(" ", raw)
    raw = _ANNOTATION_INLINE_RE.sub(" ", raw)
    raw = raw.replace("\u3000", " ").replace("\xa0", " ")
    raw = _WS_RE.sub(" ", raw).strip()
    # 中文（含日文假名）之间的空格是 ASR 切词残留，去掉；英文单词间空格保留
    raw = re.sub(r"(?<=[\u3400-\u9fff\u3040-\u30ff])\s+(?=[\u3400-\u9fff\u3040-\u30ff])", "", raw)
    raw = raw.strip(" -–—·•|/\\_=+*#")
    return raw.strip()


def is_noise_line(text: Any) -> bool:
    """判断一行是否是不该进歌词的噪声（标注行、制作人员、字幕水印、纯符号）。"""
    cleaned = clean_lyric_text(text)
    if not cleaned:
        return True
    if _PUNCT_ONLY_RE.match(cleaned):
        return True
    if _ANNOTATION_ONLY_RE.match(str(text).strip()):
        return True
    if _SPAM_RE.search(cleaned):
        return True
    if _CREDIT_RE.match(cleaned):
        return True
    # 没有任何可唱字符（中日文/拉丁字母）
    if not (_CJK_RE.search(cleaned) or _LATIN_RE.search(cleaned) or _KANA_RE.search(cleaned)):
        return True
    return False


def is_interjection_line(text: Any) -> bool:
    """判断是否整行都是语气词（"Oh~" / "Whoa~" / "Oh oh oh"）。

    唱歌里的语气词是真实人声内容，不能当噪声删除；这里只做标记，
    让渲染层用弱样式展示、让评估脚本可以单独统计。
    """
    cleaned = clean_lyric_text(text)
    if not cleaned:
        return False
    tokens = re.findall(r"[a-zA-Z]+", cleaned)
    if not tokens:
        return False
    return all(token.lower() in _INTERJECTION_TOKENS for token in tokens)


# ---------------------------------------------------------------------------
# 音节切分
# ---------------------------------------------------------------------------


def _english_syllables(word: str) -> list[str]:
    """英文单词按元音组粗切音节（启发式，仅供逐字对齐，不追求语言学正确）。"""
    core = re.sub(r"[^A-Za-z0-9']", "", word)
    if len(core) <= 3:
        return [core] if core else ([word] if word.strip() else [])
    lower = core.lower()

    groups: list[tuple[int, int]] = []
    index = 0
    while index < len(lower):
        if lower[index] in _EN_VOWELS:
            end = index
            while end + 1 < len(lower) and lower[end + 1] in _EN_VOWELS:
                end += 1
            groups.append((index, end))
            index = end + 1
        else:
            index += 1

    # 结尾不发音的 e（love / make），但 -le 结尾保留（little）
    if len(groups) > 1 and lower.endswith("e") and not lower.endswith("le"):
        groups = groups[:-1]
    if len(groups) <= 1:
        return [core]

    cuts: list[int] = [0]
    for (_, group_end), (next_start, _) in zip(groups, groups[1:]):
        between_start = group_end + 1
        consonants = next_start - between_start
        if consonants <= 1:
            cut = next_start
        else:
            cut = between_start + 1
        cuts.append(cut)
    cuts.append(len(core))

    parts: list[str] = []
    for left, right in zip(cuts, cuts[1:]):
        piece = core[left:right]
        if piece:
            parts.append(piece)
    return parts if len(parts) > 1 else [core]


def split_syllables(text: Any, language: str | None = None) -> list[str]:
    """把一段歌词切成“对齐单元”序列。

    - 中文/日文：每个汉字/假名一个音节，连续拉丁字母/数字聚成一个单元；
    - 英文：按空格分词后再按元音组切音节；
    - 标点不参与对齐。
    """
    cleaned = clean_lyric_text(text)
    if not cleaned:
        return []
    lang = (language or _detect_language(cleaned)).lower()

    if lang.startswith("zh") or lang.startswith("ja"):
        tokens: list[str] = []
        buffer = ""
        for char in cleaned:
            if _CJK_RE.match(char) or _KANA_RE.match(char):
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                tokens.append(char)
            elif _LATIN_ALNUM_RE.match(char):
                buffer += char
            else:
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
        if buffer:
            tokens.append(buffer)
        return tokens

    syllables: list[str] = []
    for word in re.split(r"\s+", cleaned):
        word = word.strip("'\"")
        if not word:
            continue
        if not _LATIN_ALNUM_RE.search(word):
            continue
        if re.search(r"[\u3400-\u9fff\u3040-\u30ff]", word):
            syllables.extend(split_syllables(word, "zh"))
        else:
            syllables.extend(_english_syllables(word))
    return syllables


# ---------------------------------------------------------------------------
# 词级时间戳规整
# ---------------------------------------------------------------------------


def _sanitize_words(
    words: Iterable[dict], line_start: float | None, line_end: float | None
) -> list[dict]:
    """保证 words 时间单调、合法、落在行区间内（宽松夹取，不重排语义）。"""
    cleaned: list[dict] = []
    for word in words:
        text = str(word.get("text") or "").strip()
        if not text:
            continue
        start = _as_float(word.get("start"))
        end = _as_float(word.get("end"))
        if start is None:
            start = cleaned[-1]["end"] if cleaned else line_start
        if end is None:
            end = start + 0.12 if start is not None else None
        if start is None or end is None:
            continue
        if line_start is not None and start < line_start:
            start = line_start
        if line_end is not None and end > line_end:
            end = line_end
        if end <= start:
            end = start + 0.02
        if cleaned and start < cleaned[-1]["end"]:
            start = cleaned[-1]["end"]
            if end <= start:
                end = start + 0.02
        entry = {"start": round(start, 3), "end": round(end, 3), "text": text}
        if "space_before" in word:
            entry["space_before"] = bool(word.get("space_before"))
        cleaned.append(entry)
    return cleaned


def _fill_word_times(
    entries: list[dict], seg_start: float, seg_end: float
) -> list[dict]:
    """把缺时间戳的音节按前后已知时间线性插值补齐。

    entries 每项: {"text": str, "start": float|None, "end": float|None}
    """
    if not entries:
        return []
    count = len(entries)
    # 已知时间点（用 start 作为锚）
    known: list[tuple[int, float]] = [
        (index, entry["start"]) for index, entry in enumerate(entries) if entry["start"] is not None
    ]
    if not known:
        step = max((seg_end - seg_start) / max(count, 1), 0.02)
        for index, entry in enumerate(entries):
            entry["start"] = seg_start + index * step
            entry["end"] = entry["start"] + step
    else:
        first_index, first_start = known[0]
        step = max((first_start - seg_start) / max(first_index, 1), 0.02) if first_index else 0.0
        for index in range(first_index):
            entries[index]["start"] = seg_start + index * step
            entries[index]["end"] = entries[index]["start"] + step
        for (left_index, left_start), (right_index, right_start) in zip(known, known[1:]):
            span = right_index - left_index
            gap = max(right_start - left_start, 0.02)
            for offset in range(1, span):
                entries[left_index + offset]["start"] = left_start + gap * offset / span
        last_index, last_start = known[-1]
        step = max((seg_end - last_start) / max(count - last_index, 1), 0.02)
        for offset in range(0, count - last_index):
            index = last_index + offset
            if entries[index]["start"] is None:
                entries[index]["start"] = last_start + step * offset
    # end 缺失则由下一个 start 或段尾补齐
    for index, entry in enumerate(entries):
        if entry.get("end") is None or entry["end"] <= entry["start"]:
            if index + 1 < count and entries[index + 1].get("start") is not None:
                entry["end"] = entries[index + 1]["start"]
            else:
                entry["end"] = seg_end
            if entry["end"] <= entry["start"]:
                entry["end"] = entry["start"] + 0.02
    return entries


def build_words(text: Any, start: float, end: float, language: str | None = None) -> list[dict]:
    """在没有词级时间戳时，把文本按音节均分到 [start, end]。"""
    syllables = split_syllables(text, language)
    if not syllables:
        return []
    start = float(start)
    end = max(float(end), start + 0.02)
    step = (end - start) / len(syllables)
    return [
        {
            "start": round(start + index * step, 3),
            "end": round(start + (index + 1) * step, 3),
            "text": syllable,
            "space_before": index == 0,
        }
        for index, syllable in enumerate(syllables)
    ]


def words_to_text(words: Sequence[dict], language: str | None = None) -> str:
    """把音节级 words 拼回文本（同一个词的音节之间不插空格）。

    英文按音节切开后（"de"+"ploy"），如果一律用空格连接会得到 "de ploy"，
    看起来像模型把词拆坏了。``space_before`` 标记音节是否是词首，据此决定空格。
    """
    pieces: list[str] = []
    for word in words or []:
        if not isinstance(word, dict):
            continue
        text = str(word.get("text") or "")
        if not text:
            continue
        if pieces and not word.get("space_before", True):
            pieces[-1] += text
        else:
            pieces.append(text)
    return _join_text(pieces, language)


_LATIN_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'’\-]*$")


def merge_split_words(raw_words: Sequence[dict]) -> list[dict]:
    """合并被 whisper 拆开的子词（``" lyr"`` + ``"ics"`` → ``"lyrics"``）。

    whisper 用 **前导空格** 标记词边界：带空格的 token 是新词，不带空格的
    token 是上一个词被切出来的子词。中文/日文 token 本来就不带空格，所以
    只对纯拉丁字母 token 做合并，避免把整句中文并成一个词。
    """
    merged: list[dict] = []
    for raw in raw_words or []:
        if not isinstance(raw, dict):
            continue
        token = str(raw.get("word") or raw.get("text") or "")
        stripped = token.strip()
        if not stripped:
            continue
        previous = merged[-1] if merged else None
        is_subword = (
            previous is not None
            and not token[:1].isspace()
            and bool(_LATIN_TOKEN_RE.match(stripped))
            and bool(_LATIN_TOKEN_RE.match(str(previous.get("word") or "").strip()))
        )
        if is_subword and previous is not None:
            previous["word"] = str(previous.get("word") or "").rstrip() + stripped
            end = _as_float(raw.get("end"))
            if end is not None:
                previous["end"] = end
            continue
        merged.append(dict(raw, word=token))
    return merged


def _words_from_whisper(
    raw_words: Sequence[dict], seg_start: float, seg_end: float, language: str | None
) -> list[dict]:
    """把 whisper 的 word 序列转成音节级 words（中文按字切、英文按音节切）。"""
    entries: list[dict] = []
    for raw in merge_split_words(raw_words):
        if not isinstance(raw, dict):
            continue
        token = str(raw.get("word") or raw.get("text") or "")
        syllables = split_syllables(token, language)
        if not syllables:
            continue
        token_start = _as_float(raw.get("start"))
        token_end = _as_float(raw.get("end"))
        if token_start is None:
            token_end = None
        elif token_end is None or token_end <= token_start:
            token_end = token_start + 0.12
        step = ((token_end - token_start) / len(syllables)) if token_start is not None else None
        for index, syllable in enumerate(syllables):
            # space_before：词首音节才需要空格，其余音节属于同一个词
            space_before = index == 0
            if step is None:
                entries.append({"text": syllable, "start": None, "end": None, "space_before": space_before})
            else:
                entries.append(
                    {
                        "text": syllable,
                        "start": token_start + index * step,
                        "end": token_start + (index + 1) * step,
                        "space_before": space_before,
                    }
                )
    entries = _fill_word_times(entries, seg_start, seg_end)
    return _sanitize_words(entries, seg_start, seg_end)


# ---------------------------------------------------------------------------
# whisper segment -> 歌词行
# ---------------------------------------------------------------------------


def _segment_confidence(raw: dict) -> float:
    """用 avg_logprob / no_speech_prob 估一个 0..1 的句子置信度。"""
    avg_logprob = _as_float(raw.get("avg_logprob"))
    if avg_logprob is None:
        confidence = 0.55
    else:
        confidence = math.exp(min(0.0, avg_logprob))
    no_speech = _as_float(raw.get("no_speech_prob"))
    if no_speech is not None:
        # 高 no_speech_prob 说明该段可能只是伴奏/噪声，降低置信度但不直接丢弃
        confidence *= _clamp(1.0 - no_speech * 0.5, 0.35, 1.0)
    return round(_clamp(confidence, 0.0, 1.0), 3)


def normalize_segments(
    raw_segments: Sequence[dict],
    *,
    language: str | None = None,
    no_speech_threshold: float = 0.5,
    logprob_threshold: float = -1.0,
) -> list[dict]:
    """清洗 whisper 原始 segment：去噪、规整时间与词级时间戳。

    ``no_speech_prob > no_speech_threshold`` 且 ``avg_logprob < logprob_threshold``
    的段视为“没有歌声、模型在硬编”，整段丢弃（两个条件同时满足才丢，
    避免误伤轻声演唱的真歌词）。
    """
    segments: list[dict] = []
    for raw in raw_segments or []:
        if not isinstance(raw, dict):
            continue
        no_speech = _as_float(raw.get("no_speech_prob"))
        avg_logprob = _as_float(raw.get("avg_logprob"))
        if (
            no_speech is not None
            and avg_logprob is not None
            and no_speech > no_speech_threshold
            and avg_logprob < logprob_threshold
        ):
            LOGGER.debug(
                "丢弃低质量 segment [%.2f-%.2f]：no_speech=%.2f avg_logprob=%.2f",
                _as_float(raw.get("start")) or 0.0,
                _as_float(raw.get("end")) or 0.0,
                no_speech,
                avg_logprob,
            )
            continue
        text = clean_lyric_text(raw.get("text"))
        if is_noise_line(text):
            continue
        start = _as_float(raw.get("start"))
        end = _as_float(raw.get("end"))
        if start is None or end is None or end <= start:
            continue
        lang = language if language and not str(language).startswith("auto") else None
        words = _words_from_whisper(raw.get("words") or [], start, end, lang)
        if not words:
            words = build_words(text, start, end, lang)
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "words": words,
                "confidence": _segment_confidence(raw),
            }
        )
    segments.sort(key=lambda item: item["start"])
    return segments


def _ends_with_terminal(text: str) -> bool:
    stripped = text.rstrip(" \t\"'”’")
    return bool(stripped) and stripped[-1] in _TERMINAL_PUNCT


def _split_segment(segment: dict, *, max_len: int, pause: float, language: str | None) -> list[dict]:
    """把一个 whisper segment 按停顿/长度拆成若干歌词行；无需拆时原样返回。"""
    words = [word for word in segment.get("words") or [] if word.get("text")]
    text = segment["text"]
    if len(split_syllables(text, language)) <= max_len and not any(
        (later["start"] - earlier["end"]) > pause
        for earlier, later in zip(words, words[1:])
    ):
        return [segment]

    units: list[dict] = []
    bucket: list[dict] = []
    bucket_start = segment["start"]

    def flush() -> None:
        nonlocal bucket, bucket_start
        if not bucket:
            return
        unit_text = words_to_text(bucket, language)
        if unit_text and not is_noise_line(unit_text):
            units.append(
                {
                    "start": round(bucket_start, 3),
                    "end": round(bucket[-1]["end"], 3),
                    "text": unit_text,
                    "words": list(bucket),
                    "confidence": segment["confidence"],
                }
            )
        bucket = []

    for word in words:
        if bucket:
            gap = word["start"] - bucket[-1]["end"]
            if gap > pause or len(split_syllables(words_to_text(bucket, language), language)) >= max_len:
                flush()
                bucket_start = word["start"]
        if not bucket:
            bucket_start = word["start"]
        bucket.append(word)
        if _ends_with_terminal(word["text"]):
            flush()
    flush()
    return units or [segment]


def group_segments_into_lines(
    segments: Sequence[dict],
    *,
    max_gap: float = 0.5,
    max_len: int = 30,
    pause: float = 0.55,
    language: str | None = None,
) -> list[dict]:
    """断句 + 合并：按停顿/标点/长度切行，再把过短的碎片合并回行。"""
    if not segments:
        return []
    units: list[dict] = []
    for segment in segments:
        units.extend(_split_segment(segment, max_len=max_len, pause=pause, language=language))

    merged: list[dict] = []
    for unit in units:
        if merged:
            previous = merged[-1]
            gap = unit["start"] - previous["end"]
            combined_text = _join_text([previous["text"], unit["text"]], language)
            combined_len = len(split_syllables(combined_text, language))
            can_merge = (
                0.0 <= gap <= max_gap
                and not _ends_with_terminal(previous["text"])
                and combined_len <= max_len
            )
            if can_merge:
                previous["end"] = unit["end"]
                previous["text"] = combined_text
                previous["words"] = list(previous["words"]) + list(unit["words"])
                previous["confidence"] = round(
                    min(previous["confidence"], unit["confidence"]), 3
                )
                continue
        merged.append(dict(unit))
    return merged


def _normalized_key(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text).lower()


def dedupe_lines(lines: Sequence[dict], *, near_window: float = 0.5) -> list[dict]:
    """去重与抖动过滤。

    1. 完全重复（同时间区间 + 同文本）直接丢弃；
    2. 相邻两行文本相同（去掉标点后）且间隔很小时，视为 whisper 循环幻觉，丢弃后一行；
       间隔较大的重复（真正的副歌重复）必须保留。
    """
    result: list[dict] = []
    seen: set[tuple[int, int, str]] = set()
    for line in sorted(lines, key=lambda item: item["start"]):
        text = clean_lyric_text(line.get("text"))
        if not text or is_noise_line(text):
            continue
        key = (int(round(line["start"] * 10)), int(round(line["end"] * 10)), _normalized_key(text))
        if key in seen:
            continue
        if result:
            previous = result[-1]
            same_text = _normalized_key(previous["text"]) == _normalized_key(text)
            gap = line["start"] - previous["end"]
            # 完全相同且紧邻/重叠：whisper 在同一段音乐上重复输出
            if same_text and gap <= near_window:
                continue
            # 后一行完全被前一行包含且文本相同：重复输出
            if same_text and line["end"] <= previous["end"] + near_window:
                continue
        seen.add(key)
        result.append(
            {
                "start": round(float(line["start"]), 3),
                "end": round(float(line["end"]), 3),
                "text": text,
                "words": list(line.get("words") or []),
                "confidence": round(_as_float(line.get("confidence")) or 0.5, 3),
            }
        )
    return result


_INTERJECTION_WORD = r"(?:oh|ooh|ah|ahh|uh|la|na|hey|woah|whoa|yeah|yea|mm|mmm|hmm)"
_INTERJECTION_RUN_HEAD = re.compile(
    rf"^\s*(?:{_INTERJECTION_WORD}\b[\s,~.!?\-]*)+", re.IGNORECASE
)
_INTERJECTION_RUN_TAIL = re.compile(
    rf"(?:[\s,~.!?\-]*\b{_INTERJECTION_WORD})+\s*$", re.IGNORECASE
)


def strip_repeated_edge_interjections(text: Any, *, min_run: int = 3) -> str:
    """去掉行首/行尾成串的语气词（"oh oh oh …" 是 whisper 的典型幻觉边角）。

    中间的真歌词不动；少于 ``min_run`` 个的语气词保留（"Oh~" 是歌词的一部分）。
    """
    cleaned = clean_lyric_text(text)
    for pattern in (_INTERJECTION_RUN_HEAD, _INTERJECTION_RUN_TAIL):
        match = pattern.search(cleaned)
        while match is not None:
            run_text = match.group(0)
            count = len(re.findall(rf"\b{_INTERJECTION_WORD}\b", run_text, re.IGNORECASE))
            if count < min_run:
                break
            cleaned = (cleaned[: match.start()] + " " + cleaned[match.end() :]).strip()
            match = pattern.search(cleaned)
    return clean_lyric_text(cleaned)


def split_long_lines(
    lines: Sequence[dict],
    *,
    max_duration: float = 10.0,
    gap: float = 0.45,
    max_syllables: int = 30,
    language: str | None = None,
) -> list[dict]:
    """把过长/中间有换气停顿的歌词行拆成更短的显示行。

    whisper 经常把一个乐句连续输出成一整行（10 秒以上），弹唱谱一行太长不好读。
    这里只在“词间停顿够大”或“行太长”处拆，不动词内部时间。
    """
    result: list[dict] = []
    for line in lines:
        words = [word for word in line.get("words") or [] if isinstance(word, dict)]
        if len(words) < 2:
            result.append(line)
            continue
        line_start = float(line.get("start") or 0.0)
        line_end = float(line.get("end") or line_start)
        groups: list[list[dict]] = [[]]
        for word in words:
            if groups[-1]:
                previous = groups[-1][-1]
                previous_end = _as_float(previous.get("end"))
                word_start = _as_float(word.get("start"))
                gap_ok = (
                    previous_end is not None
                    and word_start is not None
                    and word_start - previous_end >= gap
                )
                group_start = _as_float(groups[-1][0].get("start"))
                long_ok = (
                    word_start is not None
                    and group_start is not None
                    and word_start - group_start >= max_duration
                )
                text_so_far = words_to_text(groups[-1], language)
                syllables_so_far = len(split_syllables(text_so_far, language))
                if gap_ok or long_ok or syllables_so_far >= max_syllables:
                    groups.append([])
            groups[-1].append(word)
        if len(groups) < 2:
            result.append(line)
            continue
        for group in groups:
            text = clean_lyric_text(words_to_text(group, language))
            if not text:
                continue
            start = _as_float(group[0].get("start"))
            end = _as_float(group[-1].get("end"))
            result.append(
                {
                    "start": round(start if start is not None else line_start, 3),
                    "end": round(end if end is not None else line_end, 3),
                    "text": text,
                    "words": list(group),
                    "confidence": round(
                        min(
                            [_as_float(word.get("confidence")) or _as_float(line.get("confidence")) or 0.5 for word in group]
                            + [_as_float(line.get("confidence")) or 0.5]
                        ),
                        3,
                    ),
                }
            )
    return result


# ---------------------------------------------------------------------------
# whisper 在音乐上的两类典型幻觉的清洗
# ---------------------------------------------------------------------------


def is_repetition_hallucination(text: Any, *, min_tokens: int = 4, dominance: float = 0.7) -> bool:
    """判断是否是 whisper 在音乐上的重复幻觉（"oh oh oh oh oh" 这类长串单字重复）。

    真歌词里的 "Oh~" 只有一两个 token，会保留；四个以上 token 且某一个 token
    占比超过 ``dominance`` 时几乎都是背景音乐/和声造成的循环输出。
    """
    tokens = [token.lower() for token in re.findall(r"[a-zA-Z]+", clean_lyric_text(text))]
    if len(tokens) < min_tokens:
        return False
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return max(counts.values()) / len(tokens) >= dominance


def drop_repetition_hallucinations(lines: Sequence[dict]) -> list[dict]:
    """丢掉 whisper 重复幻觉行（保留其他所有行）。"""
    return [
        line for line in lines if not is_repetition_hallucination(line.get("text"))
    ]


def resolve_line_overlaps(lines: Sequence[dict], *, min_duration: float = 0.15) -> list[dict]:
    """修复歌词行时间重叠：按时间排序后裁剪边界，太短的残行直接丢弃。"""
    result: list[dict] = []
    for line in sorted(lines, key=lambda item: (item["start"], item["end"])):
        current = dict(line)
        if result:
            previous = result[-1]
            if current["start"] < previous["end"] - 0.02:
                previous["end"] = round(max(previous["start"], current["start"]), 3)
                if previous["end"] - previous["start"] < min_duration:
                    result.pop()
                elif result:
                    previous = result[-1]
                current["start"] = round(max(current["start"], previous["end"] if result else 0.0), 3)
        if current["end"] - current["start"] >= min_duration:
            result.append(current)
    return result


# ---------------------------------------------------------------------------
# 供对齐层使用：行 -> 音节序列
# ---------------------------------------------------------------------------


def line_syllables(line: dict, *, language: str | None = None) -> list[dict]:
    """把一行歌词展开成带时间的音节列表（对齐层的输入）。

    返回 [{"text": "一", "start": 1.2, "end": 1.4}, ...]，时间单调递增。
    有词级时间戳就按词分配，没有就整行均分。
    """
    line_start = _as_float(line.get("start"))
    line_end = _as_float(line.get("end"))
    text = clean_lyric_text(line.get("text"))
    words = [word for word in line.get("words") or [] if isinstance(word, dict)]

    entries: list[dict] = []
    if words:
        for word in words:
            word_text = word.get("text")
            syllables = split_syllables(word_text, language)
            if not syllables:
                continue
            word_start = _as_float(word.get("start"))
            word_end = _as_float(word.get("end"))
            if word_start is None:
                entries.append({"text": syllables, "start": None, "end": None})
                continue
            if word_end is None or word_end <= word_start:
                word_end = word_start + 0.12 * len(syllables)
            step = (word_end - word_start) / len(syllables)
            for index, syllable in enumerate(syllables):
                entries.append(
                    {
                        "text": syllable,
                        "start": word_start + index * step,
                        "end": word_start + (index + 1) * step,
                    }
                )
    else:
        syllables = split_syllables(text, language)
        if not syllables:
            return []
        start = line_start if line_start is not None else 0.0
        end = line_end if line_end is not None and line_end > start else start + 0.3 * len(syllables)
        build = build_words(text, start, end, language)
        entries = [{"text": word["text"], "start": word["start"], "end": word["end"]} for word in build]

    if not entries:
        return []

    # 缺时间戳的按前后插值；已知的保持
    flattened: list[dict] = []
    for entry in entries:
        if isinstance(entry.get("text"), list):
            for syllable in entry["text"]:
                flattened.append({"text": syllable, "start": entry.get("start"), "end": entry.get("end")})
        else:
            flattened.append(entry)

    fallback_start = line_start if line_start is not None else 0.0
    fallback_end = (
        line_end if line_end is not None and line_end > fallback_start else fallback_start + 0.3 * len(flattened)
    )
    filled = _fill_word_times(flattened, fallback_start, fallback_end)
    return _sanitize_words(filled, line_start, line_end)


# ---------------------------------------------------------------------------
# whisper 后端
# ---------------------------------------------------------------------------


def _import_faster_whisper():
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # pragma: no cover - 依赖环境
        raise LyricsEngineUnavailable(f"faster-whisper 不可用：{exc}") from exc
    return WhisperModel


def _import_whisper():
    """旧引擎兜底（openai-whisper 在 py312/Windows 上通常装不了）。"""
    try:
        import whisper  # type: ignore
    except Exception as exc:  # pragma: no cover - 依赖环境
        raise LyricsEngineUnavailable(f"openai-whisper 不可用：{exc}") from exc
    return whisper


def _whisper_cache_dir() -> str:
    configured = os.environ.get("AITSM_WHISPER_DOWNLOAD_ROOT") or os.environ.get(
        "AITSM_WHISPER_CACHE"
    )
    if configured:
        return configured
    return str(Path.home() / ".cache" / "whisper")


def _whisper_model_name() -> str:
    return os.environ.get("AITSM_WHISPER_MODEL", DEFAULT_WHISPER_MODEL).strip() or DEFAULT_WHISPER_MODEL


def _resolve_device() -> str:
    configured = (os.environ.get("AITSM_WHISPER_DEVICE") or "").strip().lower()
    if configured in {"cpu", "cuda"}:
        return configured
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_compute_type(device: str) -> str:
    configured = (os.environ.get("AITSM_WHISPER_COMPUTE") or "").strip()
    if configured:
        return configured
    return "float16" if device == "cuda" else "int8"


def _initial_prompt(language: str | None) -> str | None:
    """音乐场景 initial_prompt：只用祈使句元语言，英文/自动检测一律不加。

    whisper 的 initial_prompt 是**当前解码的前文上下文**：prompt 越像歌词，
    模型越容易顺着它“接着唱”。实测英文 prompt
    "The following are the lyrics of a song, transcribed line by line with
    punctuation." 会被复读成整句假歌词（红线：不得编造歌词）。
    英文及其它语言返回 None（实测不加 prompt 反而更准）。
    """
    if not language:
        return None
    lang = str(language).lower()
    if lang.startswith("zh"):
        return "转写歌曲人声，只输出歌词行。"
    if lang.startswith("ja"):
        return "歌の人声を書き起こす。歌詞の行だけを出力する。"
    # 英文 / 其它语言：prompt 会被当成歌词复读，不加
    return None


def _content_words(text: Any) -> list[str]:
    """取出用于“prompt 复读”比对的内容词（字母/数字/汉字，忽略标点与空白）。"""
    return [item.lower() for item in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", str(text or ""))]


def is_prompt_echo(text: Any, prompt: str | None, *, min_run: int = 4) -> bool:
    """判断文本是否在复读 initial_prompt（连续 ``min_run`` 个内容词命中）。

    whisper 复读 prompt 时很少一字不差（会改标点、断行），所以按连续词串比对。
    """
    if not prompt:
        return False
    prompt_words = _content_words(prompt)
    text_words = _content_words(text)
    if len(prompt_words) < min_run or len(text_words) < min_run:
        return False
    grams = {
        tuple(prompt_words[index : index + min_run])
        for index in range(len(prompt_words) - min_run + 1)
    }
    return any(
        tuple(text_words[index : index + min_run]) in grams
        for index in range(len(text_words) - min_run + 1)
    )


_META_LYRIC_PATTERNS = (
    re.compile(r"the\s+lyr\s*ics?\s+of\s+(this|the|a)\s+song", re.IGNORECASE),
    re.compile(r"lyrics?\s+of\s+(this|the|a)\s+song", re.IGNORECASE),
    re.compile(r"transcribed\s+line\s+by\s+line", re.IGNORECASE),
    re.compile(r"following\s+are\s+the\s+lyrics", re.IGNORECASE),
)


def is_meta_lyric_hallucination(text: Any) -> bool:
    """判断是否“关于歌词的元描述”而非歌词本身（prompt 复读或模型自述）。

    这类句子是 whisper 在间奏/纯音乐段的经典幻觉，必须整行丢弃，不能进谱。
    """
    value = clean_lyric_text(text)
    if not value:
        return False
    return any(pattern.search(value) for pattern in _META_LYRIC_PATTERNS)


# 中文歌曲的「署名/元信息」行。实测 whisper 在片头静音段会凭空吐出「詞曲李宗盛」
# 这类署名（本曲真实作词作曲并不是他）——属于编造内容，必须整行丢弃。
_ZH_CREDIT_WORDS = (
    "作词", "作詞", "作曲", "詞曲", "词曲", "编曲", "編曲", "演唱", "原唱", "制作人", "製作人",
    "出品", "监制", "監製", "混音", "母带", "母帶", "和声", "和聲", "录音", "錄音", "发行", "發行",
    "策划", "策劃", "制作", "製作", "监棚", "監棚", "配唱",
)
_ZH_CREDIT_RE = re.compile(
    r"^[\s\[【(（]*(?:" + "|".join(_ZH_CREDIT_WORDS) + r")\s*[:：\-—、]?\s*\S*"
)


def is_zh_credit_line(text: Any) -> bool:
    """判断是否为中文署名/元信息行（不该出现在歌词里）。"""
    value = clean_lyric_text(text)
    if not value:
        return False
    return bool(_ZH_CREDIT_RE.match(value))



# ---------------------------------------------------------------------------
# 静音幻觉闸门 / 碎行合并 / 繁简统一
# ---------------------------------------------------------------------------

def _load_mono_audio(path: str) -> tuple[Any, int] | None:
    """读单声道 float32 波形（失败返回 None，闸门自动跳过）。"""
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data[:, 0], int(sr)
    except Exception:  # pragma: no cover - 读不了就放弃闸门，不影响主流程
        return None


def _segment_rms(audio: Any, sr: int, start: float, end: float) -> float:
    import numpy as np  # type: ignore

    lo = max(0, int(float(start or 0.0) * sr))
    hi = min(len(audio), int(float(end or 0.0) * sr))
    if hi <= lo:
        return 0.0
    chunk = audio[lo:hi].astype("float64")
    return float(np.sqrt(np.mean(chunk * chunk)))


def drop_silent_lines(
    lines: Sequence[dict],
    audio_path: str,
    *,
    floor_ratio: float = 0.08,
    abs_floor: float = 3e-4,
) -> list[dict]:
    """丢弃落在「近静音」区间的歌词行。

    whisper 在纯音乐/静音段会凭空生成整行文字（实测中文明明在片头静音处吐出
    「詞曲李宗盛」，而该段 vocals stem 的 RMS 只有整体均值的 0.15%）。
    这属于编造歌词，踩到项目红线，必须按能量证据丢掉。

    判据用**相对整体中位数**而不是绝对阈值：不同歌曲响度差很大，
    安静的真唱（低声段）要保住，静音里的幻觉要丢弃。
    """
    if not lines:
        return list(lines)
    loaded = _load_mono_audio(audio_path)
    if loaded is None:
        return list(lines)
    audio, sr = loaded
    if not len(audio):
        return list(lines)

    rms_values: list[float] = []
    for line in lines:
        rms_values.append(_segment_rms(audio, sr, line.get("start"), line.get("end")))
    ordered = sorted(rms_values)
    median = ordered[len(ordered) // 2] if ordered else 0.0
    threshold = max(abs_floor, floor_ratio * median)

    kept: list[dict] = []
    for line, rms in zip(lines, rms_values):
        if rms <= threshold:
            LOGGER.warning(
                "丢弃静音段歌词行 [%.2f-%.2f]（rms=%.5f ≤ %.5f）：%s",
                float(line.get("start") or 0.0),
                float(line.get("end") or 0.0),
                rms,
                threshold,
                (line.get("text") or "")[:40],
            )
            continue
        kept.append(line)
    return kept


def _text_units(text: Any) -> int:
    """粗略"字数"：中文按字，拉丁按词。"""
    value = clean_lyric_text(text)
    if not value:
        return 0
    cjk = sum(1 for ch in value if "\u4e00" <= ch <= "\u9fff")
    if cjk:
        return cjk
    return len([chunk for chunk in re.split(r"[\s\-]+", value) if chunk])


def merge_tiny_lines(
    lines: Sequence[dict],
    *,
    language: str | None = None,
    min_units: int | None = None,
    max_gap: float = 2.5,
) -> list[dict]:
    """把只有 1–2 个字/词的碎行并回相邻行。

    whisper 分句时会在换气处切出「去」「息」这种单字行，谱面上会变成
    孤零零一个字占一整行，可读性很差；时间上紧邻（gap ≤ max_gap）就合并。
    """
    items = [dict(line) for line in lines]
    if not items:
        return []
    if min_units is None:
        # 中文按字（3 字以下算碎），拉丁按词（2 词以下算碎）
        lang = (language or "").lower()
        min_units = 2 if lang.startswith(("en", "ja")) else 3
    merged: list[dict] = []
    for index, line in enumerate(items):
        if _text_units(line.get("text")) >= min_units:
            merged.append(line)
            continue
        prev = merged[-1] if merged else None
        nxt = items[index + 1] if index + 1 < len(items) else None
        if prev is not None and float(line.get("start") or 0.0) - float(prev.get("end") or 0.0) <= max_gap:
            prev["text"] = (str(prev.get("text") or "").rstrip() + str(line.get("text") or "").strip())
            prev["end"] = max(float(prev.get("end") or 0.0), float(line.get("end") or 0.0))
            continue
        if nxt is not None and float(nxt.get("start") or 0.0) - float(line.get("end") or 0.0) <= max_gap:
            nxt["text"] = (str(line.get("text") or "").strip() + str(nxt.get("text") or "").lstrip())
            nxt["start"] = min(float(nxt.get("start") or 0.0), float(line.get("start") or 0.0))
            continue
        merged.append(line)
    return merged


def to_simplified_lines(
    lines: Sequence[dict], *, language: str | None = None
) -> tuple[list[dict], bool]:
    """把歌词里的繁体字统一成简体（中文歌默认行为）。

    whisper 对普通话音频经常整段输出繁体（实测同一首歌里「淚光」与「阳光」混排），
    大陆用户读起来很别扭。用 zhconv 做逐行转换；装了才转，没装就原样返回
    （不因为可选依赖缺失改变输出）。需要保留繁体时设 ``AITSM_KEEP_TRADITIONAL=1``。

    日文歌不转：zhconv 会把汉字当成繁体中文一并改掉，日文假名也识别不准。
    """
    items = [dict(line) for line in lines]
    if not items:
        return items, False
    if os.environ.get("AITSM_KEEP_TRADITIONAL", "").strip().lower() in {"1", "true", "yes"}:
        return items, False
    lang = (language or "").lower()
    if lang.startswith(("ja", "ko")):
        return items, False
    joined = "".join(str(line.get("text") or "") for line in items)
    if not any("\u4e00" <= ch <= "\u9fff" for ch in joined):
        return items, False
    if any("\u3040" <= ch <= "\u30ff" for ch in joined):  # 有假名 → 当日语处理，不转
        return items, False
    try:
        import zhconv  # type: ignore
    except Exception:
        return items, False
    changed = False
    for line in items:
        text = str(line.get("text") or "")
        if not text:
            continue
        converted = zhconv.convert(text, "zh-cn")
        if converted != text:
            changed = True
            line["text"] = converted
    return items, changed


def faster_whisper_segments_to_raw(segments: Iterable[Any]) -> list[dict]:
    """把 faster-whisper 的 segment 对象转成与 openai-whisper 同形的 dict。

    只依赖属性读取（start/end/text/words/avg_logprob/no_speech_prob），
    因此可以用简单假对象做离线单测。
    """
    raw_segments: list[dict] = []
    for segment in segments or []:
        words: list[dict] = []
        for word in getattr(segment, "words", None) or []:
            words.append(
                {
                    "word": getattr(word, "word", "") or "",
                    "start": getattr(word, "start", None),
                    "end": getattr(word, "end", None),
                    "probability": getattr(word, "probability", None),
                }
            )
        raw_segments.append(
            {
                "start": getattr(segment, "start", None),
                "end": getattr(segment, "end", None),
                "text": getattr(segment, "text", "") or "",
                "words": words,
                "avg_logprob": getattr(segment, "avg_logprob", None),
                "no_speech_prob": getattr(segment, "no_speech_prob", None),
            }
        )
    return raw_segments


def _load_faster_whisper_model(device: str, compute_type: str):
    """加载并缓存模型（首次 CPU 加载约 10-20s，不能每首歌重载）。"""
    WhisperModel = _import_faster_whisper()
    model_name = _whisper_model_name()
    key = (model_name, device, compute_type)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached, model_name
    started = time.time()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    LOGGER.info("faster-whisper 模型加载完成：%s（%.1fs）", model_name, time.time() - started)
    _MODEL_CACHE[key] = model
    return model, model_name


def decode_audio_16k(path: str, *, duration: float | None = None) -> Any:
    """用 librosa 预解码为 16kHz 单声道 numpy 数组。

    faster-whisper 传文件路径时走 PyAV（本环境 PyAV 版本不兼容会崩
    ``open() got an unexpected keyword argument 'metadata_errors'``），
    因此统一先自己解码再传数组。
    """
    try:
        import librosa  # type: ignore
    except Exception as exc:  # pragma: no cover - 依赖环境
        raise LyricsEngineUnavailable(f"librosa 不可用，无法预解码音频：{exc}") from exc
    try:
        audio, _ = librosa.load(path, sr=_WHISPER_SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise LyricsEngineUnavailable(f"音频解码失败（{path}）：{exc}") from exc
    if duration is not None and duration > 0:
        max_samples = int(round(float(duration) * _WHISPER_SAMPLE_RATE))
        if 0 < max_samples < len(audio):
            audio = audio[:max_samples]
    return audio


def _transcribe_with_faster_whisper(
    path: str,
    *,
    language: str | None,
    device: str,
    compute_type: str,
    duration: float | None,
    progress: Callable[..., Any] | None,
) -> tuple[list[dict], str]:
    """加载 faster-whisper 并转写；返回 (raw_segments, 实际使用的 device)。"""
    _emit_progress(
        progress, 0.05, f"歌词 ASR：加载 faster-whisper（{device}/{compute_type}）"
    )
    model, model_name = _load_faster_whisper_model(device, compute_type)
    _emit_progress(progress, 0.15, "歌词 ASR：解码音频（librosa → numpy，绕过 PyAV）")
    audio = decode_audio_16k(path, duration=duration)
    _emit_progress(progress, 0.25, "歌词 ASR：识别中")
    started = time.time()
    segments_iter, info = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        beam_size=5,
        word_timestamps=True,
        # Silero VAD 是为“说话”训练的：唱歌的基频/音色会被判成非语音，
        # 实测开启后 60-105s 片段从 8 段变成 0 段，音乐场景必须关闭。
        vad_filter=False,
        initial_prompt=_initial_prompt(language),
        # 音乐场景必须关：否则背景音乐会让模型陷入重复同一句的幻觉循环
        condition_on_previous_text=False,
        # 温度回退链：单用 0.0 时模型容易卡在 "oh oh oh oh" 的循环里，
        # 允许回退到更高温度能跳出退化重复（实测 CER 0.68 → 0.49）。
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        # 唱歌的“语音概率”天然低于说话，默认 0.6 偏严
        no_speech_threshold=0.5,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )
    raw_segments = faster_whisper_segments_to_raw(segments_iter)
    LOGGER.info(
        "faster-whisper 转写完成：model=%s device=%s compute=%s language=%s elapsed=%.1fs segments=%d",
        model_name,
        device,
        compute_type,
        getattr(info, "language", None),
        time.time() - started,
        len(raw_segments),
    )
    return raw_segments, device


def _run_faster_whisper(
    path: str,
    *,
    language: str | None,
    duration: float | None,
    progress: Callable[..., Any] | None,
) -> dict:
    """faster-whisper 优先；CUDA 不可用（如缺 cuDNN 9）时自动退回 CPU/int8。"""
    device = _resolve_device()
    compute_type = _resolve_compute_type(device)
    try:
        raw_segments, used_device = _transcribe_with_faster_whisper(
            path,
            language=language,
            device=device,
            compute_type=compute_type,
            duration=duration,
            progress=progress,
        )
    except LyricsEngineUnavailable:
        raise
    except Exception as exc:
        if device != "cuda":
            raise
        LOGGER.warning("faster-whisper CUDA 推理失败，降级 CPU/int8：%s", exc)
        _emit_progress(progress, 0.1, "歌词 ASR：CUDA 不可用，降级 CPU")
        raw_segments, used_device = _transcribe_with_faster_whisper(
            path,
            language=language,
            device="cpu",
            compute_type="int8",
            duration=duration,
            progress=progress,
        )
    _emit_progress(progress, 0.9, "歌词 ASR：清洗与断句")
    # model / device 必须回传给 manifest：不同代模型（small vs medium）的 CER 差 0.10 以上，
    # 产物里不写型号就无法解释两份谱面为什么不一样（本项目踩过这个坑）。
    return {
        "segments": raw_segments,
        "engine": "faster-whisper",
        "model": _whisper_model_name(),
        "device": used_device,
    }


def _run_openai_whisper(
    path: str,
    *,
    language: str | None,
    duration: float | None,
    progress: Callable[..., Any] | None,
) -> dict:
    """旧引擎兜底（openai-whisper）。"""
    whisper = _import_whisper()
    model_name = _whisper_model_name()
    device = _resolve_device()
    _emit_progress(progress, 0.05, f"歌词 ASR：加载 whisper 模型 {model_name}（{device}）")

    started = time.time()
    model = whisper.load_model(model_name, device=device, download_root=_whisper_cache_dir())
    _emit_progress(progress, 0.15, "歌词 ASR：解码音频")
    audio = whisper.load_audio(path)
    if duration is not None and duration > 0:
        max_samples = int(round(float(duration) * _WHISPER_SAMPLE_RATE))
        if max_samples > 0:
            audio = audio[:max_samples]

    _emit_progress(progress, 0.25, "歌词 ASR：识别中（whisper 不提供中间进度）")
    result = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        word_timestamps=True,
        initial_prompt=_initial_prompt(language),
        condition_on_previous_text=False,
        temperature=0.0,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        verbose=False,
    )
    LOGGER.info(
        "whisper 转写完成：model=%s device=%s elapsed=%.1fs segments=%d",
        model_name,
        device,
        time.time() - started,
        len(result.get("segments") or []),
    )
    _emit_progress(progress, 0.9, "歌词 ASR：清洗与断句")
    result.setdefault("engine", "openai-whisper")
    result.setdefault("model", model_name)
    result.setdefault("device", device)
    return result


def _run_whisper(
    path: str,
    *,
    language: str | None,
    duration: float | None,
    progress: Callable[..., Any] | None,
) -> dict:
    """加载 ASR 引擎并转写，返回与 openai-whisper 同形的 result dict。

    优先 faster-whisper（py312/Windows 有预编译 wheel，且音乐场景支持 VAD），
    失败再退回 openai-whisper；两者都不可用时抛 LyricsEngineUnavailable。
    """
    try:
        return _run_faster_whisper(path, language=language, duration=duration, progress=progress)
    except LyricsEngineUnavailable as exc:
        LOGGER.warning("faster-whisper 不可用，尝试 openai-whisper：%s", exc)
        first_error = exc
    except Exception as exc:  # 推理失败也应兜底别的引擎，但不要吞掉真正原因
        LOGGER.warning("faster-whisper 推理失败，尝试 openai-whisper：%s", exc)
        first_error = exc
    try:
        return _run_openai_whisper(path, language=language, duration=duration, progress=progress)
    except LyricsEngineUnavailable as exc:
        raise LyricsEngineUnavailable(
            f"没有可用的歌词 ASR 引擎（faster-whisper: {first_error}；openai-whisper: {exc}）"
        ) from first_error


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


def transcribe_lyrics(
    vocals_wav: str,
    *,
    duration: float | None = None,
    language: str = "auto",
    progress: Callable[..., Any] | None = None,
) -> list[dict]:
    """从（人声）音频转写带时间戳的歌词行。

    参数:
        vocals_wav: 人声 stem 或整曲 WAV 路径（整曲时准确率下降，由调用方告警）。
        duration: 音频时长（秒），用于截断和片头片尾保护；None 时用文件全部内容。
        language: "auto" / "zh" / "en" / "ja"；默认 auto（whisper 自动检测，
            避免英文歌被中文 initial_prompt 带偏）。
        progress: 可选进度回调，接受 dict 或 (fraction, message)。

    返回:
        [{"start", "end", "text", "words": [{"start","end","text"}], "confidence"}]
        whisper 不可用、音频不可读或识别失败时返回 []（不编造歌词）。
    """
    path = str(vocals_wav or "").strip()
    if not path or not Path(path).is_file():
        LOGGER.warning("歌词 ASR 输入文件不存在：%s", path)
        _emit_progress(progress, 0.0, f"歌词 ASR：输入音频不存在 {path}")
        return []

    language = (language or "auto").strip() or "auto"
    language_lower = language.lower()
    whisper_language: str | None = (
        None if language_lower in {"auto", "detect", "none", "-"} else language_lower
    )
    if duration is None:
        duration = _probe_duration(path)

    try:
        result = _run_whisper(
            path, language=whisper_language, duration=duration, progress=progress
        )
    except LyricsEngineUnavailable as exc:
        LOGGER.warning("歌词 ASR 降级：%s", exc)
        _emit_progress(progress, 1.0, f"歌词 ASR 不可用：{exc}")
        _LAST_RUN_INFO.clear()
        _LAST_RUN_INFO.update({"engine": "none", "model": None, "device": None})
        return []
    except Exception as exc:  # 权重下载 / 推理 / 解码失败都不应炸掉整条流水线
        LOGGER.warning("歌词 ASR 失败，返回空歌词：%s", exc, exc_info=True)
        _emit_progress(progress, 1.0, f"歌词 ASR 失败：{exc}")
        return []

    # 记录本次实际使用的引擎/型号/设备（写进 manifest，供复现与质量归因）
    _LAST_RUN_INFO.clear()
    _LAST_RUN_INFO.update(
        {key: result.get(key) for key in ("engine", "model", "device") if result.get(key)}
    )
    _LAST_RUN_INFO.setdefault("engine", "whisper")

    try:
        segments = normalize_segments(result.get("segments") or [], language=whisper_language)
        lines = group_segments_into_lines(segments, language=whisper_language)
        lines = dedupe_lines(lines)
        lines = drop_repetition_hallucinations(lines)
        lines = resolve_line_overlaps(lines)
        # 一行 10 秒以上不好读：按换气停顿/长度拆成显示行
        lines = split_long_lines(lines, language=whisper_language)
        lines = resolve_line_overlaps(lines)
        # 去掉行首/行尾成串的语气词幻觉（"oh oh oh oh I stand…"）
        stripped_lines: list[dict] = []
        for line in lines:
            stripped = strip_repeated_edge_interjections(line.get("text"))
            if stripped:
                line["text"] = stripped
                stripped_lines.append(line)
        lines = stripped_lines
        # 红线：不得编造歌词。丢弃“复读 initial_prompt”、“关于歌词的元描述”行、
        # 以及中文署名行（whisper 会在片头静音段凭空吐出「詞曲某某」这类内容）。
        prompt = _initial_prompt(whisper_language)
        kept_lines: list[dict] = []
        for line in lines:
            text = line.get("text")
            if (
                is_prompt_echo(text, prompt)
                or is_meta_lyric_hallucination(text)
                or is_zh_credit_line(text)
            ):
                LOGGER.warning(
                    "丢弃 prompt 复读/元描述/署名行 [%.2f-%.2f]：%s",
                    line.get("start") or 0.0,
                    line.get("end") or 0.0,
                    line.get("text"),
                )
                continue
            kept_lines.append(line)
        lines = kept_lines
        # 红线：静音段里生成的行一律丢弃（用 vocals 能量做证据）
        lines = drop_silent_lines(lines, path)
        # 碎行合并 → 中文统一简体
        lines = merge_tiny_lines(lines, language=whisper_language)
        lines, simplified = to_simplified_lines(lines)
        _LAST_RUN_INFO["text_converted_to_simplified"] = bool(simplified)
        for line in lines:
            line["interjection"] = is_interjection_line(line.get("text"))
    except Exception as exc:  # pragma: no cover - 清洗层异常不应炸掉流水线
        LOGGER.warning("歌词清洗失败：%s", exc, exc_info=True)
        return []

    _emit_progress(progress, 1.0, f"歌词 ASR 完成：{len(lines)} 行")
    return lines


def _probe_duration(path: str) -> float | None:
    """尽力探测音频时长；失败返回 None（交给 whisper 自己解码）。"""
    try:
        import soundfile  # type: ignore

        info = soundfile.info(path)
        if info and info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        import librosa  # type: ignore

        return float(librosa.get_duration(path=path))
    except Exception:
        return None
