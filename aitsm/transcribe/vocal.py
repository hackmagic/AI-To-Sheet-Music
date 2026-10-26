"""人声联合转录层（v2 第 1A 层）：词、音高、时值、词-音符对齐一次产出。

为什么需要这一层
----------------
v1 的人声信息由两条独立链路拼出来：whisper 只出「歌词行 + 时间戳」，librosa/pyin
只出「音高 + 音段」，最后靠 :mod:`aitsm.pipeline.align` 把它们对齐。两条链路的切分
粒度天然不同（whisper 按语义断句、pyin 按能量/基频突变切片），对齐只能做「近似同时
刻」的粗匹配，一字多音、装饰音、拖腔处必然错位。

VocalParse 这类模型把四件事放在同一次解码里产出：token 串里既有音高 ``<P_68>``、
时值 ``<NOTE_4>``、速度 ``<BPM_89>``，也有歌词原文。切分粒度天然一致，不需要再做
跨链路对齐。

本模块的三块职责
----------------
1. **纯字符串解析**：:func:`parse_vocalparse_tokens` / :func:`note_value_to_quarters`
   把 token 串转成 :class:`VocalEvent`，无模型依赖，可在没有 torch 的环境离线单测；
2. **时间轴铺排**：:func:`vocal_events_to_notes` 把事件铺到绝对秒轴上，产出
   :class:`aitsm.contracts.MelodyNote`；
3. **模型适配与降级**：:func:`is_available` / :func:`transcribe_vocal`。**可用才用**，
   不可用或推理失败就显式降级为既有的 whisper(+pyin) 三段式。绝不联网下载权重，也
   绝不假装成功：``engine`` 记实际生效的实现，``requested`` 记架构首选
   ``VocalParse``，降级原因写进 ``warnings``。

时间与单位约定
--------------
* 对外时间单位一律是**秒**（见 :mod:`aitsm.contracts`）；时值 token 表示「几分音符」，
  两者的换算只发生在 :func:`vocal_events_to_notes` 内；
* token 串自身不带绝对时间：时间轴以 ``beatgrid.first_downbeat_sec`` 为锚点，按每个
  事件当时的 BPM 累加时值。没有 beatgrid 时锚点为 0、BPM 用 120，仍然返回结果——
  但调用方应当知道这只是「没有节拍证据」时的默认值，不是识别结论。

公共契约::

    parse_vocalparse_tokens(token_str) -> list[VocalEvent]
    note_value_to_quarters(value) -> float
    vocal_events_to_notes(events, beatgrid) -> list[MelodyNote]
    is_available() -> bool
    transcribe_vocal(vocals_wav, *, duration=None, language="auto", beatgrid=None,
                     device="auto", progress=None, whisper_model="medium") -> VocalResult

重依赖（torch / transformers / pipeline 各阶段）全部在函数内惰性导入，``import
aitsm.transcribe.vocal`` 本身只花几毫秒。
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..contracts import BeatGrid, LyricLine, LyricWord, MelodyNote, VocalResult
from . import qwen_backend

LOGGER = logging.getLogger(__name__)

__all__ = [
    "VocalEvent",
    "VocalParseError",
    "VocalParseIncompatibleError",
    "parse_vocalparse_tokens",
    "last_parse_orphans",
    "note_value_to_quarters",
    "vocal_events_to_notes",
    "is_available",
    "vocalparse_dir",
    "transcribe_vocal",
    "REQUESTED_ENGINE",
    "FALLBACK_ENGINE",
    "VOCALPARSE_DIR_ENV",
    "VOCALPARSE_VOCAB_SIZE",
]

#: 架构首选引擎（写进 ``VocalResult.requested`` / provenance）。
REQUESTED_ENGINE = "VocalParse"
#: 降级路径的实际引擎名（whisper 出歌词 + pyin 出音高，再对齐）。
FALLBACK_ENGINE = "whisper+pyin"
#: 本地权重目录的环境变量。本层**只读本地目录**，绝不联网拉权重。
VOCALPARSE_DIR_ENV = "AITSM_VOCALPARSE_DIR"
#: 复用的既有 whisper 型号环境变量（降级路径临时覆盖，用完恢复）。
WHISPER_MODEL_ENV = "AITSM_WHISPER_MODEL"

#: 没有 BPM 证据时的默认速度（与 contracts.BeatGrid 的默认值一致）。
DEFAULT_BPM = 120.0
#: 用户没写 NOTE 时按四分音符处理（缺时值比丢音更可救）。
DEFAULT_NOTE_VALUE = "4"
#: token 解码出的音符没有逐音概率可用：写 1.0 表示「确实由模型给出」，
#: 而不是「高置信度」——质量面板不应据此免检。
DEFAULT_EVENT_CONFIDENCE = 1.0
#: 合法的 BPM 上界：再高就不是人声转录能处理的速度了，当噪声丢弃。
_MAX_BPM = 400.0
#: 附点 = 1.5 × 基准时值。
DOTTED_RATIO = 1.5
#: 可接受的时值分母范围（1 = 全音符 … 64 = 六十四分音符）。
_MIN_NOTE_BASE = 1.0
_MAX_NOTE_BASE = 64.0
#: 换算结果的可接受区间（四分音符为单位）。
_MIN_QUARTERS = 1.0 / 16.0
_MAX_QUARTERS = 16.0

#: ``<P_68>`` / ``<NOTE_4.>`` / ``<BPM_89>``；大小写、下划线两侧空白、属性分隔都容错。
_TOKEN_RE = re.compile(
    r"<\s*(?P<kind>[A-Za-z]+)\s*(?:_\s*(?P<value>[^<>]*?))?\s*>",
    re.IGNORECASE,
)
#: 畸形标签（``<P_70`` 少了右尖括号，或 ``<FOO_bar>`` 这类未知标签）连同它吞掉的裸文本
#: 一起从歌词流里剔除：宁可少一个音节，也不要把标签当歌词唱出来。
_TEXT_NOISE_RE = re.compile(r"<[^<>]*>?")
#: 「像歌词」的字符：拉丁/数字、CJK、假名、谚文。纯标点（含 ``|`` 分隔符）不是歌词。
_WORDISH_RE = re.compile(
    r"[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]"
)

# ---- 真实模型输出里的结构性噪声（实测于 pymaster/VocalParse，不是文档推测） ----

#: CoT 模式下模型先输出纯歌词，再用 ``<|file_sep|>`` 分隔结构化序列。
#: 只解析分隔符**之后**的部分；没有分隔符时整串都算结构化序列。
_COT_SEPARATOR_RE = re.compile(r"<\|file_sep\|>", re.IGNORECASE)
#: 序列开头的语言/任务前缀，如 ``language Chinese<asr_text>``。
_SEQUENCE_PREFIX_RE = re.compile(
    r"^\s*(?:language\s*[A-Za-z]*\s*)?(?:<asr_text>)?\s*", re.IGNORECASE
)
#: 模型用来表示「这里没有歌词」的占位符与自我修正标记：不是歌词，必须剔除。
_NON_LYRIC_TOKENS = frozenset(
    {
        "sp",
        "sil",
        "rest",
        "cursorposition",
        "equal",
        "nolyric",
        "no-lyric",
    }
)
#: 包裹型噪声（``(Equal)`` / ``[SP]`` 之类）：整段丢弃。
_BRACKETED_NOISE_RE = re.compile(r"[\(\[（【]\s*(?:equal|sp|cursorposition|sil)\s*[\)\]）】]", re.IGNORECASE)
#: ``<P_0>`` 在 VocalParse 里表示休止/无音高（实测大量出现），不能当成 MIDI 0 输出。
_REST_PITCH = 0

#: VocalParse 的提示词：只要求 token 序列，不要解释（解释会混进歌词流）。
_VOCALPARSE_PROMPT = (
    "Transcribe the singing voice in this audio into VocalParse tokens. "
    "Output only the token sequence: <BPM_x>, then for every sung note "
    "<P_midi> lyrics text <NOTE_value>. Do not add explanations."
)


class VocalParseError(RuntimeError):
    """VocalParse 适配失败（权重/处理器/推理/解码任一环节）。

    只在本层内部使用：调用方（:func:`transcribe_vocal`）捕获它并显式降级，
    绝不把异常抛给上层流水线。
    """


# --------------------------------------------------------------------------- #
# 事件与 token 解析
# --------------------------------------------------------------------------- #


@dataclass
class VocalEvent:
    """VocalParse 解码出的一个事件：一个音高 + 一条歌词 + 一个时值。

    ``note_value`` 保存 token 原文（``"4"`` / ``"8"`` / ``"4."``），这样下游复算时
    不会因为提前浮点化而丢掉「附点」这一信息；换算见 :func:`note_value_to_quarters`。

    ``bpm`` 是该事件**当时**生效的速度（``<BPM_89>`` 之后的所有事件都带 89），
    在第一个 BPM token 之前解码出的事件为 ``None``（没有速度证据，不编造）。
    """

    lyric: str = ""
    midi_pitch: int = 60
    note_value: str = DEFAULT_NOTE_VALUE
    bpm: float | None = None

    def __post_init__(self) -> None:
        self.lyric = _clean_lyric_text(self.lyric)
        self.midi_pitch = max(0, min(127, _as_int(self.midi_pitch, 60)))
        self.note_value = _normalize_note_value_token(self.note_value)
        bpm = _as_float(self.bpm)
        self.bpm = bpm if bpm is not None and bpm > 0 else None


def parse_vocalparse_tokens(token_str: str) -> list[VocalEvent]:
    """把 VocalParse 的 token 串解析成事件列表（纯字符串处理，不依赖模型）。

    支持的 token（大小写、空白、下划线容错）::

        <P_68>      音高（MIDI 音号；超出 0..127 视为噪声丢弃）
        <NOTE_4>    时值（1/2/4/8/16 …，尾部 ``.`` 或 ``d`` 表示附点）
        <BPM_89>    速度（浮点可用，如 <BPM_89.5>；超出 (0, 400] 丢弃）

    歌词是 token 之间的自由文本，规则只有两条：

    * 文本优先挂到**当前事件**上（常见写法 ``<P_68>你<NOTE_4>``，以及
      ``<P_68><NOTE_4>你`` 都成立）；同一事件上连续的文本会拼接，因此
      ``<P_68>你<NOTE_4>好`` 得到 lyric=``"你好"``（多字符歌词是合法的）；
    * 还没有任何音高时就出现的文本先缓存，挂到**下一个**音高事件上
      （``你<P_68>`` 这种倒序写法）。

    缺失/畸形的输入一律按噪声处理而不是抛异常：无 ``>`` 的半个标签、未知标签、
    非数字音高、非法时值都不会污染歌词流。时值出现在所有音高之前时没有可挂的音符，
    直接丢弃。
    """
    text = str(token_str or "")

    # ---- 真实输出预处理（顺序重要，每步都有实测依据） ----
    # 1) CoT 模式：`<|file_sep|>` 之前是纯歌词、之后才是结构化序列，只取后者。
    #    没有分隔符时整串都当结构化序列（标准 interleaved 输出）。
    separator = _COT_SEPARATOR_RE.search(text)
    if separator is not None:
        text = text[separator.end():]
    # 2) 序列开头可能带 `language Chinese<asr_text>` 之类前缀。
    text = _SEQUENCE_PREFIX_RE.sub("", text, count=1)
    # 3) `(Equal)` / `[SP]` 这类包裹型噪声整段丢弃（否则会被当成歌词）。
    text = _BRACKETED_NOISE_RE.sub(" ", text)

    events: list[VocalEvent] = []
    current: VocalEvent | None = None
    pending = ""
    bpm: float | None = None
    cursor = 0
    #: 孤立的 ``<NOTE_x>``（前面没有 ``<P_y>``）。正常输出里不该出现；一旦出现，
    #: 说明模型把「休止」写成了纯时值，静默跳过会让后续音符整体前移。
    orphan_notes: list[str] = []

    for match in _TOKEN_RE.finditer(text):
        chunk = text[cursor : match.start()]
        cursor = match.end()

        cleaned = _clean_lyric_text(_TEXT_NOISE_RE.sub(" ", chunk))
        if cleaned and not _is_non_lyric_placeholder(cleaned):
            # 两种真实写法都要支持：
            #   a) `你 <P_68> <NOTE_4>`（VocalParse 实测输出：词在音高之前）
            #   b) `<P_68> <NOTE_4> 你`（文档示例：词在音高之后）
            # 判据：当前事件是否**已经有音高但还没有词**——若还没有词，说明这段
            # 文本属于它；否则文本属于下一个尚未出现的音高。
            if current is not None and not current.lyric:
                current.lyric = _join_lyric(current.lyric, cleaned)
            else:
                pending = _join_lyric(pending, cleaned)

        kind = (match.group("kind") or "").strip().upper()
        value = re.sub(r"\s+", "", match.group("value") or "")

        if kind == "BPM":
            bpm_value = _as_float(value)
            if bpm_value is not None and 0 < bpm_value <= _MAX_BPM:
                bpm = bpm_value
            continue

        if kind == "P":
            pitch = _parse_pitch(value)
            if pitch is None:
                continue
            # 真实输出（实测）：歌词在音高 token **之前**，如 `Let's <P_58> <NOTE_DOT_8>`。
            # 因此这一段文本要挂到**即将创建**的这个事件上，而不是上一个事件。
            current = VocalEvent(
                lyric=pending,
                midi_pitch=pitch,
                note_value="",
                bpm=bpm,
            )
            pending = ""
            events.append(current)
            continue

        if kind == "NOTE":
            if current is None:
                # 时值必须挂在某个音高上；孤立的 <NOTE_x> 没有音乐含义。
                # 但它也可能意味着「模型把休止写成了纯时值」——那样静默跳过会让
                # 后续音符整体前移。这里记一笔，便于上层用 stats 判断。
                orphan_notes.append(value)
                continue
            if not value or not _parse_note_value(value)[1]:
                continue
            current.note_value = value
            continue
        # 未知标签：当作噪声跳过

    # 尾部文本：挂到最后一个事件上（没有音高时无处可挂，只能丢弃）
    tail = _clean_lyric_text(_TEXT_NOISE_RE.sub(" ", text[cursor:]))
    if tail and not _is_non_lyric_placeholder(tail):
        pending = _join_lyric(pending, tail)
    if pending and current is not None:
        current.lyric = _join_lyric(current.lyric, pending)

    if orphan_notes:
        # 不静默：这些时值没挂上任何音高。调用方可用 stats 判断模型输出是否异常。
        LOGGER.warning(
            "VocalParse 输出里有 %d 个未挂到音高的 <NOTE_> token（%s）："
            "可能是模型把休止写成了纯时值，这些时值已被跳过。",
            len(orphan_notes),
            ", ".join(orphan_notes[:5]),
        )
        _LAST_PARSE_ORPHANS.clear()
        _LAST_PARSE_ORPHANS.extend(orphan_notes)
    else:
        _LAST_PARSE_ORPHANS.clear()

    return events


def _is_non_lyric_placeholder(text: str) -> bool:
    """``SP`` / ``CursorPosition`` / ``(Equal)`` 这类占位符不是歌词。

    实测 VocalParse 生成序列里会夹带这些自我修正标记；把它们当歌词会把整行
    歌词张冠李戴到某个音符上，比留空更糟。
    """
    candidate = re.sub(r"[\(\)\[\]（）【】\s]+", "", str(text or "")).strip().lower()
    return not candidate or candidate in _NON_LYRIC_TOKENS


def note_value_to_quarters(value: Any) -> float:
    """把一个「几分音符」的时值换算成四分音符数（可记谱值的浮点表示）。

    例子::

        note_value_to_quarters("1")  == 4.0    # 全音符
        note_value_to_quarters(4)    == 1.0    # 四分音符
        note_value_to_quarters("8")  == 0.5    # 八分音符
        note_value_to_quarters("4.") == 1.5    # 附点四分（也接受 "4d"）
        note_value_to_quarters("")   == 1.0    # 缺失时值 → 四分音符

    无法解析、非正数、超范围（分母 > 64）的值一律退回四分音符而不是抛异常：
    解析层不承担「拒绝整首歌」的权力，缺一个时值不该让整首谱子消失。
    """
    quarters, ok = _parse_note_value(value)
    return quarters


# --------------------------------------------------------------------------- #
# 事件 -> 绝对时间轴上的音符
# --------------------------------------------------------------------------- #


def vocal_events_to_notes(
    events: Sequence[VocalEvent] | None,
    beatgrid: BeatGrid | Mapping[str, Any] | None = None,
) -> list[MelodyNote]:
    """把事件序列铺到绝对秒轴上，返回 :class:`aitsm.contracts.MelodyNote` 列表。

    时间轴规则：

    * 锚点 = ``beatgrid.first_downbeat_sec``（无 beatgrid 时为 0）；
    * 每个事件的时长 = ``时值(四分音符) × 60 / BPM``，BPM 优先用事件自带的
      （``<BPM_89>`` 之后的事件都带它），否则用 beatgrid 的 BPM，再否则 120；
    * 事件首尾相接：下一个事件从上一个事件结束处开始（token 串本身不含绝对时间，
      也不能假装知道哪里该有休止）；
    * 歌词为空的音符 ``syllable_index=None``（它不该参与词-音符对齐计数）。
    """
    anchor, fallback_bpm = _time_axis(beatgrid)
    notes: list[MelodyNote] = []
    cursor = anchor
    emitted = 0

    for index, event in enumerate(_iter_events(events)):
        bpm = event.bpm if event.bpm else fallback_bpm
        seconds = note_value_to_quarters(event.note_value or DEFAULT_NOTE_VALUE) * 60.0 / bpm
        end = cursor + seconds
        # <P_0> 在 VocalParse 里是「休止/无音高」（实测大量出现），不是 MIDI 0。
        # 保留时间推进（休止占据时值），但**不产出音符**——绝不伪造一个音高。
        if int(event.midi_pitch) == _REST_PITCH:
            cursor = end
            continue
        notes.append(
            MelodyNote(
                start=round(cursor, 6),
                end=round(end, 6),
                midi=event.midi_pitch,
                confidence=DEFAULT_EVENT_CONFIDENCE,
                lyric=event.lyric,
                syllable_index=emitted if event.lyric else None,
                source="vocalparse",
            )
        )
        emitted += 1
        cursor = end

    return notes


# --------------------------------------------------------------------------- #
# 可用性（只读本地权重，绝不联网）
# --------------------------------------------------------------------------- #


def vocalparse_dir() -> Path | None:
    """返回可用的本地权重目录；未配置 / 不存在 / 空目录都返回 ``None``。"""
    raw = (os.environ.get(VOCALPARSE_DIR_ENV) or "").strip()
    if not raw:
        return None
    try:
        path = Path(raw).expanduser()
        if not path.is_dir():
            return None
        if not any(path.iterdir()):
            return None
    except OSError as exc:  # 权限不足 / 路径过长等：按不可用处理
        LOGGER.debug("VocalParse 权重目录不可读：%s", exc)
        return None
    return path


def is_available() -> bool:
    """VocalParse 是否**能**用：本地权重目录 + ``qwen_asr`` 后端路径都成立。

    为什么不是 ``transformers``：Qwen3-ASR 的建模代码不在标准 transformers 里，
    官方通过 ``qwen-asr`` 包分发。只装 transformers 时通用类会按
    151936/3584 建图，与 checkpoint 的 152101/2048 冲突（实测 315 个 key MISSING）。

    这里走 :func:`aitsm.transcribe.qwen_backend.probe`，它同时检查
    「包已安装」与「三个子模块路径还在」——后者能提前发现 ``qwen-asr`` 升级
    导致路径变化，而不是等到推理中途才抛 ImportError。绝不联网。
    """
    if vocalparse_dir() is None:
        return False
    return bool(qwen_backend.probe()["available"])


# --------------------------------------------------------------------------- #
# 对外主入口
# --------------------------------------------------------------------------- #


def transcribe_vocal(
    vocals_wav: str | Path | None,
    *,
    duration: float | None = None,
    language: str = "auto",
    beatgrid: BeatGrid | Mapping[str, Any] | None = None,
    device: str = "auto",
    progress: Callable[..., Any] | None = None,
    whisper_model: str = "medium",
) -> VocalResult:
    """人声联合转录：优先 VocalParse，不可用/失败则降级 whisper(+pyin)。

    参数
    ----
    vocals_wav:
        人声 stem（推荐）或整曲 WAV 路径。文件不存在时返回空结果 + 告警，不抛异常。
    duration:
        音频时长（秒），用于截断/片头片尾保护；``None`` 表示用文件全部内容。
    language:
        ``"auto"`` / ``"zh"`` / ``"en"`` / ``"ja"``；只有降级路径的 ASR 用得到。
    beatgrid:
        节拍骨架。VocalParse 路径用它做时间轴锚点，降级路径把它转成既有旋律模块
        认识的 ``beats`` dict。``None`` 时用 0 秒锚点 + 120 BPM 默认值。
    device:
        ``"auto"`` / ``"cpu"`` / ``"cuda"``；只影响 VocalParse 的模型放置。
    progress:
        可选进度回调，接受 dict（``{"stage", "percent", "message"}``）或
        ``(fraction, message)`` 两种协议，异常一律吞掉。
    whisper_model:
        降级路径要用的 whisper 型号（如 ``"medium"`` / ``"small"``）。既有
        :func:`aitsm.pipeline.lyrics.transcribe_lyrics` 只从环境变量读型号，因此这里
        在调用期间临时设置 ``AITSM_WHISPER_MODEL`` 并在 ``finally`` 里恢复——
        不改模块全局状态，也不影响并发调用之外的行为。

    返回
    ----
    :class:`aitsm.contracts.VocalResult`：``engine`` 为 ``"vocalparse"`` 或
    ``"whisper+pyin"``，``requested`` 恒为 ``"VocalParse"``；降级原因、被丢弃的
    旋律告警都写在 ``warnings`` 里，实际子引擎型号在 ``alignment`` 里。
    """
    started = time.monotonic()
    warnings: list[str] = []
    path = str(vocals_wav or "").strip()

    _emit_progress(progress, 0.0, "人声联合转录：检查输入")
    if not path or not Path(path).is_file():
        warnings.append(f"人声音频不存在：{path or '(空路径)'}；未做任何识别")
        _emit_progress(progress, 1.0, "人声联合转录：缺少输入，已跳过")
        return VocalResult(
            engine="none",
            requested=REQUESTED_ENGINE,
            warnings=warnings,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    if is_available():
        _emit_progress(progress, 0.05, "人声联合转录：VocalParse 推理")
        try:
            primary = _transcribe_with_vocalparse(
                path,
                duration=duration,
                language=language,
                beatgrid=beatgrid,
                device=device,
                progress=progress,
            )
        except Exception as exc:  # 适配层任何失败都不许炸掉流水线
            reason = f"{type(exc).__name__}: {exc}"
            warnings.append(f"VocalParse 推理失败（{reason}），已降级为 {FALLBACK_ENGINE}")
            LOGGER.warning("VocalParse 推理失败，降级：%s", reason)
        else:
            if primary.notes or primary.lyrics:
                primary.warnings = [*warnings, *primary.warnings]
                primary.elapsed_seconds = round(time.monotonic() - started, 3)
                _emit_progress(
                    progress,
                    1.0,
                    f"人声联合转录完成：VocalParse，{len(primary.notes)} 个音",
                )
                return primary
            # 能加载但什么都没解出来：多半是权重/提示词不匹配，交给降级路径兜底
            warnings.append(
                f"VocalParse 未解出任何音符或歌词，已降级尝试 {FALLBACK_ENGINE}"
            )
            LOGGER.warning("VocalParse 输出为空，降级")
    else:
        reason = _unavailable_reason()
        warnings.append(f"VocalParse 不可用（{reason}），已降级为 {FALLBACK_ENGINE}")
        LOGGER.info("VocalParse 不可用：%s", reason)

    return _transcribe_with_fallback(
        path,
        duration=duration,
        language=language,
        beatgrid=beatgrid,
        progress=progress,
        whisper_model=whisper_model,
        warnings=warnings,
        started=started,
    )


# --------------------------------------------------------------------------- #
# 首选路径：VocalParse（本地权重）
# --------------------------------------------------------------------------- #


#: VocalParse 权重里的键前缀 —— 该 checkpoint 由早期 Qwen3-ASR 实现保存，
#: 键形如 ``thinker.audio_tower.*`` / ``thinker.model.*``；
#: 当前 transformers 的 ``Qwen3ASRForConditionalGeneration`` 期望
#: ``model.audio_tower.*`` / ``model.language_model.*``。这里做显式重映射，
#: 而不是指望库自动兼容（实测库的 ``validate`` 会把它们全部报成 MISSING）。
_VOCALPARSE_KEY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("thinker.audio_tower.", "audio_tower."),
    ("thinker.lm_head.", "lm_head."),
    ("thinker.model.", "model."),
)

#: ``qwen_asr`` 的版本/路径/词表/mrope 常量都集中在
#: :mod:`aitsm.transcribe.qwen_backend`——那里 pin 了版本、做了路径探测，
#: 升级 qwen-asr 时只需要改那一个文件。
VOCALPARSE_VOCAB_SIZE = qwen_backend.VOCAB_SIZE
DEFAULT_MROPE = qwen_backend.DEFAULT_MROPE

#: VocalParse 推理采样率（Whisper 编码器要求 16kHz）
SAMPLE_RATE_16K = 16000
#: 单次推理的最大窗口（秒）。模型 card 明确「>30s 应先分段」，后文按此切分。
VOCALPARSE_MAX_WINDOW = 30.0


class VocalParseIncompatibleError(VocalParseError):
    """权重与已安装的 qwen_asr 版本结构不兼容（不是运行期偶发错误）。"""


#: 最近一次 :func:`parse_vocalparse_tokens` 里「没挂到音高的 <NOTE_> token」。
#: 正常输出里应为空；非空说明模型把休止写成了纯时值，需要上层注意时序偏移。
_LAST_PARSE_ORPHANS: list[str] = []


def last_parse_orphans() -> list[str]:
    """最近一次解析遗留的孤立时值 token（诊断用，正常应为空列表）。"""
    return list(_LAST_PARSE_ORPHANS)


def _remap_vocalparse_state_dict(raw: Mapping[str, Any]) -> dict[str, Any]:
    """把 checkpoint 的键名映射到 ``qwen_asr`` 的 Thinker 模型。

    实测（2026-10，真实权重 + qwen_asr 0.0.6）：

    * ``thinker.audio_tower.*`` → ``audio_tower.*``（**去掉** ``model.`` 前缀）
    * ``thinker.model.*`` → ``model.*``（语言模型部分保留 ``model.``）
    * ``lm_head.weight`` 与 ``model.embed_tokens.weight`` 共享，权重文件里没有
      前者，需要显式补上（tied weights）。

    纯函数、不依赖 torch，便于离线单测。
    """
    remapped: dict[str, Any] = {}
    for key, value in raw.items():
        target = str(key)
        for source_prefix, target_prefix in _VOCALPARSE_KEY_PREFIXES:
            if target.startswith(source_prefix):
                target = target_prefix + target[len(source_prefix):]
                break
        remapped[target] = value
    # 语言模型头与词嵌入共享权重（checkpoint 里不存 lm_head）
    embed = remapped.get("model.embed_tokens.weight")
    if embed is not None and "lm_head.weight" not in remapped:
        remapped["lm_head.weight"] = embed
    return remapped


def _load_qwen_asr_backend() -> dict[str, Any]:
    """按文件路径加载 ``qwen_asr`` 的三个建模子模块（实现在 qwen_backend）。

    版本 pin、路径探测、空壳父包都集中在 :mod:`aitsm.transcribe.qwen_backend`；
    这里只把它的错误类型翻译成本层的 ``VocalParseError``，保证调用方只需处理
    一种异常。
    """
    try:
        return qwen_backend.load_backend()
    except qwen_backend.QwenBackendError as exc:
        raise VocalParseError(str(exc)) from exc


#: 进程级模型缓存：VocalParse 是 2B 参数、加载约 35s、占约 8GB 内存。
#: 同一进程里连续转录多段音频时若每次重新加载，很容易把内存打满并触发
#: 后续调用失败（实测：连跑两段时第二段降级）。这里按「权重目录 + 设备」缓存。
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def _load_vocalparse_model(model_dir: Path, *, device_name: str, torch: Any) -> Any:
    """加载 VocalParse 权重（走 ``qwen_asr`` 的 Thinker 建模代码），带进程级缓存。

    为什么必须用 ``Qwen3ASRThinkerForConditionalGeneration``：该 checkpoint 的
    键名是 ``thinker.*``，且 audio ``output_dim`` 是 2048、词表是 152101——
    都对应 Qwen3-ASR 的 **Thinker** 结构；通用 ``Qwen3ASRForConditionalGeneration``
    期望的是 3584 / 151936，两者不可能直接对上。

    任何形状/键不一致都抛 :class:`VocalParseIncompatibleError`，调用方据此降级。
    """
    from safetensors.torch import load_file  # type: ignore

    cache_key = (str(Path(model_dir).resolve()), str(device_name))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    modules = _load_qwen_asr_backend()
    config_module = modules["configuration_qwen3_asr"]
    modeling_module = modules["modeling_qwen3_asr"]

    weights_path = model_dir / "model.safetensors"
    if not weights_path.is_file():
        raise VocalParseIncompatibleError(f"{model_dir} 缺少 model.safetensors")

    try:
        config = config_module.Qwen3ASRConfig.from_pretrained(str(model_dir))
    except Exception as exc:
        raise VocalParseIncompatibleError(f"配置读取失败：{type(exc).__name__}: {exc}") from exc

    thinker_config = config.thinker_config
    text_config = thinker_config.text_config
    if getattr(text_config, "rope_scaling", None) is None:
        # checkpoint 没写 rope_scaling，建模代码要求它存在
        text_config.rope_scaling = dict(DEFAULT_MROPE)
    # 扩展词表（AST token）必须与权重一致，否则 embed_tokens 形状对不上
    text_config.vocab_size = VOCALPARSE_VOCAB_SIZE

    # mRoPE 少了 mrope_section 就是**位置编码静默失效**：不报错，但音符时序会
    # 系统性偏移。这里显式校验，不信任默认值。
    mrope_problems = qwen_backend.validate_mrope(text_config)
    if mrope_problems:
        raise VocalParseIncompatibleError(
            "VocalParse 位置编码配置不可用：" + "；".join(mrope_problems)
        )

    try:
        raw = load_file(str(weights_path))
    except Exception as exc:
        raise VocalParseError(f"读取权重失败：{type(exc).__name__}: {exc}") from exc
    remapped = _remap_vocalparse_state_dict(raw)

    thinker_cls = getattr(modeling_module, "Qwen3ASRThinkerForConditionalGeneration")
    model = thinker_cls(thinker_config)
    state = model.state_dict()

    mismatched = [
        f"{key}: 权重 {tuple(tensor.shape)} vs 模型 {tuple(state[key].shape)}"
        for key, tensor in remapped.items()
        if key in state and tuple(state[key].shape) != tuple(tensor.shape)
    ]
    missing = [key for key in state if key not in remapped]
    unexpected = [key for key in remapped if key not in state]

    if mismatched or missing or unexpected:
        detail = []
        if mismatched:
            detail.append("形状不匹配 " + "；".join(mismatched[:3]))
        if missing:
            detail.append(f"权重缺 {len(missing)} 个键（如 {missing[0]}）")
        if unexpected:
            detail.append(f"多出 {len(unexpected)} 个键（如 {unexpected[0]}）")
        raise VocalParseIncompatibleError(
            "VocalParse 权重与已安装的 qwen_asr 版本不兼容：" + "；".join(detail)
        )

    state.update(remapped)
    model.load_state_dict(state, strict=True)
    if hasattr(model, "to"):
        model = model.to(device_name)
    if hasattr(model, "eval"):
        model = model.eval()
    # 处理器（feature extractor + tokenizer）随权重目录一起提供
    try:
        processor = modules["processing_qwen3_asr"].Qwen3ASRProcessor.from_pretrained(
            str(model_dir)
        )
    except Exception as exc:
        raise VocalParseError(f"处理器加载失败：{type(exc).__name__}: {exc}") from exc
    _MODEL_CACHE[cache_key] = (model, processor)
    return model, processor


def _split_windows(
    audio: Any, sample_rate: int, max_seconds: float
) -> list[tuple[float, Any]]:
    """把整段音频切成 ≤``max_seconds`` 的窗口，返回 ``[(起始秒, 片段), ...]``。

    为什么必须分段：VocalParse 的模型 card 明确写「>30s 应先分段」。整段输入时
    后半段的音高/歌词质量会明显下降（模型按 30s 上下文训练）。

    不足一个窗口时返回单段（``[(0.0, audio)]``），保持短音频的原行为。
    """
    total = int(len(audio))
    window = max(1, int(sample_rate * float(max_seconds)))
    if total <= window:
        return [(0.0, audio)]
    windows: list[tuple[float, Any]] = []
    start = 0
    while start < total:
        end = min(total, start + window)
        windows.append((start / float(sample_rate), audio[start:end]))
        if end >= total:
            break
        start = end
    return windows


def _generate_once(
    model: Any,
    processor: Any,
    chunk: Any,
    torch: Any,
    device_name: str,
) -> str:
    """对一段音频跑一次 generate，返回解码后的 token 串。"""
    # 必须用仓库自带的 chat template 拼出带 <|audio_start|>…占位符的文本，
    # 否则 processor 会报 "You need to specify either a text input"（实测）。
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": "转录音高、时值与歌词"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], audio=[chunk], sampling_rate=16000, return_tensors="pt")
    inputs = {
        key: (value.to(device_name) if hasattr(value, "to") else value)
        for key, value in dict(inputs).items()
    }
    prompt_length = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=2048)
    return str(processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0])


def _dominant_bpm(events: Sequence[VocalEvent]) -> float | None:
    """取出现次数最多的 BPM（多段时用它代表全曲 tempo）。

    为什么不是「最后一个」：多段拼接后最后一段可能是慢速尾奏/清唱，
    用最后一个会把全曲 tempo 带偏。出现次数最多的更稳健；并列时取先出现的。
    """
    counts: dict[float, int] = {}
    order: list[float] = []
    for event in events:
        bpm = event.bpm
        if not bpm:
            continue
        if bpm not in counts:
            counts[bpm] = 0
            order.append(bpm)
        counts[bpm] += 1
    if not counts:
        return None
    best = max(order, key=lambda value: (counts[value], -order.index(value)))
    return float(best)


def _transcribe_with_vocalparse(
    path: str,
    *,
    duration: float | None,
    language: str,
    beatgrid: BeatGrid | Mapping[str, Any] | None,
    device: str,
    progress: Callable[..., Any] | None,
) -> VocalResult:
    """用本地 VocalParse 权重做一次联合转录（只在 :func:`is_available` 为真时调用）。

    走 ``qwen_asr`` 的 **Thinker** 建模代码（见 :func:`_load_vocalparse_model`），
    一次 ``generate`` 拿到 AST token 串，再交给 :func:`parse_vocalparse_tokens`。

    实测（2026-10，CPU，14s 音频）：加载 34.7s、生成 117.3s。任何一步失败都抛
    :class:`VocalParseError`，由 :func:`transcribe_vocal` 显式降级。
    """
    model_dir = vocalparse_dir()
    if model_dir is None:
        raise VocalParseError(f"未配置可用的本地权重目录（{VOCALPARSE_DIR_ENV}）")

    # 重依赖一律在这里惰性导入，保证 import aitsm.transcribe.vocal 是毫秒级
    import torch  # type: ignore

    device_name = _resolve_device(device)
    try:
        model, processor = _load_vocalparse_model(model_dir, device_name=device_name, torch=torch)
    except VocalParseError:
        raise
    except Exception as exc:
        raise VocalParseError(f"本地权重加载失败：{type(exc).__name__}: {exc}") from exc

    try:
        from aitsm.pipeline.lyrics import decode_audio_16k  # 复用既有解码，采样率一致

        audio = decode_audio_16k(str(path), duration=duration)
    except Exception as exc:
        raise VocalParseError(f"音频解码失败：{type(exc).__name__}: {exc}") from exc

    _emit_progress(progress, 0.15, "人声联合转录：VocalParse 生成 token")
    # VocalParse 官方建议单次输入 ≤30s（模型 card 的 Limitations：
    # "Long audio segments (> 30s) should be pre-segmented before inference"）。
    # 整段直接塞进去会让后半段的音高/歌词质量明显下降，所以这里按窗口切分，
    # 每段独立解码后按段内偏移拼回统一时间轴。
    windows = _split_windows(audio, SAMPLE_RATE_16K, VOCALPARSE_MAX_WINDOW)
    decoded_chunks: list[tuple[float, str]] = []
    try:
        for index, (offset, chunk) in enumerate(windows):
            if callable(progress):
                _emit_progress(
                    progress,
                    0.15 + 0.7 * (index / max(1, len(windows))),
                    f"VocalParse 解码第 {index + 1}/{len(windows)} 段",
                )
            decoded_chunks.append((offset, _generate_once(model, processor, chunk, torch, device_name)))
    except Exception as exc:
        raise VocalParseError(f"推理失败：{type(exc).__name__}: {exc}") from exc

    # 单段时保持原来的行为（直接用它的 BPM）；多段时取各段里出现最多的 BPM，
    # 避免「最后一段恰好是慢速尾奏」把全曲 tempo 带偏。
    if len(decoded_chunks) == 1:
        events = parse_vocalparse_tokens(decoded_chunks[0][1])
        notes = vocal_events_to_notes(events, beatgrid)
    else:
        events = []
        notes = []
        for offset, text in decoded_chunks:
            chunk_events = parse_vocalparse_tokens(text)
            if not chunk_events:
                continue
            events.extend(chunk_events)
            # 段内时间轴从 0 开始，拼回绝对时间要加上该段起点
            for note in vocal_events_to_notes(chunk_events, None):
                notes.append(
                    MelodyNote(
                        start=round(note.start + offset, 6),
                        end=round(note.end + offset, 6),
                        midi=note.midi,
                        confidence=note.confidence,
                        lyric=note.lyric,
                        syllable_index=None,
                        source="vocalparse",
                    )
                )
        notes.sort(key=lambda item: (item.start, item.midi))
        # 跨段后重新编音节序号，供下游做词-音符对齐计数
        cursor = 0
        for note in notes:
            if note.lyric:
                note.syllable_index = cursor
                cursor += 1

    if not events:
        preview = re.sub(r"\s+", " ", str(" / ".join(text for _offset, text in decoded_chunks)))[:160]
        raise VocalParseError(f"模型输出里没有可解析的 VocalParse token：{preview!r}")

    lyrics = _lyric_lines_from_notes(notes)
    token_bpm = _dominant_bpm(events)
    bpm = token_bpm if token_bpm is not None else _grid_bpm(beatgrid)

    _emit_progress(progress, 0.9, f"人声联合转录：解析出 {len(notes)} 个音")
    return VocalResult(
        notes=notes,
        lyrics=lyrics,
        bpm=bpm,
        engine="vocalparse",
        requested=REQUESTED_ENGINE,
        alignment={
            "engine": "vocalparse",
            "language": language,
            "events": len(events),
            "notes_with_lyric": sum(1 for note in notes if note.lyric),
            "lyric_lines": len(lyrics),
            "bpm_source": "token" if token_bpm is not None else ("beatgrid" if bpm else "none"),
        },
    )


# --------------------------------------------------------------------------- #
# 降级路径：whisper(+pyin) 三段式
# --------------------------------------------------------------------------- #


def _transcribe_with_fallback(
    path: str,
    *,
    duration: float | None,
    language: str,
    beatgrid: BeatGrid | Mapping[str, Any] | None,
    progress: Callable[..., Any] | None,
    whisper_model: str,
    warnings: list[str],
    started: float,
) -> VocalResult:
    """既有三段式降级：歌词 ASR → 旋律提取 → 词-旋律对齐 → 转 contracts。

    每一段都单独兜异常：任何一段失败都只丢掉它自己的产物（并告警），不会让另外
    两段白跑——这是 v1 已经验证过的降级纪律。
    """
    beats = _beats_payload(beatgrid)

    _emit_progress(progress, 0.1, "人声联合转录：歌词 ASR（whisper）")
    lines, lyrics_info = _fallback_lyrics(
        path,
        duration=duration,
        language=language,
        progress=progress,
        whisper_model=whisper_model,
        warnings=warnings,
    )

    _emit_progress(progress, 0.55, "人声联合转录：旋律提取（pyin）")
    melody = _fallback_melody(path, beats, duration=duration, warnings=warnings)

    _emit_progress(progress, 0.85, "人声联合转录：词-旋律对齐")
    melody = _fallback_align(melody, lines, beats, warnings=warnings)

    notes = _notes_from_melody(melody)
    lyric_lines = _lyric_lines_from_lines(lines)
    if not notes:
        warnings.append("旋律为空：未识别出任何音符（可能无人声演唱或基频提取失败）")
    if not lyric_lines:
        warnings.append("歌词为空：未识别出歌词（whisper 不可用、无演唱或语言不符）")

    alignment = dict(melody.get("alignment") or {})
    alignment.update(
        {
            "engine": FALLBACK_ENGINE,
            "language": language,
            "notes": len(notes),
            "lyrics": len(lyric_lines),
            "notes_with_lyric": sum(1 for note in notes if note.lyric),
            "lyrics_engine": str(lyrics_info.get("engine") or "none"),
            "lyrics_model": str(lyrics_info.get("model") or ""),
            "melody_engine": str(melody.get("engine") or "none"),
            "whisper_model": str(whisper_model or ""),
            "beatgrid": "provided" if beatgrid is not None else "absent",
        }
    )

    _emit_progress(
        progress,
        1.0,
        f"人声联合转录完成：{FALLBACK_ENGINE}，{len(notes)} 个音 / {len(lyric_lines)} 行歌词",
    )
    return VocalResult(
        notes=notes,
        lyrics=lyric_lines,
        bpm=_grid_bpm(beatgrid),
        engine=FALLBACK_ENGINE,
        requested=REQUESTED_ENGINE,
        alignment=alignment,
        warnings=warnings,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )


def _fallback_lyrics(
    path: str,
    *,
    duration: float | None,
    language: str,
    progress: Callable[..., Any] | None,
    whisper_model: str,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """调既有歌词 ASR；返回 (歌词行, 实际运行信息)。失败返回 ([], {})。"""
    try:
        from aitsm.pipeline import lyrics as lyrics_mod
    except Exception as exc:
        warnings.append(f"歌词模块不可用（{type(exc).__name__}: {exc}），跳过歌词识别")
        return [], {}

    model_override = str(whisper_model or "").strip()
    previous = os.environ.get(WHISPER_MODEL_ENV)
    changed = bool(model_override) and model_override != previous
    if changed:
        os.environ[WHISPER_MODEL_ENV] = model_override
    try:
        raw_lines = lyrics_mod.transcribe_lyrics(
            path, duration=duration, language=language, progress=progress
        )
    except Exception as exc:
        warnings.append(f"歌词识别失败（{type(exc).__name__}: {exc}）")
        return [], {}
    finally:
        if changed:
            if previous is None:
                os.environ.pop(WHISPER_MODEL_ENV, None)
            else:
                os.environ[WHISPER_MODEL_ENV] = previous

    lines: list[dict[str, Any]] = []
    if isinstance(raw_lines, list):
        lines = [item for item in raw_lines if isinstance(item, Mapping)]
    elif isinstance(raw_lines, Mapping):
        candidate = raw_lines.get("lines")
        if isinstance(candidate, list):
            lines = [item for item in candidate if isinstance(item, Mapping)]
    elif raw_lines is not None:
        warnings.append(f"歌词模块返回了意外类型 {type(raw_lines).__name__}，按空歌词处理")

    info: dict[str, Any] = {}
    getter = getattr(lyrics_mod, "last_run_info", None)
    if callable(getter):
        try:
            info = dict(getter() or {})
        except Exception:  # 记录元信息失败不影响主链路
            info = {}
    if not lines:
        info.setdefault("engine", "none")
    return [dict(item) for item in lines], info


def _fallback_melody(
    path: str,
    beats: Mapping[str, Any],
    *,
    duration: float | None,
    warnings: list[str],
) -> dict[str, Any]:
    """调既有旋律提取；返回 melody dict（含 notes/engine/alignment/warnings）。"""
    try:
        from aitsm.pipeline import melody as melody_mod
    except Exception as exc:
        warnings.append(f"旋律模块不可用（{type(exc).__name__}: {exc}），跳过旋律提取")
        return {"notes": [], "engine": "none", "warnings": []}

    try:
        raw = melody_mod.transcribe_melody(path, dict(beats), duration=duration)
    except Exception as exc:
        warnings.append(f"旋律提取失败（{type(exc).__name__}: {exc}）")
        return {"notes": [], "engine": "none", "warnings": []}

    if isinstance(raw, (list, tuple)):
        raw = {"notes": [item for item in raw if isinstance(item, Mapping)]}
    if not isinstance(raw, Mapping):
        warnings.append(f"旋律模块返回了意外类型 {type(raw).__name__}，按空旋律处理")
        return {"notes": [], "engine": "none", "warnings": []}

    melody = dict(raw)
    for item in melody.get("warnings") or []:
        text = str(item).strip()
        if text:
            warnings.append(f"旋律：{text}")
    return melody


def _fallback_align(
    melody: Mapping[str, Any],
    lines: Sequence[Mapping[str, Any]],
    beats: Mapping[str, Any],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    """把歌词音节挂到旋律音符上；任一前提不满足就原样返回旋律。"""
    result = dict(melody)
    notes = [item for item in (result.get("notes") or []) if isinstance(item, Mapping)]
    if not lines or not notes:
        return result
    try:
        from aitsm.pipeline.align import align_lyrics_to_melody
    except Exception as exc:
        warnings.append(f"对齐模块不可用（{type(exc).__name__}: {exc}），歌词与音符分开返回")
        return result

    try:
        aligned = align_lyrics_to_melody(dict(result), [dict(item) for item in lines], dict(beats))
    except Exception as exc:
        warnings.append(
            f"歌词-旋律对齐失败（{type(exc).__name__}: {exc}），歌词与音符分开返回"
        )
        return result

    if isinstance(aligned, Mapping):
        merged = dict(aligned)
        merged.setdefault("engine", result.get("engine"))
        return merged
    if isinstance(aligned, (list, tuple)):
        merged = dict(result)
        merged["notes"] = [item for item in aligned if isinstance(item, Mapping)]
        return merged
    warnings.append("对齐模块返回了意外类型，歌词与音符分开返回")
    return result


# --------------------------------------------------------------------------- #
# 既有结构 -> contracts
# --------------------------------------------------------------------------- #


def _notes_from_melody(melody: Mapping[str, Any] | None) -> list[MelodyNote]:
    """把旋律 dict（``notes[*]``，可能已带 lyric）转成 :class:`MelodyNote`。"""
    if not isinstance(melody, Mapping):
        return []
    engine = str(melody.get("engine") or "").strip() or "pyin"
    notes: list[MelodyNote] = []
    for item in melody.get("notes") or []:
        if not isinstance(item, Mapping):
            continue
        midi = _as_float(item.get("midi"))
        if midi is None:
            continue
        start = _as_float(item.get("start"))
        end = _as_float(item.get("end"))
        if start is None:
            continue
        if end is None or end < start:
            end = start
        notes.append(
            MelodyNote(
                start=start,
                end=end,
                midi=int(round(midi)),
                confidence=_as_float(item.get("confidence")) or 0.0,
                lyric=str(item.get("lyric") or "").strip(),
                syllable_index=item.get("syllable_index"),
                source=str(item.get("source") or engine),
            )
        )
    return notes


def _lyric_lines_from_lines(lines: Sequence[Mapping[str, Any]] | None) -> list[LyricLine]:
    """把既有歌词行 dict 转成 :class:`LyricLine`（逐字时间戳转 :class:`LyricWord`）。"""
    result: list[LyricLine] = []
    for item in lines or []:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        words: list[LyricWord] = []
        for entry in item.get("words") or []:
            if not isinstance(entry, Mapping):
                continue
            word_text = str(entry.get("text") or "").strip()
            if not word_text:
                continue
            start = _as_float(entry.get("start"))
            end = _as_float(entry.get("end"))
            words.append(
                LyricWord(
                    start=start if start is not None else 0.0,
                    end=end if end is not None else (start if start is not None else 0.0),
                    text=word_text,
                )
            )
        start = _as_float(item.get("start"))
        end = _as_float(item.get("end"))
        result.append(
            LyricLine(
                start=start if start is not None else 0.0,
                end=end if end is not None else (start if start is not None else 0.0),
                text=text,
                words=words,
                confidence=_as_float(item.get("confidence")) or 0.0,
                interjection=bool(item.get("interjection")),
            )
        )
    return result


def _lyric_lines_from_notes(notes: Sequence[MelodyNote]) -> list[LyricLine]:
    """从带歌词的音符**保守地**切出歌词行。

    VocalParse 的 token 流没有显式断句信息，所以这里只在「模型明确给出无词音符」处
    断行（那是它自己标的旋律/间奏边界），绝不按时间间隔猜断句。每个音一个
    :class:`LyricWord`，行内文本是这些音节的拼接。
    """
    lines: list[LyricLine] = []
    run: list[MelodyNote] = []

    def flush() -> None:
        if not run:
            return
        lines.append(
            LyricLine(
                start=run[0].start,
                end=run[-1].end,
                text="".join(note.lyric for note in run),
                words=[
                    LyricWord(start=note.start, end=note.end, text=note.lyric)
                    for note in run
                ],
                confidence=min(note.confidence for note in run),
            )
        )
        run.clear()

    for note in notes:
        if note.lyric:
            run.append(note)
        else:
            flush()
    flush()
    return lines


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def _parse_note_value(value: Any) -> tuple[float, bool]:
    """解析「几分音符」，返回 (四分音符数, 是否解析成功)。"""
    if isinstance(value, bool):
        return 1.0, False
    if isinstance(value, (int, float)):
        text = str(value)
    elif value is None:
        return 1.0, False
    else:
        text = re.sub(r"\s+", "", str(value))
    if not text:
        return 1.0, False

    dotted = False
    if text[-1] in ".dD":
        # 附点两种写法都接受：<NOTE_4.> 与 <NOTE_4d>
        dotted = True
        text = text[:-1]
    if not text:
        return 1.0, False

    try:
        base = float(text)
    except ValueError:
        return 1.0, False
    if not (_MIN_NOTE_BASE <= base <= _MAX_NOTE_BASE):
        return 1.0, False

    quarters = 4.0 / base * (DOTTED_RATIO if dotted else 1.0)
    quarters = min(_MAX_QUARTERS, max(_MIN_QUARTERS, quarters))
    return round(quarters, 6), True


def _normalize_note_value_token(value: Any) -> str:
    """把时值 token 规范化成 ``"4"`` / ``"4."`` 这种紧凑写法（空则空串）。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return re.sub(r"\s+", "", str(value))


def _parse_pitch(value: str) -> int | None:
    """解析 ``<P_68>`` 的音高；只接受 0..127 的整数 MIDI 音号。"""
    text = (value or "").strip()
    if not text or not text.lstrip("+-").isdigit():
        return None
    pitch = int(text)
    if 0 <= pitch <= 127:
        return pitch
    return None


def _clean_lyric_text(value: Any) -> str:
    """规范歌词文本：折叠空白、去首尾空白、丢掉「只有标点」的碎片。"""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    if not _WORDISH_RE.search(text):
        # "|" / "," / "。" 这类分隔符不是歌词，混进来会让音节计数虚高
        return ""
    return text


def _join_lyric(left: str, right: str) -> str:
    """拼接同一事件上的多段歌词：西文之间补空格，中文/日文直接连写。"""
    if not left:
        return right
    if not right:
        return left
    if (
        left[-1].isascii()
        and left[-1].isalnum()
        and right[0].isascii()
        and right[0].isalnum()
    ):
        return f"{left} {right}"
    return f"{left}{right}"


def _iter_events(events: Sequence[VocalEvent] | None) -> Iterable[VocalEvent]:
    """逐个产出事件；容忍 Mapping 形式的输入，其余类型跳过。"""
    for item in events or []:
        if isinstance(item, VocalEvent):
            yield item
        elif isinstance(item, Mapping):
            yield VocalEvent(
                lyric=item.get("lyric") or "",
                midi_pitch=item.get("midi_pitch", item.get("midi", 60)),
                note_value=item.get("note_value", DEFAULT_NOTE_VALUE),
                bpm=item.get("bpm"),
            )


def _time_axis(
    beatgrid: BeatGrid | Mapping[str, Any] | None,
) -> tuple[float, float]:
    """返回 (时间轴锚点秒, 兜底 BPM)。缺失/非法值退回 (0, 120)。"""
    if beatgrid is None:
        return 0.0, DEFAULT_BPM
    if isinstance(beatgrid, BeatGrid):
        anchor = _as_float(beatgrid.first_downbeat_sec)
        bpm = _as_float(beatgrid.bpm)
    elif isinstance(beatgrid, Mapping):
        anchor = _as_float(beatgrid.get("first_downbeat_sec"))
        bpm = _as_float(beatgrid.get("bpm"))
        if anchor is None:
            downbeats = beatgrid.get("downbeats") or []
            if downbeats:
                anchor = _as_float(downbeats[0])
    else:
        return 0.0, DEFAULT_BPM
    anchor = max(0.0, anchor) if anchor is not None else 0.0
    if bpm is None or bpm <= 0:
        bpm = DEFAULT_BPM
    return anchor, bpm


def _grid_bpm(beatgrid: BeatGrid | Mapping[str, Any] | None) -> float | None:
    """只读 beatgrid 自己声明的 BPM；没有就返回 None（不编造速度）。"""
    if beatgrid is None:
        return None
    if isinstance(beatgrid, BeatGrid):
        value = _as_float(beatgrid.bpm)
    elif isinstance(beatgrid, Mapping):
        value = _as_float(beatgrid.get("bpm"))
    else:
        return None
    return value if value and value > 0 else None


def _beats_payload(beatgrid: BeatGrid | Mapping[str, Any] | None) -> dict[str, Any]:
    """把 BeatGrid 转成既有 pipeline 认识的 ``beats`` dict。

    没有 beatgrid 时**不伪造 bpm**：只给空网格，量化器会退回「保留原始时刻」。
    """
    if isinstance(beatgrid, BeatGrid):
        payload: dict[str, Any] = beatgrid.to_dict()
    elif isinstance(beatgrid, Mapping):
        payload = dict(beatgrid)
        payload["beats"] = [item for item in (payload.get("beats") or []) if _as_float(item) is not None]
        payload["downbeats"] = [
            item for item in (payload.get("downbeats") or []) if _as_float(item) is not None
        ]
        if not _grid_bpm(payload):
            payload.pop("bpm", None)
    else:
        payload = {"beats": [], "downbeats": []}
    payload.setdefault("beats_per_bar", 4)
    payload.setdefault("time_signature", "4/4")
    return payload


def _resolve_device(device: Any) -> str:
    """把 ``"auto"`` 解析成具体设备；无 torch 时按 CPU 处理。"""
    value = str(device or "").strip().lower()
    if value in {"cpu", "cuda"}:
        return value
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _can_import_transformers() -> bool:
    """transformers 是否可导入（用 find_spec 探测，避免几秒的导入代价）。"""
    try:
        return importlib.util.find_spec("transformers") is not None
    except (ImportError, AttributeError, ValueError):  # 损坏的 sys.modules 条目
        return False


def _unavailable_reason() -> str:
    """给 warnings 用的人类可读原因（说明为什么降级）。"""
    raw = (os.environ.get(VOCALPARSE_DIR_ENV) or "").strip()
    if not raw:
        return f"未设置 {VOCALPARSE_DIR_ENV}（本地权重目录），本层不会联网下载"
    if vocalparse_dir() is None:
        return f"{VOCALPARSE_DIR_ENV}={raw} 不是可用（存在且非空）的权重目录"
    if not _can_import_transformers():
        return "transformers 不可导入"
    return "未知原因"


def _emit_progress(
    progress: Callable[..., Any] | None, fraction: float, message: str
) -> None:
    """宽松进度协议：优先 dict，其次 (fraction, message)；回调异常一律吞掉。"""
    if not callable(progress):
        return
    ratio = min(1.0, max(0.0, _as_float(fraction) or 0.0))
    payload = {
        "stage": "vocal",
        "percent": round(ratio * 100.0, 1),
        "message": message,
    }
    try:
        progress(payload)
        return
    except TypeError:
        pass
    except Exception:  # pragma: no cover - 回调本身不应影响主流程
        LOGGER.debug("人声转录进度回调异常", exc_info=True)
        return
    try:
        progress(ratio, message)
    except Exception:  # pragma: no cover - 同上
        LOGGER.debug("人声转录进度回调异常", exc_info=True)


def _as_float(value: Any) -> float | None:
    """宽松转 float：失败、bool、非有限值都返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _as_int(value: Any, default: int = 0) -> int:
    """宽松转 int：失败或 bool 返回 default。"""
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default
