#!/usr/bin/env python
"""DSH Skill 入口：一次调用完成「音频 -> 弹唱谱」并输出结构化摘要。

设计要点：
- **进程隔离**：分析会加载 torch / demucs / whisper，绝不能跑在 agent 主进程里，
  因此这里用 subprocess 调 ``python -m aitsm.cli generate``。
- **路径无关**：项目根由 ``_locate.py`` 解析（环境变量 -> 配置 -> 开发布局 -> vendor -> pip），
  脚本被复制到 ``~/.config/opencode/skills/`` 后仍能找到 aitsm。
- **结构化返回**：stdout 只输出一个 JSON 对象（UTF-8），agent 直接读；
  人类可读的摘要在 ``report.md``，进度日志走 stderr。
- **不吞错**：任何阶段失败都返回 ``{"ok": false, "error": ..., "hint": ...}``，退出码非 0。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _locate  # noqa: E402

# 告警严重度：high = 会误导用户弹错，必须明说；medium = 建议复核；low = 信息性
SEVERITY = {
    "high": [
        "separate.failed",
        "lyrics.hallucination",
        "lyrics.empty",
        "lyrics.no_vocals",
        "key.invalid",
        "key.invalid_option",
        "structure.degenerate",
        "structure.failed",
        "chords.no_chords",
        "chords.missing_bars",
        "score.schema",
        "score.validation",
        "analysis.schema",
        "melody.module_missing",
        "lyrics.module_missing",
        "melody.failed",
        "lyrics.failed",
        "melody.align_failed",
        "key.refine_failed",
    ],
    "medium": [
        "chords.low_confidence",
        "chords.coverage_gap",
        "chords.too_many",
        "lyrics.low_confidence",
        "lyrics.missing",
        "lyrics.coverage",
        "lyrics.skipped",
        "lyrics.vocals_fallback_mix",
        "key.relative_ambiguity",
        "key.chroma_only",
        "key.capo_high",
        "key.non_guitar_friendly",
        "structure.coarse",
        "melody.range",
        "melody.low_confidence",
        "melody.overlap",
        "separate.disabled",
        "guitar.voicing_mismatch",
        "options.instrument",
    ],
}
_LEVEL_ORDER = {"high": 0, "medium": 1, "low": 2}
_LEVEL_LABEL = {"high": "必须告知", "medium": "建议复核", "low": "信息"}


# ------------------------------------------------------------------ 工具


def severity_of(warning: dict) -> str:
    code = str(warning.get("code") or "")
    for level, codes in SEVERITY.items():
        if code in codes:
            return level
    level = str(warning.get("level") or "").lower()
    if level in ("error", "critical"):
        return "high"
    if level == "warn":
        return "medium"
    return "low"


def mmss(seconds: float | int | None) -> str:
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "?"
    return f"{total // 60:02d}:{total % 60:02d}"


def bar_span_text(bars: list[int], sections: list[dict]) -> str:
    """把小节号压成「第 12–18、32 小节（副歌）」这种可读形式。"""
    if not bars:
        return ""
    ordered = sorted(set(int(b) for b in bars))
    parts: list[str] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append(f"{start}" if start == prev else f"{start}–{prev}")
        start = prev = value
    parts.append(f"{start}" if start == prev else f"{start}–{prev}")
    text = "第 " + "、".join(parts) + " 小节"
    labels: list[str] = []
    for section in sections:
        low, high = section.get("start_bar"), section.get("end_bar")
        if low is None or high is None:
            continue
        if any(int(low) <= b <= int(high) for b in ordered):
            label = str(section.get("label") or "")
            if label and label not in labels:
                labels.append(label)
    if labels:
        text += "（" + "、".join(labels[:3]) + "）"
    return text


def section_at(bar_index: int, sections: list[dict]) -> str:
    for section in sections:
        low, high = section.get("start_bar"), section.get("end_bar")
        if low is not None and high is not None and int(low) <= bar_index <= int(high):
            return str(section.get("label") or "")
    return ""


def time_of_bar(bar: dict) -> tuple[float | None, float | None]:
    try:
        return float(bar.get("start_sec")), float(bar.get("end_sec"))
    except (TypeError, ValueError):
        return None, None


# ------------------------------------------------------------------ needs_review


def _manifest_engines(manifest: dict, analysis: dict | None = None) -> dict:
    """从 manifest / analysis 里取出「引擎与可用性」字典。

    **这里踩过一次真实的跨模块契约冲突**：``aitsm/service.py`` 合并 manifest 之后，
    ``manifest["stages"]["analysis"]`` 是分析层写的**阶段记录 list**
    （``[{"name": "ingest", "elapsed_seconds": ...}, ...]``），而早期版本这里是
    service 写的**摘要 dict**（``{"duration_sec":..., "engines": {...}}``）。
    本脚本原来直接 ``.get("engines")``，一旦 service 层改动就崩在
    ``AttributeError: 'list' object has no attribute 'get'``。

    所以按可靠性从高到低依次尝试，任何一层拿到非空 dict 就返回：
    1. ``manifest["tools"]`` —— pipeline 一直写在这里，最稳定；
    2. ``manifest["stages"]["analysis_summary"]["engines"]`` —— service 层摘要；
    3. ``manifest["stages"]["analysis"]["engines"]`` —— 旧版 service 摘要；
    4. ``analysis["tools"]`` —— 直接读分析结果兜底。
    """
    if not isinstance(manifest, dict):
        manifest = {}

    tools = manifest.get("tools")
    if isinstance(tools, dict) and tools:
        return tools

    stages = manifest.get("stages")
    if isinstance(stages, dict):
        for key in ("analysis_summary", "analysis"):
            block = stages.get(key)
            if isinstance(block, dict):
                engines = block.get("engines")
                if isinstance(engines, dict) and engines:
                    return engines

    if isinstance(analysis, dict):
        tools = analysis.get("tools")
        if isinstance(tools, dict):
            return tools
    return {}


def collect_needs_review(score: dict, manifest: dict, analysis: dict) -> list[dict]:
    """把分散在 score/manifest/analysis 里的问题汇成「用户需要确认」的结构化列表。

    每条：severity / code / what / where / bars / start_sec / end_sec / action
    按严重程度排序（high 在前），同级按出现位置（小节号 / 时间）升序。
    """
    sections = score.get("sections") or []
    bars = score.get("bars") or []
    quality = score.get("quality") or {}
    items: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(
        *,
        severity: str,
        code: str,
        what: str,
        action: str,
        where: str = "",
        bar_list: list[int] | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
    ) -> None:
        key = (code, where)
        if key in seen:
            return
        seen.add(key)
        items.append(
            {
                "severity": severity,
                "code": code,
                "what": what,
                "where": where,
                "bars": sorted(set(int(b) for b in (bar_list or []))),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "action": action,
            }
        )

    # 1) 分轨降级：影响面最大，最靠前
    stems = analysis.get("stems") or {}
    engine = str(stems.get("engine") or "")
    if engine and engine not in ("demucs", "htdemucs"):
        add(
            severity="high",
            code="separate.degraded",
            what=f"人声/伴奏分离降级（{engine}），旋律与和弦都建立在粗糙分离上",
            action="建议告知用户：这首歌的旋律与和弦可靠性低于正常水平，需要人工复核",
        )

    # 2) 逐条处理 quality.warnings
    for warning in quality.get("warnings") or []:
        if not isinstance(warning, dict):
            continue
        code = str(warning.get("code") or "")
        message = str(warning.get("message") or "").strip()
        if not code:
            continue
        level = severity_of(warning)
        target = warning.get("target") or warning.get("path") or ""
        where = str(target) if target and target != "bars" else ""
        add(
            severity=level,
            code=code,
            what=message or code,
            action=_action_for(code),
            where=where,
        )

    # 3) 和弦低置信度：定位到具体小节 + 时间
    low_conf_bars: list[dict] = []
    threshold = 0.55
    for bar in bars:
        confs = [
            float(chord["confidence"])
            for chord in (bar.get("chords") or [])
            if isinstance(chord.get("confidence"), (int, float))
        ]
        if confs and sum(confs) / len(confs) < threshold:
            low_conf_bars.append(bar)
    if low_conf_bars:
        numbers = [int(bar.get("index", 0)) for bar in low_conf_bars]
        start_sec, _ = time_of_bar(low_conf_bars[0])
        _, end_sec = time_of_bar(low_conf_bars[-1])
        add(
            severity="medium",
            code="chords.bar_low_confidence",
            what=f"{len(numbers)} 个小节的和弦置信度低于 {threshold:.2f}",
            action="在网页里点这些小节对照原曲听一遍，必要时重跑或人工改",
            where=bar_span_text(numbers, sections),
            bar_list=numbers,
            start_sec=start_sec,
            end_sec=end_sec,
        )

    # 4) 歌词：低置信度行（定位到时间）、覆盖率
    lyric_lines = analysis.get("lyrics") or []
    low_lines = [
        line
        for line in lyric_lines
        if isinstance(line, dict) and float(line.get("confidence") or 1.0) < 0.4
    ]
    if low_lines:
        add(
            severity="medium",
            code="lyrics.line_low_confidence",
            what=f"{len(low_lines)}/{len(lyric_lines)} 行歌词识别置信度低于 0.40",
            action="交付时附上这几行原文供用户校对，或建议以原曲为准",
            where=f"{mmss(low_lines[0].get('start'))}–{mmss(low_lines[-1].get('end'))}",
            start_sec=low_lines[0].get("start"),
            end_sec=low_lines[-1].get("end"),
        )
    alignment = ((analysis.get("melody") or {}).get("alignment") or {})
    coverage = alignment.get("coverage")
    if isinstance(coverage, (int, float)) and coverage < 0.6:
        add(
            severity="medium",
            code="lyrics.coverage_low",
            what=f"旋律音符与歌词只对齐了 {coverage * 100:.0f}%（其余音符无词，留空不编造）",
            action="说明「有旋律但没匹配到词的部分是空的」，不要当成遗漏",
        )

    # 5) 旋律音域
    melody_range = ((analysis.get("melody") or {}).get("range") or {})
    low_midi, high_midi = melody_range.get("low_midi"), melody_range.get("high_midi")
    if isinstance(low_midi, int) and isinstance(high_midi, int):
        if not (48 <= low_midi and high_midi <= 90):
            add(
                severity="medium",
                code="melody.range_wide",
                what=f"记谱音域 MIDI {low_midi}–{high_midi}，超出常规人声 48–90",
                action="可能八度判错，建议用 --transpose 或换调重跑",
            )

    # 6) 段落结构
    if len(sections) > 16:
        add(
            severity="medium",
            code="structure.fragmented",
            what=f"切成 {len(sections)} 段，偏碎",
            action="说明段落仅供导航，不影响演奏",
        )
    short = [
        int(section["start_bar"])
        for section in sections
        if section.get("start_bar") is not None
        and section.get("end_bar") is not None
        and int(section["end_bar"]) - int(section["start_bar"]) + 1 < 4
    ]
    if short:
        add(
            severity="low",
            code="structure.short_sections",
            what=f"{len(short)} 个段落短于 4 小节",
            action="信息性，可忽略",
            where=bar_span_text(short, sections),
            bar_list=short,
        )

    # 7) 引擎信息（manifest 里的降级记录）
    engines = _manifest_engines(manifest, analysis)
    lyrics_engine = (engines.get("lyrics") or {}).get("engine") if isinstance(engines.get("lyrics"), dict) else None
    if lyrics_engine in (None, "none") and not (engines.get("lyrics") or {}).get("available"):
        add(
            severity="high",
            code="lyrics.none",
            what="本次没有歌词（whisper 不可用或被关闭）",
            action="明确告知用户只有旋律没有词，不要假装有歌词",
        )
    melody_engine = (engines.get("melody") or {}).get("engine") if isinstance(engines.get("melody"), dict) else None
    if melody_engine in (None, "none") and not (engines.get("melody") or {}).get("available"):
        add(
            severity="high",
            code="melody.none",
            what="本次没有旋律（pyin 不可用或被关闭）",
            action="明确告知用户只有和弦进行，谱面没有主旋律",
        )

    # 排序：严重度 -> 起始位置（小节号优先，其次秒）
    def sort_key(item: dict) -> tuple:
        bars_at = item.get("bars") or []
        position = bars_at[0] if bars_at else 10**6
        start = item.get("start_sec")
        start_at = float(start) if isinstance(start, (int, float)) else 10**9
        return (_LEVEL_ORDER.get(item["severity"], 3), position, start_at)

    items.sort(key=sort_key)
    return items


def _action_for(code: str) -> str:
    table = {
        "separate.failed": "必须告知：旋律与和弦质量会明显下降",
        "separate.disabled": "说明本次未分轨，旋律可能锁到贝斯上",
        "key.relative_ambiguity": "说明按关系大调标注；用户若知道原曲是小调，可用 --key 指定重跑",
        "key.chroma_only": "说明调性仅由 chroma 推断，可能不准",
        "key.capo_high": "说明 capo 品数偏高，用户可改 --capo",
        "key.non_guitar_friendly": "说明原调对吉他不够友好，已自动选替代调",
        "lyrics.missing": "说明有旋律但未匹配到词的部分留空（不编造）",
        "lyrics.coverage": "说明歌词覆盖率，未覆盖部分留空",
        "lyrics.low_confidence": "提醒用户歌词需校对，建议以原曲为准",
        "lyrics.skipped": "说明跳过歌词识别的原因",
        "lyrics.hallucination": "高度警惕：模型可能编造了歌词，建议整段标为待核对",
        "lyrics.empty": "告知本次未识别到歌词",
        "lyrics.no_vocals": "告知没有人声轨，无法产出歌词与旋律",
        "chords.low_confidence": "列出低置信度小节，建议试听核对",
        "chords.missing_bars": "列出没有和弦的小节，说明是留白",
        "chords.coverage_gap": "说明哪些时段没有和弦",
        "chords.no_chords": "告知未识别出和弦",
        "structure.coarse": "信息性，不影响演奏",
        "structure.degenerate": "说明谱面按单段处理",
        "melody.range": "可能八度错，建议换调或整体移调重跑",
        "melody.silent_bars": "信息性：纯伴奏小节",
        "melody.span": "信息性：音域跨度大属正常",
        "bars.tempo_changes": "说明已按小节写入局部 tempo",
        "bars.pickup": "信息性：有弱起小节",
        "audio.resampled": "信息性，无需处理",
        "score.validation": "读 quality.warnings 里的具体条目并转述",
        "score.schema": "说明产物结构有告警，需人工确认",
    }
    return table.get(code, "在交付说明中如实转述这一条")


# ------------------------------------------------------------------ report.md


def build_report(
    score: dict,
    manifest: dict,
    analysis: dict,
    out_dir: Path,
    needs_review: list[dict],
    located: dict | None = None,
) -> str:
    meta = score.get("meta") or {}
    quality = score.get("quality") or {}
    key = meta.get("key") or {}
    sections = score.get("sections") or []
    bars = score.get("bars") or []

    def conf(value) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "—"

    lines: list[str] = [f"# 弹唱谱报告：{meta.get('title') or '未命名'}", ""]

    # ---- 一、结论先行
    lines.append("## 一、交付结论")
    overall = quality.get("overall")
    verdict = "可直接使用"
    if isinstance(overall, (int, float)):
        if overall >= 0.70:
            verdict = "质量良好，可直接使用"
        elif overall >= 0.55:
            verdict = "基本可用，个别处需复核（见第二节）"
        else:
            verdict = "可用性有限，**必须先人工复核**（见第二节）"
    lines.append(f"综合质量 **{conf(overall)}**（相对质量分，不是准确率）→ {verdict}")
    high = [item for item in needs_review if item["severity"] == "high"]
    medium = [item for item in needs_review if item["severity"] == "medium"]
    if high:
        lines.append(f"- **{len(high)} 项必须告知用户**")
    if medium:
        lines.append(f"- {len(medium)} 项建议复核")
    lines.append("")

    # ---- 二、基本信息
    lines.append("## 二、基本信息")
    tempo = meta.get("tempo")
    ts = meta.get("time_signature") or []
    lines.append(f"- 调性：**{key.get('play') or '?'} {key.get('mode') or ''}**"
                 f"（原曲检测 {key.get('detected') or '?'}，置信度 {conf(key.get('confidence'))}）")
    lines.append(f"- 变调夹：**{meta.get('capo')}** 品　速度：**{tempo} BPM**　拍号：{'/'.join(map(str, ts)) or '?'}")
    lines.append(f"- 规模：{len(bars)} 小节 · {len(sections)} 段落 · 难度 {meta.get('difficulty')}")
    lines.append(f"- 和弦事件：{sum(len(bar.get('chords') or []) for bar in bars)} 个"
                 f"　旋律音符：{sum(len(bar.get('vocal') or []) for bar in bars)} 个")
    aligned = sum(1 for bar in bars for note in (bar.get("vocal") or []) if note.get("lyric"))
    lines.append(f"- 带词音符：{aligned} 个")
    lines.append("")

    # ---- 三、段落
    lines.append("## 三、段落结构")
    for section in sections:
        lines.append(
            f"- {section.get('label')}（第 {section.get('start_bar')}–{section.get('end_bar')} 小节，"
            f"{mmss(section.get('start_sec'))}–{mmss(section.get('end_sec'))}，"
            f"节奏型 {section.get('pattern_id') or '—'}）"
        )
    lines.append("")

    # ---- 四、需要确认（按严重程度排序）
    lines.append("## 四、需要确认（按严重程度排序）")
    if not needs_review:
        lines.append("未发现需要人工复核的问题。")
    else:
        current = None
        for item in needs_review:
            if item["severity"] != current:
                current = item["severity"]
                lines.append("")
                lines.append(f"### {_LEVEL_LABEL.get(current, current)}（{current}）")
            where = f"　位置：{item['where']}" if item.get("where") else ""
            lines.append(f"- `{item['code']}` {item['what']}{where}")
            if item.get("action"):
                lines.append(f"  - 建议：{item['action']}")
    lines.append("")

    # ---- 五、质量分项
    lines.append("## 五、质量分项（0–1，相对质量分）")
    stages = quality.get("stages") or {}
    thresholds = {"beats": 0.75, "key": 0.65, "chords": 0.60, "melody": 0.60,
                  "lyrics": 0.50, "structure": 0.65, "lyric_match": 0.60}
    for name, value in stages.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        base = thresholds.get(name)
        flag = ""
        if base is not None:
            flag = " ✅" if number >= base else (" ⚠️ 偏低" if number >= base - 0.15 else " ❌ 需复核")
        lines.append(f"- {name}: {number:.2f}{flag}")
    lines.append("")
    lines.append("> `melody` 与 `lyrics` 的口径与和弦/节拍不可比，单独看更有意义。")
    lines.append("")

    # ---- 六、降级与引擎
    engines = _manifest_engines(manifest, analysis)
    if engines:
        lines.append("## 六、引擎与降级记录")
        for name in ("torch", "demucs_model", "python", "ffmpeg"):
            if name in engines:
                lines.append(f"- {name}: {engines[name]}")
        for name in ("lyrics", "melody"):
            info = engines.get(name)
            if isinstance(info, dict):
                lines.append(f"- {name}: engine={info.get('engine')} available={info.get('available')}"
                             f" count={info.get('line_count') or info.get('note_count')}")
        stems = analysis.get("stems") or {}
        if stems.get("engine"):
            lines.append(f"- 分轨: engine={stems.get('engine')} device={stems.get('device')} "
                         f"cached={stems.get('cached')}")
        lines.append("")

    # ---- 七、产物
    lines.append("## 七、产物")
    for name, why in (
        ("score.json", "网页乐谱数据（唯一数据源）"),
        ("score.musicxml", "MuseScore / Guitar Pro 可打开"),
        ("score.mid", "MIDI，可播放或导入 DAW"),
        ("analysis.json", "分析证据（调试用）"),
        ("manifest.json", "输入哈希 / 耗时 / 引擎版本"),
        ("report.md", "本文件"),
        ("playback.mp3", "播放音源"),
    ):
        path = out_dir / name
        lines.append(f"- `{name}`{'（未生成）' if not path.is_file() else ''} — {why}")
    stems_dir = out_dir / "stems"
    if stems_dir.is_dir():
        lines.append(f"- `stems/` — 分离出的人声/鼓/贝斯/伴奏")
    lines.append("")

    # ---- 八、能力边界（如实写死，避免 agent 过度承诺）
    lines.append("## 八、能力边界（交付时不要超出）")
    lines.append("- 置信度是**相对质量分**，不是概率：说「置信度 0.60」而不是「准确率 60%」。")
    lines.append("- 旋律依赖人声分轨；无分轨时会锁到贝斯/伴奏上，不可信。")
    lines.append("- 关系大小调（如 Bm vs D）在音频上常难区分，默认按关系大调标注。")
    lines.append("- 失真吉他、大量七和弦的色彩和声会判得偏保守（宁给三和弦不给错和弦）。")
    lines.append("- 暂不处理速度/拍号的中途变化，不支持多声部人声。")
    lines.append("- 全流程本地 CPU：4 分钟歌冷启动约 4.5 分钟，分轨缓存命中约 1 分钟。")
    elapsed = manifest.get("duration_sec")
    if isinstance(elapsed, (int, float)):
        lines.append(f"- 本次实际耗时：{elapsed:.1f}s")
    lines.append("")

    # ---- 九、下一步
    lines.append("## 九、下一步命令")
    lines.append("```bash")
    lines.append(f"# 打开网页查看器（移调 / capo / 打印）")
    lines.append(f'python scripts/serve.py --score "{out_dir / "score.json"}"')
    lines.append("")
    lines.append("# 换个调 / capo 重跑")
    lines.append(f'python scripts/generate.py "<audio>" --out "<dir>" --key G --capo 2')
    lines.append("```")
    if located:
        lines.append("")
        lines.append(f"> 项目根：`{located.get('aitsm_root')}`（来源：{located.get('aitsm_root_source')}）")
        lines.append(f"> 解释器：`{located.get('python')}`（来源：{located.get('python_source')}）")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main


def fail(payload: dict, code: int) -> int:
    payload.setdefault("ok", False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


def main() -> int:
    _locate.force_utf8()

    parser = argparse.ArgumentParser(description="音频 -> 弹唱谱（DSH skill 入口）")
    parser.add_argument("audio", help="输入音频或视频文件")
    parser.add_argument("--out", default=None, help="输出目录（默认自动选可写位置）")
    parser.add_argument("--key", default=None, help="指定调，如 C / G / Am")
    parser.add_argument("--capo", type=int, default=None, help="指定变调夹品位 0-9")
    parser.add_argument("--difficulty", default="elementary",
                        choices=["beginner", "elementary", "intermediate", "advanced"])
    parser.add_argument("--language", default="auto")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-separation", action="store_true")
    parser.add_argument("--no-lyrics", action="store_true")
    parser.add_argument("--no-melody", action="store_true")
    parser.add_argument("--force", action="store_true", help="输出目录非空时仍清空重跑")
    parser.add_argument(
        "--whisper-model",
        default=None,
        help="歌词模型：medium（默认，准）、small（约快 2 倍，CER 高约 0.10）或 large-v3（最准，很慢）",
    )
    parser.add_argument(
        "--quality",
        default=None,
        choices=["standard", "high", "max"],
        help="质量档：standard/fast（默认）；high 用微调分轨模型（约 4 倍慢）；max 再加多次位移与 large-v3 歌词",
    )
    args = parser.parse_args()

    # ---- 1) 定位项目与解释器（失败时给可执行的修复建议）
    try:
        located = _locate.resolve(__file__)
    except _locate.LocateError as exc:
        return fail({
            "ok": False,
            "error": str(exc),
            "hint": exc.hint,
            "tried": exc.tried,
        }, 2)
    root = Path(located["aitsm_root"])

    # ---- 2) 校验输入
    source = Path(args.audio).expanduser()
    if not source.is_file():
        return fail({"ok": False, "error": f"找不到文件：{source}",
                     "hint": "确认路径存在；Windows 路径请用正斜杠或双反斜杠。"}, 2)
    source = source.resolve()

    # ---- 3) 准备输出目录
    if args.out:
        out_dir = Path(args.out).expanduser()
        if not out_dir.is_absolute():
            out_dir = (Path.cwd() / out_dir)
        out_dir = out_dir.resolve()
        ok, error = _locate.ensure_dir(out_dir)
        if not ok:
            fallback, tried = _locate.default_out_dir(root, __file__)
            return fail({
                "ok": False,
                "error": f"输出目录不可写：{out_dir}",
                "reason": error,
                "hint": "换一个可写目录，或用这些可写位置之一。",
                "writable_candidates": [item["path"] for item in tried if item["writable"]],
                "suggested_out": str(fallback) if fallback else None,
            }, 2)
    else:
        out_dir, tried = _locate.default_out_dir(root, __file__)
        if out_dir is None:
            return fail({
                "ok": False,
                "error": "找不到可写的输出目录",
                "hint": "用 --out 指定一个可写目录，或设置环境变量 AITSM_OUT_DIR。",
                "tried": tried,
            }, 2)

    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            return fail({
                "ok": False,
                "error": f"输出目录非空：{out_dir}",
                "hint": "换一个 --out，或加 --force 覆盖（会删除该目录内容）。",
            }, 2)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 4) 子进程跑 aitsm CLI（进程隔离）
    cmd = [
        located["python"], "-m", "aitsm.cli", "generate", str(source),
        "--out", str(out_dir),
        "--difficulty", args.difficulty,
        "--language", args.language,
    ]
    if args.key:
        cmd += ["--key", args.key]
    if args.capo is not None:
        cmd += ["--capo", str(args.capo)]
    if args.device:
        cmd += ["--device", args.device]
    if args.no_separation:
        cmd += ["--no-separation"]
    if args.no_lyrics:
        cmd += ["--no-lyrics"]
    if args.no_melody:
        cmd += ["--no-melody"]
    if args.whisper_model:
        cmd += ["--whisper-model", args.whisper_model]
    if args.quality:
        cmd += ["--quality", args.quality]

    env = _locate.child_env(root)
    try:
        proc = subprocess.run(
            cmd, cwd=str(root), env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except OSError as exc:
        return fail({
            "ok": False,
            "error": f"无法启动分析子进程：{exc}",
            "hint": f"检查解释器是否可用：{located['python']}",
            "command": cmd,
        }, 3)

    if proc.returncode != 0:
        return fail({
            "ok": False,
            "error": "生成失败",
            "exit_code": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-3000:],
            "stdout_tail": (proc.stdout or "")[-3000:],
            "command": cmd,
            "hint": "先跑 scripts/doctor.py --json 检查环境；上面的 stderr 是原始报错。",
        }, proc.returncode or 1)

    # ---- 5) 读产物
    score_path = out_dir / "score.json"
    if not score_path.is_file():
        return fail({"ok": False, "error": "score.json 未生成",
                     "stdout_tail": (proc.stdout or "")[-2000:],
                     "stderr_tail": (proc.stderr or "")[-2000:]}, 1)
    try:
        score = json.loads(score_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return fail({"ok": False, "error": f"score.json 解析失败：{exc}"}, 1)
    manifest: dict = {}
    if (out_dir / "manifest.json").is_file():
        try:
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
    analysis: dict = {}
    if (out_dir / "analysis.json").is_file():
        try:
            analysis = json.loads((out_dir / "analysis.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            analysis = {}

    # ---- 6) 生成 report.md
    needs_review = collect_needs_review(score, manifest, analysis)
    report = build_report(score, manifest, analysis, out_dir, needs_review, located)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    # ---- 7) 结构化摘要（agent 读这个）
    meta = score.get("meta") or {}
    quality = score.get("quality") or {}
    key = meta.get("key") or {}
    sections = score.get("sections") or []
    bars = score.get("bars") or []
    overall = quality.get("overall")
    if isinstance(overall, (int, float)):
        usable = overall >= 0.70
        verdict = ("good" if overall >= 0.70 else "fair" if overall >= 0.55 else "poor")
    else:
        usable, verdict = False, "unknown"

    summary = {
        "ok": True,
        "out_dir": str(out_dir),
        "aitsm_root": located["aitsm_root"],
        "python": located["python"],
        "files": {
            name: str(out_dir / name)
            for name in ("score.json", "score.musicxml", "score.mid",
                         "report.md", "manifest.json", "analysis.json", "playback.mp3")
            if (out_dir / name).is_file()
        },
        "summary": {
            "title": meta.get("title"),
            "key": key.get("play"),
            "mode": key.get("mode"),
            "key_detected": key.get("detected"),
            "key_confidence": key.get("confidence"),
            "capo": meta.get("capo"),
            "tempo": meta.get("tempo"),
            "time_signature": meta.get("time_signature"),
            "bars": len(bars),
            "sections": len(sections),
            "difficulty": meta.get("difficulty"),
            "chord_events": sum(len(bar.get("chords") or []) for bar in bars),
            "melody_notes": sum(len(bar.get("vocal") or []) for bar in bars),
            "lyric_notes": sum(
                1 for bar in bars for note in (bar.get("vocal") or []) if note.get("lyric")
            ),
            "quality_overall": overall,
            "quality_stages": quality.get("stages") or {},
            "duration_sec": (score.get("meta") or {}).get("source", {}).get("duration_sec"),
            "elapsed_sec": manifest.get("duration_sec"),
        },
        "usable": usable,
        "verdict": verdict,
        "needs_review": needs_review,
        "needs_review_counts": {
            level: sum(1 for item in needs_review if item["severity"] == level)
            for level in ("high", "medium", "low")
        },
        "sections": [
            {
                "label": section.get("label"),
                "start_bar": section.get("start_bar"),
                "end_bar": section.get("end_bar"),
                "start_sec": section.get("start_sec"),
                "end_sec": section.get("end_sec"),
                "pattern_id": section.get("pattern_id"),
            }
            for section in sections
        ],
        "report_path": str(out_dir / "report.md"),
        "view_command": f'python scripts/serve.py --score "{out_dir / "score.json"}"',
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # 兜底：绝不给 agent 一堆 traceback
        _locate.force_utf8()
        print(json.dumps({
            "ok": False,
            "error": "skill 脚本内部错误",
            "traceback": traceback.format_exc()[-3000:],
        }, ensure_ascii=False, indent=2))
        raise SystemExit(1)
