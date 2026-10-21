"""用带 ground truth 的合成歌曲客观评估扒谱准确率。

为什么需要客观评估
------------------
真实歌曲没有标准答案，"听着像不像"无法量化，也无法判断一次算法改动
是变好还是变坏。合成歌曲的 BPM、拍号、调性、和弦、段落都是写下去的，
因此可以逐项打分：

    BPM 误差 %       —— 速度准确度
    key_hit          —— 调性是否命中（调名 + 大小调）
    sig_hit          —— 拍号是否命中
    chord_accuracy   —— 和弦命中率（按时间加权，不是按数量）
    section_accuracy —— 段落边界准确度

和弦命中率的计算方式
--------------------
按 0.25 秒步长沿时间轴采样，每个采样点分别取 ground truth 与识别结果里
**当时生效的和弦**，比较根音 + 和弦性质（大三/小三/七和弦等归到同一类）。
这样错一个和弦不会因为"后面全对"而被稀释，也不会因为"和弦事件数量不同"
而产生偏差。

用法::

    python -m aitsm.testsupport.evaluate --analysis work/synth/waltz/analysis.json \
        --truth work/synth/waltz.groundtruth.json
    python -m aitsm.testsupport.evaluate --all work/synth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

NOTE_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# 和弦性质归并：评估时把写法差异归到同一族，避免"maj7 认成 maj"被算全错
QUALITY_FAMILY = {
    "": "maj", "maj": "maj", "M": "maj", "major": "maj",
    "m": "min", "min": "min", "minor": "min",
    "7": "dom7", "dom7": "dom7",
    "maj7": "maj7", "M7": "maj7", "Δ": "maj7",
    "m7": "m7", "min7": "m7",
    "dim": "dim", "dim7": "dim",
    "aug": "aug", "+": "aug",
    "sus2": "sus", "sus4": "sus", "sus": "sus",
    "6": "6", "m6": "m6",
    "9": "ext", "add9": "ext", "madd9": "ext", "5": "power",
}


def parse_symbol(symbol: str) -> tuple[int, str, int | None] | None:
    """解析和弦符号 -> (根音 pc, 性质族, 低音 pc 或 None)。"""
    text = (symbol or "").strip().replace("♯", "#").replace("♭", "b")
    if not text:
        return None
    # 拆斜杠低音
    bass_pc = None
    if "/" in text:
        head, _, bass = text.partition("/")
        text = head.strip()
        bass = bass.strip()
        if bass in NOTE_PC:
            bass_pc = NOTE_PC[bass]
    # 根音：字母 + 可选升降号
    if not text or text[0] not in "ABCDEFG":
        return None
    root = text[0]
    rest = text[1:]
    while rest and rest[0] in "#b":
        root += rest[0]
        rest = rest[1:]
    if root not in NOTE_PC:
        return None
    quality = QUALITY_FAMILY.get(rest.strip(), "other")
    return NOTE_PC[root], quality, bass_pc


def _active_at(events: list[dict[str, Any]], t: float, start_key: str, end_key: str) -> dict[str, Any] | None:
    """取时间点 t 处生效的事件（按 start 排序后线性扫描即可）。"""
    current = None
    for event in events:
        start = float(event.get(start_key) or 0.0)
        end = float(event.get(end_key) or 0.0)
        if start <= t < end:
            return event
        if start > t:
            break
        current = event
    return current if current and float(current.get(end_key) or 0) <= t else None


def score_analysis(analysis: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    """对一份 analysis.json 打分。"""
    report: dict[str, Any] = {"profile": truth.get("profile"), "title": truth.get("title")}

    # ---- BPM ----
    truth_bpm = float(truth.get("bpm") or 0)
    got_bpm = float((analysis.get("beats") or {}).get("bpm") or 0)
    bpm_error = abs(got_bpm - truth_bpm) / truth_bpm * 100 if truth_bpm else 100.0
    # 半速/倍速也算命中（音乐上等价），但要记录
    ratio = got_bpm / truth_bpm if truth_bpm else 0
    octave_note = ""
    if 0.45 <= ratio <= 0.55:
        octave_note = "识别为半速"
    elif 1.9 <= ratio <= 2.1:
        octave_note = "识别为倍速"
    report["bpm"] = {
        "truth": truth_bpm,
        "got": round(got_bpm, 2),
        "error_pct": round(bpm_error, 2),
        "hit": bpm_error <= 3.0 or bool(octave_note),
        "note": octave_note,
    }

    # ---- 拍号 ----
    truth_sig = list(truth.get("time_signature") or [])
    got_sig = (analysis.get("beats") or {}).get("time_signature") or ""
    got_sig_list = [int(p) for p in str(got_sig).split("/") if p.isdigit()]
    report["time_signature"] = {
        "truth": truth_sig,
        "got": got_sig_list or got_sig,
        "hit": truth_sig == got_sig_list,
    }

    # ---- 调性 ----
    truth_key = truth.get("key") or {}
    got_key = analysis.get("key") or {}
    truth_pc = NOTE_PC.get(truth_key.get("tonic", ""), None)
    got_pc = NOTE_PC.get(got_key.get("tonic", ""), NOTE_PC.get(got_key.get("display_name", "")[:2], None))
    if got_pc is None and got_key.get("tonic_pc") is not None:
        got_pc = int(got_key["tonic_pc"])
    # 大小调命中判定：调名相同且调式相同；或相对大小调（音乐上近义）算"部分命中"
    exact = truth_pc == got_pc and truth_key.get("mode") == got_key.get("mode")
    relative = truth_pc is not None and got_pc is not None and (
        (got_pc - truth_pc) % 12 == 3 or (truth_pc - got_pc) % 12 == 3
    )
    report["key"] = {
        "truth": f"{truth_key.get('tonic')} {truth_key.get('mode')}",
        "got": f"{got_key.get('tonic')} {got_key.get('mode')}",
        "hit": exact,
        "relative_only": bool(relative and not exact),
    }

    # ---- 和弦命中率（按时间加权）----
    truth_chords = truth.get("chords") or []
    got_chords = sorted(analysis.get("chords") or [], key=lambda c: float(c.get("start") or 0))
    duration = float(truth.get("duration") or 0)
    # 采样策略：在每个「真值和弦区间」内部均匀取点，并跳过边界两侧的
    # EDGE_MARGIN 秒。原因：识别结果的边界通常与真值差几十毫秒（量化、
    # 起音检测延迟），若在网格上固定取样，会把这些几厘米的偏移记成整段
    # 和弦错误，得出荒谬的低分。真正要衡量的是「这段时间弹的是哪个和弦」。
    EDGE_MARGIN = 0.12
    STEP = 0.25
    total = 0
    matched = 0
    root_only = 0
    detail_misses: list[dict[str, Any]] = []
    for chord in truth_chords:
        seg_start = float(chord.get("start_sec") or 0.0)
        seg_end = float(chord.get("end_sec") or 0.0)
        lo = seg_start + EDGE_MARGIN
        hi = seg_end - EDGE_MARGIN
        if hi <= lo:
            # 极短和弦（小于 2×margin）取中点
            lo = hi = (seg_start + seg_end) / 2
        t = lo
        expected_symbol = f"{chord['root']}{'' if chord['quality'] == 'maj' else chord['quality']}"
        expected_parsed = parse_symbol(expected_symbol)
        while t <= hi:
            actual = _active_at(got_chords, t, "start", "end")
            total += 1
            if actual is not None and expected_parsed:
                act = parse_symbol(actual.get("symbol") or "")
                if act:
                    if expected_parsed[0] == act[0] and expected_parsed[1] == act[1]:
                        matched += 1
                    elif expected_parsed[0] == act[0]:
                        root_only += 1
                        if len(detail_misses) < 8:
                            detail_misses.append({"t": round(t, 1), "expected": expected_symbol,
                                                  "got": actual.get("symbol")})
                    elif len(detail_misses) < 8:
                        detail_misses.append({"t": round(t, 1), "expected": expected_symbol,
                                              "got": actual.get("symbol")})
                elif len(detail_misses) < 8:
                    detail_misses.append({"t": round(t, 1), "expected": expected_symbol,
                                          "got": actual.get("symbol")})
            elif len(detail_misses) < 8:
                detail_misses.append({"t": round(t, 1), "expected": expected_symbol, "got": None})
            t += STEP
    report["chords"] = {
        "exact_accuracy": round(matched / total, 4) if total else 0.0,
        "root_accuracy": round((matched + root_only) / total, 4) if total else 0.0,
        "samples": total,
        "misses": detail_misses,
    }

    # ---- 段落边界准确度 ----
    truth_sections = truth.get("sections") or []
    got_sections = sorted(analysis.get("sections") or [], key=lambda s: float(s.get("start") or 0))
    truth_starts = [round(float(s["start_sec"]), 1) for s in truth_sections]
    got_starts = [round(float(s.get("start") or 0), 1) for s in got_sections]
    # 边界命中：在 ±1.5 秒内找到对应边界
    hits = 0
    for want in truth_starts:
        if any(abs(want - g) <= 1.5 for g in got_starts):
            hits += 1
    report["sections"] = {
        "truth_count": len(truth_sections),
        "got_count": len(got_sections),
        "boundary_hit_rate": round(hits / len(truth_starts), 3) if truth_starts else 0.0,
        "truth_bounds": truth_starts,
        "got_bounds": got_starts,
    }

    # ---- 综合 ----
    weights = {"bpm": 0.15, "time_signature": 0.10, "key": 0.15, "chords": 0.40, "sections": 0.20}
    score = (
        weights["bpm"] * (1.0 if report["bpm"]["hit"] else 0.0)
        + weights["time_signature"] * (1.0 if report["time_signature"]["hit"] else 0.0)
        + weights["key"] * (1.0 if report["key"]["hit"] else (0.5 if report["key"]["relative_only"] else 0.0))
        + weights["chords"] * report["chords"]["exact_accuracy"]
        + weights["sections"] * report["sections"]["boundary_hit_rate"]
    )
    report["overall"] = round(score, 4)
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [f"# {report.get('title')} ({report.get('profile')})", ""]
    bpm = report["bpm"]
    flag = "OK" if bpm["hit"] else "FAIL"
    lines.append(f"BPM       {bpm['truth']:.1f} -> {bpm['got']:.1f}  误差 {bpm['error_pct']:.1f}%  [{flag}] {bpm['note']}")
    sig = report["time_signature"]
    lines.append(f"拍号      {sig['truth']} -> {sig['got']}  [{'OK' if sig['hit'] else 'FAIL'}]")
    key = report["key"]
    key_flag = "OK" if key["hit"] else ("PARTIAL(关系调)" if key["relative_only"] else "FAIL")
    lines.append(f"调性      {key['truth']} -> {key['got']}  [{key_flag}]")
    ch = report["chords"]
    lines.append(f"和弦      精确 {ch['exact_accuracy']*100:.1f}%  含根音 {ch['root_accuracy']*100:.1f}%  "
                 f"(采样 {ch['samples']})")
    if ch["misses"]:
        for miss in ch["misses"][:4]:
            lines.append(f"          {miss['t']:6.1f}s  期望 {miss['expected']:<8} 得到 {miss['got']}")
    se = report["sections"]
    lines.append(f"段落      边界命中 {se['boundary_hit_rate']*100:.1f}%  "
                 f"({se['truth_count']} -> {se['got_count']} 段)")
    lines.append("")
    lines.append(f"综合得分  {report['overall']*100:.1f} / 100")
    return "\n".join(lines)


def evaluate_pair(analysis_path: Path, truth_path: Path) -> dict[str, Any]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    return score_analysis(analysis, truth)


def main() -> int:
    parser = argparse.ArgumentParser(description="客观评估扒谱准确率")
    parser.add_argument("--analysis", help="analysis.json 路径")
    parser.add_argument("--truth", help="groundtruth.json 路径")
    parser.add_argument("--all", help="批处理根目录，自动配对 <name>/analysis.json 与 <name>.groundtruth.json")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    for stream_name in ("stdout", "stderr"):
        reconfigure = getattr(getattr(sys, stream_name, None), "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    reports: list[dict[str, Any]] = []
    if args.all:
        root = Path(args.all)
        for truth_path in sorted(root.glob("*.groundtruth.json")):
            name = truth_path.name.replace(".groundtruth.json", "")
            analysis_path = root / name / "analysis.json"
            if not analysis_path.is_file():
                print(f"跳过 {name}：找不到 {analysis_path}", file=sys.stderr)
                continue
            reports.append(evaluate_pair(analysis_path, truth_path))
    elif args.analysis and args.truth:
        reports.append(evaluate_pair(Path(args.analysis), Path(args.truth)))
    else:
        parser.error("需要 --analysis/--truth 或 --all")

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        for report in reports:
            print(format_report(report))
            print()
        if len(reports) > 1:
            avg = sum(r["overall"] for r in reports) / len(reports)
            print(f"=== {len(reports)} 首平均综合得分：{avg*100:.1f} / 100 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
