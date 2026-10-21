# 质量判读指南

本文件说明如何读 `score.json` 的 `quality` 字段、`report.md`、以及 `generate.py` 输出的 `needs_review` 数组，判断结果是否可用、哪些部分需要用户人工复核。

## 一句话原则

**置信度是相对质量分，不是概率。** 对所有用户说「置信度 0.60」，绝不说「准确率 60%」。

## 核心字段

| 字段 | 范围 | 含义 |
| --- | --- | --- |
| `quality.overall` | 0–1 | 综合分：`beats` / `key` / `chords` / `melody` / `lyrics` 五项等权平均（**不含** `lyric_match`） |
| `quality.stages.beats` | 0–1 | 节拍：拍点置信度 + 覆盖率 + downbeat 完备度 |
| `quality.stages.key` | 0–1 | 调性：chroma 与和弦两路融合 |
| `quality.stages.chords` | 0–1 | 和弦：平均置信度 0.6 + 时间覆盖 0.25 + 调内占比 0.15 |
| `quality.stages.melody` | 0–1 | 旋律：逐帧稳定度 + 能量 + pyin 概率加权 |
| `quality.stages.lyrics` | 0–1 | 歌词：whisper avg_logprob 与行数 |
| `quality.stages.lyric_match` | 0–1 | 旋律音符与歌词音节的对齐率（1 − 未匹配/总数） |

> `overall` 是五项**等权**平均，因此歌词低会把总分拖下来。判断谱面本身能不能弹，看 `beats` / `key` / `chords` 三项更准。

## 阈值与应对

| 指标 | 正常 | 需要提醒用户 | 应视为失败 |
| --- | --- | --- | --- |
| `overall` | ≥ 0.70 | 0.55–0.70 | < 0.55 |
| `beats` | ≥ 0.75 | 0.55–0.75 | < 0.55（速度可能判错，整谱时值全错） |
| `key` | ≥ 0.65 | 0.45–0.65 | < 0.45（调号可能错，和弦全偏） |
| `chords` | ≥ 0.60 | 0.45–0.60 | < 0.45（和弦不可信） |
| `melody` | ≥ 0.60 | 0.40–0.60 | < 0.40 |
| `lyrics` | ≥ 0.50 | 0.30–0.50 | < 0.30 |
| `lyric_match` | ≥ 0.60 | 0.40–0.60 | < 0.40（大量音符没有词） |
| 段落数 | 5–12 | 3–4 或 13–16 | < 3（未切分）或 > 16（过碎） |
| 单段长度 | ≥ 4 小节 | 2–4 小节 | < 2 小节（碎片） |
| 旋律音域 | MIDI 55–84 | 50–54 / 85–88 | 超出 48–90（八度错） |
| 歌词行数 | ≥ 8 | 1–7 | 0（无歌词或有演唱但未识别） |

## `needs_review` 数组（agent 主要看这个）

`generate.py` 已把散落在 `quality.warnings` / `analysis.json` / `manifest.json` 的问题汇成统一结构，**并按严重程度排序**（`high` → `medium` → `low`，同级按小节号/时间升序）。

```jsonc
{
  "severity": "high",           // high 必须告知 | medium 建议复核 | low 信息性
  "code": "chords.bar_low_confidence",
  "what": "2 个小节的和弦置信度低于 0.55",
  "where": "第 32–33 小节（主歌 1）",   // 已带小节号与段落名，可直接转述
  "bars": [32, 33],
  "start_sec": 74.2,
  "end_sec": 79.1,
  "action": "在网页里点这些小节对照原曲听一遍，必要时重跑或人工改"
}
```

`where` 字段已经排好人话（`第 12–18、32 小节（副歌、主歌 2）`），可以直接抄进交付说明。

`needs_review_counts` 给出三档计数，适合放在交付说明开头。

## 告警码速查

### high（必须告知用户）

| code | 含义 | 该做什么 |
| --- | --- | --- |
| `separate.failed` / `separate.degraded` | 分轨失败，用混音降级 | **必须告知**：旋律与和弦质量显著下降 |
| `lyrics.hallucination` | 检测到疑似编造歌词 | **高度警惕**：整段标为待核对，不要当真 |
| `lyrics.none` / `lyrics.empty` / `lyrics.no_vocals` | 没有歌词 | 明确说明只有旋律没有词 |
| `melody.none` | 没有旋律 | 明确说明只有和弦进行 |
| `key.invalid` | 调性无效 | 说明并建议 `--key` 重跑 |
| `chords.no_chords` / `chords.missing_bars` | 没有和弦 | 说明哪些小节是留白 |
| `score.schema` / `score.validation` / `analysis.schema` | 产物结构告警 | 读具体条目转述，不要忽略 |

### medium（建议复核）

| code | 含义 | 该做什么 |
| --- | --- | --- |
| `key.relative_ambiguity` | 关系大小调难区分 | 说明按大调标注；用户知道是小调就用 `--key` |
| `key.chroma_only` | 仅凭 chroma 定调 | 提醒调性可能不准 |
| `key.capo_high` | capo 品数偏高 | 说明并给出 `--capo` 用法 |
| `key.non_guitar_friendly` | 原调对吉他不够友好 | 说明已自动选替代调 |
| `chords.low_confidence` / `chords.bar_low_confidence` | 和弦置信度低 | 列出小节号，建议试听核对 |
| `lyrics.low_confidence` / `lyrics.line_low_confidence` | 歌词置信度低 | 提醒校对，给出时间区间 |
| `lyrics.missing` / `lyrics.coverage` | 有旋律但无词 | 说明留空是设计，不是遗漏 |
| `lyrics.skipped` | 跳过歌词识别 | 说明原因 |
| `structure.coarse` | 段落偏长未细分 | 信息性，不影响演奏 |
| `melody.range` | 音域超出建议范围 | 可能八度错，建议换调或 `--transpose` |

### low（信息性）

`audio.resampled`、`bars.pickup`、`bars.tempo_changes`、`melody.silent_bars`、`melody.span`、`sections.merged`、`sections.kind_fixed`、`structure.short_sections`

这些通常不必单独讲，除非用户主动问。

## 幻觉歌词的识别

whisper 在音乐场景容易产生三类幻觉：

1. **复读 prompt**：把 `initial_prompt` 当歌词继续唱，例如 "The lyrics of this song are…"。
   → 已通过不给英文歌加 prompt + `is_prompt_echo` 过滤修复。
2. **重复循环**：整段反复输出同一句，尤其副歌和声密集处。
   → 已通过 `condition_on_previous_text=False` + `drop_repetition_hallucinations` 处理。
3. **凭空句子**：与歌曲内容无关的完整句子。
   → 需要与 LRC 或用户提供的歌词比对；单段 CER > 0.9 高度可疑。

**发现幻觉时**：不要静默删除后宣称正常。写明「第 X 段歌词疑似识别错误，建议以原曲为准」，或把该段留空并标注。

## 何时建议用户手工修正

| 问题 | 处理 |
| --- | --- |
| 调性判错 | `--key` 指定重跑 |
| capo 不适合嗓音 | `--capo` 指定，或在网页里点 CAPO |
| 和弦低置信度小节 | 网页里对照原曲判断 |
| 歌词错 | 目前无前端歌词编辑器，建议以原曲为准，或导出 MusicXML 后用 MuseScore 改 |
| 旋律八度错 | `--transpose 12` / `--transpose -12` 重跑 |

## 不要做的事

- 不要因为 `overall` 低就拒绝交付——低质量 + 明确说明，比不给结果更有用。
- 不要把 `confidence` 当概率报给用户。
- 不要隐去 `needs_review` 里的小节号 / 时间定位。
- 不要在没跑成功的情况下描述产物内容。
- 不要声称「MusicXML 已在 MuseScore 验证」——本机没有 MuseScore，只做过 XSD 校验。
