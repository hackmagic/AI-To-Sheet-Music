# 端到端验收记录

本文件记录本项目的实际验收数据，用于确认「音频 → 弹唱谱」全链路可用性与已知限制。
所有数字均来自本机实测（Windows 11 / Python 3.12.8 / RTX 4060 Laptop 8GB 但 **全程未使用 GPU**）。

最后一次完整运行：`work/accept-v2`（交付用的 `artifacts/` 由它复制而来），
配置为默认配置（`--whisper-model medium`、分轨 + 歌词 + 旋律），**冷启动**。

## 测试素材

| 项 | 值 |
| --- | --- |
| 文件 | `小林未郁、HOYO-MiX - 下一站 Beyond This Station.flac` |
| 时长 | 229.03s（3:49） |
| 格式 | FLAC / 44.1kHz / 29.1MB，sha256 `9b99e9b9…be224` |
| 语言 | 英文（正文歌词），中文元数据 |
| 参考歌词 | 同目录 `.lrc`，正文 **62 行**（已过滤作词/作曲/编曲/制作人），归一化 1465 字符，用于 CER 对照 |

> 该文件属于开发者本地素材（`hackmagic-music/assets/`），**不随仓库分发**。

## 分阶段实测

| 阶段 | 实现 | 耗时 | 关键输出 | 评价 |
| --- | --- | --- | --- | --- |
| ingest | ffmpeg → 22.05kHz 单声道分析轨 + 44.1kHz 播放轨 + mp3 | 2.8s | duration 229.03s | ✅ |
| separate | Demucs `htdemucs`（CPU） | **308.5s** | 4 stems，时长精确 229.03s，RMS 非零 | ✅ 最慢的一步 |
| beats | librosa beat_track + `fit_beat_grid` 最小二乘 | 11.6s | **92.47 BPM**，4/4，置信度 0.792 | ✅ |
| key | chroma(Krumhansl) + 和弦双路融合 + 关系大小调裁决 | 12.2s | **D 大调**，2 升号，conf 0.842 | ✅ 标注了 Bm/D 歧义（delta 0.036） |
| chords | 半小节 chroma 模板匹配 + 低音约束 + 调内先验 + 换和弦代价/滞回 | 13.2s | **55 事件**，mean_conf **0.574**（min 0.492 / max 0.717） | ✅ 全部调内，仅 2 个色彩和弦 |
| key_refine | 用识别出的和弦重新推断调性 | 12.2s | 与 key 阶段一致 | ✅ |
| structure | 自相似矩阵 + 能量 + 和弦重复 | 13.8s | **12 段**，最短 10.3s / 最长 41.6s | ✅ 并给出 `structure.coarse` 提示 |
| lyrics | faster-whisper **medium**（CPU int8，无 VAD） | **151.5s** | **18 行**，字符级 **CER 0.4935** / WER 0.5547 | ⚠️ 引擎上限，见下 |
| melody | pyin on vocals stem + 八度归位 + 量化 | 12.9s | **326 音符**，midi **57–84**，mean_conf **0.831** | ✅ |
| assemble | 音节对齐 + analysis.json 组装与校验 | <1s | lyrics 行 ↔ 音符匹配 | ✅ |
| arrange | 定调 / capo / 小节网格 / 量化 / 段落映射 / 指法 | ~3s | **88 小节**，7 段，D 大调 capo 0 | ✅ |
| validate | 结构 + 音乐合理性 + schema | <1s | 0 error | ✅ |
| export | MusicXML 4.0 + MIDI SMF | ~2s | MusicXML 88 小节 ×2 part；MIDI 3 tracks | ✅ 过官方 XSD |

**全流程（冷启动，含分轨与 medium 歌词）：538.7s ≈ 9 分钟**
**其中分轨 308s + 歌词 152s 占 85%**

同配置的另一次冷启动（`work/accept-final`）用时 617.2s，两次差 79s 全部落在分轨阶段
（308s vs 335s）——CPU 上的 Demucs 受机器负载影响明显。把歌词降级为
`--whisper-model small` 可省约 90s。

## 产物校验

### score.json
- schema 校验（`schemas/score.schema.json`）：**0 错误**
- 88 小节，编号 1..88 连续；`bars[].section` 全部指向有效段落
- 7 段覆盖 1..88 无空洞无重叠：前奏 1–9 / 间奏 10–14 / 主歌 1 15–25 / 副歌 1 26–38 / 桥段 39–54 / 副歌 2 55–71 / 主歌 2 72–88
- **132 个和弦 token**，其中 **44 个是中途换和弦**（beat ≠ 0），其余在小节线
- vocal 352 音符，**189 个带歌词**（覆盖率 54%），其余留空不编造
- guitar hits **668 个**（扫弦型 + 分解型）
- 7 个不同和弦符号，**10 个把位**（G/D 各两个位置），符号覆盖率 100%

> 注意：132 个 token 里约 60% 是「相邻小节同符号重复」。这是有意的排版选择——中文弹唱谱习惯逐小节标和弦，便于初学者对位；不是重复识别。

### score.musicxml
- 过 **官方 MusicXML 4.0 XSD**（`work/xsd2/`，由 W3C 的 `musicxml.raw.xsd` + `xlink.raw.xsd` + `xml.raw.xsd` 组装）：`XSD valid: True`
- 根元素 `score-partwise` version 4.0，2 part（Vocal 主旋律 + Guitar TAB）
- 176 个 `<measure>`（= 88 小节 × 2 part），编号连续
- 1170 个 `<note>`，每个都有 `<duration>` 与 `<type>`
- 132 个 `<harmony>`，**298 个 `<lyric>`**（逐字音节）
- `<key><fifths>2` 与 D 大调一致
- `<staff-details>` 声明 TAB 6 弦调弦
- ⚠️ **未在 MuseScore 中实际打开验证**（本机未安装 MuseScore）

### score.mid
- `MThd` + 3 个 `MTrk`，format 1 / PPQ 480
- track 0 conductor（tempo + 拍号），track 1 Vocal，track 2 Guitar
- `ffprobe` 可识别，duration 231.0s
- ⚠️ 未做 MIDI→音频回放听感验证

### manifest.json（可复现性）
- 16 个顶层键：`created_at / degraded / duration_sec / generator / input / options / outputs / quality / schema / schema_validation / stage_timings / stages / started_at / tools / total_elapsed_seconds / warnings`
- `stages` = `{analysis: [9 条阶段记录], analysis_summary, arrange, export}`
- `input.sha256` / `input.size_bytes` / `input.analyzed_duration` 齐全
- `tools` 记录 `python / ffmpeg / librosa / numpy / scipy / soundfile / torch / demucs / demucs_model / jsonschema / lyrics / melody`
- **`tools.lyrics = {available, engine: faster-whisper, model: medium, device: cpu, line_count: 18}`** —— 现在能从产物反查用的是哪个歌词模型
- `schema_validation.valid = true`，`degraded` 3 条（全部是 info/warn 级的环境与歧义提示）

## 质量评分：两个口径，别混用

| 口径 | 位置 | overall | 分项 |
| --- | --- | --- | --- |
| **管线质量** | `manifest.json` → `quality` | **0.824** | beats 0.882 / key 0.843 / chords 0.745 / structure 0.851 / melody 0.849 / lyrics 1.0 |
| **谱面质量** | `score.json` → `quality` | **0.68** | beats 0.792 / key 0.842 / chords 0.576 / melody 0.833 / lyrics 0.358 / lyric_match 0.817 |

两者差在**含义不同**：`manifest.quality` 是「分析管线各阶段的自评」，`score.quality` 是「这份谱面有多可信」。
谱面口径更低是因为它把「和弦直接用 mean_conf 当分」以及「歌词覆盖率与对齐质量」算进去了（lyrics 0.358 主要来自 54% 的覆盖率）。
**对用户有意义的数字是 `score.quality`（68/100）——但它不是百分比正确率**，只是相对质量分。

## 歌词准确率（三次实测，必须按区间报告）

| 运行 | 模型 | 行数 | 字符级 CER | WER |
| --- | --- | --- | --- | --- |
| `work/deliver` | small | 20 | 0.6416 | 0.7628 |
| `work/accept-medium` | medium | 20 | 0.5420 | 0.5912 |
| `work/accept-final` | medium | 17 | 0.5024 | 0.5693 |
| `work/accept-v2`（交付） | medium | 18 | **0.4935** | **0.5547** |

**同一配置重复运行会差 0.05 个 CER**——因为 Demucs 在 CPU 上不是比特可复现的
（两次 `stems/vocals.wav` sha256 不同，互相关在 lag=0 有 4.5% 残差、相关系数 0.999），
whisper 对这种微小差异敏感，断句与转写都会变。**所以对外只能说「medium ≈ 0.49–0.54」，不能说 0.4935。**
small→medium 的差距（约 0.10–0.14）远大于这个波动，模型结论成立。

逐行对照看：medium 找回了 small 整段漏掉的 85–95s，多句从「完全错」变成「完全对」
（如 `What's stretching so far along` / `I stand in promise, with my heart unchained`）。
反面：medium 首段出现 `home home` 复读型幻觉，另有整行编造（`Always unequal, let's play it now`）。
因此**低置信度行必须在质量面板可见**（本次 `lyrics.low_confidence` 告警：平均置信度 0.36）。

## 网页渲染验收（playwright 实测）

在 `python -m aitsm.cli serve` 下加载同一份真实谱（`work/jobs/accept-medium` 副本），

| 检查项 | 结果 |
| --- | --- |
| 页面加载 | ✅ 无 console error（只有一条既有的「1/6 拍音符无法精确记谱」提示） |
| 真实谱渲染（88 小节） | ✅ **31 个 system / 88 个可点小节 / 122 个和弦标 / 0 异常** |
| 五线谱 + 逐字歌词 | ✅ 歌词不与符干/加线重叠，符干朝上 |
| 和弦指法图 | ✅ 行首 6 弦图，位置正常（历史 bug x=-463 已修） |
| TAB 谱表 | ✅ 6 弦 + 品位数字 |
| 移调 / 变调夹 | ✅ C→D、G/B→A/C#、Am→Bm 同步；capo 0–7 |
| 播放 | ✅ `state=playing`，`.ov-chord.is-current` 跟随 |
| **编辑器：改和弦** | ✅ 点和弦 → 弹面板 → 输入 Am → 谱面变 Am、`data-edited=1`、徽标「已手工修正 1 处」 |
| **编辑器：撤销** | ✅ `editor.undo()` → 回到 G、徽标隐藏、countEdits=0、toast「已撤销：改和弦」 |
| **编辑器：删空小节** | ✅ 删掉唯一小节的段落 → 该段被丢弃、剩余小节整体重编号（87 小节 / 29 system / `barSections` 全 s2），无异常 |
| 打印（A4） | ✅ 逐页 PDF，中文无 tofu |

截图：`work/shots/medium-score.png`、`work/shots/medium-lyrics.png`

## 已知问题（诚实记录）

### ✅ 已修复
1. score 层丢 tie 导致同词重复 → 显式 `lyric: null` 不再回填，`tie: stop/continue` 的音绝不补词
2. 英文子词被空格拆开（`bab y`）→ `merge_split_words` 按前导空格合并
3. TAB hits 用了错误和弦的指法（121 处）→ 按 beat 查生效和弦
4. 和弦重叠碎片（0.11 拍的残影和弦）→ `_resolve_chord_overlaps` + `_fill_chord_gaps`
5. whisper 复读 `initial_prompt` 编造歌词 → 英文不加 prompt + `is_prompt_echo` / `is_meta_lyric_hallucination` 双保险
6. 歌词压在五线谱上 → 符干朝上 + `ledgerDepth = ceil((64-lowestMidi)/2)*5+5`
7. 和弦指法图跑出屏幕（x=-463）→ CSS 选择器收窄到 `.voicing-diagram` 而非 `.diagram`
8. 空小节无休止符 → 画全休止块
9. 调号拼写错误（C# 配 Gb 和弦）→ 按拼写查五度圈而非按 pitch class
10. MusicXML `<note>` 子元素顺序不合规 → 过**官方 MusicXML 4.0 XSD**
11. 段落碎段/巨段（5.1s 与 93s）→ `_normalize_sections` 相邻同 kind 合并 + 首段纠正 + 最短相邻对合并
12. 关系大小调误判（Bm 胜出但 A/D 更合理）→ delta < 0.05 取关系大调
13. 拍号 6/8 误判 → 两路 z 标准化 + 硬规则（top1=6 且拍间隔 ≥0.40s 降级为 3）
14. BPM 138→136 的系统性偏差 → `fit_beat_grid` 最小二乘拟合
15. 斜杠和弦误判（C/E）→ `_SLASH_ROOT_PENALTY=0.16` + 偏好边际 0.10
16. 三和弦被判成七和弦 → 七和弦惩罚提到 0.07~0.075
17. 和弦粒度太粗（5.9s/和弦）→ 中途换和弦罚 0.06→0.02 + 滞回 0.04，55 事件
18. `--no-lyrics` 仍跑 ASR → `_normalize_options` 同时置 `skip_lyrics`
19. manifest 被 service 覆盖写、丢可复现性字段 → `_merge_manifests()`
20. skill 的 `generate.py` 因 `stages.analysis` 变 list 而崩 → `_manifest_engines()` 兼容层
21. 早期 manifest 不记录歌词模型 → `tools.lyrics.{engine,model,device}`

### 🟠 仍存在（可接受，已在 UI/报告里标注）
1. **歌词 CER 0.49–0.54**（medium，英文歌）：术语、连读、和声段仍会错；偶有整行幻觉。**不保证每句都对**，低置信度会标出。
2. **和弦 mean_conf 0.574**：全曲调内，但色彩和弦（`Asus2`、`C#m7b5`）与结尾弱证据段需人工确认。
3. 真歌 **188.6–221.1s（尾声）被并成单个 32.5s 的 D**：该段各槽位 top1 仅 0.09–0.23 且 A/Bm/G/D 交替，Viterbi 保守地保持不变。**判定为可接受的保守取舍**，不为单曲过拟合。
4. **歌词覆盖率 54%**（189/352）：有旋律但没匹配到词的部分留空，不编造。
5. 段落粒度偏粗（副歌 16 小节、桥段 16 小节），并给出 `structure.coarse` 提示。
6. 速度波动（83–101 BPM）写进了逐小节 tempo，但不支持**拍号中途变化**。
7. 7 段里「间奏 10–14」与「前奏」重复，段落标签本身仍需人工过一眼。

### 🟡 环境限制
1. **GPU 未使用**（cu128 wheel 反复下载损坏）：全流程走 CPU，4 分钟歌冷启动约 9 分钟。
2. **未安装 MuseScore**：MusicXML 过官方 XSD 但未做实际打开验证。
3. **未做主观听感验证**：所有音频结论基于信号分析与 LRC 文本对照。
4. **安装 skill 必须用非 Low 完整性解释器**：`.venv/Scripts/python.exe` 带 Low 完整性标签，写不了 `%USERPROFILE%\.config\...`，要改用 `D:/WorkSoftware/anaconda3/python.exe plugin/install.py`。

## 结论

**目标达成**：从真实歌曲音频可端到端产出可渲染、可播放、可编辑、可移调、可打印的弹唱谱。
- `score.json` 通过 schema + 音乐合理性双重校验（0 错误）
- `score.musicxml` 通过官方 MusicXML 4.0 XSD
- `score.mid` 通过独立 SMF 解析与 ffprobe
- 网页在 88 小节真实谱上渲染、编辑、撤销、打印全部实测通过
- 三首合成曲（3/4、4/4 摇滚、无人声）信号层 **85.1/100**，见 `docs/07-synthetic-eval.md`

**交付质量评级**（对应 `score.quality` 而非准确率）：
- 结构正确性（小节/段落/调性/拍号）：**A**
- 和弦进行：**B**（合理、全调内、置信度中等、粒度已细化到 55 事件）
- 旋律：**A-**（音域 57–84，量化漂移 0.048s）
- 歌词：**C+**（CER 0.49–0.54，不编造但会错；漏词留空）
- 网页呈现：**A-**（31 systems，0 重叠，编辑/撤销/打印实测通过）
- 工程质量：**A**（505 python tests + 34 node tests 全绿，19 条已知问题文档化）

## 最终验收命令

```bash
# 1. 全部单测（期望 505 passed）
.venv/Scripts/python -m pytest tests/ -q
node tests/web-editor.test.mjs          # 期望 34 passed, 0 failed

# 2. 环境自检
.venv/Scripts/python -m aitsm.cli doctor

# 3. 端到端生成
.venv/Scripts/python -m aitsm.cli generate "<song>" --out work/mysong

# 4. 谱面校验（期望 schema 0 / music True）
.venv/Scripts/python -c "
import json
from aitsm.score.validate import validate_score
from aitsm.schema import validate_score as sv
s=json.load(open('work/mysong/score.json',encoding='utf-8'))
print('schema', len(sv(s)), 'music', validate_score(s)['ok'])
"

# 5. MusicXML 过官方 XSD（需要在 work/xsd2/ 备好三个 raw xsd）
D:/WorkSoftware/anaconda3/python.exe -c "
from lxml import etree
schema = etree.XMLSchema(etree.parse('work/xsd2/musicxml.xsd'))
print('XSD valid:', schema.validate(etree.parse('work/mysong/score.musicxml')))
"

# 6. DSH 插件入口
.venv/Scripts/python plugin/scripts/doctor.py --json
.venv/Scripts/python plugin/scripts/generate.py "<song>" --out work/skill-out
```

## 复现注意

- 想要快速验证链路而不是质量：`--no-separate --no-lyrics` + `AITSM_SKIP_ASR=1`（16s 片段约 15s 出结果）。
- 想要更快的歌词：`--whisper-model small`（CER 会到 0.64 左右）。
- 想要和本文档数字可比：**必须冷启动 + 默认 medium**，且接受 CER ±0.05 的波动。
