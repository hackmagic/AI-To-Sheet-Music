# 整曲实测质量（v2，真实模型全开）

> 2026-10 实测。源：`小林未郁、HOYO-MiX - 下一站 Beyond This Station`（FLAC，29.1 MB，
> 229.03 秒）。四层专用模型全部真实运行，`degraded_layers = []`。

## 1. 各层产出

| 层 | engine | 产出 |
| --- | --- | --- |
| 分离 | `demucs-onnx`（htdemucs_ft / bag_infer） | 4 轨 44.1kHz |
| 人声 | `vocalparse`（qwen_asr Thinker） | **420 个旋律音 / 56 行歌词** |
| 伴奏 | `sheetsage2` | **188 个和弦 / 181 个强拍 / 20 个段落** |
| 量化 | `duet` | 独立算，无降级 |
| 编配 | `dp` + `tart-heuristic` | 90 小节 / 10 段 / 164 个和弦落位 |

耗时（CPU，16 核）：

| 阶段 | 耗时 |
| --- | --- |
| 分离（ONNX，229s 音频） | ~950 s |
| SheetSage2 | ~130 s |
| VocalParse（8 个 30s 窗口） | ~3300 s |
| 编配 + 导出 | ~5 s |
| **总计** | **4547 s（75.8 分钟）** |

## 2. 实测暴露并修复的三个真问题

跑完整曲才暴露出来的，短片段冒烟测不到：

### 2.1 拍号误判：4/4 被报成 2/4

`beat.lab` 里拍点间隔 0.64s（93.75 BPM），强拍间隔 1.28s，模型据此报 **2/4**。
后果：**181 个“小节”**，而整曲只有 229 秒（即每小节 1.26 秒）——真实歌曲不可能这么短。

交叉验证：旧 librosa 管线独立判定 **4/4 / 92.47 BPM**，与 93.75 几乎一致，
差异只在拍号。

修复（`accompaniment_stage._sanity_check_meter`）：

- 判据不是「强拍/拍点比」——那个是**自洽**的（2:1，确实符合 2/4）；
- 真正可疑的是**每小节秒数**：`2/4` 且 < 1.6 s/小节 → 判定为「把八分音符当成了拍」；
- 回退：拍号改 **4/4**，BPM **保持 93.75**（不是反推成 187.5，那是八分音符速度），
  downbeats 隔一个取一个 → **91 小节 × 2.55 s ≈ 229 s**，与整曲时长吻合。

修复后 `analysis` 与 `score` **双双通过校验**。

### 2.2 段落 kind 不在 schema 枚举里

SheetSage2 输出 `pre-chorus`（带连字符），schema 只认 `prechorus`。
实测造成 **3 处 `sections/*/kind` 非法 → `analysis 未通过 schema 校验`**（error 级）。

修复（`_normalize_section_kind`）：把 `pre-chorus` / `pre chorus` / `post-chorus` /
`breakdown` / `guitar solo` 等变体映射到标准 kind；**未知标签一律落 `other`，不猜**。

### 2.3 所有段落共用同一个节奏型

20 个段落的 energy 全是 **0.50** —— 因为 `contracts.Section.energy` 默认 0.5，
而 SheetSage2 不提供段落能量，适配器又只在 `energy` 为空时才取先验，
于是「默认值 0.5」被当成了真实数据。

修复：按 `kind` 给**类型先验**（intro 0.20 / verse 0.45 / prechorus 0.60 /
chorus 0.75 / interlude 0.35 / bridge 0.50 ...）。这不是编造，而是类型先验，
且写在 `analysis.sections[].energy` 里可见。

## 3. 修复后的谱面质量

```
拍号 4/4  BPM 93.75  91 小节  20 段落（→ 编配合并成 10 段）
谱面质量 0.94   analysis 校验通过   score 校验通过
```

段落结构与节奏型（音乐上合理）：

| 段落 | energy | 节奏型 |
| --- | --- | --- |
| 前奏（1–18） | 0.20 | `ballad8` 分解八分 |
| 主歌 1（19–30） | 0.45 | `pop8` 民谣八分 |
| 预副歌 1（31–34） | 0.60 | `sync8` 切分八分 |
| 副歌 1（35–38） | 0.75 | `drive8` 摇滚八分 |
| 间奏 1（39–42） | 0.35 | `folk4` 四分扫弦 |
| 主歌 2（43–50） | 0.45 | `pop8` |
| 预副歌 2（51–58） | 0.60 | `sync8` |
| 副歌 2（59–67） | 0.75 | `drive8` |
| 间奏 2（68–72） | 0.35 | `folk4` |
| 副歌 3（73–90） | 0.75 | `drive8` |

和弦进行（B 小调，样例）：

```
小节 1: Gmaj7
小节 2: Gmaj7 (0.0) → D/A (2.0)
小节 3: D/A (0.0)   → Bm7 (2.0)
```

## 4. 交付产物

| 文件 | 大小 | 内容 |
| --- | --- | --- |
| `score.json` | 550 KB | 90 小节 / 353 旋律音（134 带词） |
| `score.musicxml` | 482 KB | 180 measure / 1260 note / 164 harmony / **772 组 string+fret（TAB）** / 185 lyric / 114 direction |
| `score.mid` | 25 KB | 可播放 |
| `payload.json` | 709 KB | timing 90 小节 / 353 音符 / 164 指法图 |
| `timing.json` | 78 KB | 绝对时间轴 |

MusicXML 是 **4.0** 版，TAB 声部完整（772 组 `<technical><string>/<fret>`）。

## 5. 诚实的限制

| 项 | 实测 |
| --- | --- |
| 歌词 | VocalParse 解出 56 行、353 个旋律音里 134 个挂上了词。**未做人工校对**，中文/英文混唱处仍可能错字 |
| 和弦 | 188 个事件来自 SheetSage2 联合解码；`Gmaj7`/`D/A`/`Bm7` 这类带色彩的和弦是模型判断，置信度未标注（SheetSage2 不输出逐事件置信度） |
| 拍号 | 已按数据自洽性修正为 4/4；但**中途变速/变拍号**仍不支持（SheetSage2 只给一个全局拍号） |
| 段落 | 20 个原始段落 → 编配层合并为 10 段；`interlude` 与 `chorus` 的区分依赖模型标签 |
| 耗时 | CPU 75.8 分钟/首（VocalParse 占 73%）。GPU 是唯一现实解法 |

## 6. 复现命令

```bash
export AITSM_VOCALPARSE_DIR=models/VocalParse \
       AITSM_SHEETSAGE2_DIR=models/SheetSage2 \
       AITSM_DUET_DIR=models/DuET \
       AITSM_DEMUCS_ONNX_DIR=models/htdemucs_ft_onnx \
       HF_HOME=.cache/hf

# 整曲（约 76 分钟，CPU）
.venv/Scripts/python.exe -X utf8 work/demo_full_song.py

# 仅用已有 analysis 重建交付物（秒级，用于修正下游问题）
.venv/Scripts/python.exe -X utf8 work/rebuild_demo.py
```
