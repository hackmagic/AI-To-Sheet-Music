# 模型对比实测记录

本文件记录关键模型选择的实测依据，避免"凭感觉调参"。

## Whisper 模型：small vs medium

**测试条件**：同一首歌（《下一站 Beyond This Station》229.03s）的 Demucs vocals stem，
CPU int8 推理，`vad_filter=False`、`condition_on_previous_text=False`、`beam_size=5`、
`language=None`（自动检测）。参考文本为歌曲配套 LRC（62 行正文）。

| 指标 | small | medium | 结论 |
| --- | --- | --- | --- |
| **CER** | 0.6546 | **0.4976** | medium 降低 24% 错误率 |
| 语言检测置信度 | 0.25 | **0.56** | medium 更确信是英文 |
| 识别行数 | 20 | **26** | medium 覆盖更全（真值 62 行，两者都偏少） |
| 模型加载 | 2.5s | 5.5s | 可接受 |
| 推理耗时 | 149.4s（0.65× 实时） | 348.3s（1.52× 实时） | 慢 2.3 倍 |
| avg_logprob 中位 | -0.601 | -0.569 | 略好 |

**逐行对比**（前 5 行）：

| 时间 | small | medium | LRC 原文 |
| --- | --- | --- | --- |
| 30.0s | Let's get it up! | Let's get it up! | Let's pave it out |
| 43.8/45.1s | What stress chance so far along? | Was dressed chance so far along? | What's stretching so far along |
| 50.4/50.0s | Oh the rays of our mutual bond | **Oh, the rays of our mutual bond** | The rails or our mutual bond? |
| 62-64s | We're trapped through the quiet land | **Whoa, we drive through the quiet land to carry us across the sil…** | We track through the quiet land / To carry us across the sand |
| 67-70s | We're curious across the sky | **One day I will fall away, look out, I don't believe the way** | One day, our weight, will fall away / Look out! Ride on… |

medium 明显更接近原文，尤其第 4、5 行已经能对上多个短语；small 则整句跑偏。

### 决策

**默认使用 `medium`**。设置位置：
- `aitsm/config.py` → `Settings.whisper_model` 默认值
- `aitsm/pipeline/lyrics.py` → `DEFAULT_WHISPER_MODEL`

**需要更快时**（例如批量处理、CI）：
```bash
AITSM_WHISPER_MODEL=small python -m aitsm.cli generate ...
# 或
python -m aitsm.cli generate --whisper-model small ...
```

**代价（实测，229s 歌曲）**：专用 benchmark 脚本里 small 149.4s、medium 348.3s（含模型加载）；
但在正式管线里 medium 的 `lyrics` 阶段只花 **151–157s**（`work/accept-final` / `work/accept-v2`），
因为 benchmark 脚本为了公平对比用了 `beam_size=5` + 逐词时间戳 + 独立进程加载，
而管线的参数不同。**要报成本请用管线实测的全流程数字**：

| 配置 | 全流程冷启动 | 其中分轨 | 其中歌词 |
| --- | --- | --- | --- |
| medium（默认） | 约 9–10 分钟 | 约 5 分钟 | 约 2.5 分钟 |
| small | 约 6–7 分钟 | 约 5 分钟 | 约 1 分钟 |

> 本机 CPU 上的 Demucs 受负载影响很大：实测 `separate` 阶段在 140–335s 之间浮动。
> 早期记录过「small 全流程 4.5 分钟」，那是分轨较快的一次运行。报耗时请用区间。

### 剩余差距

即使 medium，CER 仍有 0.498，原因是：
1. **演唱 vs 说话**：whisper 训练数据以说话为主，唱腔的连读、拖长音、颤音不符合其先验
2. **和声干扰**：vocals stem 里仍混有和声与混响，尤其在副歌
3. **英文歌 + 非母语演唱**：这首歌是日系制作团队演唱英文，发音本身不标准
4. **缺少上下文**：whisper 没有歌词先验（如果用户能提供歌词文本，可以做 forced alignment，准确率能到 0.9+，但那是另一个功能）

**不计划继续换更大的模型**（large-v3 约 3GB，CPU 推理会超过 10 分钟，收益递减）。

## 分轨模型：htdemucs（4 源）vs htdemucs_6s（6 源）

未做对比。当前用 `htdemucs`（4 源：vocals/drums/bass/other），因为：
- 本项目是弹唱谱，只需要 vocals 做旋律/歌词，其它 3 源用于和弦与节拍
- `htdemucs_6s` 多出 guitar/piano 两源，但官方标注质量低于 4 源版本
- 6 源模型推理时间约 1.8 倍

如果需要吉他 riff 转录（P2 功能），再评估 6 源模型。

## 合成歌曲评估基线

见 `docs/07-synthetic-eval.md`。

---

## v2 专用模型实测（2026-10，CPU-only）

这一节记录**换成四个专用模型之后**的实测事实。条件：整首
《下一站 Beyond This Station》（229.03s，FLAC 44.1kHz），
`transformers==4.57.6`、`qwen-asr==0.0.6`、`HF_HOME=.cache/hf`、CPU（16 核），GPU 未启用。

### 四层耗时与产出

| 层 | 实际 engine | 耗时 | 产出 |
| --- | --- | --- | --- |
| 分离 | `demucs-onnx`（htdemucs_ft，bag_infer） | ~950 s | 4 轨 44.1kHz |
| 人声 | `vocalparse`（qwen_asr Thinker，8×30s 窗口） | ~3300 s | **420 旋律音 / 56 行歌词** |
| 伴奏 | `sheetsage2` | ~130 s | **188 和弦 / 181 强拍 / 20 段落** |
| 量化 | `duet` | 秒级 | 独立算，无降级 |
| 编配 + 导出 | `dp` + `tart-heuristic` | ~5 s | 90 小节 / 164 和弦落位 |
| **合计** | — | **4547 s（75.8 分钟）** | `degraded_layers == []` |

**结论**：VocalParse 占 **73%** 耗时，是 GPU 迁移的唯一优先项；
DuET 与编配层可以永久留在 CPU。

### 各模型单独验证（14s 片段）

| 模型 | 关键事实 |
| --- | --- |
| htdemucs_ft ONNX | 4 sub-model × 3 chunk = 12 次推理，**57.7s**（RTF 2.21）；**输入必须是 44.1kHz 双声道**，22.05kHz 分析音轨要先升采样 |
| SheetSage2 | 首次加载 3.6s（有缓存后）；输出 `score.abc` + `beat/chord/key/structure.lab` + `melody_vocal.mid` |
| VocalParse | 加载 34.7s、生成约 120s/12s 音频；verse 段 123 音符 / 26 带词，BPM 89 来自 `<BPM_>` token |
| DuET | 12 音 performance MIDI：`input_coverage=1.0`、8 小节 / 21 音符 / divisions=480 |

### 与 v1 内核的对比（同一首歌）

| 维度 | v1 内核 | v2 专用模型 |
| --- | --- | --- |
| 歌词行数 | 26（whisper medium） | **56**（VocalParse） |
| 旋律音数 | 395（pyin） | **420** |
| 和弦事件 | 69（chroma 模板） | **188**（SheetSage2 联合解码） |
| 段落数 | 11 | **20** |
| 每层可追溯性 | 只有 `manifest.degraded` | **`symbolic.provenance` 逐层 requested vs engine** |
| 耗时 | 9–10 分钟 | 76 分钟 |
| 和弦逐事件置信度 | 有 | **无**（SheetSage2 不输出） |

> 行数/音符更多**不等于更准**——v2 没有人工标注的 ground truth 可比。
> 唯一可交叉验证的是**拍号**：SheetSage2 报 2/4，旧 librosa 管线报 4/4 / 92.47 BPM
> （v2 报 93.75 BPM），两者 BPM 几乎一致、拍号不同 → 判定 SheetSage2 误判，
> 已在适配器里按「每小节秒数」纠正为 4/4。详见 `13-demo-quality.md` §2.1。

### TART / Noise2Fret 的可用性核查

| 项目 | HF 仓库 | GitHub | 权重 |
| --- | --- | --- | --- |
| Noise2Fret | 无 | `RiccardoVib/Noise2Fret` ✅ 可克隆 | **无**（仓库内无 `.pt/.ckpt/.safetensors`，README 也无外部链接） |
| TART | 无 | 论文 arXiv 2609.11904 **未给出代码仓库链接** | 无从获取 |

因此：Noise2Fret 只移植了**公式**（`aitsm/arrangement/playability.py`，纯 numpy 五项损失）；
TART 的 audio-conditioned T5（AudioFret）无法复现，`provenance.engine` 如实写 `tart-heuristic`。
