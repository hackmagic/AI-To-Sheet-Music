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
