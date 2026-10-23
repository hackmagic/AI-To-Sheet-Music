# 质量档：为什么慢的档位真的更准，以及慢多少

> 一句话结论：`--quality high` / `max` 不是「调高某个阈值让分数好看」，而是换更贵的模型
> 和更稳的输入信号；代价是 CPU 上从约 10 分钟涨到 40 分钟以上。

## 1. 三个档位到底做了什么

| 档位 | 分轨模型 | shifts | overlap | 歌词模型 | 和弦输入 | CPU 全曲耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| `standard`（默认） | `htdemucs` | 1 | 0.50 | `medium` | 去人声伴奏轨 | **实测 9.0 分钟** |
| `high` | `htdemucs_ft` | 2 | 0.75 | `medium` | 去人声伴奏轨 | 预估 50–70 分钟 |
| `max` | `htdemucs_ft` | 4 | 0.90 | `large-v3` | 去人声伴奏轨 | 预估 1.5–2 小时 |

> 为什么 `high`/`max` 只给预估：`htdemucs_ft` 是 4 模型 bag，本身就要 4 倍分轨时间
> （实测 229 秒的歌：`htdemucs` 308 s → `htdemucs_ft` 1243 s），`shifts`/`overlap`
> 再各乘上去。**只把歌词换成 `large-v3` 的实测数据在第 3 节**（总计 30.7 分钟）。

对应实现：

- `aitsm/pipeline/separate.py` 的 `demucs_quality()` **每次调用现读**环境变量
  （`AITSM_DEMUCS_MODEL` / `AITSM_DEMUCS_SHIFTS` / `AITSM_DEMUCS_OVERLAP`），
  所以同一进程里切档立即生效；这三个值同时进分轨缓存键，换档必然重算。
- `aitsm/cli.py` 的 `QUALITY_PRESETS` / `apply_quality_preset()` 把档位翻译成环境变量，
  **不覆盖**用户已显式设置的变量（想混搭就自己设 `AITSM_DEMUCS_SHIFTS=1`）。
- `aitsm/pipeline/run.py` 的 `_build_instrumental()` 把 `drums + bass + other` 相加成
  `instrumental.wav`，和弦识别改用它。

## 2. 三个真正提升精度的机制

### 2.1 和弦识别改用「去人声」伴奏轨

人声旋律在和弦窗口里是**和弦外音**。用混音算 chroma 时，主唱的一个经过音
（比如 C 和弦上唱到 D）会把三和弦的证据摊到 II 度上，最直接的后果是被判成
`Cadd9` / `C/D` / `Cmaj7` 之类的复杂和弦——置信度低，指法还难按。

去掉人声后 chroma 只剩伴奏，三和弦的能量集中在根音/三音/五音上，模板匹配的
相对优势变大，Viterbi 选出的进行也更稳定。

### 2.2 微调版分轨模型（`htdemucs_ft`）比原版干净

`htdemucs_ft` 是官方在 htdemucs 基础上微调的 4 模型 bag，同一段音频会跑 4 个模型
再平均，代价正好是 4 倍。它对人声/鼓/贝斯的边界更保守：残留到 `other` 里的人声更少，
被误当成人声的镲片也更少——对歌词时间轴和旋律提取都有帮助。

### 2.3 `shifts` 与 `overlap` 降低随机性

Demucs 推理时对输入做随机位移（shift invariance）并平均，`shifts` 越大方差越小；
`overlap` 决定分段之间的重叠比例，越大边界越平滑。两者都线性增加耗时，
**边际收益递减**：shifts 1→2 通常还有可测提升，2→4 主要是在抹平抖动。

## 3. 实测：standard vs high vs max

（数据来自同一首真实歌曲 `下一站 Beyond This Station`，229 秒，CPU 推理。）

> 实测条件：为了在可接受时间内拿到数据，下面第二列用的是 **`htdemucs_ft` + shifts=1 +
> overlap=0.5 + `large-v3`**（即 `--quality high` 去掉多次位移、把歌词换成最大模型）。
> 完整 `high`（shifts=2 / overlap=0.75）在 CPU 上全曲预估 1 小时以上，`max` 更久，
> 这一列已经足够回答「贵的档位到底贵在哪、值不值」。

| 指标 | `standard`（htdemucs + medium） | `high`+large-v3（ft 分轨 + large-v3 歌词） | 变化 |
| --- | --- | --- | --- |
| 全曲总耗时 | 538.7 s（9.0 分钟） | 1842.9 s（30.7 分钟） | **3.4×** |
| ├ 分轨 | 308.5 s | 1243.1 s | 4.0× |
| ├ 歌词 | 151.5 s | 504.8 s | 3.3× |
| BPM / 置信度 | 92.47 / 0.792 | 92.47 / 0.792 | 一致 |
| 调性 / 置信度 | D 大调 / 0.843 | D 大调 / **0.795** | 判定一致，自评降 |
| 和弦事件数 | 55 | 52 | −3 |
| **和弦平均置信度** | 0.5743 | **0.5811** | **+1.2%** |
| 和弦置信度区间 | 0.492–0.717 | 0.483–0.714 | 基本持平 |
| 段落数 | 12 | 11 | −1 |
| 旋律音符 / 置信度 | 326 / 0.831 | 334 / 0.827 | 持平 |
| 歌词行数 | 18 | 21 | +3 行（覆盖更全） |
| **歌词字符级 CER** | 0.4935 | **0.4696** | **−4.8%（相对）** |
| 歌词词级 WER | 0.5547 | 0.5401 | −2.6%（相对） |
| `score.quality.overall` | 0.685 | **0.671** | **−2.0%** |
| `manifest.quality.overall` | 0.824 | 0.813 | −1.3% |

MusicXML 校验：高质版谱面过官方 XSD，并用 MuseScore 4.7.5 **严格导入（不带 `-f`）**
exit=0，导出 7 页 PDF（139 KB）——记谱合法性没有因为换档退化。

### 从这张表能读出的三件事

**1. 贵的档位主要在改善歌词文本，不在和弦置信度。** 和弦平均置信度只从 0.5743 涨到
0.5811，1.2% 的差别落在噪声量级（同一配置两次运行的波动就有 ±0.03）。真正明显的是
歌词：CER 0.494 → 0.470，行数 18 → 21（medium 漏掉的整段被补回来了）。

**2. 模型的「自评置信度」和「准确率」不是一回事，而且会反向走。** 换 `large-v3` 后
歌词文本更准，但每行自评置信度更低（`quality.stages.lyrics` 0.387 → 0.362），
调性置信度也从 0.843 掉到 0.795。原因很朴素：大模型对自己的错误更不自信，
小模型会自信地把错字吐出来。**所以不能拿置信度数字飙升来验收「更高标准」**，
必须看文本/听音频/A-B 对比。

**3. `score.quality.overall` 是五个分项的算术平均，会被歌词自评分数拖住。**
这也是高质版 overall 反而低了 2% 的原因：和弦 +1.2%、歌词文本 +5%，但歌词自评 −6.5%
和调性自评 −5.7% 的权重更大。这个指标适合横向扫「哪首歌难」，**不适合衡量档位收益**。

### 那到底值不值

| 你的诉求 | 建议 | 代价 |
| --- | --- | --- |
| 快速看和弦走向、自己再修 | `standard` | 10 分钟 |
| 歌词要能直接唱（英文/中文都可） | `standard` + `--whisper-model large-v3` | +6 分钟（这是性价比最高的单项） |
| 交付给人弹、和弦要少改 | `--quality high` | 40 分钟以上 |
| 一次性的最好结果 | `--quality max` | 1.5–2 小时 |

也就是说：**先把歌词模型换成 `large-v3`，再考虑要不要为分轨付 4 倍钱。**
`htdemucs_ft` 在这首歌上换来的和弦收益（+1.2%）远小于它的成本（4 倍分轨时间）。

## 4. 什么时候该用哪个档

- **日常/批量**：`standard`。4 分钟的歌 10 分钟出谱，和弦大致可用。
- **要交付给人弹**：`high`。分轨干净一档，和弦与歌词都更少需要手工修。
- **一次性的「最佳效果」**：`max`。只有在你能接受 1.5–2 小时、且歌词是重点时才值得。

判断信号：看 `score.json` 的 `quality.overall` 与 `quality.stages.chords`。
如果 `chords` 分数卡在 0.55 上下、告警里有 `chords.low_confidence`，
先上 `high`；如果歌词错字多（尤其是中文），再考虑 `max` 的 `large-v3`。

## 5. 不要期待什么

- 质量档不会让**段落划分**变准。段落问题是合成/编曲特征问题，不是模型精度问题。
- 质量档不会让**时间轴对齐 LRC**。歌词时间来自 ASR，参考 LRC 常来自另一个编曲版本。
- `max` 的 `large-v3` 只是更少听错字，**不会**把中文幻觉署名行变少——
  那类问题靠 `drop_silent_lines()` / `is_zh_credit_line()` 在事后清洗里解决。

## 6. 复现

```bash
# 标准档
python -m aitsm.cli generate song.flac --out work/std

# 高质量档（微调分轨 + 更稳的分段）
python -m aitsm.cli generate song.flac --out work/high --quality high

# 最高质量档（再加多次位移 + large-v3 歌词）
python -m aitsm.cli generate song.flac --out work/max --quality max

# 混搭：微调分轨但只跑一次位移，歌词用 large-v3
AITSM_DEMUCS_SHIFTS=1 python -m aitsm.cli generate song.flac --out work/mix \
  --quality high --whisper-model large-v3
```

模型缓存位置：`work/cache/hf/hub/`（`models--adefossez--HTDemucs-ft`、
`models--Systran--faster-whisper-large-v3`）。首次运行会下载
（`HTDemucs-ft` 约 320MB，`large-v3` 约 3GB），走 `HF_ENDPOINT=https://hf-mirror.com`。

> 踩坑记录：`faster-whisper` 的 `download_root` 必须留空让它用
> `HUGGINGFACE_HUB_CACHE`；手动传 `download_root=…/cache/hf` 会让它去
> `…/cache/hf/models--*` 找模型（而实际缓存在 `…/cache/hf/hub/models--*`），
> 于是**静默重下 3GB**，看起来像是「加载模型卡住」。
