# 环境与已知坑（实测记录）

本文记录在本机（Windows 11 / RTX 4060 Laptop 8GB / Python 3.12.8）上实际踩到的坑与可用解法，
避免后续开发者和 DSH 插件用户重复踩。

## 1. 缓存目录不可写 → 必须重定向

**症状**：`PermissionError: [WinError 5] 'C:\Users\<user>/.cache/torch'`

**原因**：沙箱/企业环境限制用户目录写入；torch、huggingface_hub、numba 默认都往 `~/.cache` 写。

**解法**：`aitsm/config.py` 的 `prepare_cache_dirs()` 在 `aitsm/__init__.py` 顶层调用，
把下列环境变量重定向到 `work/cache/`：

```
TORCH_HOME, HF_HOME, HUGGINGFACE_HUB_CACHE, XDG_CACHE_HOME, MPLCONFIGDIR, NUMBA_CACHE_DIR
```

**要求**：任何会加载 torch / librosa / huggingface_hub 的代码，必须保证 `aitsm` 包先被 import。
不要用 `python aitsm/pipeline/xxx.py` 直接跑脚本，用 `python -m aitsm.pipeline.run`。

## 1b. 技能装不上不是权限配置问题，是 `Low` 完整性标签的**继承**问题

**症状**：`plugin/install.py` 从 `.venv` 的 python 启动时，报
`PermissionError: [WinError 5] 拒绝访问: 'C:\Users\<user>\.agents\skills\...'`；
更迷惑的是它还会把**每一个**候选解释器都报成「写不了」，
连平时在命令行里明明能用的 `D:\WorkSoftware\anaconda3\python.exe` 也一样。

**原因**：`icacls .venv/Scripts/python.exe` 显示它带
`Mandatory Label\Low Mandatory Level:(I)(NW)` —— 这是个**打在 exe 上的完整性标签**，
不是目录权限。带这个标签的解释器启动后整个进程跑在 **Low** 下，于是写不了 `%USERPROFILE%`。

关键在于 **Windows 子进程继承父进程的完整性级别**，不会因为换了个 exe 就升回来：

| 验证 | 结果 |
| --- | --- |
| bash（Medium）直接跑 anaconda python 写 `~/.agents/skills` | ✅ 成功 |
| anaconda python **由 venv(Low) python 派生**再写同一目录 | ❌ WinError 5 |
| venv python 由 **anaconda(Medium)** 派生再写 | ❌ WinError 5 |

也就是说「自动挑一个能写的解释器再重跑自己」这条路在 Low 父进程下是**结构性无效**的，
不是脚本 bug、更不是机器上没一个能写的解释器。

**解法**：在**普通终端**（Medium）里用系统 python 跑安装，别从 `.venv` 里跑：

```bash
D:/WorkSoftware/anaconda3/python.exe plugin/install.py
```

`install.py` 已按这个事实重写探测逻辑：先用「本进程自己写一个探针文件」判断
（不派生、无继承问题），判据失败且当前是 Low 时**不再尝试换解释器重试**，
而是直接列出可用的 Medium 解释器与可复制的命令；只有当前**不是** Low 时才走原来的自动切换。
装到项目盘（`--dest K:/.../.agents/skills`）不受这个限制。

> 顺带一提：`runs on Low` 也解释了为什么 `%USERPROFILE%` 下一些目录 bash 能写、python 不能 ——
> bash 是 Medium，从它启动的 python 才是 Low。判断「能不能写」一律要**实测本进程**，
> 不能看「这个目录的 ACL 看起来允许」。

**连带影响：装完的技能目录里 `work/` 用不了（但有兜底）。**
`~/.agents/skills/audio-to-chord-sheet/work/out` 在 Low 解释器下 `mkdir` 就被拒
（在 `%USERPROFILE%` 下建目录本身要 Medium 权限，给目录打 Low 标签也救不回来）。
`_locate.writable_candidates()` 会按顺序实测，落到 `K:/.../work/skill-out`（项目盘可写）：

```text
out_candidates[0] skill 目录 work/out        writable=False  (WinError 5)
out_candidates[1] 项目 work/skill-out        writable=True   ← 实际生效
```

所以 **`doctor.py` 的 `paths.default_out_dir` 会显示仓库里的 `work/skill-out`，
不是技能目录下的 `work/out`** —— 这是预期行为，不是配错了。生成时也可以显式
`--out <可写目录>` 或设 `AITSM_OUT_DIR`。

## 2. HuggingFace Xet 后端返回 401

**症状**：`RuntimeError: File reconstruction error: CAS Client Error: HTTP status 401`（`cas-server.xethub.hf.co`）

**解法**：`prepare_cache_dirs()` 里设置 `HF_HUB_DISABLE_XET=1`，回退到普通 HTTP 下载。

## 3. `librosa.onset.onset_detect` 进程级段错误

**症状**：调用后进程直接崩溃（exit code 不同，无 Python traceback），`except` 捕不到。

**环境**：librosa 1.0.0 + numba 0.68.0 + Windows。

**解法**：**禁止使用** `librosa.onset.onset_detect`。改用谱通量 + `librosa.util.peak_pick` 或纯 numpy 挑峰。
已验证可用的函数：`beat_track`、`chroma_cqt`、`recurrence_matrix`、`pyin`、`piptrack`、`stft`、`rms`、`times_like`、`frames_to_time`。

## 4. librosa 1.0.0 与 0.10 的 API 差异

- `librosa.filters.mel` 需显式传 `sr=` 与 `n_fft=`
- `librosa.core.*` 旧命名空间大多移除，统一用顶层 `librosa.xxx`
- 短音频会警告 `Empty filters detected in mel frequency basis`，需按 `sr` 调整 `n_mels` / `fmax`

## 5. numpy 2.x 移除的别名

`np.float_`、`np.complex_`、`np.alltrue`、`np.trapz` 均已移除；`np.trapz` → `np.trapezoid`。
从旧项目复制代码时要检查。

## 6. `openai-whisper` 装不上

**症状**：py312 Windows 没有预编译 wheel，pip 走源码构建，编译 `tiktoken`/`tokenizers` 时卡死（>10 分钟无产物）。

**解法**：改用 **`faster-whisper`**（CTranslate2 后端）：
- 有 py312 win 预编译 wheel，安装秒级
- 支持 `word_timestamps`、`vad_filter`
- 比 openai-whisper 快约 4 倍
- 中文效果更好

## 7. `faster-whisper` + PyAV 版本不兼容

**症状**：`model.transcribe("file.mp3")` 抛 `TypeError: open() got an unexpected keyword argument 'metadata_errors'`

**原因**：`faster_whisper.audio.decode_audio` 用了旧版 PyAV 的 API，而装的 `av 19.0.1` 已改签名。

**解法**：**不要传文件路径**，先自己解码成 numpy（16kHz 单声道）再传入：

```python
import librosa
y, sr = librosa.load(path, sr=16000, mono=True)
segments, info = model.transcribe(y, language=None, word_timestamps=True, ...)
```

## 8. `vad_filter=True` 在音乐上返回 0 段

**症状**：整首歌唱下来 `segments` 为空，但 `vad_filter=False` 时正常。

**原因**：Silero VAD 是为**说话**训练的，唱歌的基频/音色被判成非语音。

**解法**：音乐场景**必须** `vad_filter=False`。这是本项目歌词识别的关键开关。

**实测对比**（`work/full16k.wav`，60-105s / 120-165s）：

| vad_filter | 段数 |
| --- | --- |
| False | 8 / 12 |
| True | 0 / 0 |

## 9. 音频路径 vs numpy 输入的质量差异

同一首歌同一时间段：

| 输入 | 识别结果 |
| --- | --- |
| 整曲混音 | `"I'm on your way"` / `"We truck to the point that we're curious"`（明显错乱） |
| Demucs vocals stem | `"I stormed the silence with the hearts away"`（对照 LRC 原文 `I storm the silence, with our hearts awake`） |

**结论**：优先用 vocals stem，没有 stem 才降级用混音 + 告警。

## 10. torch CUDA 版下载困难

**症状**：`download.pytorch.org` 只有 ~357 KB/s，2.86GB 的 cu128 wheel 反复中断；阿里云镜像
`mirrors.aliyun.com/pytorch-wheels/cu128/` 到 1.1 MB/s 但长连接仍会被切断，`curl -C -` 续传后
文件仍是坏的（`BadZipFile`）。

**当前决定**：**CPU 版 torch 已够用**（4 分钟歌全流程 5-8 分钟）。GPU 作为可选优化写进文档，
不作为必需项。需要 GPU 的用户请按 pytorch.org 官方指引安装匹配版本。

**校验 wheel 完整性的方法**：
```python
import zipfile
zipfile.ZipFile("torch-xxx.whl").testzip()   # 返回 None 才是完整的
```

## 11. Windows 控制台中文乱码

**症状**：CLI 打印的中文变成 `����Ŀ¼` 或直接抛 `UnicodeEncodeError`。

**解法**：`aitsm/cli.py` 的 `_force_utf8_stdio()` 在 `main()` 最前面把 stdout/stderr 重配置为 UTF-8。

## 12. whisper `initial_prompt` 会被当成歌词复读

**症状**：歌词里出现 `"The lyrics of this song are the lyrics of a song, transcribed line by line with…"`
这种与歌曲内容完全无关的句子。

**原因**：whisper 的 `initial_prompt` 是**当前解码的前文上下文**。prompt 文本越像歌词，
模型越容易顺着它"接着唱"。原先的英文 prompt 恰好是这句元描述本身。

**解法**（`aitsm/pipeline/lyrics.py`）：
- `_initial_prompt`：**英文/auto 一律返回 `None`**（不加 prompt 实测反而更准）；zh/ja 用祈使句元语言
  （"转写歌曲人声，只输出歌词行。"），不含可被当成歌词的名词短语。
- 双保险：`is_prompt_echo`（连续 4 个内容词命中 prompt 即丢）+ `is_meta_lyric_hallucination`
  （正则命中 `the lyr ics of this song` / `transcribed line by line` 等元描述即丢）。

**这是红线问题**：它不是"准确率低"，而是**编造不存在的内容**。

## 13. 英文音节被空格拆开

**症状**：`"deploy anti-curious"` 显示成 `"dep loy an ti cur ious"`。

**原因**：whisper 的 `words[]` 里，**词边界由前导空格标记**。拼接时无脑加空格会把子词拆开。

**解法**：`merge_split_words`（按前导空格合并，只对纯拉丁 token 生效，中文不受影响）
+ `words_to_text` / `space_before` 标记。

## 14. MusicXML `<note>` 子元素顺序必须严格

**症状**：MuseScore 能打开但严格解析器报错。

**原因**：MusicXML 4.0 XSD 要求 `<note>` 的子元素顺序为
`duration → tie → voice → type → dot → time-modification → notations → lyric`。
另外 `<harmony text="...">` 的 `text` 属性**不存在**于 XSD（只有 `<kind text="...">` 允许）。

**解法**：`_append_duration()` + `_append_note_type()` 按序写入；删除 harmony 上的 `text` 属性。
已用官方 XSD 验证通过（`tests/test_export.py` 有子元素顺序断言）。

## 15. 分轨不是比特可复现，导致歌词 CER 有 ±0.04 波动

**症状**：同一首歌、同一套参数跑两遍，歌词行数 17 vs 20、字符级 CER 0.502 vs 0.542。

**原因**：Demucs 在 CPU 上多线程归约存在浮点非确定性。两次运行的 `stems/vocals.wav`
sha256 不同，互相关在 lag=0 时仍有 4.5% 残差（相关系数 0.999，不是时间偏移）。
whisper 对这种微小差异足够敏感，于是断句与转写结果不同。

**结论**：报告歌词质量必须给**区间**（本次实测 0.50–0.54），不能拿单次运行的小数点后三位
当准确率。做 A/B 对比（small vs medium）时，差距（约 0.14）远大于波动（0.04），结论仍成立。

## 16. 跨模块改数据结构必须同步改下游脚本

**症状**：整首歌跑完、所有产物都写好了，`plugin/scripts/generate.py` 才崩在
`AttributeError: 'list' object has no attribute 'get'`，用户拿到 `{"ok": false}`。

**原因**：`aitsm/service.py` 的 manifest 合并把 `manifest["stages"]["analysis"]`
从「service 写的摘要 dict」改成「pipeline 写的阶段记录 **list**」，而 skill 里的
`generate.py` 还按 dict 在用：`((manifest.get("stages") or {}).get("analysis") or {}).get("engines")`。
`aitsm` 的 498 条测试全绿 —— 因为它们都在仓库内，测不到被复制走的脚本。

**解法**：`generate.py` 里改成 `_manifest_engines()`，按可靠性降级尝试
`manifest["tools"]` → `stages["analysis_summary"]["engines"]` → `stages["analysis"]["engines"]` → `analysis["tools"]`；
并新增 `tests/test_plugin_generate.py`（16 条）在仓库内锁住这个契约。

**教训**：改共享 JSON 结构的形状，等于改所有消费者的隐式接口。要么版本化（`schema` 字段），要么保留旧字段，要么给下游写兼容层。

## 17. skill 目录是复制品，改仓库不重装不生效

**症状**：修好 `plugin/scripts/generate.py` 后重跑仍报同样的错。

**原因**：DSH 加载的是 `%USERPROFILE%\.agents\skills\audio-to-chord-sheet\scripts\generate.py`
（`plugin/install.py` 复制过去的副本），不是仓库里的那份。

**解法**：改 `plugin/` 之后**必须**重跑 `python plugin/install.py`，再用
`diff` 确认副本一致。DSH 会自动重扫 skill 目录，不需要重启进程。
`install.py` 现在会自己校验副本 `SKILL.md` 与仓库逐字节一致，不一致直接非 0 退出。

## 17b. 技能装在 `~/.config/opencode/skills` 里 —— DSH 根本不扫这个目录

**症状**：`%USERPROFILE%\.config\opencode\skills\audio-to-chord-sheet\` 里文件齐全、
内容也对，但会话的技能清单里没有 `audio-to-chord-sheet`，怎么重装、重启都不出现。

**原因**：DSH 0.2.x 的 `@deepseek-ai/dsh-skill-filesystem` 只扫四个根：
`<projectRoot>/.dsh/skills`、`<projectRoot>/.agents/skills`、`~/.dsh/skills`、`~/.agents/skills`。
`~/.config/opencode/skills` 是 opencode 时代的位置，0.2.x 源码里已经没有这个路径。
**装错目录不会报错，只会静默不加载。**

**解法**：用当前 `plugin/install.py` 重装（默认目标已改为 `~/.agents/skills`），
确认 `doctor.py --json` 的 `ready: true`，再回到会话看技能清单里是否出现
`audio-to-chord-sheet`。

旧副本**不会自动消失**，留着有真实风险：别的 agent 工具（opencode / claude code 等）
仍会扫 `~/.config/opencode/skills`，于是出现两个同名、内容不同版本的技能，
谁生效取决于工具。清理用：

```bash
python plugin/install.py --remove-legacy
```

它把旧目录**改名**为 `audio-to-chord-sheet.bak-<时间戳>`（可回滚、`work/` 产物不丢），
而不是 `rmtree`。默认不清理 —— 用户数据不能无声消失。

契约细节与依据见 [docs/03-dsh-plugin.md](03-dsh-plugin.md) 的「DSH 0.2.x 技能发现契约与版本」。

## 18. 早期 manifest 没记录 whisper 型号

**症状**：两份谱面歌词差别明显，但 `manifest.json` 里只有 `engine: "whisper"`，看不出用的 small 还是 medium。

**原因**：`transcribe_lyrics()` 的公开契约是「返回歌词行列表」（渲染层依赖），
引擎元信息没有出口；`tools.lyrics` 只写了 `engine` 与 `line_count`。

**解法**：`lyrics.py` 增加模块级 `_LAST_RUN_INFO` + `last_run_info()`，
`_transcribe_lyrics()` 把它合并进 payload，`tools.lyrics` 落 `engine / model / device`。
（`tests/test_manifest_engines.py` 7 条锁住这条链路。）

## 19. 中文歌的三类特有缺陷（whisper 普通话输出）

用《遥远的你》（`K:/Music/221小伙伴 - 遥远的你.flac`，216s）跑真实中文歌，
英文歌从未出现的三类问题一次全冒出来：

| 缺陷 | 现象 | 根因 |
| --- | --- | --- |
| 繁简混排 | 同一份谱里「淚光/無邊無際/遊蕩/鑰匙/過去」与「阳光/喜欢/风」并存 | whisper 对普通话常整段输出繁体，且同一首歌里不一致 |
| 单字碎行 | 「去」「息」「息」「底」各占一行 | 换气处被切成 1 字 segment，谱面上一字一行 |
| 幻觉署名行 | 第 0 行「詞曲李宗盛」（真实作词是「王明明/李亚然」） | 该段 `vocals.wav` RMS **0.00022**（整体 0.1495，仅 0.15%）→ 纯静音段编造 |

**解法**（`aitsm/pipeline/lyrics.py`，都在 `transcribe_lyrics` 的清洗链里，按顺序执行）：

1. `is_zh_credit_line()` —— 整行是「作词:/作曲:/詞曲:/制作公司:/和声编写:」这类署名就丢；
   用关键词 + 长度双门限，避免把「我曾经和制作人聊过」这种正片歌词误杀。
2. `drop_silent_lines(lines, audio_path)` —— 按**相对整体 RMS 中位数**判静音
   （`threshold = max(3e-4, 0.08 × median)`），低于阈值的行丢弃。
   用相对阈值而不是绝对阈值，是为了保住「低声真唱」的段落。
3. `merge_tiny_lines(lines, language=...)` —— 中文 <3 字、拉丁 <2 词的碎行并入相邻行
   （优先并到前一行，首行则并到后一行），`max_gap=2.5s` 防止跨越间奏乱并。
4. `to_simplified_lines()` —— 用 `zhconv` 转 `zh-cn`；`AITSM_KEEP_TRADITIONAL=1` 可关闭；
   日/韩文（有假名、language=ja/ko）跳过，避免把日文汉字改错；没装 zhconv 就原样返回。

**效果**（同一首歌、同一参考 LRC）：繁体行 **8 → 0**、<3 字碎行 **4 → 0**、
幻觉署名行 **1 → 0**；文本召回率 recall **0.731 → 0.835**、precision **0.523 → 0.606**。

## 20. 中文歌不能用英文歌的 CER 口径

**症状**：中文歌算出来 CER 0.67，比英文歌的 0.50 差一大截，但实际识别质量并不更差。

**原因**：两层

1. **参考歌词版本可能不是同一个编曲**。本曲 LRC 首行标在 2.08s，但音频 `vocals.wav`
   0–15s RMS 恒为 0.0002（人声从 ~14.8s 起）；按内容对齐算出的偏移还在 16.5s→19s→21s
   之间漂移，说明不是 uniform shift。→ **中文歌不能做逐行时间比对，只能做文本比对。**
2. **副歌重复三遍**，Levenshtein 对重复文本敏感，单看 CER 会同时惩罚「漏词」和「多词」。

**解法**：`work/zh-eval.py` 同时报三个数——`lev_cer`（和英文歌同口径，便于横向看）、
`recall`（参考里有多少字被识别出来 = 漏词）、`precision`（识别出的字里有多少对 = 幻觉）。
结论要按三元组报，不要只报一个 CER。

## 21. MuseScore 实机打开才发现的速度/段落标记问题

**症状**（只有真的用 MuseScore 打开才看得到，XSD 与自研校验都发现不了）：

1. 谱面速度标记显示小数：`♩= 87.591`；
2. 速度和段落标记在**两个 part 各出一份**，MuseScore 里重复显示；
3. 逐小节速度抖动（83/85/86/88/89/90/92/95/97/99/101）会堆出 **25 个可见 ♩= 标记**。

**解法**（`aitsm/score/export/musicxml.py`）：

- `_append_tempo(..., visible=True, label_bpm=None)`：可见时 `<per-minute>` 写 `round()` 的整数，
  `<sound tempo>` 始终写精确值；`visible=False` 时用 `<metronome print-object="no">` + `<sound>`。
- `_append_rehearsal` 只写主旋律 part（P1），TAB part 不再重复。
- 可见性节流：与上次显示的速度差 <15% 就不打印（`<sound tempo>` 仍然每小节写，回放精确）。

**踩坑**：`visible=False` 一开始只写 `<direction><sound/></direction>`，
XSD 立刻报 `Element 'sound': This element is not expected. Expected is ( direction-type )`
—— `direction` 里 `<direction-type>` 必须排在 `<sound>` 之前。

**验证**：`artifacts/score.musicxml` 过官方 MusicXML 4.0 XSD，并在 **MuseScore Studio 4.7.5**
里实际打开：`♩ = 92` 只出现一次且是整数、「前奏」只出现一次、TAB 谱表无速度/段落标记。

## 22. MuseScore 4 的无头导出（已推翻「不能无头」的旧结论）

**结论修正**：MuseScore 4 **可以**无头导出（`-o` 足够；`-f` 只用于忽略告警强制转换），
但 `--help`/`-v` 在 Windows 上确实会把进程挂住（后台常驻），且**转换失败时它 exit>0 却不打印原因**——
错误只写进日志文件。

**可靠调用**（PowerShell，Windows 上不要用 Git bash 直接调，参数解析容易错）：

```powershell
Start-Process 'C:\Program Files\MuseScore 4\bin\MuseScore4.exe' `
  -ArgumentList '-o','out.pdf','in.musicxml' -Wait -PassThru | Select-Object ExitCode
```

失败时的唯一诊断渠道是日志：`%LOCALAPPDATA%\MuseScore\MuseScore4\logs\*.log`
（搜 `ConverterController::convertFile`、`failed load notation`）。

GUI 自动化仍可用作交叉验证：`work/shot.ps1`（`SystemInformation.VirtualScreen` +
`Graphics.CopyFromScreen`）截屏；打开文件用 `AppActivate(pid)` + `SendKeys('^o')` +
剪贴板粘贴路径 + `{ENTER}`。坑：鼠标点击对 MuseScore 无效（键盘有效）；
缩放快捷键 `^-`/`^+` 在 SendKeys 下无效；打开 454KB / 1170 音符的文件后窗口会短暂「未响应」，等约 45s 恢复。

## 23. 节奏时值没有对应音符形状（`<type>` 与 `<duration>` 矛盾）

**症状**：`MuseScore4.exe -o out.pdf artifacts/score.musicxml` 返回 **exit 1320**、
不产出 PDF，也不打印任何原因。日志里是：

```
failed load notation, err: [2009] Incomplete measure: Full score, measure 2, staff 1.
Found: 10/8.  Expected: 9/8.
```

（分母 8 = 二分音符单位，即「找到 5 拍、期望 4.5 拍」。）

**根因**：量化器给的时值（3.5 拍、3.25 拍、2.75 拍、1/6 拍、1/3 拍…）没有单一音符形状，
早期实现取「最接近的 type」写进 `<type>`（3.5 拍 → `whole`），`<duration>` 却写真实值。
XML 自身合法、XSD 也过，但宿主软件按 `<type>` 重排小节就装不下 → 判定不完整小节并**拒绝打开**。
更危险的是 `-f` 会静默修补，等于打印一份节奏错的谱。

**解法**（`aitsm/score/export/musicxml.py`）：

1. `_segment_duration(dur, position=)` 把任意时值贪心拆成**合法的二进制片段**，必须落在整拍/半拍网格上；
   长音拆成多个 `<note>` 并用 `<tie type="start"/stop">` 相连，歌词只挂第一段。
2. `_snap_to_grid()` + `_normalize_events()`：`RHYTHM_GRID = 0.125` 拍（三十二分音符），
   起音按网格吸附并保持严格递增、时值裁到下一个起音为止、越界事件丢弃 → 三连音位移全部消失，
   谱面不再需要 `time-modification`（试过保留三连音标记：混合时值组的 `<tuplet>` 会让 MuseScore
   把小节算错，报 `Found: 49/48. Expected: 131/128`，仍然 exit 1320）。
3. 小节末尾补休止符也走同一条拆段路径，保证每小节时值**精确等于**拍号。

**验证**：`work/audit-musicxml.py` 检查四份谱面 → 0 个时值矛盾、0 个小节不满；
不带 `-f` 的严格转换对 `artifacts/score.musicxml`（88 小节 / 1375 音符 / 218 个延音线）、
`work/zh-medium2`、`work/accept-v2`、`work/deliver` 全部 **exit 0**，并生成 7 页 PNG 与 PDF。
回归测试：`tests/test_export.py` 新增 4 条（时值/type 一致、长音拆段带延音线、三连音吸附、
每小节时值求和）。

**教训**：`-f` 成功不能当作验证通过；XSD 过了也不代表宿主软件能打开。

## 23b. 「两处副本不同步」这一类坑（三种形态，同一个病根）

这三条症状看起来毫不相干，病根是同一个：**同一份数据存在两个位置，写的一方不知道另一方存在**。

| 形态 | 症状 | 病根 |
| --- | --- | --- |
| ① 技能目录是复制品 | 改了 `plugin/` 却不生效 | 运行的是 `~/.agents/skills/` 的副本，不是仓库文件 |
| ② 单测写真实仓库文件 | 跑完 `pytest` 工作区就多出脏 diff | 测试直接写了被 git 跟踪的 `web/voicings.json` |
| ③ 查看器写回 `samples/` | 打开一次预览，`samples/preview.json` 就变了 | `serve.py` 兜底分支把预览文件复制进了仓库 |

**② 实测案例**：`tests/test_voicings.py` 的 fixture docstring 写着「写到临时文件，避免单测污染
`web/voicings.json`」，但实际传的是 `REPO_ROOT/"web"/"voicings.json"`；而 `build_voicing_db()`
末尾真的会 `target.write_text(...)` 落盘。后果是**每跑一次 `pytest`，被 git 跟踪的
`web/voicings.json` 就被改写一次**（只有 `generated_at` 变）。历史上已经因此产生过一个专门的还原提交
`f7d7c9d`「还原 web/voicings.json 的 generated_at 时间戳噪声（内容未变）」——不是理论问题，是反复发生。
已修为 `tmp_path_factory` / `tmp_path`，验证 226 passed 且 `git status web/voicings.json` 为空。

**③ 实测案例**：`plugin/scripts/serve.py` 在「score 同目录写不了」时兜底复制到
`<repo>/samples/preview.json`，而 `samples/preview.json` **是被 git 跟踪的**（`git ls-files` 可见），
于是打开一次预览就在仓库里留下脏 diff。已修为落到 `tempfile.mkdtemp()`。

**通用规则**：
- 会写盘的 `build_*` / `export_*` 函数，在单测里**必须**传 `tmp_path`，绝不能用 `REPO_ROOT`。
- 脚本的兜底写入**不能**落到仓库内被跟踪的路径；要么 temp，要么 `.gitignore` 覆盖的 `work/`。
- 复制型部署（技能目录）必须校验副本，`install.py` 现在会逐字节比对 `SKILL.md`。
- 自查命令：改完跑一遍 `git status --porcelain`，出现意料之外的文件就是这一类。

## 24. 环境自检

```bash
.venv/Scripts/python -m aitsm.cli doctor          # 项目自身
.venv/Scripts/python plugin/scripts/doctor.py     # DSH 插件入口
```

输出会列出 ffmpeg/ffprobe 路径、各引擎可用性、模型缓存、schema 文件、工作目录。

---

# v2 专用模型链路的坑（实测记录）

以下坑都是 v2 接入四个专用模型时**实际踩到并已修复**的。共同教训：
「权重能下载」≠「能加载」≠「能跑」≠「输出可信」——这四件事要分别验证。

## 25. `vocalparse` 这个包**从来不存在于 PyPI**

**症状**：`pip install vocalparse` → 404；找不到 VocalParse 的安装方式。

**真相**：VocalParse 不是独立分发的 pip 包。它是基于 **Qwen3-ASR** 的微调权重，
而 Qwen3-ASR 的**建模代码**由 `qwen-asr` 包分发（PyPI 上有，最新 0.0.6）。

**解法**：

```bash
.venv/Scripts/python.exe -m pip install --only-binary :all: qwen-asr==0.0.6
```

**重要**：只装 `transformers` 时 `qwen3_asr` 这个 model_type 能被识别，但通用
`Qwen3ASRForConditionalGeneration` 按 `vocab=151936 / output_dim=3584` 建图，
与 checkpoint 的 `152101 / 2048` 冲突——实测 **315 个 key MISSING**。

## 26. Qwen3-ASR 权重前缀与通用类不匹配

**症状**：`size mismatch`、大量 `MISSING`，加载后推理输出乱码。

**原因**：VocalParse 的 checkpoint 是 **Thinker 结构**：

- 键名 `thinker.audio_tower.*` / `thinker.model.*` / `thinker.lm_head.*`；
- audio `output_dim = 2048`（通用类期望 3584）；
- 词表 `152101`（基础 151936 + 约 165 个 AST token：`<P_n>` / `<NOTE_x>` / `<BPM_n>`）；
- `config.json` **没有** `rope_scaling`，但建模代码要求它存在。

**解法**（`aitsm/transcribe/vocal.py`）：

```python
from qwen_asr...modeling_qwen3_asr import Qwen3ASRThinkerForConditionalGeneration
config = Qwen3ASRConfig.from_pretrained(dir).thinker_config
config.text_config.vocab_size = 152101
config.text_config.rope_scaling = {"type": "mrope", "mrope_section": [24, 20, 20]}
# 键映射：thinker.audio_tower.→audio_tower.  thinker.model.→model.  thinker.lm_head.→lm_head.
# lm_head 与 embed_tokens tied（checkpoint 里根本没有 lm_head）
```

`mrope_section=[24,20,20]` 不是猜的，是 `qwen_asr` 建模代码自己的缺省值；
设错会让位置编码**静默失效**，音符时序系统性偏移。适配器现在会在加载前断言它。

## 27. `import qwen_asr` 会拉 `nagisa` + `gradio`

**症状**：`ModuleNotFoundError: No module named 'nagisa'`；`pip install nagisa` 卡住（Windows）。

**原因**：包 `__init__` 连带导入推理封装（强制对齐要 `nagisa`）与 demo 依赖（`gradio`），
这些对建模推理毫无作用。

**解法**：**按文件路径**只加载三个建模子模块，不执行 `__init__`：

```python
# 造三个空壳父包，让相对导入成立，再 exec_module
for name in ("qwen_asr", "qwen_asr.core", "qwen_asr.core.transformers_backend"): ...
spec = importlib.util.spec_from_file_location(full_name, backend / f"{stem}.py")
```

**必须配版本 pin + 路径探测**：`qwen-asr` 仍是 0.0.x，子模块路径随时可能变。
`aitsm/transcribe/qwen_backend.py` 把版本号、子模块名、词表大小、mrope 集中成常量，
`check_import_paths()` 在**探测阶段**就能发现路径变化（而不是推理中途抛 ImportError）。

## 28. DuET 的 `vocabulary.json` 在 Windows 检出成 CRLF

**症状**：`ValueError: Model vocabulary does not match the installed package.`

**原因**：DuET 用 sha256 校验 `vocabulary.json`，而 git 在 Windows 上把 LF 检出成 CRLF，
哈希自然对不上。

**解法**：把该文件按 **LF** 重写；仓库内 `vendor/DuET/src/duet/vocabulary.json` 已转 LF。
CI/新机器检出后要确认（`git config core.autocrlf` 不应用于该文件）。

## 29. `duet-score` 装不上（Windows 构建目录权限）

**症状**：`error: could not create 'build\lib\duet': 拒绝访问`；`WinError 5`。

**解法**：不 pip 安装，把源码放 `vendor/DuET`，由 `aitsm/fusion/duet.py` 把
`vendor/DuET/src` 加进 `sys.path` 直接用（`gitignore` 已忽略 `vendor/`）。

## 30. SheetSage2 的 `BartDecoder` 签名变化（transformers 5.x）

**症状**：`TypeError: BartDecoder.__init__() got an unexpected keyword argument 'embed_tokens'`

**原因**：SheetSage2 权重按 **transformers 4.45.2** 导出，5.x 移除了该参数。

**解法**：锁 `transformers==4.57.6`（同时满足 VocalParse 的 4.57.6 与 SheetSage2）。
这是 `requirements.txt` 里必须精确 pin 的两处之一。

## 31. SheetSage2 的 `trust_remote_code` 连不上 Hub

**症状**：`OSError: We couldn't connect to 'https://hf-mirror.com/' ... couldn't find them in cached files`
（明明权重已下到本地目录）。

**原因**：`trust_remote_code=True` 走 HF 动态模块缓存，其目录结构（`transformers_modules/<repo>/<hash>`）
与本地权重目录名不匹配，于是它以为要重新联网。

**解法**：**本地包导入**——把权重目录复制成
`<HF_HOME>/pkgs/ss2/sheetsage2/`（含 `__init__.py`），然后
`from sheetsage2.modeling_sheetsage2 import SheetSage2Model`。`trust_remote_code` 只作兜底。

## 32. SheetSage2 把 4/4 判成 2/4

**症状**：整曲 229 秒被切成 **181 个「小节」**（每小节 1.26 秒），谱面小节数翻倍、和弦落位全错。

**原因**：`beat.lab` 里拍点间隔 0.64s（93.75 BPM），强拍间隔 1.28s。模型据此报 `2/4`——
它自身是**自洽**的（强拍/拍点 = 2:1，确实符合 2/4），所以不能靠这个判据发现。
真正破绽是**每小节秒数**：2/4 且 < 1.6s/小节，意味着把八分音符当成了拍。

**解法**（`pipeline/accompaniment_stage._sanity_check_meter`）：

1. 判据用「每小节秒数」+「声明的拍号是否在常见集合」；
2. 回退 **4/4**，**BPM 保持不变**（93.75 是对的；若按 `bar_seconds/4` 反推会得到 187.5，
   那是八分音符速度）；
3. downbeats 隔一个取一个 → 91 小节 × 2.55s ≈ 229s，与整曲时长吻合；
4. 「少见但自洽」的拍号（如 5/4）**只提示不改写**，不能把正确结果改错。

**交叉验证**：旧 librosa 管线独立判 4/4 / 92.47 BPM，与 93.75 几乎一致，差异只在拍号。

## 33. SheetSage2 的段落名不在 schema 枚举里

**症状**：`analysis 未通过 schema 校验（3 处）：sections/6/kind: 'pre-chorus' is not one of [...]`

**原因**：SheetSage2 输出 `pre-chorus`（带连字符），schema 只认 `prechorus`。

**解法**：`_normalize_section_kind()` 把 `pre-chorus` / `pre chorus` / `post-chorus` /
`breakdown` / `guitar solo` 等变体映射到标准 kind；**未知标签一律落 `other`，不猜**。

## 34. 段落能量默认值被当成真实数据

**症状**：20 个段落全部用同一个节奏型（`pop8`），谱面毫无层次。

**原因**：`contracts.Section.energy` 默认 0.5，而 SheetSage2 不提供段落能量，
适配器又只在 `energy` 为空时才取先验 → 「默认值 0.5」被当成真实测量值。

**解法**：按 `kind` 给**类型先验**（intro 0.20 / verse 0.45 / prechorus 0.60 /
chorus 0.75 / interlude 0.35 / bridge 0.50 ...）。修复后前奏→分解、副歌→摇滚扫弦。

## 35. 多进程部署会 OOM

**症状**：`uvicorn --workers 4` 时机器被打爆。

**原因**：VocalParse 是 2B 参数，加载后常驻约 **8.5 GB**；4 个 worker 各加载一份 ≈ 34 GB。

**解法**：**单 worker + 内部队列**（uvicorn 默认就是 1）。要横向扩展就把 VocalParse
拆成独立推理服务，模型只加载一份。`aitsm/deployment.py` 把这条约束做成了可执行检查，
`cli doctor` 会打印模型内存估算与部署建议。详见 `12-operations.md` §2。

## 36. 连续转录两段时第二段降级

**症状**：同一进程先后转录两段音频，第二段回落 `whisper+pyin`。

**原因**：每次调用都重新加载 2B 权重（约 8 GB），内存压力导致后续失败。

**解法**：`_MODEL_CACHE` 按「权重目录 + 设备」缓存，同进程复用。

## 37. 设备选择：有显卡 ≠ 能用 GPU

**症状**：机器有 RTX 4060，但 `torch.cuda.is_available()` 为 False。

**原因**：装的是 `torch 2.x+cpu`（编译时没带 CUDA）。这两件事必须**分别告知用户**：
「没有卡」和「有卡但 torch 是 CPU 版」的解法完全不同。

**解法**：`aitsm/devices.py` 同时读 torch 与 `nvidia-smi`，`/api/health.devices.summary`
给出人话结论；前端下拉在无 GPU 时标注「GPU（本机不可用）」并说明原因；
后端 `resolve_device()` 校验并回落 CPU，写 `device.resolved` 告警。
**绝不假装在用 GPU。**

## 38. 一份 2.5 GB 的 CUDA torch 不能随便降级

**说明**：启用 GPU 需要 `--index-url .../cu124` 的 torch（本机 2.6.0+cu124，约 2.5 GB），
而从 2.14.1 降下来会波及已验证的 demucs / transformers / qwen-asr 组合。

**做法**：v2 交付时**没有**擅自降级；改为如实报告 GPU 不可用 + 给出启用命令。
真要启用时建议新建一个 venv 验证，而不是在已验证环境里直接替换。

## 39. GitHub HTTPS 被阻断时怎么推送

**症状**：`git push` → `Failed to connect to github.com:443`（curl 也返回 `000`），
但 `gh api` 正常（`api.github.com` 返回 200）。

**解法**：`scripts/push_via_api.py` 走 **Git Data API**（blob → tree → commit → ref）。
其中两个必须注意的点：

1. **内容要塞 stdin**：base64 后的文件内容做成命令行参数会触发
   `WinError 206（文件名或扩展名太长）`，必须 `gh api --input -` 从 stdin 传 JSON；
2. **必须读 git 对象库**（`git cat-file blob HEAD:path`）而不是工作区文件：
   工作区是 CRLF、git 里是 LF，直接上传工作区字节会让远端 blob sha 与本地提交不一致。

API 重建的 commit sha 与本地不同（tree 相同）。恢复直连后
`git fetch origin && git reset --mixed origin/main` 对齐即可。
