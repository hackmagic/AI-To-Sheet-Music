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

**原因**：DSH 加载的是 `%USERPROFILE%\.config\opencode\skills\audio-to-chord-sheet\scripts\generate.py`
（`plugin/install.py` 复制过去的副本），不是仓库里的那份。

**解法**：改 `plugin/` 之后**必须**重跑 `python plugin/install.py`，再用
`diff` 确认副本一致。DSH 会自动重扫 skill 目录，不需要重启进程。

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

## 24. 环境自检

```bash
.venv/Scripts/python -m aitsm.cli doctor          # 项目自身
.venv/Scripts/python plugin/scripts/doctor.py     # DSH 插件入口
```

输出会列出 ffmpeg/ffprobe 路径、各引擎可用性、模型缓存、schema 文件、工作目录。

