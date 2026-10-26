# v2 五层流水线：专用模型优先、缺权重时显式降级

> 本文描述 v2 的重写目标与落地现状。v1 的模块接口见 `docs/02-architecture.md`，
> 仍然有效：v2 把「转录」与「编配」拆成两层，中间用一份**符号契约**连接。

## 1. 为什么要拆

v1 里「音频分析 → 编配」是同一条链上的连续函数，编配层能读到音频派生的任何东西。
v2 的硬约束是：

- **转录层**只产符号（节拍/调性/和弦/段落/旋律/歌词），必须来自该任务最强的专用组件；
- **编配层**只吃符号（MusicXML / `symbolic.json`），不许碰音频、不许做基频估计；
- 每层记录 `requested`（架构首选组件）与 `engine`（本机实际跑的），差距写进
  `provenance` 与 `manifest.degraded`，**降级可见、可审计**；
- 大模型（Claude/GPT-4o 一类）只允许出现在编配层的符号决策里，绝不进入音频转录链路。

## 2. 层与代码

| 层 | 架构首选 | 代码入口 | 本机可用条件 |
| --- | --- | --- | --- |
| 0 分离 | htdemucs_ft | `aitsm/separation/engine.py::separate` | `AITSM_DEMUCS_ONNX_DIR` 指向含 `bag_infer.py` 的权重目录（只需 numpy+onnxruntime+soundfile）；否则 torch+demucs |
| 1A 人声联合转录 | VocalParse | `aitsm/transcribe/vocal.py::transcribe_vocal` | `AITSM_VOCALPARSE_DIR` + transformers 本地权重 |
| 1B 伴奏联合转录 | SheetSage2 | `aitsm/transcribe/accompaniment.py::transcribe_accompaniment` | `AITSM_SHEETSAGE2_DIR` + transformers |
| 2 骨架融合 + 量化 | SheetSage2 拍点 + DuET | `aitsm/fusion/skeleton.py::fuse`、`aitsm/fusion/quantize.py::quantize_symbolic` | `AITSM_DUET_DIR` + transformers |
| 3 编配（唯一自研，纯符号） | DP 指法 + TART rerank + 规则节奏型 + 可选 LLM 评审 | `aitsm/arrangement/decisions.py::build_plan`，接入点在 `aitsm/score/arrange.py` | 永远可用（纯 CPU） |
| 4 渲染 | MusicXML + JSON payload + VexFlow / abcjs | `aitsm/render/payload.py`、`aitsm/render/timing.py`、`aitsm/render/abc.py` | 永远可用 |

数据流：

```text
audio
  └─(0) separation/engine ──> stems{vocals,other}
        ├─(1A) transcribe/vocal ─────────┐
        └─(1B) transcribe/accompaniment ─┤
                                          ▼
                        pipeline/symbolic_stage.build_symbolic
                     (2) fusion.skeleton.fuse + fusion.quantize
                                          ▼
                     symbolic.json + symbolic.musicxml   ← 契约 aitsm.symbolic/v1
                                          ▼
                     score/arrange.py  ← arrangement/decisions.build_plan（3）
                                          ▼
                    score.json + score.musicxml + payload.json（4）
```

## 3. 符号契约（`aitsm/contracts.py`）

`SCHEMA = "aitsm.symbolic/v1"`。核心类型：

- `BeatGrid(bpm, beats, downbeats, beats_per_bar, time_signature, confidence, pickup_beats, first_downbeat_sec, source)`
- `KeySignature(tonic, mode, confidence, fifths)`
- `ChordEvent(start, end, symbol, confidence, source)`
- `Section(start, end, kind, label, energy, repeat_of, confidence)`
- `MelodyNote(start, end, midi, confidence, lyric, syllable_index, source)`
- `LyricWord` / `LyricLine(start, end, text, words, confidence, interjection)`
- `VocalResult(notes, lyrics, bpm, engine, requested="VocalParse", alignment, warnings)`
- `AccompanimentResult(beats, key, chords, sections, melody_skeleton, abc, events, engine, requested="SheetSage2")`
- `SeparationResult(engine, model, device, stems, warnings, detail)`
- `SymbolicScore(...)`：`from_analysis` / `to_analysis` / `to_musicxml` / `from_musicxml` /
  `save(dir)` → `symbolic.json` + `symbolic.musicxml` / `load(dir)`（优先 MusicXML + sidecar）
- `Provenance(layer, engine, requested, model, version, ok, cached, elapsed_seconds, detail, warnings)`

`SymbolicScore.to_analysis()` 产出的形状与 v1 `analysis.json` 兼容，所以 v1 的
`arrange()` 不改造也能吃符号契约；v2 编配增量是叠在上面的（见第 6 节）。

读回 MusicXML 的已知损失（有 sidecar 时全部还原）：歌词行按间隙重新分组、置信度与
精确浮点时间、provenance、段落 `energy/repeat_of`、单声部限制（叠音）、只写全局 tempo。

## 4. 降级矩阵（本机无权重时的实际行为）

| 层 | requested | 本机 engine | 后果 |
| --- | --- | --- | --- |
| 分离 | htdemucs_ft | `htdemucs` 或 `demucs-fallback` | 人声残留略多；`detail` 里带 `hq_stems/shifts/overlap` |
| 1A 人声 | VocalParse | `whisper+pyin` | 歌词与旋律是两个独立模型，联合解码的时值一致性没有（**已可切到真实 VocalParse**，见 §4.2） |
| 1B 伴奏 | SheetSage2 | **`sheetsage2`（已实测跑通）** | 联合解码节拍/调性/和弦/段落，全局一致 |
| 2 量化 | DuET | **`duet`（已实测跑通）** | 真实 performance MIDI → 记谱节奏 |
| 2 融合 | fusion.skeleton | `rule-skeleton` | 规则补齐旋律空档，`source="skeleton"` 置信度 ×0.85 |
| 3 指法 | DP + TART | `fingering-dp` + `tart-heuristic` | TART 只用符号启发式打分，不做音频条件 |
| 3 节奏型 | 规则 + LLM | `rules` | 无 LLM 建议，纯段落能量 + 拍号匹配 |
| 3 LLM 评审 | Claude/GPT-4o | 停用 | 仅在 `AITSM_LLM_URL` + `AITSM_LLM_ADVISOR=1` 时联网 |

降级判定在 `pipeline/symbolic_stage.py::_is_degraded`：`ok=False` 或
`engine` 不在该层的架构引擎别名集合里，就进 `degraded_layers`。

### 4.1 权重下载（镜像 + 幂等）

```bash
.venv/Scripts/python.exe scripts/fetch_models.py --list          # 看哪些已就绪
.venv/Scripts/python.exe scripts/fetch_models.py --only sheetsage2 duet
.venv/Scripts/python.exe scripts/fetch_models.py --all
```

脚本固定 `HF_ENDPOINT=https://hf-mirror.com`、关闭 Xet 传输（Windows 上会报
`I/O error 5`），下载后校验关键文件，并打印对应的环境变量。

| 仓库 | 本地目录 | 体积 | 层 |
| --- | --- | --- | --- |
| `m-a-p/SheetSage2` | `models/SheetSage2` | 0.23 GB | 1B 伴奏联合转录 |
| `Wayne42/DuET` | `models/DuET` | 0.43 GB | 2 节奏量化 |
| `pymaster/VocalParse` | `models/VocalParse` | 4.1 GB | 1A 人声联合转录 |
| `StemSplitio/htdemucs-ft-onnx` | `models/htdemucs_ft_onnx` | ~0.2 GB | 0 分离（免 PyTorch） |

### 4.2 版本坑（实测记录，避免重复踩）

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| `DuET: Model vocabulary does not match` | `vocabulary.json` 被 Windows 检出成 **CRLF**，sha256 对不上 | 按 LF 重写（见 `vendor/DuET/src/duet/vocabulary.json`） |
| `duet-score` 装不上（`could not create build\lib\duet`） | pip 构建目录权限 | 源码放 `vendor/DuET`，走 `sys.path` 导入（`aitsm/fusion/duet.py` 已内置） |
| SheetSage2 `BartDecoder.__init__() got an unexpected keyword argument 'embed_tokens'` | 权重按 transformers **4.45.2** 导出，5.x 改了签名 | 装 `transformers==4.57.6`（同时满足 VocalParse 的 4.57.6 与 SheetSage2） |
| SheetSage2 `couldn't connect to hf-mirror` | `trust_remote_code` 的模块缓存路径与本地目录名不匹配 | 改为**本地包导入**（复制到 `.cache/pkgs/ss2/sheetsage2`），`trust_remote_code` 仅作兜底 |
| `mir_eval` / `pretty_midi` / `music21` 缺失 | 可选依赖 | `pip install mir_eval pretty_midi music21` |
| VocalParse 权重加载后 315 个 key 报 MISSING | checkpoint 用 `thinker.*` 前缀，标准 transformers 的 `Qwen3ASRForConditionalGeneration` 期望 `model.*` | **根因是缺 `qwen-asr` 包**：Qwen3-ASR 建模代码不在 transformers 里。装 `qwen-asr`（PyPI 上有）后用 `Qwen3ASRThinkerForConditionalGeneration` + 键重映射 |
| VocalParse `embed_tokens 152101 vs 151936`、`projector 2048 vs 3584` | checkpoint 是 Thinker 结构；通用类默认按 151936 / 3584 建图 | 用 Thinker 类；`vocab_size` 覆盖成 **152101**（AST token 扩展）；`rope_scaling` 补 mrope 默认值；`lm_head` 与词嵌入 tied |
| `import qwen_asr` 报缺 `nagisa` / `gradio` | 包 `__init__` 会连带导入推理封装与 demo 依赖 | 适配器**按文件路径**只加载 `configuration/processing/modeling` 三个子模块，不执行 `__init__` |
| 连续转录两段时第二段降级 | 每次重新加载 2B 权重（~8GB），内存压力大 | 加进程级模型缓存（按「权重目录 + 设备」） |

### 4.3 hf-mirror 上的仓库核实（2026-10 实测）

| 仓库 | 存在 | 大小 | 备注 |
| --- | --- | --- | --- |
| `pymaster/VocalParse` | ✅ | 4.09 GB | 有 `ngoctham/VocalParse` 副本 |
| `m-a-p/SheetSage2` | ✅ | 0.23 GB | 代码 + 权重同仓（142 文件），另有 GGUF 版 |
| `Wayne42/DuET` | ✅ | 0.43 GB | 自定义导出格式，需配 GitHub 代码 |
| `adefossez/HTDemucs-ft` | ✅ | 4 个 safetensors | 官方 torch 权重，需 demucs 加载 |
| `StemSplitio/htdemucs-ft-onnx` | ✅ | 四轨 ONNX ×2（fp32/fp16）+ 官方 `bag_infer.py` | 首选路径；**不需要** `demucs_onnx` 包（`bag_infer` 只用 numpy+onnxruntime+soundfile） |
| `Wayne42/DuET-demo` | ⚠️ 401 | — | 是 Space 不是 model 仓库，API 查模型会 401 |
| TART / Noise2Fret | ❌ 无 HF | — | 只有 GitHub 代码，无独立模型仓库 |

## 5. 运行时开关

### 5.1 模型权重与后端（v2 核心）

| 环境变量 | 作用 |
| --- | --- |
| `AITSM_DEMUCS_ONNX_DIR` | 指向含 `*.onnx` + `bag_infer.py` 的目录，走 ONNX 推理（免完整 PyTorch） |
| `AITSM_DEMUCS_MODEL` | 显式覆盖分离模型名，默认 `htdemucs`；`--quality high/max` 会设成 `htdemucs_ft` |
| `AITSM_DEMUCS_SHIFTS` | 随机位移次数（越大越稳越慢） |
| `AITSM_DEMUCS_OVERLAP` | 分段重叠率（0–0.95） |
| `AITSM_VOCALPARSE_DIR` | VocalParse 权重目录 |
| `AITSM_SHEETSAGE2_DIR` | SheetSage2 权重目录 |
| `AITSM_SHEETSAGE2_PKG_DIR` | SheetSage2 代码副本目录（默认 `<HF_HOME>/pkgs/ss2`） |
| `AITSM_SHEETSAGE2_FIRST` | 设 `0` 关闭「SheetSage2 接管节拍/调性/和弦/段落」，强制旧管线 |
| `AITSM_DUET_DIR` | DuET 量化权重目录 |
| `AITSM_DUET_VENDOR` | DuET 源码根（默认 `vendor/DuET`） |
| `AITSM_TART_DIR` | TART 权重目录（**本机不可用**；缺权重时 rerank 走 `tart-heuristic`） |
| `HF_HOME` | HF 缓存根；本机指向仓库内 `.cache/hf`（用户目录不可写） |

### 5.2 推理设备

| 环境变量 | 作用 |
| --- | --- |
| `AITSM_DEVICE` | 默认推理设备 `auto` / `cpu` / `cuda`；也可在页面上选（`options.device`） |
| `AITSM_WHISPER_DEVICE` | 仅歌词 ASR 的设备覆盖 |
| `AITSM_WHISPER_MODEL` | 歌词模型档位（`tiny`/`base`/`small`/`medium`/`large-v3`） |
| `AITSM_WHISPER_COMPUTE` | ASR 计算类型（如 `float16`/`int8`） |

设备解析顺序：`options.device` → `AITSM_DEVICE` → `auto` → `aitsm.devices.resolve_device()`
校验（GPU 不可用则回落 CPU 并写 `device.resolved` 告警）。

### 5.3 编配层 LLM 顾问（默认关）

| 环境变量 | 作用 |
| --- | --- |
| `AITSM_LLM_URL` | 第三方大模型 HTTP 端点；未设置时编配层**不联网** |
| `AITSM_LLM_ADVISOR` | 设为 `1` 才启用 LLM 评审（默认关） |
| `AITSM_LLM_MODEL` | 请求里使用的模型名 |
| `AITSM_LLM_API_KEY` | 鉴权头 |

### 5.4 降级路径的调参开关（v1 内核，保留）

| 环境变量 | 作用 |
| --- | --- |
| `AITSM_CHORD_FUSION` / `_HARMONIC` / `_SMOOTHING` / `_REPETITION` | 和弦识别特征融合 / 谐波加权 / 序列平滑 / 重复一致性 |
| `AITSM_RHYTHM_QUANTIZER` | 量化器选择（`hmm` 等） |
| `AITSM_QUANTIZE_MAX_SHIFT` | HMM 允许的最大位移（拍） |
| `AITSM_MELODY_TRIPLETS` / `_LOW_MIDI` / `_HIGH_MIDI` | 旋律三连音开关与音域约束 |
| `AITSM_F0_BANDPASS` / `_LOW` / `_HIGH` / `_SMOOTH` / `_MAX_TRANSITION` / `_SWITCH_PROB` | 基频带通与平滑参数 |
| `AITSM_SKIP_ASR` | 跳过歌词识别（离线调试） |
| `AITSM_KEEP_TRADITIONAL` | 中文歌词不转简体 |

### 5.5 服务与任务

| 环境变量 | 作用 |
| --- | --- |
| `AITSM_HOST` / `AITSM_PORT` | 服务监听地址/端口 |
| `AITSM_WORK_DIR` | 工作根目录（默认 `work/`） |
| `AITSM_MAX_UPLOAD_MB` / `AITSM_MAX_DURATION_SEC` | 上传体积/时长上限 |
| `AITSM_WORKERS` / `AITSM_MAX_WORKERS` | 任务线程数（**注意**：这是任务并发，不是 uvicorn worker；大模型下不要开多个 uvicorn worker，见 `12-operations.md`） |
| `AITSM_KEEP_JOBS` | 保留多少个历史任务目录 |
| `AITSM_FFMPEG` / `AITSM_FFPROBE` | 显式指定二进制路径 |
| `AITSM_ENABLE_SEPARATION` / `_LYRICS` / `_MELODY` | 各阶段总开关（等价于 CLI 的 `--no-*`） |

`--quality {standard,high,max}` 只调分离/ASR 的精度旋钮，不改分层结构。

## 6. 编配层（v2 增量）

`arrangement/decisions.py::build_plan(score, *, options=None, advisor=None, meta=None,
max_candidates=8, weights=None, tuning=STANDARD_TUNING) -> ArrangementPlan`

- 指法：DP 在候选把位 DAG 上求最短路，成本
  `position_shift 1.2 / hand_span 0.5 / open_string 0.15(奖励) / switch 0.8 / melody_cover 2.0(奖励)`，
  不可演奏 `+25`；TART 适配器作为候选 rerank（本机为 `tart-heuristic`）。
- 节奏型：按段落能量 + 拍号选模板；拍号不符的模板被拒绝并写
  `arrangement.strum_meter_mismatch`（12/8 不能被 4 拍模板覆盖）。
- LLM：只返回结构化 JSON（`{"section","suggested_strumming","chord_substitutions":[{"original","suggested","reason"}]}`），
  当候选池参与评分，不直接写谱；异常一律吞掉并回落 `rules`。

接入点（`score/arrange.py`）：`options.arrangement = "auto" | "off"`、`options.llm_advisor`；
plan 命中不了的 `(start_sec, symbol)` 自动回退 v1 的 `_choose_voicing`，因此
「新决策层挂掉」只会让谱面回到旧行为，不会失败。`score["arrangement"]` 记录
`choices/strumming/total_cost/engines/applied`。

## 7. 渲染与 API

`payload.json`（`aitsm.payload/v1`）一次给全前端要的东西：

```text
schema title meta sections stats musicxml abc timing chord_diagrams audio_url
timing = {bars, notes, chords, lyrics}   # 播放跟随用
chord_diagrams = {tuning, capo, voicings, bars[{index, chords[{beat,symbol,voicing_id,diagram}]}], count}
```

前端主渲染 VexFlow（Stave + TabStave + 歌词 + VexChords）；`web/js/abcjs-fallback.js`
在 VexFlow 失败或只有 ABC 时兜底（懒加载 vendor 再 CDN，本地 3s / CDN 3.5s 超时）。

同一份时间轴另存为独立的 `timing.json`（内容与 `payload.timing` 完全一致），
播放器/第三方不必解析整个 payload：

```text
timing.json = { bars[{index,start,end}], notes[{start,end,midi,lyric,bar}],
                chords[{start,end,symbol,bar}], lyrics[{start,end,text,bar}] }
```

`web/js/main.js` 消费策略：任务完成后并行取 `score.json` 与 `payload.json`，
`web/js/payload.js`（纯函数、node 可测）负责归一化——ABC 取 payload 版本
（`pickAbc`），播放/高亮的绝对时间用 payload 时间轴覆盖视图（`attachTimeline`，
只改视图不改 `raw`，编辑与导出仍是原始数据）；payload 缺失或 409 时静默退回
`score.json` 原有行为。

新增路由：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/jobs/{id}/symbolic` | 符号契约（provenance 在这里） |
| GET | `/api/jobs/{id}/payload` | 渲染 payload |
| GET | `/api/jobs/{id}/timing` | 绝对时间轴（`timing.json`） |
| GET | `/api/jobs/{id}/export/symbolic` | 转录层原生 `symbolic.musicxml` |
| GET | `/api/jobs/{id}/export/payload` | `payload.json` |
| GET | `/api/jobs/{id}/export/timing` | `timing.json` |

`analysis.json` 新增 `symbolic`（`available/stats/paths/provenance/warnings/degraded_layers`），
`manifest.outputs.symbolic` 登记产物路径；符号阶段在 `pipeline/run.py` 里以进度 99.2 上报，
失败只写 `symbolic.failed` 告警，不影响 `analysis.json` 主交付。

## 8. 验收

- `pytest tests/ -q` → **921 passed, 1 skipped**（v2 新增 23 个测试文件；
  skipped 那条是「本地无 VocalParse 权重时跳过真实权重校验」，离线环境正常）。
- 前端纯数据层：`node tests/web-payload.test.mjs` → 12 passed；`node tests/web-editor.test.mjs` → 34 passed。
- 量化 + 契约冒烟（artifacts/analysis.json）→ `symbolic.json` + `symbolic.musicxml`
  （158 KB，89 小节，每小节 divisions 总和 == 96）。
- 整曲真实交付（229s，四层全开）→ 90 小节 / 164 和弦落位 / 质量 0.94，
  `analysis` 与 `score` 双双通过校验，详见 [`13-demo-quality.md`](13-demo-quality.md)。
- 端到端脚本：`work/v2_e2e_run.py`（16s 截取，CPU）、`work/demo_full_song.py`（整曲）。

## 9. 多曲目基准

权重到位前先把「怎么量」固定下来，避免只凭单曲手感判断模型好坏：

```bash
.venv/Scripts/python -m aitsm.cli bench samples/ --out work/bench
```

- 每首曲子写到 `work/bench/NN-<名字>/`，内部就是一次完整 `run_job` 的产物；
- 汇总写 `benchmark.json`（机器读）与 `benchmark.md`（人读）；
- 表格列：时长、耗时、分离/人声/伴奏/量化四层的 **实际 engine**、总分与分项质量、
  小节数、降级层。engine 后面带 `⚠` 表示该层没跑到架构首选组件；
- 「环境」段记录 `demucs` 后端、`vocalparse`/`sheetsage2`/`duet` 是否可用，
  这样换机器或换权重后的数字才有解释力；
- 单曲失败不会中断整批（失败行写明原因），CLI 退出码在有失败时为 1。

对比方式：先跑一次存 `work/bench-before`，设置好 `AITSM_DEMUCS_ONNX_DIR` /
`AITSM_VOCALPARSE_DIR` / `AITSM_SHEETSAGE2_DIR` / `AITSM_DUET_DIR` 后再跑
`work/bench-after`，两张 `benchmark.md` 逐行对比——只有 `⚠` 消失、且对应分项上升，
才说明专用模型真的接上了。

## 10. 实测结果（真实权重，16s 真实片段）

配置：`transformers==4.57.6`、`AITSM_SHEETSAGE2_DIR=models/SheetSage2`、
`AITSM_DUET_DIR=models/DuET`、`HF_HOME=.cache/hf`、CPU。

```text
$ .venv/Scripts/python.exe work/v2_e2e_run.py     # 端到端，683s（16s 音频，CPU）
$ .venv/Scripts/python.exe work/v2_layers_run.py  # 逐层验证
```

| 层 | requested | engine | 状态 |
| --- | --- | --- | --- |
| 分离 | htdemucs_ft | **`demucs-onnx`（bag_infer）** | ✅ 真实首选模型 |
| 1A 人声 | VocalParse | **`vocalparse`（qwen_asr Thinker）** | ✅ 真实模型 |
| 1B 伴奏 | SheetSage2 | **`sheetsage2`** | ✅ 真实模型 |
| 2 量化 | DuET | **`duet`** | ✅ 真实模型 |
| 3 编配 | DP + TART | `dp` + `tart-heuristic` | 纯符号（TART 无 HF 权重） |

**`analysis.symbolic.degraded_layers` = `[]`**——四层全部命中架构首选组件。

VocalParse 实测（12s 有人声片段，CPU）：加载 34.7s + 生成 ~120s；verse 段解出
**123 个音符 / 26 个带词**、BPM 89（来自 `<BPM_>` token）；16s 前奏片段只解出
2 个音（SheetSage2 也把那段判为 `intro`，两者互相印证，不是模型故障）。

htdemucs_ft ONNX 实测（14s 音频，CPU）：4 个 sub-model × 3 chunk = 12 次 ONNX 推理，
**57.7s**（RTF 2.21），输出 4 轨 44.1kHz。注意 ONNX 输入必须是 44.1kHz 双声道，
流水线的 22.05kHz 分析音轨会先升采样（`separation.onnx_input_44k.wav`）。

SheetSage2 在实际 14s 片段上的输出（`score.abc` / `*.lab`）：

```text
M:4/4  L:1/16  Q:1/4=186  K:Bm
"B7sus4"z4A4f2e2d4|"Gmaj7"d16-|"Gmaj7"d8z2e2f2a2-|"D/A"a16-|
beat.lab     44 个拍点，4/4
chord.lab    11 个和弦（含 N=无和弦）
structure.lab chorus / interlude
melody_vocal.mid  16 个旋律音
```

DuET 在 12 音 performance MIDI 上：`input_coverage=1.0`，8 小节 / 21 音符 / divisions=480。

完整细节与踩过的版本坑见 §4.1–4.3。

## 11. 当前仍缺什么

| 缺口 | 原因 | 解锁方式 |
| --- | --- | --- |
| GPU 加速 | 本机 CPU-only，VocalParse 生成一段 12s 音频约 2 分钟 | 装 CUDA 版 torch 即可自动用 GPU |
| TART rerank | 论文只放 GitHub 代码，无 HF 权重 | 需自行克隆并适配 T5 弦位模块 |
| Noise2Fret 损失 | 同上，只有 GitHub | 可从代码提取五项损失作为 DP 成本项（当前用等价启发式） |
| 长音频分段 | VocalParse 官方建议 >30s 先分段 | 流水线按 `duration` 截取；整曲需要外层切窗 |

> **不再缺**：VocalParse 已跑通。关键在于它不是标准 transformers 模型——
> Qwen3-ASR 建模代码由 `qwen-asr` 包分发（PyPI 上有，非 `vocalparse` 包），
> 且必须用 `Qwen3ASRThinkerForConditionalGeneration` 加载、词表按 152101 覆盖。
> 详见 §4.2 的版本坑表。
