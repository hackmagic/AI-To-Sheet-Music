# 总体架构与接口契约（v1 内核 + v2 分层）

> v2 的转录/编配分层、符号契约与降级矩阵见 [`11-v2-pipeline.md`](11-v2-pipeline.md)；
> 部署约束、GPU 迁移与实测质量见 [`12-operations.md`](12-operations.md)、
> [`13-demo-quality.md`](13-demo-quality.md)。
>
> **这里的 v1 模块接口仍然有效**：v2 没有抛弃它们，而是在外层加了
> 「符号契约 + 可插拔模型后端」。v1 的 `pipeline/*` 现在是 v2 的**降级路径**
> （`beats+chroma` / `whisper+pyin` / `hmm`）。

## 1. 目录结构

```text
AI-To-Sheet-Music/
├── aitsm/
│   ├── __init__.py
│   ├── config.py            # 路径/环境变量/模型选择
│   ├── api.py               # FastAPI：上传、任务、状态、产物、静态站点
│   ├── jobs.py              # 内存任务队列 + 进度事件 + 产物目录
│   ├── cli.py               # generate / serve / bench / doctor / voicings / demo
│   ├── media.py             # ffmpeg/ffprobe 封装（ingest）
│   ├── schema.py            # JSON Schema 加载与校验
│   ├── service.py           # 端到端编排（分析→编配→校验→导出→payload）
│   ├── contracts.py         # ★ v2 符号契约（aitsm.symbolic/v1）与 Provenance
│   ├── devices.py           # ★ CPU/GPU 探测（/api/health.devices）
│   ├── deployment.py        # ★ 模型内存预算与多 worker OOM 检查
│   ├── benchmark.py         # ★ 多曲目基准（逐层实际 engine + 质量对比）
│   │
│   ├── separation/          # ★ 层 0：分离
│   │   └── engine.py        #   htdemucs_ft：ONNX bag_infer → torch demucs → HPSS
│   ├── transcribe/          # ★ 层 1：联合转录
│   │   ├── vocal.py         #   VocalParse（Thinker 类 + 30s 分段）/ whisper+pyin
│   │   ├── qwen_backend.py  #   qwen-asr 版本 pin、路径探测、mrope 校验
│   │   └── accompaniment.py #   SheetSage2（lab 解析）/ beats+chroma
│   ├── fusion/              # ★ 层 2：骨架融合与量化
│   │   ├── skeleton.py      #   fuse / snap / performance MIDI
│   │   ├── musicxml_io.py   #   符号契约的 MusicXML 写读
│   │   ├── quantize.py      #   DuET 优先 / HMM 降级
│   │   └── duet.py          #   DuET 适配器（权重 + vendor 源码）
│   ├── arrangement/         # ★ 层 3：编配决策（纯符号，不读音频）
│   │   ├── fingering_dp.py  #   DP 指法全局最优
│   │   ├── tart_rerank.py   #   TART rerank（本机 heuristic）
│   │   ├── strumming_rules.py  # 段落节奏型
│   │   ├── llm_advisor.py   #   第三方大模型评审（默认关）
│   │   ├── playability.py   # ★ Noise2Fret 五项损失（numpy 移植）
│   │   └── decisions.py     #   build_plan → ArrangementPlan
│   ├── render/              # ★ 层 4：渲染
│   │   ├── payload.py       #   aitsm.payload/v1
│   │   ├── timing.py        #   绝对时间轴
│   │   └── abc.py           #   ABC 兜底
│   │
│   ├── pipeline/            # v1 内核（现为 v2 的降级路径）
│   │   ├── ingest.py        # F1/F2
│   │   ├── separate.py      # F4  Demucs + 降级（separation/engine 的底层）
│   │   ├── beats.py         # F5
│   │   ├── keydet.py        # F6
│   │   ├── chords.py        # F7
│   │   ├── structure.py     # F8
│   │   ├── lyrics.py        # F9
│   │   ├── melody.py        # F10 + F14 音节对齐
│   │   ├── quantize.py      # HMM 节奏量化（fusion/quantize 的降级路径）
│   │   ├── align.py         # 词-音符对齐
│   │   ├── accompaniment_stage.py  # ★ SheetSage2 接入 + 拍号/段落校验
│   │   ├── symbolic_stage.py       # ★ 转录层 → SymbolicScore
│   │   └── run.py           # 编排：产出 analysis.json
│   ├── testsupport/         # 合成曲评估工具（不参与线上链路）
│   │   ├── synth_song.py    # 三个 profile 的合成曲 + ground truth
│   │   └── evaluate.py      # 五维加权打分
│   ├── score/               # 编配引擎（v1 内核；v2 在其上叠加决策层）
│   │   ├── model.py         # 读写 score.json、时间/音高工具
│   │   ├── voicings.py      # F12 和弦指法库与选择
│   │   ├── strum.py         # F13 节奏型库与选择
│   │   ├── arrange.py       # F11/F15/F16 弹唱谱编配（含 v2 决策接入）
│   │   ├── validate.py      # F23 质量网关
│   │   └── export/
│   │       ├── musicxml.py  # F24
│   │       └── midi.py      # F24
├── schemas/
│   ├── analysis.schema.json
│   └── score.schema.json
├── scripts/                 # ★ fetch_models.py（下载专用模型）、push_via_api.py
├── models/                  # ★ 本地权重（git 忽略；用 scripts/fetch_models.py 获取）
├── vendor/                  # ★ 第三方源码（DuET、Noise2Fret 调研；git 忽略）
├── web/                     # 前端（无构建步骤，原生 ES Module + 本地 vendor）
│   ├── index.html
│   ├── css/app.css
│   ├── js/{main,api,renderer,chorddiagram,player,transpose,print,editor,ui,music}.js
│   ├── js/payload.js        # ★ payload 归一化 / 时间轴覆盖
│   ├── js/abcjs-fallback.js # ★ VexFlow 失败时的 abcjs 兜底
│   ├── js/musicxml.js      # ★ 本地解析导入的 MusicXML（DOMParser，无依赖）
│   └── vendor/vexflow.js    # 从 npm 复制，离线可用
├── samples/                 # 手工构造的示例 score/analysis，供前端独立开发
├── tests/                   # 45 个测试文件（921 条 pytest + 46 条 node）
├── plugin/                  # DSH skill（SKILL.md + scripts + references）
├── artifacts/               # 真实运行的交付样例
├── docs/
└── work/                    # 运行产物（git 忽略）
```

标 ★ 的是 v2 新增模块。**v1 内核没有被删掉**：专用模型不可用时，`pipeline/*` 与
`score/*` 保证链路仍能出谱，只是 `provenance.engine` 会如实标成降级实现。

## 2. 数据契约

| 文件 | schema | 产出方 | 说明 |
| --- | --- | --- | --- |
| `analysis.json` | `aitsm.analysis/v1` | `pipeline/run.py` | 分析**证据**，不含编配决定；含 `symbolic` 摘要 |
| `symbolic.json` | `aitsm.symbolic/v1` | `pipeline/symbolic_stage.py` | ★ 符号契约：节拍/调性/和弦/段落/旋律/歌词 + `provenance` |
| `symbolic.musicxml` | MusicXML 4.0 | `fusion/musicxml_io.py` | ★ 转录层原生 MusicXML（无 TAB） |
| `score.json` | `aitsm.score/v1` | `score/arrange.py` | 前端渲染与所有导出的唯一输入 |
| `score.musicxml` | MusicXML 4.0 | `score/export/musicxml.py` | 含五线谱 + TAB + 和弦 + 歌词 |
| `payload.json` | `aitsm.payload/v1` | `render/payload.py` | ★ 网页渲染数据（MusicXML + ABC + timing + 指法图） |
| `timing.json` | — | `render/timing.py` | ★ 绝对时间轴（同步播放/第三方消费） |
| `manifest.json` | `aitsm.manifest/v1` | `run.py` + `service.py` | 输入哈希、工具版本、耗时、降级原因 |
| `web/voicings.json` | — | `score/voicings.py` | 前端移调后重新选指法 |

时间单位约定：
- 音频时间：秒（float），绝对时间轴。**`symbolic.json` 与 `payload.timing` 也是秒**。
- 谱面时间：**四分音符拍**（quarter-note beats），小节内相对（beat 0 = 小节线）。
  时值可为 `number` 或 `"1/3"` 这样的分数字符串（三连音）。
- 音高：MIDI 号（整数），`score` 中一律为**记谱音高**（即 `play_key` 视角；capo 已从发声音高扣除）。

降级可追溯性（v2）：
- `symbolic.provenance[]`：每层 `{layer, engine, requested, ok, elapsed_seconds, detail}`；
- `analysis.symbolic.degraded_layers`：`engine` 不等于架构首选组件的层名列表；
- `manifest.degraded[]`：所有 warn/error 级告警。

## 3. 模块公共接口（并行开发契约，不得随意改名）

```python
# aitsm/pipeline/ingest.py
def ingest(source: Path, work_dir: Path, *, max_seconds: float | None = None) -> dict:
    """返回 {"source_wav": str, "playback_wav"/"playback_mp3": str, "duration": float,
             "sample_rate": int, "channels": int, "sha256": str, "original_name": str}"""

# aitsm/pipeline/separate.py
def separate(source_wav: Path, work_dir: Path, *, device: str = "auto") -> dict:
    """返回 {"engine": "demucs"|"fallback", "stems": {"vocals": str, "drums": str, "bass": str, "other": str},
             "warnings": [str]}"""

# aitsm/pipeline/beats.py
def analyze_beats(source_wav: Path | str) -> dict:
    """返回 {"bpm": float, "confidence": float, "beats": [float], "downbeats": [float],
             "beats_per_bar": int, "time_signature": "4/4", "beat_strength": [float]}"""

# aitsm/pipeline/keydet.py
def detect_key(source_wav: str, chords: list[dict] | None = None) -> dict:
    """返回 {"tonic": "C", "mode": "major", "confidence": float, "num_sharps_flats": 0}"""

# aitsm/pipeline/chords.py
def recognize_chords(source_wav: str, beats: dict, *, key: dict | None = None,
                     total_seconds: float | None = None) -> list[dict]:
    """返回 [{"start": 1.2, "end": 3.4, "symbol": "C", "confidence": 0.72, "source": "template"}]"""

# aitsm/pipeline/structure.py
def analyze_structure(source_wav: str, beats: dict, chords: list[dict], total_seconds: float) -> list[dict]:
    """返回 [{"start": 0.0, "end": 16.2, "kind": "intro", "label": "前奏", "energy": 0.2, "repeat_of": None}]"""

# aitsm/pipeline/lyrics.py
def transcribe_lyrics(vocals_wav: str, *, duration: float | None = None, language: str = "zh",
                      progress: Callable | None = None) -> list[dict]:
    """返回 [{"start": 1.2, "end": 4.5, "text": "…", "words": [{"start":..,"end":..,"text":".."}],
              "confidence": 0.8}]"""

# aitsm/pipeline/melody.py
def transcribe_melody(vocals_wav: str, beats: dict, *, duration: float | None = None) -> dict:
    """返回 {"notes": [{"start": 1.2, "end": 1.8, "midi": 64, "confidence": 0.8,
                        "onset": 1.21, "syllables": None}], "warnings": [str]}"""

def align_lyrics_to_melody(melody: dict, lyrics: list[dict], beats: dict) -> list[dict]:
    """给每个音符挂 lyrics 音节，返回 notes（含 lyric 字段）"""

# aitsm/pipeline/run.py
def run_pipeline(source: Path, work_dir: Path, options: dict,
                 progress: Callable[[dict], None] | None = None) -> dict:
    """编排全流程，写 analysis.json 与 manifest.json，返回 analysis dict。"""

# aitsm/score/arrange.py
def arrange(analysis: dict, options: dict) -> dict:
    """返回 score dict（aitsm.score/v1）。options: {key, capo, difficulty, style, transpose}"""

# aitsm/score/validate.py
def validate_score(score: dict) -> dict:
    """返回 {"ok": bool, "errors": [...], "warnings": [...], "stats": {...}}"""

# aitsm/score/export/musicxml.py
def export_musicxml(score: dict, out_path: Path) -> dict
# aitsm/score/export/midi.py
def export_midi(score: dict, out_path: Path) -> dict
# aitsm/score/voicings.py
def build_voicing_db() -> dict  # 写出 web/voicings.json 并返回
```

### 3.1 v2 层接口（★ 新增，均为「可用才用，否则显式降级」）

```python
# aitsm/contracts.py —— 层间唯一数据出口
@dataclass
class SymbolicScore:                      # 转录层产物 = 编配层输入
    beats: BeatGrid; key: KeySignature
    chords: list[ChordEvent]; sections: list[Section]
    melody: list[MelodyNote]; lyrics: list[LyricLine]
    meta/audio/source: dict; provenance: list[Provenance]; warnings: list[dict]
    # from_analysis / to_analysis / to_musicxml / from_musicxml / save / load / stats

@dataclass
class Provenance:                          # 每层「请求了什么 vs 实际用了什么」
    layer: str; engine: str; requested: str; ok: bool; elapsed_seconds: float | None
    detail: dict; warnings: list[str]

# aitsm/separation/engine.py —— 层 0
def separate(source_wav, work_dir, *, device="auto", model=None, prefer_onnx=True) -> SeparationResult
def onnx_available() -> bool               # bag_infer 脚本 + onnxruntime 就绪即 True

# aitsm/transcribe/vocal.py —— 层 1A
def is_available() -> bool                 # 权重目录 + qwen_asr 后端路径都成立
def transcribe_vocal(vocals_wav, *, duration=None, language="auto", beatgrid=None,
                     device="auto", progress=None, whisper_model="medium") -> VocalResult
def parse_vocalparse_tokens(token_str) -> list[VocalEvent]   # 纯字符串，可离线单测

# aitsm/transcribe/qwen_backend.py —— 依赖边界（版本 pin + 路径探测）
def probe() -> dict                        # {available, version, pinned_version, paths, reason}
def check_import_paths() -> dict           # 三个建模子模块是否还在预期位置
def validate_mrope(config) -> list[str]     # mrope_section 缺失/形状错 → 问题列表

# aitsm/transcribe/accompaniment.py —— 层 1B
def is_available() -> bool
def transcribe_accompaniment(audio_path, *, work_dir, full_mix=None,
                             device="auto", progress=None) -> AccompanimentResult
def parse_events(events) -> AccompanimentResult  # 规范化任意形状的 events

# aitsm/fusion/skeleton.py + quantize.py + musicxml_io.py —— 层 2
def fuse(vocal: VocalResult, accompaniment: AccompanimentResult, *, meta=None,
         audio=None, source=None) -> SymbolicScore
def snap_notes_to_beats(notes, beatgrid, *, tolerance=0.06) -> list[MelodyNote]
def write_performance_midi(score, path) -> dict
def quantize_symbolic(score, *, work_dir=None, device="auto") -> tuple[SymbolicScore, Provenance]
def write_musicxml(score, *, include_lyrics=True) -> str
def read_musicxml(text, *, sidecar=None) -> SymbolicScore

# aitsm/arrangement/decisions.py —— 层 3（纯符号，不读音频）
def build_plan(score, *, options=None, advisor=None, meta=None,
               max_candidates=8, weights=None, tuning=STANDARD_TUNING) -> ArrangementPlan

# aitsm/render/*.py —— 层 4
def build_payload(score, *, symbolic=None, audio_url=None, abc=None) -> dict
def build_timing(score) -> dict
def write_abc(source, *, title=None) -> str

# aitsm/devices.py —— CPU/GPU 判定（供 /api/health）
def detect_devices() -> dict               # {default, gpu_usable, summary, torch, ...}
def resolve_device(requested) -> str        # auto/cuda 不可用时回落 cpu
def device_note(requested, resolved) -> str # 给用户看的一句话

# aitsm/deployment.py —— 部署约束
def check_deployment(enabled=None, *, workers=None) -> dict
def model_memory_report(enabled=None) -> dict

# aitsm/benchmark.py —— 多曲目基准
def run_benchmark(sources, *, out_dir, options=None, runner=None, progress=None) -> dict
```

契约要点：**编配层只接受 `SymbolicScore` 或其纯数据摘要**，不得 import
`aitsm.pipeline`、不得读音频；`arrangement/*` 是这个约束的执行者。

## 4. 服务端 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/jobs` | multipart 上传音频 + options JSON（含 `device`），创建任务，返回 `{job_id}` |
| GET | `/api/jobs` | 任务列表（`?limit=`） |
| GET | `/api/jobs/{id}` | 任务状态 + 阶段进度 + 产物 URL |
| GET | `/api/jobs/{id}/score` | `score.json` |
| GET | `/api/jobs/{id}/analysis` | `analysis.json`（调试/质量面板） |
| GET | `/api/jobs/{id}/manifest` | `manifest.json` |
| GET | `/api/jobs/{id}/symbolic` | ★ `symbolic.json`（符号契约，含 `provenance`） |
| GET | `/api/jobs/{id}/payload` | ★ `payload.json`（网页渲染数据） |
| GET | `/api/jobs/{id}/timing` | ★ `timing.json`（绝对时间轴） |
| GET | `/api/jobs/{id}/export/{fmt}` | `musicxml` / `midi` / `json` / ★ `payload` / ★ `timing` / ★ `symbolic` |
| GET | `/api/voicings` | 和弦指法库 |
| GET | `/api/health` | 健康检查 + **`devices`（CPU/GPU 可用性）** |
| GET | `/api/demo` | 示例谱（`?sample=1` 回退手写样例） |

原曲播放音源由 `service.py` 写进 `score.meta.source.audio_url`（形如
`/jobs/<id>/playback.mp3`），前端直接消费，不再依赖单独的 `/media` 路由。

任务状态：`queued → ingesting → separating → analyzing → transcribing → arranging → done|failed`。
所有阶段通过 `jobs.py` 的 `update(job_id, stage, percent, message)` 上报，前端轮询。

**设备选择**：POST 的 `options.device` 取 `auto` / `cpu` / `cuda`。后端在
`pipeline/run.py` 里用 `aitsm.devices.resolve_device()` 校验——请求 GPU 但本机不可用时
**回落 CPU** 并写 `device.resolved` 告警（level 视情况为 warn/info），绝不假装在用 GPU。

## 5. 前端结构

- `main.js`：上传/示例加载、任务轮询、状态机、工具栏（移调/变调夹/显示开关/导出/打印）；
  ★ 设备下拉（`opt-device`）与 `/api/health.devices` 联动，`?job=<id>` 可直接打开某个任务。
- `payload.js`（★）：`normalizePayload` / `attachTimeline` / `pickAbc` / `diagramFor`——
  纯函数、node 可测，负责把 `payload.json` 归一化并用它的时间轴覆盖视图。
- `renderer.js`：VexFlow 渲染。每系统 = 主旋律五线谱（Treble）+ 歌词（StaveText）+ 和弦符号（ChordSymbol/TextNote）+ 可选 TAB 谱表。小节按 time signature 排布，4 小节/行。
- `abcjs-fallback.js`（★）：VexFlow 渲染失败时懒加载 abcjs（先本地 vendor 再 CDN）兜底。
- `musicxml.js`（★）：用浏览器 `DOMParser` 本地解析用户导入的 MusicXML（MuseScore / Guitar Pro 导出），
  产出与后端 payload 兼容的 score 形状后走同一渲染链路；不支持的布局/多 part/装饰音**如实告警**，
  不引入 vexml 等 npm 依赖（前端无构建步骤）。
- `chorddiagram.js`：纯 SVG 和弦框（6 弦 5 品，含横按、空弦/闷弦标记）。
- `player.js`：WebAudio 合成（旋律方波/三角波 + 和弦扫弦噪声包络 + 节拍器），按 score 事件调度；可选 `<audio>` 原曲同步，光标行高亮。
- `transpose.js`：和弦符号移调、旋律音高移调、capo 重算、重新选指法（读 voicings.json）。
- `print.js`：`@media print` 分页控制 + 页眉页脚。

## 6. 分工（并行写域）

> 这是**项目初期**的并行分工记录（v1 阶段由多 agent 并行实现）。
> v2 重写阶段各层由 5 个 teammate 分担（vocal-parse / accomp / fusion / arrange / render），
> 现在全部已合并进主线，下表保留作历史与写域边界参考。

| 角色 | 写域 | 交付 |
| --- | --- | --- |
| lead | `aitsm/{api,jobs,config,cli,schema}.py`, `docs/`, `schemas/`, `README.md`, 集成与验收 | 骨架、契约、端到端串联、最终 demo |
| audio-core | `aitsm/{media.py,pipeline/{ingest,separate,beats,keydet,chords,structure,run}.py}`, `tests/test_{beats,chords,structure}.py` | analysis.json 全链路 |
| melody-lyrics | `aitsm/pipeline/{lyrics,melody,align}.py`, `tests/test_{lyrics,melody}.py` | 歌词 + 旋律 + 对齐 |
| score-engine | `aitsm/score/**`, `schemas/score.schema.json` 的意见回传, `tests/test_score*.py` | score.json + voicings + 导出 |
| web-render | `web/**`, `samples/demo-score.json` | 可交互网页乐谱 |
| plugin-docs | `plugin/**`, `docs/dsh-plugin.md`, `docs/05-usage.md` | DSH 插件方案与使用文档 |

跨域修改须先经 lead 同意；公共契约变更必须同步 `schemas/` 与本文档。

## 7. 依赖与运行

- Python 3.12，venv 在 `.venv`。
- 核心：`numpy scipy librosa soundfile fastapi uvicorn python-multipart pydantic requests pyyaml jsonschema`
- 重型可选：`torch torchaudio demucs openai-whisper`（缺失时自动降级并在 manifest 记录）
- ★ **v2 专用模型依赖（必须 pin）**：
  - `transformers==4.57.6` —— SheetSage2 按 4.45.2 导出、5.x 的 `BartDecoder` 签名变化会直接报错；
    VocalParse 的 checkpoint 在 4.57.6 实测可用。
    见 [requirements.txt](../requirements.txt) 与 [`12-operations.md`](12-operations.md) §1。
  - `qwen-asr==0.0.6` —— Qwen3-ASR 建模代码不在标准 transformers 里。它的 `__init__` 会连带
    要 `nagisa`/`gradio`，本仓库按文件路径只加载三个建模子模块绕开（`transcribe/qwen_backend.py`）。
- 其他可选：`mir_eval`（SheetSage2 和弦表示）、`pretty_midi`（MIDI 解析）、`mido`。
- 前端：`vexflow`（vendored 到 `web/vendor/`），无打包器；abcjs 兜底走懒加载。
- 启动：`python -m aitsm.cli serve --port 7863`，或 `python -m aitsm.cli generate <file> --out work/demo`。
- 专用模型权重：`python scripts/fetch_models.py --list` 看状态，`--all` 下载（走 hf-mirror）。
- 环境自检：`python -m aitsm.cli doctor` —— 会打印 v2 四层就绪情况、模型内存估算与部署建议。
