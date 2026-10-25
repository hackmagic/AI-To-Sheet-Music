# 总体架构与接口契约（v1）

> v2 的转录/编配分层、符号契约与降级矩阵见 [`11-v2-pipeline.md`](11-v2-pipeline.md)。
> 本文的模块接口仍然有效：v2 在其上加了一层显式的 `symbolic` 契约与可插拔模型后端。

## 1. 目录结构

```text
AI-To-Sheet-Music/
├── aitsm/
│   ├── __init__.py
│   ├── config.py            # 路径/环境变量/模型选择
│   ├── api.py               # FastAPI：上传、任务、状态、产物、静态站点
│   ├── jobs.py              # 内存任务队列 + 进度事件 + 产物目录
│   ├── cli.py               # python -m aitsm.cli generate ...
│   ├── media.py             # ffmpeg/ffprobe 封装（ingest）
│   ├── schema.py            # JSON Schema 加载与校验
│   ├── pipeline/
│   │   ├── ingest.py        # F1/F2
│   │   ├── separate.py      # F4  Demucs + 降级
│   │   ├── beats.py         # F5
│   │   ├── keydet.py        # F6
│   │   ├── chords.py        # F7
│   │   ├── structure.py     # F8
│   │   ├── lyrics.py        # F9
│   │   ├── melody.py        # F10 + F14 音节对齐
│   │   └── run.py           # 编排：产出 analysis.json
│   ├── testsupport/         # 合成曲评估工具（不参与线上链路）
│   │   ├── synth_song.py    # 三个 profile 的合成曲 + ground truth
│   │   └── evaluate.py      # 五维加权打分
│   ├── score/
│   │   ├── model.py         # 读写 score.json、时间/音高工具
│   │   ├── voicings.py      # F12 和弦指法库与选择
│   │   ├── strum.py         # F13 节奏型库与选择
│   │   ├── arrange.py       # F11/F15/F16 弹唱谱编配
│   │   ├── validate.py      # F23 质量网关
│   │   └── export/
│   │       ├── musicxml.py  # F24
│   │       ├── midi.py      # F24
│   │       └── pdf.py       # 打印（浏览器端为主，服务端可选）
├── schemas/
│   ├── analysis.schema.json
│   └── score.schema.json
├── web/                     # 前端（无构建步骤，原生 ES Module + 本地 vendor）
│   ├── index.html
│   ├── css/app.css
│   ├── js/{main,api,renderer,chorddiagram,lyrics,player,transpose,print}.js
│   └── vendor/vexflow.js    # 从 npm 复制，离线可用
├── samples/                 # 手工构造的示例 score/analysis，供前端独立开发
├── tests/
├── plugin/                  # DSH skill（SKILL.md + scripts + references）
├── artifacts/               # 一次真实运行的交付样例（score/analysis/manifest/musicxml/mid）
├── docs/
└── work/                    # 运行产物（git 忽略）
```

## 2. 数据契约

- `analysis.json`：`aitsm.analysis/v1`，见 `schemas/analysis.schema.json`。由 `pipeline/run.py` 产出，只包含“证据”，不含编配决定。
- `score.json`：`aitsm.score/v1`，见 `schemas/score.schema.json`。由 `score/arrange.py` 产出，是前端渲染与所有导出的唯一输入。
- `web/voicings.json`：由 `score/voicings.py` 导出，前端移调后重新选指法用。
- `manifest.json`：输入哈希、工具版本、各阶段耗时与降级原因。

时间单位约定：
- 音频时间：秒（float），绝对时间轴。
- 谱面时间：**四分音符拍**（quarter-note beats），小节内相对（beat 0 = 小节线）。时值可为 `number` 或 `"1/3"` 这样的分数字符串（三连音）。
- 音高：MIDI 号（整数），`score` 中一律为**记谱音高**（即 `play_key` 视角；capo 已从发声音高扣除）。

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

## 4. 服务端 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/jobs` | multipart 上传音频 + options JSON，创建任务，返回 `{job_id}` |
| GET | `/api/jobs/{id}` | 任务状态 + 阶段进度 + 产物 URL |
| GET | `/api/jobs/{id}/score` | `score.json` |
| GET | `/api/jobs/{id}/analysis` | `analysis.json`（调试/质量面板） |
| GET | `/api/jobs/{id}/manifest` | `manifest.json` |
| GET | `/api/jobs/{id}/export/{fmt}` | `musicxml` / `midi` / `json` |
| GET | `/media/{id}/playback.mp3` | 原曲播放音源（用于同步播放） |
| GET | `/api/voicings` | 和弦指法库 |

任务状态：`queued → ingesting → separating → analyzing → transcribing → arranging → done|failed`。
所有阶段通过 `jobs.py` 的 `update(job_id, stage, percent, message)` 上报，前端轮询。

## 5. 前端结构

- `main.js`：上传/示例加载、任务轮询、状态机、工具栏（移调/变调夹/显示开关/导出/打印）。
- `renderer.js`：VexFlow 渲染。每系统 = 主旋律五线谱（Treble）+ 歌词（StaveText）+ 和弦符号（ChordSymbol/TextNote）+ 可选 TAB 谱表。小节按 time signature 排布，4 小节/行。
- `chorddiagram.js`：纯 SVG 和弦框（6 弦 5 品，含横按、空弦/闷弦标记）。
- `player.js`：WebAudio 合成（旋律方波/三角波 + 和弦扫弦噪声包络 + 节拍器），按 score 事件调度；可选 `<audio>` 原曲同步，光标行高亮。
- `transpose.js`：和弦符号移调、旋律音高移调、capo 重算、重新选指法（读 voicings.json）。
- `print.js`：`@media print` 分页控制 + 页眉页脚。

## 6. 分工（并行写域）

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
- 核心：`numpy scipy librosa soundfile fastapi uvicorn python-multipart pydantic requests pyyaml`
- 重型可选：`torch torchaudio demucs openai-whisper`（缺失时自动降级并在 manifest 记录）
- 前端：`vexflow`（vendored 到 `web/vendor/`），无打包器。
- 启动：`python -m aitsm.cli serve --port 7863`，或 `python -m aitsm.cli generate <file> --out work/demo`。
