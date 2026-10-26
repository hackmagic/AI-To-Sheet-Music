# v2 运维与可维护性

> 主链路已经跑通（见 `11-v2-pipeline.md`）。本文记录**让这条链路可维护地跑下去**
> 需要遵守的约束：依赖边界、部署形态、已知脆弱点、迁移路径。

## 1. 依赖边界（必须 pin 的两处）

| 依赖 | 约束 | 为什么 |
| --- | --- | --- |
| `transformers` | **`==4.57.6`** | SheetSage2 权重按 4.45.2 导出，5.x 的 `BartDecoder.__init__` 去掉 `embed_tokens` 参数后直接报错；VocalParse 的 checkpoint 在 4.57.6 实测可用 |
| `qwen-asr` | **`==0.0.6`** | 仍是 0.0.x，子模块路径/类名会变。适配器按**文件路径**加载 `configuration/processing/modeling` 三个子模块 |

`requirements.txt` 里这两条都写死。`aitsm/transcribe/qwen_backend.py` 把
**版本号、子模块名、词表大小、mrope 配置**集中成常量——升级 `qwen-asr` 时只改那一个文件。

### 为什么不能只装 transformers

Qwen3-ASR 的建模代码**不在标准 transformers 里**。只装 transformers 时：

- `qwen3_asr` 这个 model_type 能被识别（4.57.6）；
- 但通用 `Qwen3ASRForConditionalGeneration` 按 `vocab=151936 / output_dim=3584` 建图，
  与 checkpoint 的 `152101 / 2048` 冲突；
- 加上 checkpoint 的 `thinker.*` 前缀，实测 **315 个 key MISSING**。

正解是用 `qwen_asr` 里的 **`Qwen3ASRThinkerForConditionalGeneration`**，并：

```python
text_config.vocab_size = 152101                     # AST token 扩展
text_config.rope_scaling = {"type": "mrope", ...}   # checkpoint 没写
remap["lm_head.weight"] = remap["model.embed_tokens.weight"]  # tied
```

### 为什么按文件路径加载而不是 `import qwen_asr`

`qwen_asr/__init__.py` 会连带导入 `nagisa`（强制对齐用的日文分词器）与可选
`gradio`（demo 用）。这两样对建模推理没有作用，却让依赖变重、在没装的机器上直接失败
（实测 `nagisa` 在 Windows 上安装会卡住）。因此适配器：

1. 用 `find_spec` 拿到包根；
2. 造三个**空壳父包**（让相对导入成立）；
3. 只 `exec_module` 三个建模子模块。

`qwen_backend.check_import_paths()` 会在**探测阶段**验证这三条路径，
`qwen-asr` 升级导致路径变化时会明确报出来，而不是等到推理中途抛 `ImportError`。

## 2. 模型内存与部署形态（会 OOM 的地方）

| 模型 | 估计常驻内存 |
| --- | --- |
| VocalParse（2B） | **8.5 GB** |
| SheetSage2 | 1.5 GB |
| htdemucs_ft ONNX | 1.8 GB |
| DuET | 0.6 GB |
| 合计（含 2 GB 预留） | **约 16.9 GB** |

### 硬约束：不要用多 worker（**启动时强制拦截**）

```bash
# ✗ 会 OOM：4 个 worker × 16.9 GB ≈ 68 GB
uvicorn aitsm.api:app --workers 4

# ✓ 单 worker + 内部队列（uvicorn 默认就是 1）
python -m aitsm.cli serve --port 7863
```

这条**不是建议，是运行时强制**。`aitsm/api.py` 的 lifespan 启动钩子会调
`deployment.check_deployment()`，判定为不安全时**直接抛 `RuntimeError` 拒绝启动**：

```python
# aitsm/api.py
@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _enforce_deployment_limits()   # 多 worker + 大模型 → RuntimeError
    yield
```

- 判定依据：`WEB_CONCURRENCY`（uvicorn `--workers` 会设它）+ 模型内存表；
- 逃生开关：`AITSM_ALLOW_MULTI_WORKER=1` 可放行，但会记 warning，风险自担；
- 验证：`tests/test_deployment_enforcement.py` 用真实 `WEB_CONCURRENCY=4` 断言启动钩子会抛。

`python -m aitsm.cli doctor` 只是**再用人类可读的形式打印一遍**同样的结论
（不强制，因为 doctor 不启动服务）。

### 需要横向扩展时

把 VocalParse 拆成**独立推理服务**（模型只加载一份），主 API 通过内部 HTTP 调用。
`deployment.worker_advice()` 的 error 分支就是提示这件事。

## 3. 已知脆弱点

| 位置 | 脆弱点 | 兜底 |
| --- | --- | --- |
| `qwen_backend` | `qwen-asr` 升级后子模块路径变化 | `check_import_paths()` 在探测阶段报错；`pin` 版本 |
| `qwen_backend` | `mrope_section` 缺失 → 位置编码**静默失效**、音符时序系统性偏移 | `validate_mrope()` 在加载前断言，不合格直接抛 `VocalParseIncompatibleError` |
| `vocal._MODEL_CACHE` | 单进程缓存 8.5 GB；多进程会各存一份 | 见上面的部署约束 |
| `fusion.duet` | `vocabulary.json` 在 Windows 检出成 CRLF 会让 sha256 校验失败 | 仓库内版本已转 LF；`vendor/DuET` 随仓库走 |
| `transcribe.accompaniment` | SheetSage2 的 `trust_remote_code` 依赖 HF 模块缓存布局 | 优先本地包导入（复制到 `<HF_HOME>/pkgs/ss2/sheetsage2`），`trust_remote_code` 只作兜底 |

### 关于 `lm_head` 的 tie（已验证，不是猜测）

checkpoint 里**没有** `lm_head.weight`，且 `config.json` 明确 `tie_word_embeddings: true`。
所以与 `model.embed_tokens.weight` 共享是唯一合理解，不是“覆盖了一个不同的权重”。
这条有测试钉住（`tests/test_vocalparse_weights.py`）。

## 4. 多 BPM 与休止（真实输出暴露的问题）

VocalParse 的 token 串里可能出现**多个** `<BPM_>`（变速歌）。解析器不是“全局取第一个”，
而是**每个事件携带当时的 BPM**，时间轴按事件各自的 BPM 推进——已有测试覆盖：

```python
events = parse_vocalparse_tokens("<BPM_60> <P_60> <NOTE_4> <BPM_120> <P_62> <NOTE_4>")
# events[0].bpm == 60, events[1].bpm == 120
```

`<P_0>` 在真实输出里表示**休止/无音高**（实测大量出现），不是 MIDI 0：

- 时间轴照常推进（休止占时值）；
- **不产出音符**；
- 不做任何补音。

孤立的 `<NOTE_>`（前面没有音高）会被记入 `last_parse_orphans()` 并打 warning——
正常输出里应当为空；非空说明模型把休止写成了纯时值，需要关注时序偏移。

## 5. 各层耗时分布（GPU 迁移的依据）

`analysis.json` 的 `meta.timing_breakdown` 给出按层聚合的耗时与占比：

```json
{
  "by_layer_seconds": {"vocal": 210.0, "separation": 58.0, "accompaniment": 45.0},
  "by_layer_share":   {"vocal": 0.65, "separation": 0.18, "accompaniment": 0.14},
  "slowest_layer": "vocal",
  "gpu_hint": "最耗时的是 vocal（210.0s，65%），迁 GPU 应优先考虑它"
}
```

CPU 实测（16s 音频，全流程 683s）：

| 层 | 耗时占比 | 迁 GPU 收益 |
| --- | --- | --- |
| vocal（VocalParse） | **~65%** | **最高，优先迁** |
| separation（htdemucs_ft ONNX） | ~17% | 中（ONNX 也有 CUDA EP） |
| accompaniment（SheetSage2） | ~13% | 中 |
| quantization（DuET） | 小 | 低，可留 CPU |
| arrangement（纯符号） | 极小 | 无 |

### GPU 迁移要点

1. **VocalParse 用 `bf16`，不要 `fp16`**：Qwen 系列在 fp16 下偶有数值不稳定。
   ```python
   model = model.to("cuda", dtype=torch.bfloat16)
   ```
2. **特征提取可能成为新瓶颈**：Whisper 特征提取若留在 CPU，GPU 空转等数据。
   一并迁到 GPU（`feature_extractor` 支持 `device` 参数）。
3. **分层迁移**：只迁 VocalParse 即可拿到最大收益，其余留 CPU 以省显存
   （16.9 GB 里 VocalParse 占一半）。
4. 迁完后用同一段音频对比 `meta.timing_breakdown`，确认收益真实发生。

## 6. TART / Noise2Fret 的真实可用性（实测核查）

| 项目 | HF 仓库 | GitHub | 权重 |
| --- | --- | --- | --- |
| **Noise2Fret** | 无 | `RiccardoVib/Noise2Fret` ✅ 可克隆 | **无**：仓库内没有 `.pt/.ckpt/.safetensors`，README 也没有 Google Drive/OneDrive 链接 |
| **TART** | 无 | 论文（arXiv 2609.11904）**未给出代码仓库链接** | 无从获取 |

结论：

- **Noise2Fret 的价值在公式，不在权重**。它的
  `Noise2Fret_Aux_Losses/src/AuxiliaryLoss.py` 里五项损失是**纯 torch 数学**，
  不依赖任何 checkpoint。已移植为 `aitsm/arrangement/playability.py`（纯 numpy）：
  - `jaccard_tonal_distance` —— 音级距离
  - `fret_distance` —— 把位距离
  - `cof_chord_distance` —— 五度圈距离（按平均圈位的环距，与原文一致）
  - `string_activity_jaccard_loss` —— 弦相似性
  - `hand_span_penalty` —— 手跨可行性（排除空弦与闷弦）
- **TART 的 AudioFret（audio-conditioned T5）无法复现**：没有代码也没有权重。
  当前 `tart_rerank` 走的是确定性启式，`provenance.engine` 会如实写
  `tart-heuristic` —— 不假装跑过模型。

> 想用 TART 的正确做法：等作者放出仓库后再接，接口已预留
> （`aitsm/arrangement/tart_rerank.py` 的 `is_available()` / `rerank()`）。

## 7. 测试把边界条件锁住了

新增的测试文件与对应风险：

| 测试 | 锁住什么 |
| --- | --- |
| `test_qwen_backend.py` | 版本 pin、路径探测、mrope 断言、词表常量 |
| `test_vocalparse_edges.py` | 多 BPM 分段、孤立 NOTE、休止推进、前奏交叉验证 |
| `test_vocalparse_real_output.py` | 真实输出的 CoT 分隔、占位符噪声、歌词方向 |
| `test_playability.py` | Noise2Fret 五项损失的语义方向 |
| `test_deployment_timing.py` | 多 worker OOM 判定、耗时分布聚合 |

**交叉验证建议已固化**：`test_vocalparse_edges.py::test_intro_like_output_has_few_notes`
断言前奏片段解出的音符数 < 5，与 SheetSage2 的 `intro` 判断互相印证——
未来换模型或改解析器时，「前奏被误判成有人声」会立刻失败。
