# AI-To-Sheet-Music

用 AI 把一个音乐文件变成**可以直接照着自弹自唱的网页吉他谱**：人声主旋律 + 逐字歌词 + 和弦符号 + 吉他指法图 + 节奏型 + 六线 TAB，支持移调、变调夹、播放跟随与打印导出。

> 输入 `song.mp3` → 输出一份能打开就弹的 `score.json` + 网页乐谱 + MusicXML / MIDI。

## 它解决什么问题

网上找谱要么不准、要么没有想要的那首、要么移不到自己能唱的调。这个项目做的是从**音频本身**扒出弹唱谱需要的一切：

| 你需要的 | 怎么来的 |
| --- | --- |
| 调性和该用变调夹几品 | chroma + 和弦证据融合估计调性，再选吉他友好调（C/G/D/A/E）反推 capo |
| 和弦进行 | 半小节粒度的 chroma 模板匹配 + 低音约束 + 调内先验平滑 |
| 唱哪几句、每个字落在哪个音 | Whisper 词级时间戳 + Demucs 人声分轨 + pyin 基频提取 + DP 音节对齐 |
| 每个和弦怎么按 | 算法生成的指法库（442 个符号 / 1205 个把位，含横按推断与降号别名） |
| 右手怎么扫 | 按段落能量和速度选扫弦/分解型 |
| 段落结构 | 自相似矩阵 + 能量 + 和弦重复三重证据切分 |

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS / Linux

# 环境自检（会列出 ffmpeg、各引擎、v2 四层专用模型就绪情况、模型内存与部署建议）
.venv/Scripts/python -m aitsm.cli doctor
```

### 可选：下载 v2 专用模型权重

不装也能跑（自动降级到 v1 内核），装了才有 v2 的精度：

```bash
.venv/Scripts/python scripts/fetch_models.py --list        # 看哪些已就绪
.venv/Scripts/python scripts/fetch_models.py --only sheetsage2 duet   # 先下小的（0.7GB）
.venv/Scripts/python scripts/fetch_models.py --all         # 全套约 6GB
```

脚本默认走 `hf-mirror.com` 并关闭 Xet 传输（Windows 上会报 `I/O error 5`），
下完会打印对应的环境变量。

### 生成一份谱

```bash
.venv/Scripts/python -m aitsm.cli generate song.mp3 --out work/mysong
```

### 多曲目基准

```bash
.venv/Scripts/python -m aitsm.cli bench samples/ --out work/bench
```

每首曲子写一个子目录，最后汇总成 `benchmark.json` 与 `benchmark.md`：逐曲列出
分离/人声/伴奏/量化四层实际用的引擎（`⚠` = 没跑到架构首选组件）、质量分与规模。
权重到位前后各跑一次同一批目录，逐行对比即可判断专用模型是否真的接上了。

产物：

```text
work/mysong/
  score.json         # 网页乐谱的唯一数据源（aitsm.score/v1）
  score.musicxml     # MuseScore / Guitar Pro 可打开（五线谱 + TAB）
  score.mid          # 可播放 / 可导入 DAW
  symbolic.json      # 符号契约（节拍/调性/和弦/段落/旋律/歌词 + provenance）
  symbolic.musicxml  # 转录层原生 MusicXML（未做 TAB）
  payload.json       # 网页渲染数据（MusicXML + ABC + timing + 指法图）
  timing.json        # 绝对时间轴（同步播放/第三方消费）
  analysis.json      # 分析证据（调试与质量复核用）
  manifest.json      # 输入哈希、各阶段耗时、引擎版本、降级原因
  audio/playback.mp3
  stems/             # 分离出的人声、鼓、贝斯、伴奏（ONNX 时为 44.1kHz）
```

`symbolic.json` 的 `provenance` 会逐层写明 `requested`（架构首选）与 `engine`（实际跑的），
降级一目了然；`analysis.symbolic.degraded_layers` 是它们的汇总。

常用参数：

```bash
--key G --capo 2          # 指定调与变调夹（不给则自动选）
--difficulty beginner     # beginner / elementary / intermediate / advanced
--language zh             # 歌词语言，默认 auto 自动检测
--whisper-model small     # 歌词模型降级（默认 medium；small 快约 2.3 倍、CER 高约 0.10）
--quality high            # 质量档 standard/high/max：微调分轨、更稳分段、large-v3 歌词（更慢更准）
--no-separation           # 跳过分轨（快，但旋律与和弦质量下降）
--start 60 --duration 30  # 只处理片段（调试用）
--device cpu              # auto / cuda / cpu
```

### v2 分层与可选模型（缺权重也能跑）

流水线拆成「转录层 → 符号契约 → 编配层 → 渲染」四段。每层都记录 `requested`
（架构首选组件）与 `engine`（本机实际跑的），降级会写进 `symbolic.json` 的
`provenance` 和 `manifest.degraded`，不会静默掉质量：

| 层 | 架构首选 | 本机实测 |
| --- | --- | --- |
| 分离 | htdemucs_ft（可走 demucs-onnx） | ✅ **`demucs-onnx`（htdemucs_ft）** |
| 人声联合转录 | VocalParse（词+音高+时值+BPM 一个 decoder） | ✅ **`vocalparse` 已跑通**（需 `qwen-asr` 包） |
| 伴奏联合转录 | SheetSage2 | ✅ **`sheetsage2` 已跑通** |
| 骨架融合 + 量化 | SheetSage2 拍点 + DuET | ✅ **`duet` 已跑通** |
| 编配（纯符号） | DP 指法 + TART rerank + 规则节奏型 + 可选 LLM 评审 | `dp` + `tart-heuristic` |
| 渲染 | MusicXML + payload + VexFlow，abcjs 兜底 | 同上 |

四层专用模型全部实测跑通，`degraded_layers` 为空。CPU 上 16s 音频全流程约 11 分钟
（VocalParse 占大头），开 GPU 会快很多。

因此任务目录里会多出：

```text
  symbolic.json       # 转录层符号契约（节拍/调性/和弦/段落/旋律/歌词 + provenance）
  symbolic.musicxml   # 转录层原生 MusicXML（未做 TAB）
  payload.json        # 网页渲染数据（MusicXML + ABC + timing + 指法图）
  timing.json         # 绝对时间轴（小节/音符/和弦/歌词的秒），同步播放与第三方消费
```

网页会把 `payload.json` 一起取下来：ABC 兜底用它的版本，播放与小节高亮用它的
时间轴；`/api/jobs/{id}/payload`、`/api/jobs/{id}/timing` 也可直接给外部消费。

#### 下载并启用专用模型

```bash
# 一键下载（默认走 hf-mirror.com，幂等，可只选部分）
.venv/Scripts/python.exe scripts/fetch_models.py --list
.venv/Scripts/python.exe scripts/fetch_models.py --only sheetsage2 duet

# 下载完会打印这些环境变量：
export AITSM_SHEETSAGE2_DIR=$PWD/models/SheetSage2   # m-a-p/SheetSage2
export AITSM_DUET_DIR=$PWD/models/DuET               # Wayne42/DuET
export AITSM_VOCALPARSE_DIR=$PWD/models/VocalParse   # pymaster/VocalParse（4.1GB）
export AITSM_DEMUCS_ONNX_DIR=$PWD/models/htdemucs_ft_onnx  # StemSplitio/htdemucs-ft-onnx
export HF_HOME=$PWD/.cache/hf                        # 用户目录不可写时必需
```

`scripts/fetch_models.py` 会关掉 HF 的 Xet 传输（部分 Windows 环境报 `I/O error 5`），
下载后校验关键文件是否齐全，失败会明确报错而不是留一份半成品。

> **版本要求**：SheetSage2 按 `transformers` 4.x 导出，本仓库实测在 **4.57.6**
> 下可加载（5.x 会因 `BartDecoder` 签名变化失败）。
>
> **VocalParse 额外需要一个包**：Qwen3-ASR 的建模代码不在 transformers 里，必须装
> `qwen-asr`（PyPI 上有，**不是** `vocalparse`——那个包从来不存在于 PyPI）：
>
> ```bash
> .venv/Scripts/python.exe -m pip install --index-url https://pypi.org/simple/ qwen-asr
> ```
>
> 它的 `__init__` 会连带导入 `nagisa`/`gradio`，本仓库的适配器改为按文件路径只加载
> `configuration/processing/modeling` 三个子模块，因此无需装那两套推理封装依赖。

**模型获取方式：**

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 专用模型（HF 下载）
huggingface-cli download pymaster/VocalParse --local-dir ./models/VocalParse
huggingface-cli download m-a-p/SheetSage2 --local-dir ./models/SheetSage2  
huggingface-cli download Wayne42/DuET --local-dir ./models/DuET
git clone https://hf-mirror.com/StemSplitio/htdemucs-ft-onnx.git models/demucs-onnx

# 编配层增强组件（GitHub，无独立 HF 仓库）
git clone https://github.com/RiccardoVib/Noise2Fret.git  # Noise2Fret 五项损失函数
# TART 代码与权重从论文关联 GitHub 获取（第三阶段 T5 弦品分配器）
```

第三方大模型**只**做编配层的符号评审（指法重排建议 / 和弦替代），默认关闭：
`AITSM_LLM_URL=... AITSM_LLM_ADVISOR=1`。细节见 [docs/11-v2-pipeline.md](docs/11-v2-pipeline.md)；
依赖边界、部署约束（**不要用多 worker**，会 OOM）、GPU 迁移路径与
TART/Noise2Fret 的实测可用性见 [docs/12-operations.md](docs/12-operations.md)；
整曲实测质量（76 分钟 CPU、拍号/段落修复记录）见 [docs/13-demo-quality.md](docs/13-demo-quality.md)。

### 打开网页乐谱

```bash
.venv/Scripts/python -m aitsm.cli serve --port 7863
# 浏览器打开 http://127.0.0.1:7863
```

上传音频即可看到进度、乐谱、质量面板；也可以点「加载示例谱」先看效果——`/api/demo` 默认返回**整首真实交付谱**（`artifacts/score.json`，88 小节 / 英文原词），加 `?sample=1` 才回退到 12 小节的手写样例。

### 选 CPU 还是 GPU

上传弹窗里有一个**设备**下拉（自动 / CPU / GPU）。它读的是后端 `/api/health` 的
`devices` 字段，而不是前端猜——因为 GPU 能不能用取决于 **torch 是不是 CUDA 编译版**。

典型情况：机器有 NVIDIA 卡但 `torch` 是 CPU 版，界面会显示
「GPU（本机不可用）」并在选择时给出黄色说明，**不会让你以为在用 GPU 却等 CPU 跑一小时**。

启用 GPU：

```bash
# 先卸载 CPU 版，再装 CUDA 版（以 cu124 为例）
.venv/Scripts/python.exe -m pip uninstall -y torch torchaudio
.venv/Scripts/python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
# 重启服务后 /api/health 的 devices.gpu_usable 会变成 true
```

> VocalParse 迁 GPU 时用 **`bf16`**（Qwen 系列在 fp16 下偶有数值不稳定）；
> 特征提取若留 CPU 会成为新瓶颈。详见 [docs/12-operations.md](docs/12-operations.md)。

## 网页端能做什么

- 五线谱主旋律 + 逐字歌词 + 和弦符号 + 行首和弦指法图 + 可选 TAB 谱表
- **可直接拖入 MusicXML**（MuseScore / Guitar Pro 导出）：本地解析，不上传、不联网；
  不支持的布局/多 part/装饰音会如实列在告警里
- **手工修正**：点任何和弦就能改（调内 I–IV–V/vi/ii 快选 + 拍位 + 指法库校验），点歌词直接改字，点小节号可插入/删除小节、改拍号；60 步撤销重做；修正处打标记，导出时可选「干净版」去掉编辑痕迹
- 播放：内置合成器（旋律 + 扫弦 + 节拍器），光标跟随、小节点击跳转、循环当前段落、0.5×–1.2× 变速
- 有原曲音频时切换「原曲」模式，乐谱光标跟着原曲走
- 移调 ±6 半音、变调夹 0–7 品实时重排（和弦、旋律、调号、指法全部跟着变）
- 质量面板：总体置信度 + 分阶段置信度 + 告警（可点击定位到小节）
- 打印：A4 纵向、每行 4 小节、段落自动换行、页眉页脚

## 实测质量

| 维度 | 结果 | 依据 |
| --- | --- | --- |
| 三首合成曲（3/4、4/4 摇滚、无人声） | 综合 **85.1/100**（修复前 72.7）；BPM 误差 0%、拍号全对、调性全对、和弦 **100%** | [docs/07-synthetic-eval.md](docs/07-synthetic-eval.md) |
| 真实歌曲（229s 英文流行） | quality **0.821**、BPM 92.47 / 4/4、D 大调 conf 0.82、**69 个和弦事件**（mean_conf 0.586）、11 段、**395 个旋律音** | [docs/05-acceptance.md](docs/05-acceptance.md) |
| 歌词 | 字符级 CER **0.49–0.54**（medium；small 约 0.64）；同一配置跑三次差 0.05，因为分轨不是比特可复现 | [docs/05-acceptance.md](docs/05-acceptance.md) |
| 中文歌（216s 慢歌） | 59 小节 / 6 段、D 大调 conf 0.57；歌词 lev_cer **0.669** / recall **0.835** / precision **0.606**（中文专属清洗后） | [docs/05-acceptance.md](docs/05-acceptance.md) |
| MusicXML 落地 | 过官方 XSD + **MuseScore Studio 4.7.5 实机打开**：双谱表、和弦、逐字歌词、段落、`♩=92` 全部正确；**不带 `-f` 的严格导入 exit 0**（4 份谱面），可导 PDF/PNG | [docs/05-acceptance.md](docs/05-acceptance.md) |
| 成本 | 4 分钟歌 CPU 冷启动约 9 分钟（分轨 5 分钟 + 歌词 2.5 分钟）；`--quality high` 约 45 分钟，`max` 1.5–2 小时 | [docs/08-quality-tiers.md](docs/08-quality-tiers.md) |
| 和弦识别 | 特征融合（槽位中值 + 谱归一化）后调内三和弦干净率 **93.9% → 98.0%**、跨分轨一致率 **55.2% → 57.4%**；平均置信度仍在 0.58（瓶颈是方法本身，换更大模型无效） | [docs/09-chord-tuning.md](docs/09-chord-tuning.md) |
| 节奏与和声证据 | HMM 节奏量化把「音符时值等于合法音符形状」的比例 **63% → 95%**（谱面 100% 落在 1/4 拍网格、网页零「时值无法精确记谱」告警）；谐波加权让和弦置信度 +1.7%、三和弦纯度 +1.8 点，代价是跨分轨一致率 −7.4 点 | [docs/10-rhythm-and-harmony.md](docs/10-rhythm-and-harmony.md) |

合成曲只用于验证**信号层**（BPM / 拍号 / 调性 / 和弦），合成人声是正弦波所以不评歌词与旋律；真实歌曲的段落边界与歌词仍需人工校对。

## 架构

四段分层，**转录层与编配层严格分离**——编配只吃符号，不碰音频：

```text
音频/视频
  → ingest        解码、统一采样率、生成分析音轨与播放音源
  ├─ 转录层（专用模型优先，缺失显式降级）
  │   → separation       htdemucs_ft（ONNX bag_infer 优先 / torch demucs / HPSS）
  │   → vocal            VocalParse（qwen_asr Thinker，30s 分段）/ whisper+pyin
  │   → accompaniment    SheetSage2（拍点·调性·和弦·段落联合解码）/ beats+chroma
  │   → fusion           骨架对齐 + DuET/HMM 节奏量化
  ↓
symbolic.json + symbolic.musicxml   （符号契约 aitsm.symbolic/v1，每层写 provenance）
  ↓
  ├─ 编配层（纯符号，不读音频）
  │   → arrangement      DP 指法 / TART rerank / 节奏型规则 / 可选 LLM 评审
  │   → score.json       定调与 capo、小节网格、和弦落位、TAB、段落节奏型
  ↓
  └─ 渲染层
      → payload.json / timing.json / score.musicxml / score.mid
      → 网页（VexFlow 主渲染，abcjs 兜底）
```

每层记录 `requested`（架构首选组件）与 `engine`（实际跑的），差异写进
`symbolic.provenance` 与 `analysis.symbolic.degraded_layers`——降级不会静默。

| 文档 | 内容 |
| --- | --- |
| [docs/02-architecture.md](docs/02-architecture.md) | 模块接口契约、API 路由、数据契约 |
| [docs/11-v2-pipeline.md](docs/11-v2-pipeline.md) | 五层架构、降级矩阵、依赖 pin 与版本坑 |
| [docs/12-operations.md](docs/12-operations.md) | 部署约束（**不要多 worker**）、内存预算、GPU 迁移 |
| [docs/13-demo-quality.md](docs/13-demo-quality.md) | 整曲实测质量与拍号/段落修复记录 |
| [docs/04-known-issues.md](docs/04-known-issues.md) | 已知限制与已修问题 |
| [docs/01-requirements.md](docs/01-requirements.md) | 需求分解 |

## 与两个参考项目的关系

本项目站在两个开源项目的肩膀上，但目标不同（**自弹自唱弹唱谱**，不是和弦谱工作台，也不是五件乐队总谱）：

| 参考项目 | 本项目的用法 |
| --- | --- |
| AI-ChordCraft | 借鉴 ffmpeg 转录、和弦符号规范化、和弦区间选择、调性估计、歌词时间轴规整、Web 工作台形态。**不依赖**它的 SongFormer / MOSS-Music / essentia 等重型服务 |
| To-Sheet-Music-Skill | 借鉴「只维护紧凑 JSON、渲染器确定性生成谱面」的架构思想、分轨缓存与 manifest 证据纪律、可演奏性规划思路。目标格式从 MuseScore 五件总谱改为浏览器弹唱谱 |

## 环境要求

- Python 3.12（3.11+ 应该也可以）
- FFmpeg（解码音频/视频）
- 可选但强烈建议：`torch` + `demucs`（分轨）、`faster-whisper`（歌词）
- v2 专用模型（可选，见下文）：`transformers==4.57.6`、`qwen-asr==0.0.6`
- 磁盘：旧链路约 2 GB（Demucs 80MB + Whisper medium 1.5GB）；**v2 全套专用模型约 6 GB**
  （VocalParse 4.1 GB + SheetSage2 0.23 GB + DuET 0.43 GB + htdemucs_ft ONNX 约 1.3 GB）

缺少可选依赖时不会崩，会降级并在 `manifest.json` / `symbolic.provenance` 里写明原因。

### 安装可选重型依赖

```bash
# CUDA 用户：先按 https://pytorch.org 选择匹配版本，例如
.venv/Scripts/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
# CPU 用户
.venv/Scripts/python -m pip install torch torchaudio

.venv/Scripts/python -m pip install demucs faster-whisper
```

> 首次运行会下载模型权重，需要能访问 `dl.fbaipublicfiles.com`（Demucs）与 `huggingface.co`（Whisper）。
> 模型缓存在 `work/cache/`（而非 `~/.cache`），因为某些环境的用户目录不可写。

## 已知限制

- **歌词准确率**取决于混音质量与演唱清晰度；有和声/混响重的段落会偏差。低置信度会在质量面板标出。默认模型 medium，字符级 CER 约 0.50–0.54，**不保证每句都对**。
- **人声旋律**在无分轨时不可靠（会锁到贝斯上），必须有 vocals stem。
- **和弦识别**对失真吉他、大量七和弦的色彩和声会偏保守（有意为之：宁可给三和弦也不给错和弦），平均置信度约 0.58。
- **关系大小调**（如 Bm 与 D）在音频证据上常难以区分，默认按关系大调标注。
- 暂不处理速度/拍号的**中途变化**（会告警），不支持多声部人声。
- GPU 加速目前非必需：CPU 上 4 分钟歌（走旧 whisper 链路）约 9–10 分钟；**走 v2 全套专用模型则约 76 分钟**（VocalParse 占 73%），见下文「选 CPU 还是 GPU」。
- **中文歌**会额外做三步清洗（去署名幻觉行、合并单字碎行、繁简统一为简体）；未装 `zhconv` 时跳过繁简转换。中文不能用英文的 CER 口径衡量，按 lev_cer / recall / precision 三元组看（实测 0.669 / 0.835 / 0.606）。
- MusicXML 通过官方 XSD 校验，并已在 **MuseScore Studio 4.7.5 实机打开**核对（双谱表、和弦、逐字歌词、段落、速度标记均正确）；再用 `MuseScore4.exe -o out.pdf score.musicxml`（**不加 `-f`**）严格导入，四份谱面全部 exit 0。

### v2 专用模型链路的额外限制（实测）

| 项 | 实际情况 |
| --- | --- |
| **SheetSage2 拍号** | 实测会把 4/4 判成 2/4（把八分音符当成拍，181 个「小节」×1.26s，真实歌曲不可能）。已加**自洽性校验**：按「每小节秒数」判定并回退 4/4、BPM 保持不变。但**这类错误可能还有别的形态**，当前只覆盖了这一种 |
| **SheetSage2 段落名** | 会输出 `pre-chorus` 等带连字符的标签，不在 schema 枚举内 → 已做规范化，未知标签落 `other`（**不猜**） |
| **和弦置信度** | SheetSage2 不输出逐事件置信度，因此**无法给出「第几小节需要复核」**；质量面板的 chords 分来自别的口径 |
| **VocalParse 分段** | 模型 card 明确 >30s 应先分段；已按 30s 切窗并拼回时间轴。分段边界处可能丢/切分音节 |
| **VocalParse 词表** | checkpoint 扩展了约 165 个 AST token（词表 152101 ≠ 标准 151936），必须用 Thinker 类 + 覆盖 `vocab_size`；用标准 `AutoModel` 会静默加载失败 |
| **TART rerank** | 论文未公开代码仓库（arXiv 2609.11904 检索无 GitHub 链接），只有启发性 rerank，`provenance.engine` 如实写 `tart-heuristic` |
| **Noise2Fret** | 仓库只有代码无权重；已把五项损失公式移植为 `aitsm/arrangement/playability.py`（纯 numpy） |

详见 [docs/13-demo-quality.md](docs/13-demo-quality.md) 与 [docs/11-v2-pipeline.md](docs/11-v2-pipeline.md)。

## 作为 DSH 插件使用

`plugin/` 是一个标准的 DSH skill（`audio-to-chord-sheet`），装好后在会话里说「帮我把这首歌扒成弹唱谱」即可触发：

```bash
python plugin/install.py            # 复制到 %USERPROFILE%\.agents\skills\
python plugin/install.py --dry-run  # 只看会做什么
```

> DSH 0.2.x 的用户级技能扫描根是 `~/.agents/skills` 与 `~/.dsh/skills`
> （另有项目内的 `.agents/skills`、`.dsh/skills`）。旧教程里的
> `~/.config/opencode/skills` 在 0.2.x 已不被扫描 —— 装到那里会静默不加载。
> 依据与升级判断见 [docs/03-dsh-plugin.md](docs/03-dsh-plugin.md)。

skill 会把整条管线跑在**独立子进程**里（避免 torch/whisper 影响 agent 进程），stdout 只回一个 JSON：`ok` / `out_dir` / `summary` / `usable`（质量是否够交付）/ `verdict`（good｜fair｜poor）/ `needs_review`（按 high/medium/low 分组、带小节号与秒数）/ `files` / `report_path` / `view_command`。安装细节与质量红线见 [docs/03-dsh-plugin.md](docs/03-dsh-plugin.md) 与 [plugin/SKILL.md](plugin/SKILL.md)。

## 测试

```bash
.venv/Scripts/python -m pytest tests/ -q   # 958 条（1 skipped）：管线、v2 四层适配器、编配闭环、导出、service、基准
node tests/web-editor.test.mjs             # 34 条：编辑器纯数据层（独立历史、撤销/重做、段落重编号）
node tests/web-payload.test.mjs            # 12 条：payload 归一化、时间轴覆盖、ABC 优先级、指法图索引
node tests/web-musicxml.test.mjs           # 17 条：MusicXML 本地解析、拒绝不支持格式、真实谱回归
```

覆盖：和弦符号解析与等音、指法生成合法性、调性/移调矩阵、小节量化与延音、
**HMM 节奏量化（可记谱性/小节求和/和弦锚点）**、**旋律带通与 F0 平滑**、
**谐波加权 chroma 与序列平滑**、段落映射、schema 校验、MusicXML 结构（含官方 XSD、
速度标记取整与去重、`direction` 子元素顺序）、MIDI SMF 往返、歌词清洗（含中文署名行/
静音幻觉/碎行/繁简）、音节对齐、manifest 合并与引擎元信息、插件脚本兼容，
以及 v2 新增的：**VocalParse token 解析（真实输出噪声/CoT 分隔/多 BPM/休止/30s 分段）**、
**qwen_backend 版本 pin 与路径探测**、**SheetSage2 产物解析与拍号自洽校验**、
**MusicXML 契约往返**、**Noise2Fret 五项可演奏性损失**、**部署内存预算与多 worker OOM**、
**设备探测（CPU/GPU 判定）**、**多曲目基准汇总**、**渲染 payload/timing**。

> 有 1 条 skipped 是「本地无 VocalParse 权重时跳过真实权重校验」——离线环境下属正常，
> 装了权重会自动生效。

## 许可

见 [LICENSE](LICENSE)。参考项目各自的许可请遵守其原始条款。
