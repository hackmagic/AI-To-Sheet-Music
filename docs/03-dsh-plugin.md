# DSH 插件形态（第二阶段方案）

本文说明如何把 AI-To-Sheet-Music 从「独立 Web 应用」打包成 **DSH 插件 / Skill**，让用户在对话里直接说「把这首歌做成弹唱谱」就得到一份可打印的谱。

## 为什么适合做成 Skill 而不是纯插件 UI

DSH 的插件分两类：

| 形态 | 适合什么 | 本项目 |
| --- | --- | --- |
| **Skill**（`SKILL.md` + 脚本） | 有明确产物、需要 agent 决策与质量把关的任务 | ✅ 主形态 |
| **插件 UI**（Web 面板） | 需要复杂交互（拖拽、可视化编辑） | ✅ 辅助（复用现有 `web/`） |

扒谱是一个**需要判断与降级决策**的任务：分轨失败要不要继续？歌词置信度低要不要标出来？段落切得离谱要不要合并？这些正是 agent + skill 擅长的事，而不是一个"上传→下载"的固定按钮。所以主形态是 **Skill**，把 Web 界面作为可选的查看器。

## 目录结构

```text
dsh-skill-audio-to-sheet/
├── SKILL.md                    # 技能入口：何时用、怎么用、验收标准
├── agents/openai.yaml          # 显示名与默认 prompt
├── scripts/
│   ├── generate.py             # 主入口：调 aitsm CLI，产出并摘要报告
│   ├── serve.py                # 启动本地 Web 查看器（可选）
│   └── doctor.py               # 环境自检（依赖、模型、ffmpeg）
├── references/
│   ├── outputs.md              # 产物说明与字段含义
│   ├── quality.md              # 置信度与告警的解读、什么情况要人工复核
│   └── troubleshooting.md      # 常见失败与降级路径
└── vendor/aitsm/               # 项目本体（或 pip 安装 + 版本锁定）
```

## `SKILL.md` 草稿

```markdown
---
name: audio-to-chord-sheet
description: 把音频/视频文件转成可自弹自唱的吉他弹唱谱（五线谱+逐字歌词+和弦+指法图+TAB），
  交付网页乐谱、MusicXML、MIDI 与评分报告。用于「扒谱」「做弹唱谱」「这首歌唱什么和弦」等请求；
  不用于纯音频转文字、和弦谱以外的乐队总谱或需要 MuseScore 工程文件的任务。
---

# 音频转弹唱谱

你是负责交付成品的扒谱 agent。目标是让用户拿到**能直接照着弹唱**的谱，而不是一堆识别结果。

## 默认约定
- 输入是混音成品歌；人声用于旋律与歌词，吉他/键盘用于和声，不强行分离五件乐器。
- 默认难度 `elementary`，默认自动定调 + 自动 capo。
- 固定交付：`score.json`（网页乐谱数据）、`score.musicxml`、`score.mid`、`report.md`（质量摘要）。
- 全程本地、免费、不外传音频。

## 开始执行
1. 先运行 `python plugin/scripts/doctor.py` 检查环境（ffmpeg / torch / demucs / faster-whisper；
   v2 还会报 `transformers` / `qwen-asr` / 各专用模型权重是否就绪）。
   缺重型依赖时告知用户质量会下降，并询问是否继续（不要静默降级）。
2. 确认输入文件存在、时长在限制内（>10 分钟先提示耗时）。
3. 运行 `python plugin/scripts/generate.py <audio> --out <dir> [--key] [--capo] [--difficulty] [--device auto|cpu|cuda]`。
4. 读 `manifest.json`、`score.json.quality` 与 **`symbolic.json.provenance`**，
   判断哪些环节需要向用户说明（v2 的降级信息主要在 provenance 里）。

## 质量把关（必须做）
读 `references/quality.md`。至少检查：
- `quality.stages` 里是否有 < 0.5 的阶段，若有，在交付说明里点명具体环节。
- `chords` 是否大量落在调外（可能是定调错）。
- `lyrics` 是否为空或 CER 明显不合理（音译、乱码、纯语气词）。
- `melody.range` 是否落在合理人声音域（55-84），否则可能八度错。
- 段落数是否在 6-12 之间，是否有 <4 小节的碎片段落。

发现问题时：**先尝试自动修复**（重跑、换 stem、调参数），修不了就在交付说明里如实写明，
不要为了让结果看起来完整而隐去错误。

## 交付
输出简短说明：调性/变调夹/速度/拍号、段落结构、难度、必要改编；
列出 `score.json` / `score.musicxml` / `score.mid` 三个文件路径；
说明实际存在的限制（歌词待校对、和弦置信度低的小节编号等）。
如需可视化，运行 `python scripts/serve.py --score <dir>/score.json` 并给出本地 URL。
```

## `generate.py` 的职责（薄封装，不含算法）

```python
"""把 aitsm CLI 包成 agent 友好的单次调用：一次运行 -> 结构化摘要。"""

def main() -> int:
    # 1. 解析参数，校验输入
    # 2. subprocess 调 aitsm.cli generate（保持进程隔离，避免 GPU 状态污染 agent 进程）
    # 3. 读 score.json / manifest.json，生成 report.md（质量摘要 + 待确认项）
    # 4. stdout 输出 JSON：{ok, out_dir, files:{...}, summary:{key,capo,tempo,bars,sections}, issues:[...]}
```

要点：
- **进程隔离**：分析会加载 torch/demucs/whisper，绝不能跑在 agent 主进程里。
- **结构化返回值**：agent 需要能直接读的 JSON，而不是让它去 parse 人类日志。
- **幂等**：同输入 + 同参数 + 同版本，直接复用 `work/<hash>/` 的缓存结果。

## `report.md` 模板

```markdown
# 弹唱谱报告：<歌名>

## 基本信息
调性 D 大调（capo 0）· 92 BPM · 4/4 · 88 小节 · 12 段落 · 难度 基础

## 产物
- score.json（网页乐谱）
- score.musicxml（MuseScore / Guitar Pro，含五线谱 + TAB）
- score.mid
- symbolic.json（★ 符号契约，`provenance` 里逐层写明 requested vs engine）
- payload.json / timing.json（★ 网页渲染数据与绝对时间轴）
- 查看：`python scripts/serve.py`

## 需要你确认的
- [ ] 歌词为自动识别，CER 约 0.50，建议对照原曲校对（低置信度在第 12/28/41 小节）
- [ ] 副歌 57-70 和弦置信度 0.42，可能是同一条进行反复
- [ ] 第 35-52 小节被判为「主歌 3」长达 18 小节，可能应再切分

## 已做的改编
- 原曲有失真吉他双轨，已合并为单轨扫弦
- 前奏 9 小节保留了和弦进行，吉他动机未转 TAB（信号不足）
```

### v2 专用模型下的报告差异（skill 需要注意）

装了专用模型后，产物与口径有变化，`generate.py` / `report.md` 应据实反映：

| 项 | v1 内核 | v2 专用模型 |
| --- | --- | --- |
| 引擎 | `whisper+pyin` / `beats+chroma` | `vocalparse` / `sheetsage2` / `duet` / `demucs-onnx` |
| 降级信息 | 只看 `manifest.degraded` | **外加 `symbolic.provenance`**（逐层 `requested` vs `engine`） |
| 和弦位置 | 有逐事件 `confidence`，可指到具体小节 | SheetSage2 **不输出逐事件置信度** → 不能再说「第 N 小节和弦置信度 0.42」，只能说「该段和弦来自联合解码，无法逐小节打分」 |
| 耗时 | 4 分钟歌约 9–10 分钟 | **整曲约 76 分钟（CPU）**，VocalParse 占 73% |
| 设备 | 无 | 可通过 `--device` 传 `auto`/`cpu`/`cuda`；GPU 不可用时后端会回落并写 `device.resolved` 告警 |

**红线不变**：信号缺失就留空 + 告警，不猜、不隐藏降级、不宣称准确率。
新增一条：**不要把 `symbolic.provenance.engine != requested` 的情况说成"已用上专用模型"**。

## 打包与分发

技能目录是 DSH 扫描的 `%USERPROFILE%\.config\opencode\skills\`（本机即 `C:\Users\angel\.config\opencode\skills\`）。
`plugin/install.py` 负责复制 + 写 `aitsm-root.json` + 自检：

```bash
# 1. 依赖安装（用户机器，首次）
.venv/Scripts/python -m pip install -r requirements.txt

# 2. 技能注册 —— 必须用「能写用户目录」的解释器
D:/WorkSoftware/anaconda3/python.exe plugin/install.py
#    .venv/Scripts/python.exe 带 Low 完整性标签，写不了 %USERPROFILE%，会被拒绝；
#    install.py 会探测并明确报错，不会半途留下坏副本。

# 3. 验证
.venv/Scripts/python plugin/scripts/doctor.py --json     # ready: true
diff -q plugin/SKILL.md "$USERPROFILE/.config/opencode/skills/audio-to-chord-sheet/SKILL.md"
```

**DSH 会自动重扫技能目录，重装后无需重启进程。**
改了 `plugin/` 之后必须重跑 `install.py`——技能目录是**复制品**，改仓库不重装不生效（踩过）。

## 质量红线（不可妥协）

1. **不伪造音符**：信号缺失就留空 + 告警，绝不"猜"填充。
2. **不隐藏降级**：没分轨、没 whisper、模型缺失，必须出现在交付说明里。
3. **不宣称准确率**：`confidence` 是相对质量分，不是概率，措辞上不能写"准确率 89%"。
4. **不省略限制**：歌词待校对、和弦低置信度小节，必须列出小节号。
5. **授权**：用户音频只在本地处理；若未来加入云端，必须先明确告知并取得同意。

## 与 Web UI 的关系

Web 界面（`web/`）在插件里作为**可选的查看器**：
- agent 生成 `score.json` 后，用户可以 `serve.py` 打开看谱、移调、打印；
- 不做成插件内的常驻面板，因为乐谱需要大面积横向空间，塞进侧边栏不可用；
- 若未来做深度集成，可以让插件把 `score.json` 直接投递到 DSH 的预览面板（只读渲染）。

## 里程碑

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| M1 | 独立 CLI + Web 应用跑通真实歌曲 | ✅ 已完成 |
| M2 | `SKILL.md` + `generate.py` + `report.md` 生成 | ✅ 已完成（report 由 `build_report` 确定性渲染） |
| M3 | 放进 DSH skills 目录，用 3 首不同风格歌曲验证 | ✅ 已完成（三首合成曲 + 一首真实歌；DSH 已扫描到，可直接触发） |
| M4 | 打包分发（版本锁定、依赖检查、README） | ✅ 已完成（`install.py` + `doctor.py` + `requirements.txt` + README / 用户篇） |

### 已验证的边界（2026-10-21）

- 从**外部 cwd**（`C:/Windows/Temp/...`）调用技能的 `scripts/generate.py`，能正确定位仓库并跑通。
- `--no-lyrics` 时 `needs_review` 必有 `lyrics.none`（high），不会假装有词。
- 技能返回的 JSON 契约：`ok / out_dir / aitsm_root / python / files / summary / usable / verdict / needs_review / needs_review_counts / sections / report_path / view_command`。
- `generate.py` 对旧/新两种 `manifest` 形状都兼容（`_manifest_engines()`），并有 16 条回归测试。
