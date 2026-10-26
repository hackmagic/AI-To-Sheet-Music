# 需求分解（v1 · 自弹自唱吉他谱）

> **v2 已把「转录」与「编配」拆成两层**：转录层换用四个专用模型
> （htdemucs_ft / VocalParse / SheetSage2 / DuET），编配层只吃符号、不碰音频。
> 本文的需求范围（要做成什么样的谱、质量红线）**没有变**，变的是实现路径。
> 分层细节见 [`11-v2-pipeline.md`](11-v2-pipeline.md)，实测质量见
> [`13-demo-quality.md`](13-demo-quality.md)。

## 0. 产品一句话

输入一个音乐文件（mp3/wav/flac/m4a，或 mp4 等视频），AI 自动扒谱，在浏览器里产出**可直接照着自弹自唱的中文弹唱谱**：人声主旋律 + 逐字歌词 + 和弦符号 + 吉他指法图 + 节奏型/伴奏，并支持移调、变调夹、跟原曲同步播放、导出 MusicXML/MIDI/PDF。

长期形态：DSH 插件（第二阶段），第一阶段先交付独立可运行的 Web 应用 + CLI。

## 1. 参考项目复用映射

| 能力 | 来源 | 复用方式 |
| --- | --- | --- |
| 混合音频解码/标准化（ffmpeg/ffprobe） | AI-ChordCraft `app.py:258-310`, `src/_runtime.py:466-577` | 移植为 `aitsm/pipeline/ingest.py` |
| 和弦符号规范化、调性估计、速度规范化 | AI-ChordCraft `src/chord_recognition.py:123-564` | 移植为 `aitsm/pipeline/chords.py` 纯 numpy/librosa 版 |
| 和弦事件后处理/区间化简/段落对齐 | AI-ChordCraft `chord_recognition.py:600-786,1380-1544` | 移植，作为无重型模型时的默认引擎 |
| 段落结构 + 弱结构兜底 | AI-ChordCraft `src/structure_recognition.py`, `song_analysis.py:93-340` | 用 librosa 自相似矩阵实现本地段落切分；可选保留 SongFormer 服务接口 |
| 歌词 ASR 段落/时间轴规整 | AI-ChordCraft `src/lyric_asr.py` | 移植 coerce/dedupe/offset 逻辑，ASR 引擎换成本地 whisper |
| Web 工作台形态（上传→分析→展示） | AI-ChordCraft `app.py`, `frontend/` | 参考交互与接口形态，重写 UI 为乐谱优先 |
| 紧凑谱面事件模型 + 确定性记谱编译 | To-Sheet-Music-Skill `references/score-events.md`, `scripts/score_events.py` | 借鉴“只维护 JSON、渲染器确定性生成”的架构；本项目的渲染目标是浏览器而非 MuseScore |
| 分轨缓存与 manifest 证据纪律 | `scripts/separate_audio.py`, `references/separation.md` | 移植 manifest + 输入哈希缓存 |
| 可演奏性/TAB 弦品规划 | To-Sheet-Music-Skill `scripts/score_events.py`（DP 弦品选择） | v2 已落实为 `aitsm/arrangement/fingering_dp.py`（全局最优 DP，五项成本）+ `tart_rerank.py`（重排）；可演奏性度量见 `arrangement/playability.py`（Noise2Fret 五项损失移植） |
| 质量网关与证据记录 | `references/notation-and-qa.md`, `scripts/*_audit.py` | 转成 `aitsm/score/validate.py` 报告 + 前端告警展示 |

不照搬的部分：MOSS-Music / SongFormer / essentia / plkd-acr 等需要额外大模型或私有运行时；参考项目的五件乐队总谱与 MuseScore 导出链（本项目第一版不依赖 MuseScore 安装，直接浏览器渲染 + 可选 MusicXML）。

## 2. 目标用户与场景

- 想用吉他自弹自唱一首歌、但不会扒谱的人：上传歌曲，得到弹唱谱，直接练。
- 需要移调到适合自己嗓子的调：一键移调 + 变调夹建议。
- 想把谱子打印带到琴房：A4 打印 / PDF。
- 想继续编辑：导出 MusicXML 到 MuseScore / Guitar Pro。

## 3. 功能需求（P0 = MVP 必须；P1 = 本阶段尽量；P2 = 后续）

### P0 输入与解析
- F1 支持 mp3/wav/flac/m4a/ogg/aac 与常见视频容器（mp4/mov/mkv/webm），视频自动抽音轨。
- F2 探测时长、采样率、声道；统一转成 44.1kHz 单声道分析音频 + 保留立体声播放音源。
- F3 上传大小与时长限制可配置，默认 ≤100MB / ≤10 分钟。

### P0 音乐分析
- F4 分轨：Demucs `htdemucs`（vocals/drums/bass/other）；`other` 视为吉他/键盘伴奏。失败时降级为中置声道人声提取 + 谐波打击分离，并明确告警。
- F5 节拍与速度：BPM、beat 时刻表、每小节拍数（4/4、3/4、6/8 等启发式判定）、弱起处理。
- F6 调性：12 调 + 大小调，输出置信度与调号升降数。
- F7 和弦识别：按拍/半小节粒度输出和弦符号（支持 maj/min/7/maj7/m7/sus/dim/aug/斜杠低音），带置信度。
- F8 段落结构：前奏/主歌/预副歌/副歌/间奏/桥段/尾奏，时间边界与小节边界对齐；重复段识别。
- F9 歌词 ASR：中文/英文逐句识别，词级/字级时间戳，输出带起止时间的歌词行。
- F10 人声旋律转谱：从 vocals 轨提取基频与起音，量化为 MIDI 音高 + 时值，落在节拍网格上。

### P0 弹唱谱生成
- F11 调性简化与变调夹：在 C/G/D/A/E 等开放把位友好的调中选一个 `play_key`，给出 capo 品数；用户可手动覆盖。
- F12 和弦指法库：每个和弦符号给出 1~3 个可用把位（弦/品/指法/把位/横按），优先开放和弦与常用把位，输出指法图数据。
- F13 节奏型：按段落能量与速度选扫弦/分解型，输出可渲染的节奏型事件（下/上/切音/重音）+ 名称。
- F14 主旋律 + 歌词对齐：每个音符挂载歌词音节；一字多音用延音线，多字一音用连字符；无旋律的音节降级到和弦谱行显示。
- F15 谱面组装：小节数、段落边界、弱起、速度/拍号变化、反复记号的完整 `score.json`。
- F16 前奏/间奏/尾奏：至少给出和弦进行与节奏型；若吉他 stem 有明显单音 riff，尝试输出简化 TAB。

### P0 Web 呈现
- F17 五线谱（主旋律，含调号、拍号、速度标记）+ 歌词逐字对齐 + 和弦符号。
- F18 和弦指法图（SVG）：在每次和弦变化处显示指法图，可开关。
- F19 吉他伴奏：节奏型符号 + 可选六线 TAB 谱表（和弦扫弦实现）。
- F20 播放：内置合成器按谱面播放（旋律 + 和弦扫弦 + 节拍器），小节/音符高亮，自动滚动；可选加载原曲音频对齐播放。
- F21 移调（±6 半音）与变调夹切换实时重排和弦、和弦图、旋律。
- F22 打印样式（A4、每行 4 小节、段落标题、页脚页码）。
- F23 质量面板：调性/速度/和弦/歌词/旋律的置信度、告警与“需人工确认”标记。

### P0 导出与集成
- F24 导出 MusicXML（含歌词、和弦符号、TAB）、MIDI、JSON。
- F25 服务端保留分析中间产物（stems、analysis.json、score.json）便于复现与调试。
- F26 CLI：`python -m aitsm.cli generate <audio> --out <dir> [--key C] [--capo 2] [--no-lyrics]`，供 DSH 插件调用。

### P1
- F27 自动选择“最好唱”的调（依据人声音域与旋律最高/最低音）。
- F28 和弦简化：七和弦→三和弦、斜杠和弦→原位，按难度档位。
- F29 指弹（分解和弦）风格选项。
- F30 段落重复复用（副歌只写一次 + 反复记号）。
- F31 手动修正：前端可编辑和弦/歌词（P1 末或 P2）。

### P2（后续）
- F32 DSH 插件打包（skill + 本地服务 + 一键启动）。
- F33 五件乐队总谱（吸收 To-Sheet-Music-Skill 编制能力）。
- F34 多语种歌词、日文/英文分词与假名标注。
- F35 云端/批量处理、历史作品库。

## 4. 非功能需求

- N1 全本地免费：不依赖付费 API；不外传用户音频。
- N2 单卡 8GB（RTX 4060 Laptop）可跑；CPU 兜底路径必须存在（慢但可用）。
- N3 端到端时延目标：4 分钟歌曲，GPU ≤90s，CPU ≤8min。
- N4 可复现：输入哈希 + 工具版本 + 参数写入 `manifest.json`。
- N5 诚实性：所有低置信度结果必须显式标注，不允许静默编造音符/歌词。
- N6 模块可测：纯函数化分析步骤，关键路径有单元测试与合成音频回归测试。

## 5. 验收标准（MVP）

1. 用一首真实中文歌曲（`hackmagic-music/assets/小林未郁、HOYO-MiX - 下一站 Beyond This Station.flac`）端到端跑通，产出 `score.json` 与可打开的网页乐谱。
2. 网页正确显示：标题、调性/拍号/速度/变调夹、段落、五线谱主旋律、逐字歌词、和弦符号、和弦指法图、节奏型。
3. 播放跟随光标、小节点击跳转、移调与变调夹切换即时生效、打印预览分页正常。
4. `score.json` 通过 `schemas/score.schema.json` 校验；MusicXML 可在 MuseScore 打开；MIDI 可播放。
5. 质量面板如实反映各环节置信度与降级原因（如未分轨、未识别歌词）。

## 6. 关键风险与对策

| 风险 | 对策 |
| --- | --- |
| 中文歌词 ASR 准确率不稳 | whisper small/medium + 音乐专用 initial_prompt；词级时间戳与旋律起音交叉校验；低置信度标注“歌词待校对” |
| 混音中和弦识别错（尤其失真吉他） | 多引擎投票（模板匹配 + bass 根音约束 + 段落内和弦统计平滑）；保守化简 |
| 人声旋律八度错误/噪声 | 以 vocals stem 为主，中值滤波 + 音域约束 + 与歌词音节数交叉校验 |
| 段落结构无监督易碎 | 自相似矩阵 + 能量/音色变化 + 和弦重复模式三重证据；允许用户手工改段落（P1） |
| 模型下载失败（网络受限） | 分轨/ASR 均可降级；启动时预检并给出明确提示 |
| 扫弦型与真实演奏不符 | 明确标注“建议节奏型”，不声称逐音还原；允许按段落切换 |
