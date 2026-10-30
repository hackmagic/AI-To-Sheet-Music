# 用 DSH 技能做弹唱谱（用户篇）

这篇是给**使用者**看的：装好之后怎么用、会得到什么、出问题怎么办。
开发者视角的说明见 [03-dsh-plugin.md](03-dsh-plugin.md)。

---

## 这是什么

一个装进 DeepSeek Harness 的技能。你把一首歌的文件丢给对话，说「帮我做个弹唱谱」，
它会在本地分析这首歌，产出：

| 文件 | 用途 |
| --- | --- |
| `score.json` | 网页乐谱的数据源，用自带查看器打开、移调、打印 |
| `score.musicxml` | 用 MuseScore / Guitar Pro 打开继续编辑 |
| `score.mid` | MIDI，可以播放或导进 DAW |
| `report.md` | 质量报告：哪里准、哪里要你自己核对 |

谱面内容包括：**五线谱主旋律 + 逐字歌词 + 和弦符号 + 吉他指法图 + 六线 TAB + 段落结构 + 节奏型提示**。

全程在你自己的电脑上跑，音频不外传，不用付费 API。

---

## 第一次安装

前置条件：项目本体已经在这台机器上跑通过（`.venv` 里装好了依赖，`ffmpeg` 可用）。

在仓库根目录执行：

```bash
# 推荐：用系统 Python 执行安装（见下方「为什么不用 .venv 的 python」）
python plugin/install.py
```

它会做四件事：

1. 把 `plugin/` 复制到 `~/.agents/skills/audio-to-chord-sheet/`
2. 写入 `aitsm-root.json`，记住仓库位置和该用哪个 Python
3. 检查 skill 的 `work/` 目录能不能写，不能写就自动修正权限
4. 跑一次自检，打印结果

> 为什么是 `~/.agents/skills`：这是 DSH 0.2.x 用户级扫描根之一。
> 旧版教程写的 `~/.config/opencode/skills/` **DSH 0.2.x 根本不扫**，装到那里会静默不加载。
> 四个扫描根与依据见 [docs/03-dsh-plugin.md](03-dsh-plugin.md) 的「DSH 0.2.x 技能发现契约与版本」。

成功时最后几行看起来像：

```text
[1/4] 已复制 SKILL.md / scripts / references / agents
[2/4] 已写入 aitsm-root.json
[3/4] work/ 可写
[4/4] 自检：... scripts/doctor.py --json
    就绪：是
    项目根：K:\workspaces\hackmagic\AI-To-Sheet-Music（配置文件 aitsm-root.json）
    建议：环境完整，可产出完整弹唱谱。
```

装到别处：`python plugin/install.py --dest <skills 目录>`。

### 装完要不要重启 DSH

**不用。** 实测：文件复制进 `~/.agents/skills/` 后，当前 DSH 会话的技能列表里就出现了
`audio-to-chord-sheet`。它是在每次列技能时读取目录的。

但如果对话里已经打开很久都没生效，去 **设置 → 技能** 点一下刷新，或者重开一个会话。

### 为什么不用 `.venv` 的 python 装

有些 Windows 机器上，项目 `.venv/Scripts/python.exe` 带「低完整性」标记（沙箱或安全软件遗留）。
这种解释器**只能写它自己所在的磁盘区域**，往 `C:\Users\...` 写文件会被直接拒绝 ——
连 `mkdir` 都不行。安装要用系统 Python：

```bash
# Windows：任选其一（只要不是项目 .venv 里的那个）
python plugin/install.py
"C:/Program Files/Python312/python.exe" plugin/install.py
```

**换解释器不一定有用：关键是「从哪个终端启动」。**
Windows 的子进程会**继承父进程的完整性级别**。所以从带 Low 标签的 `.venv` 里
派生出去的任何解释器（哪怕 anaconda 本身是 Medium）都会降级成 Low，照样写不了 `%USERPROFILE%`。
正确做法是**在一个正常的（Medium）终端里直接调用系统 Python**，而不是先激活 `.venv` 再跑脚本。

自检命令：

```bash
icacls .venv/Scripts/python.exe | findstr Mandatory   # 出现 Low Mandatory Level 就是带标签
```

根因与三方对照实测见 [已知坑 1b](04-known-issues.md)。`install.py` 现在会先用
本进程实测写入（不派生子进程），确认是 Low 就直接给出可复制命令，不再做无效的换解释器重试。

安装脚本会自动把 skill 的 `work/` 目录做成低完整性可写，这样**生成**时用 venv python 就没问题。
注意 `work/` 若不可写，`generate.py` 会**自动回落到仓库的 `work/skill-out`**（已被 `.gitignore` 覆盖），
实际输出位置以 `doctor.py --json` 的 `paths.default_out_dir` 为准。

---

## 怎么触发

直接说人话就行，技能靠 `description` 里的场景匹配：

- 「帮我把这首歌扒成弹唱谱」+ 文件路径
- 「这首歌唱什么和弦？」
- 「我想要个能自弹自唱的谱」
- 「帮我把它移调到 G 调 / 变调夹夹 2 品」
- 「转成 MusicXML 或 MIDI」

**不会触发**这些（它们不是这个技能的范围）：

- 只要歌词文字稿（那是语音转文字）
- 要乐队总谱 / 多声部编配
- 要 MuseScore 工程文件（.mscz）
- 从零作曲编曲

---

## 用起来大概多久

纯 CPU 实测（Windows 11 / Python 3.12.8 / 未启用 GPU）：

| 歌曲时长 | 首次（要跑分轨） | 分轨缓存命中 |
| --- | --- | --- |
| 4 分钟（229 秒） | 约 9–10 分钟 | 约 4 分钟 |

其中**分轨约 5 分钟、歌词约 2.5 分钟**，其余阶段各十几秒。
歌词占大头是因为默认用 `medium` 模型（准确率明显更高）；
想快可以加 `--whisper-model small`，约省 90 秒，代价是歌词错误率上升约 0.10。

第一次还会下载模型权重（约 2GB：Demucs 80MB + Whisper medium 1.5GB + small 460MB，一次性）。

想快一点可以加 `--no-separation`，但**旋律会变得不可信**（会锁到贝斯上），一般不建议。

---

## 你会看到什么

对话里会给出四样东西：

1. **基本信息**：调性、变调夹、BPM、拍号、小节数、段落结构、难度
2. **三个文件路径**
3. **哪里需要你自己核对**，带具体位置，比如：
   - 「第 32–33 小节和弦置信度 0.51，建议对照原曲确认」
   - 「歌词为自动识别，平均置信度 0.30，建议以原曲为准」
4. **实际存在的限制**

「置信度」是相对质量分（0–1），**不是准确率百分比**。

### 打开网页看谱

```bash
python scripts/serve.py --score <输出目录>/score.json
```

会打印一个 `http://127.0.0.1:7874` 之类的地址。网页里能：

- 移调 ±6 半音，改 capo 0–7 品（和弦、旋律、调号、指法同步变）
- 播放跟随（内置合成器：旋律 + 扫弦 + 节拍器）
- 质量面板：总分 + 分阶段分数 + 可点击跳转的告警
- A4 打印

---

## 结果准不准（诚实版）

实测一首 3 分 49 秒的英文歌（`小林未郁、HOYO-MiX - 下一站 Beyond This Station`）：

| 部分 | 表现 | 说明 |
| --- | --- | --- |
| 结构（小节/段落/调性/拍号） | **好** | 92.47 BPM、4/4、D 大调、88 小节、7 段落 |
| 旋律 | **好** | 音域 MIDI 57–84（正常女声），量化漂移 0.048s |
| 和弦 | **中等** | 55 个和弦事件、全在调内、平均置信度 0.57；低置信度小节会标出来 |
| 歌词 | **偏弱** | 字符级 CER ≈ 0.49–0.54（medium，重复运行会差 0.05）；不会编造，但会错 |
| 网页呈现 | **好** | 移调、capo、打印、手工修正和弦/歌词都验证过 |

### 已知限制（都是真的）

- **歌词**需要你校对。模型会自动过滤掉复读和编造，但不保证每句都对。
- **和弦**对失真吉他、大量七和弦的歌会偏保守（宁可给三和弦也不给错和弦）。
- **关系大小调**（Bm vs D 这种）音频上很难区分，默认按大调标。知道原曲是小调就说「按 B 小调做」。
- **速度/拍号中途变化**不支持，会按主要速度记谱并告警。
- **多声部人声**（合唱、和声层）只取主旋律。
- **MusicXML** 通过了官方 MusicXML 4.0 XSD 校验，但**没在 MuseScore 里实际打开过**（本机没装）。打不开请把报错发出来。
- **GPU 默认没启用**，全流程 CPU。

---

## 常见问题

### 说「就绪：否」或提示找不到项目

技能脚本需要知道项目本体在哪。按提示做任一即可：

```bash
# 方式 1：环境变量
set AITSM_ROOT=K:\workspaces\hackmagic\AI-To-Sheet-Music

# 方式 2：重跑安装（会写配置文件）
python plugin/install.py
```

自检里看 `paths.aitsm_root_source`，它会告诉你是靠哪条规则找到的。

### 说输出目录不可写

这台机器上的 Python 可能只能写特定目录。用 `--out` 指一个明确可写的位置：

```bash
python scripts/generate.py "song.mp3" --out K:/workspaces/hackmagic/AI-To-Sheet-Music/work/mysong
```

### 报「找不到能 import aitsm 的 Python」

说明 `.venv` 没建好或依赖没装。到仓库根执行：

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 跑得太慢

先确认是不是每次都在重跑分轨。同一首歌、同样参数会复用 `work/` 下的缓存。
换歌、换 `--device`、改分轨参数会让缓存失效。

### 谱子看起来不对

先看 `report.md` 第四节「需要确认」，大部分问题那里已经列了小节号。
如果整首调都不对，多半是关系大小调判反了 —— 说「按 xx 小调重做」。
如果旋律整体高/低一个八度，用 `--transpose 12` 或 `--transpose -12`。

### 网页打不开

`serve.py` 是常驻进程，要放在后台跑。端口被占就换：

```bash
python scripts/serve.py --score <路径>/score.json --port 7875
```

### 想彻底卸载

技能可能出现在**两个**位置，都清掉才算彻底（旧位置 DSH 0.2.x 已不扫描，但别的 agent 工具可能还在用）：

```bash
# 新位置（DSH 0.2.x 扫描的）
rmdir /s /q %USERPROFILE%\.agents\skills\audio-to-chord-sheet

# 旧位置（opencode 时代的遗留；存在才需要删）
rmdir /s /q %USERPROFILE%\.config\opencode\skills\audio-to-chord-sheet
```

不确定哪个还在，先看一眼：

```bash
dir %USERPROFILE%\.agents\skills\audio-to-chord-sheet
dir %USERPROFILE%\.config\opencode\skills\audio-to-chord-sheet
```

**不想手动删旧副本**：`python plugin/install.py --remove-legacy` 会把它**改名**成
`audio-to-chord-sheet.bak-<时间戳>` 而不是直接删除 —— 改名可回滚，`work/` 里的产物都还在。
脚本默认**不删**（用户数据不能无声消失），只打印提示。

不动项目本体，也不改任何 DSH 配置。

---

## 文件都放哪

```text
~/.agents/skills/audio-to-chord-sheet/
├── SKILL.md                 # 技能说明（agent 读的是这个）
├── aitsm-root.json          # 安装时写的定位配置：仓库路径 + 解释器
├── scripts/
│   ├── generate.py          # 主入口：音频 → 弹唱谱（JSON 输出）
│   ├── doctor.py            # 环境自检
│   ├── serve.py             # 网页查看器
│   └── _locate.py           # 公用路径解析
├── references/quality.md    # 质量判读规则
├── agents/openai.yaml       # 显示名与默认 prompt
└── work/                    # 默认输出目录
```

产物默认落在 `work/out/`，也可以自己指定 `--out`。
