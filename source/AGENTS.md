# AI_UP · 给 AI 助手的操作手册

AI_UP 是在自己电脑上运行的数字人口播生产工具：
文案 → 克隆声音配音（阿里云百炼）→ HeyGem 对口型（本机 Docker）→ 画质还原 → 自动剪辑包装 → 竖屏 / 横屏成片 + 封面。
网页工作台：`启动工作台.bat` → http://127.0.0.1:8790 。

## 规矩
- API Key 只放在 `config.local.json`（或环境变量），不要写进其他文件、不要打印到日志或聊天里。
- 不要真删用户文件；要清理就移到 `00_素材库/_回收站/`。
- 改了代码就在本文件末尾记一笔（改了什么、为什么）。
- `app/server.py`、`app/jobs.py` 改了要重启工作台（双击 `启动工作台.bat`，启动器发现旧版本会自动关掉旧进程）；
  `app/pipeline.py`、`media.py`、`restore.py`、`avatar_mix.py` 和 `engine/` 里的代码每个任务开始时自动重新载入。

## 目录
- `00_素材库/数字人/`：数字人原片。名字带序号（`主播-1.mp4`、`主播-2.mp4`…）的两条以上自动成组，网页里显示“主播 · N 条随机混排”。
  `<名>.json` 可写 `{"use_from":1.5,"use_to":40}` 只用其中一段。
- `00_素材库/声音/`：声音录音（15~20 秒、安静、连续说话最好）；`_prepared/` 转码缓存，`_cache/` 逐句配音缓存，`音色库.json` 音色 ID。
- `00_素材库/通用素材/`：画中画素材（图片 / 短视频），`materials.json` 记关键词。
- `04_工程/<日期>_<标题>_<id>/`：每条任务的配音、对口型、字幕、剪辑日志、任务日志、标题设置、封面底图。
- `05_成片/<日期>/`：成片 mp4、srt、封面（竖屏 + 横屏）。
- `06_日报/tasks.db`：任务队列；`06_日报/工作台日志.txt`：网页接口报错。
- `07_音频纠正/`：音频纠正功能的会话文件。
- `engine/`：剪辑引擎（`run.py` 命令行入口，`aiclip/` 模块，`styles/` 风格包，`fonts/` 字体，`models/` 人脸检测模型）。

## 代码地图
- `app/server.py` + `web/index.html`：标准库网页工作台（文案与试听 / 素材与声音 / 剪辑任务 / 音频纠正 / 标题与封面面板）。
- `app/jobs.py`：SQLite 任务队列，单线程依次执行；“重新剪辑”= retry，复用已完成的阶段。
- `app/pipeline.py`：一条任务的全流程（配音 → 数字人混排 → 对口型 → 画质还原 → 字幕 → 剪辑竖屏/横屏 → 封面）。
- `app/clients/qwen_voice.py`：百炼声音复刻（注册音色 + qwen-audio 合成，支持情绪指令 / 语速）。
- `app/clients/heygem.py`：HeyGem 对口型（把文件复制到容器挂载目录，接口传容器内路径；卡死自动重启容器）。
- `app/avatar_mix.py`：同组数字人素材随机拼接（不倒放）；长素材每 10~18 秒在句间切换并记录切换点。
- `app/restore.py`：画质还原——HeyGem 只保留嘴部区域，其余贴回原片对应帧。
- `app/audiofix.py`：音频纠正（框选一句 → 用克隆声音重读 → 替换）。
- `app/emotion_ai.py` / `app/title_ai.py`：大模型识别每句情绪 / 提炼标题（没配置大模型就用规则版）。
- `app/titleset.py`：“标题与封面”面板（手动调文字、字体、位置、景深、颜色，实时预览）。
- `engine/aiclip/planner.py`：去停顿、分块、推镜、画中画、信息层、律师包装的编排，输出 timeline.json。
- `engine/aiclip/compositor.py`：逐帧合成；`hud.py`/`hudplan.py` 信息层；`law.py` 律师动效；`titlefx.py` 手动标题；`cover.py` 封面。
- `engine/aiclip/llm.py`：大模型接口（OpenAI 兼容或 Anthropic 格式），失败自动退回规则版。

## 风格包（engine/styles/*.json）
- `aiup_dark_tech`：暗夜科技。
- `aiup_orange`：暗夜橙光（橙/黑/白，信息层大字、逐条列表、重点推镜、双语字幕）。
- `law_gold`：律师·律政金（顶部金色宋体标题、律师名牌、开场金字 + 红色大字 + 打字机、每句红色重点词、免责声明）。
- 预览图放 `engine/styles/previews/<风格id>.jpg`（960×540），没有也能用。
- 统一封面照：`engine/assets/covers/my_cover.jpg`（3:4 半身照）。没有这张图时封面从视频里自动挑画面。

## 常用排错
见 skill 的 references/troubleshooting.md。

## 更新记录
- 读音纠正词典（`app/pron.py`，`00_素材库/读音纠正.txt`）：配音前把“原文”换成“读法”，字幕不变；网页“文案与试听”→“读音纠正”。
- “✨ AI 识别情绪和错字”：`emotion_ai.detect` 同时返回错别字改正（fix）和易读错多音字（pron），网页句子下方确认后生效；多音字可一键加入读音词典（`/api/pron/add`）。
- 对口型清晰度 720p / 1080p（任务字段 `lipsync_res`）：1080p 失败自动退回 720p 重试（`Job._lipsync`）。
