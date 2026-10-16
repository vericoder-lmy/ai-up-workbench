# AI_UP 数字人口播工作台

在自己电脑上运行的数字人口播生产工具：

**写文案 → 克隆声音逐句配音（可调情绪、语速）→ HeyGem 对口型 → 画质还原 → 自动剪辑包装 → 竖屏 + 横屏成片和封面**

- 网页工作台（`http://127.0.0.1:8790`），全部在本机运行，素材不出电脑（配音调用阿里云百炼接口，可选的大模型接口按你的配置调用）
- 同组数字人素材随机混排，不倒放；长素材自动换机位
- 画质还原：只保留 HeyGem 生成的嘴部，其余贴回原片，眼睛、头发保持清晰
- 风格包：暗夜橙光、暗夜科技、律师·律政金；信息卡片、重点词、画中画素材、推镜、音效自动编排
- 封面：竖屏 + 横屏两张，写好主题；“标题与封面”面板可手动调文字、字体、位置、景深、颜色
- 音频纠正：读错一个字只重配那一句；读音纠正词典；AI 识别情绪和错别字

## 怎么开始

- **没有 AI 助手**：看 [`安装说明.txt`](安装说明.txt)，按 8 步自己装。
- **有 AI 助手**（Claude Code、Cowork 等）：把整个仓库交给它，说“帮我装好这个工作台”，它会按 [`SKILL.md`](SKILL.md) 带你装。

工作台本体在 [`source/`](source/)，原理和改风格的说明在 [`references/`](references/)。

## 需要准备

Windows 10/11、NVIDIA 显卡（4GB 显存起）、16GB 内存起、Python 3.10+、Docker Desktop + HeyGem（现名 Duix.Avatar，GitHub 搜 Duix-Avatar；对口型，需自行安装，本仓库不包含）、阿里云百炼 API Key。大模型接口可选。

## 使用须知

1. 只克隆你本人，或已明确授权给你的人的形象和声音。
2. 发布到抖音等平台时，按平台要求声明“内容由 AI 生成”。
3. API Key 只放在 `source/config.local.json`（已在 `.gitignore` 里），不要提交到仓库。

## 第三方说明

- 字体：思源黑体（Source Han Sans）、Noto Serif CJK 子集（改名 LawSerif）、Inter、JetBrains Mono，均为 SIL Open Font License，见 `source/engine/fonts/licenses/`。
- 人脸检测模型：OpenCV Zoo 的 YuNet（`face_detection_yunet_2023mar.onnx`）。
- HeyGem / Duix.Avatar 是独立的开源项目，请按其仓库的许可和说明使用。
