# 风格包（engine/styles/*.json）

一个风格就是一个 JSON 文件，放进 `engine/styles/` 后网页“风格包”里就会出现（风格卡标题用 `display_name`，颜色点用 `accent`，预览图放 `engine/styles/previews/<文件名>.jpg`，960×540）。
新建风格最快的办法：复制一个最接近的（`aiup_orange.json` / `aiup_dark_tech.json` / `law_gold.json`），改名、改 `name` 和 `display_name`，再改下面这些字段。

## 常改的字段

| 字段 | 作用 |
|---|---|
| `fonts.<角色>` | 字体候选列表（按顺序找第一个存在的）。角色：`heavy` 大字、`sub` 字幕、`label` 小标、`num` 数字、`serif`/`serif_black` 宋体（律师风格）。字体文件放 `engine/fonts/` 或系统字体目录 |
| `subtitle` | 字幕：`y` 纵向位置（竖屏 1920 高的像素）、`size`、`color`、`stroke`、`shadow`、`en.show`（中英双语） |
| `hud` | 信息层（卡片大字）：`enable`、`accent`（强调色）、`white`、`dim`、`scrim`（压暗程度）、`dur`（每块时长范围）、`plus`（大字移位 / 整屏卡片版式） |
| `pacing.zoom_mode` | `key`：只在重点句推近（`zoom_key` 倍数、`zoom_gap` 最小间隔秒）；其他值：逐句按 `zoom_levels` 轮换 |
| `pacing.cut_zoom` | 长素材模式下数字人切换点的景别序列，如 `[1.0, 1.12]` |
| `media_pip` | 画中画素材小窗：`enable`、位置尺寸 `x/y/w/h`、`radius`、`border_color`、`min_gap`、`dur` |
| `grade` | 调色：`contrast`、`saturation`、`vignette`、`grain`（颗粒）、`shadow_tint` |
| `encode.crf` | 成片画质（越小越清晰、文件越大；17 左右够用） |
| `cover` | 封面：`image`（统一封面照路径，相对 engine/）、`dark`（压暗）、`max_chars`（封面大字每行字数）、`port_size`（用照片时竖屏封面尺寸，如 `[1080,1440]`） |
| `landscape` | 横屏专用的覆盖值（结构同上，只写要改的字段） |

## 律师风格（`law`，`law.enable=true` 时生效）

| 字段 | 作用 |
|---|---|
| `law.disclaimer` / `law.show_disclaimer` | 底部免责声明文字 / 是否显示 |
| `law.gold_grad` / `law.gold_grad2` | 顶部标题两行的金色渐变 |
| `law.kw_color` | 每句重点词的红色 |
| `law.port` / `law.land` | 竖屏 / 横屏的位置和字号：`title_y`、`t1_size`、`t2_size`、`name_y`、`sub_y`、`sub_size`、`gold_size`、`red_size`、`disc_y`、`scale` |

律师姓名、职称、律所在网页里填（选律政金风格时出现输入框），存在任务里，不写进风格文件。

## 改完怎么看效果

不用重新配音和对口型：在“剪辑任务”里选一条已完成的任务点“重新剪辑”。
只想快速看前几秒，可以命令行跑引擎：

```
cd engine
python run.py --video <对口型视频> --srt <字幕.srt> --style styles/<风格>.json --orientation portrait --align none --title "第一段 第二段" --out test.mp4 --preview 10
```
