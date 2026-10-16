"""AICLIP 自动剪辑 —— 命令行入口

示例：
  python run.py --video 素材.mp4 --srt 字幕.srt --ref 参考.mp4 --ref-images 截图1.jpg 截图2.jpg --out 成片.mp4
  python run.py ... --preview 20          # 只渲染前 20 秒快速预览
  python run.py ... --plan-only           # 只生成 timeline.json（可手改）
  python run.py ... --from-timeline work/timeline.json   # 按改过的时间线重渲染
"""
import argparse
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="AICLIP：参考风格复刻的口播自动剪辑")
    ap.add_argument("--video", required=True, help="待剪辑的口播视频")
    ap.add_argument("--srt", help="字幕文件（可选；不给则用本地语音识别自动生成。时间不准也没关系，会自动对齐）")
    ap.add_argument("--subtitle-only", action="store_true", help="只生成字幕，不剪辑")
    ap.add_argument("--ref", help="参考视频")
    ap.add_argument("--ref-images", nargs="*", help="参考截图")
    ap.add_argument("--style", nargs="+", default=[os.path.join(HERE, "styles", "aiup_orange.json")],
                    help="风格模板 JSON；给多个就在一条视频里按章节轮换混用（第一个为主风格）")
    ap.add_argument("--orientation", default="auto", choices=["auto", "portrait", "landscape"],
                    help="输出画幅：auto=跟随原片（默认）/ portrait=竖屏1080x1920 / landscape=横屏1920x1080")
    ap.add_argument("--brand", help="品牌信息 JSON（title/slogan/namecard_lines/disclaimer 等）")
    ap.add_argument("--title", help="顶部大标题（默认：从字幕识别“我是XX医生”）")
    ap.add_argument("--slogan", help="副标题")
    ap.add_argument("--broll", help="B-roll 素材目录（文件名=关键词，如 脂肪肝.jpg）")
    ap.add_argument("--bgm", help="背景音乐文件")
    ap.add_argument("--no-sfx", action="store_true", help="关闭转场/弹字音效")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--work", default=None, help="中间文件目录（默认 <out>_work）")
    ap.add_argument("--align", default="auto", choices=["auto", "dtw", "asr", "none"], help="字幕对齐方式")
    ap.add_argument("--encoder", default="auto", choices=["auto", "nvenc", "x264"])
    ap.add_argument("--workers", type=int, default=0, help="并行渲染进程数（默认 CPU核数-1，最多6）")
    ap.add_argument("--preview", type=float, default=0, help="只渲染前 N 秒")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--from-timeline", help="使用已有 timeline.json 渲染")
    ap.add_argument("--keep-work", action="store_true", help="保留分段视频等中间文件（默认成功后清理）")
    ap.add_argument("--no-ref-override", action="store_true", help="不使用参考分析结果覆盖模板参数")
    a = ap.parse_args()
    a.work = a.work or (os.path.splitext(a.out)[0] + "_work")
    from aiclip.pipeline import JobError, run_pipeline
    try:
        run_pipeline(a)
    except JobError as e:
        print("[输入错误]", e)
        sys.exit(2)


if __name__ == "__main__":
    mp.freeze_support()
    main()
