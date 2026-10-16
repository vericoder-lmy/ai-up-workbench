"""AI_UP 封面生成：python cover.py --video 干净口播.mp4 --timeline work/timeline.json --style styles/x.json
   --portrait 封面_竖屏.jpg --landscape 封面_横屏.jpg"""
import argparse
import json
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--portrait")
    ap.add_argument("--landscape")
    ap.add_argument("--base-dir", help="缓存封面底图的目录（工程目录）")
    ap.add_argument("--settings", help="标题设置.json（手动调节的封面参数在 cover 字段）")
    a = ap.parse_args()
    from aiclip.cover import make_covers
    tl = json.load(open(a.timeline, encoding="utf-8"))
    st = json.load(open(a.style, encoding="utf-8"))
    outs = {k: v for k, v in (("portrait", a.portrait), ("landscape", a.landscape)) if v}
    custom = None
    if a.settings and os.path.exists(a.settings):
        custom = (json.load(open(a.settings, encoding="utf-8")) or {}).get("cover") or None
    done = make_covers(a.video, tl, st, outs, os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx"),
                       base_dir=a.base_dir, custom=custom)
    print(json.dumps(done, ensure_ascii=False))


if __name__ == "__main__":
    main()
