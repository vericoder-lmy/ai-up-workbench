"""进度上报：写 work/progress.json，供网页界面/批处理读取。渲染阶段各分段另写 prog_chunk_*.txt。"""
import glob
import json
import os
import time

# 各阶段在总进度中的区间（百分比）
STAGES = {
    "准备": (0, 3), "参考分析": (3, 5), "识别字幕": (5, 16), "字幕对齐": (16, 24), "剪辑编排": (24, 27),
    "渲染画面": (27, 92), "混音合成": (92, 95), "成片质检": (95, 100), "完成": (100, 100),
}


def report(work, stage, frac=0.0, **extra):
    if not work:
        return
    a, b = STAGES.get(stage, (0, 100))
    d = {"stage": stage, "pct": round(a + (b - a) * max(0.0, min(1.0, frac)), 1), "t": time.time()}
    d.update(extra)
    tmp = os.path.join(work, "progress.json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(work, "progress.json"))
    except OSError:
        pass


def read(work):
    """读取进度（渲染阶段把各分段帧数汇总进来）。"""
    p = os.path.join(work, "progress.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    if d.get("stage") == "渲染画面" and d.get("frames_total"):
        done = 0
        for fp in glob.glob(os.path.join(work, "prog_chunk_*.txt")):
            try:
                done += int(open(fp).read().strip() or 0)
            except Exception:
                pass
        a, b = STAGES["渲染画面"]
        f = min(1.0, done / d["frames_total"])
        d["pct"] = round(a + (b - a) * f, 1)
        d["frames_done"] = done
        el = time.time() - d.get("render_start", d["t"])
        if done > 30 and el > 1:
            d["fps"] = round(done / el, 1)
            d["eta"] = round((d["frames_total"] - done) / max(0.1, d["fps"]) + 20, 0)
    return d
