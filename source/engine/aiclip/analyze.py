"""参考视频/截图分析：提取可量化的风格参数，覆盖风格模板中的对应字段。

能测出的：
  - 镜头/版式切换节奏（帧差事件 → 平均切换间隔）
  - 常驻包装层位置（时间稳定 + 边缘密集的区域 → 顶部标题栏高度、底部免责声明位置）
  - 标题栏主色（K-means 取最饱和色）
  - 字幕所在高度（白色高对比、随时间变化的文字行）
  - 截图中的圆形画中画（Hough 圆检测 → 尺寸/位置）
  - 黑场转场（亮度骤降）
配置视觉大模型后（llm.analyze_images）额外获得标题文字、章节名、特效类型等语义信息。
参考视频常是手机录屏（带抖音界面），这里按“高度对齐、水平居中裁切”映射到 9:16 画布。
"""
import os

import cv2
import numpy as np

from . import llm
from .utils import ffmpeg_bin, imread, log, probe, save_json


def _read_frames(path, fps=4.0, max_frames=240, width=270):
    import subprocess
    info = probe(path)
    h = int(round(info["h"] * width / info["w"] / 2)) * 2
    p = subprocess.Popen([ffmpeg_bin(), "-v", "error", "-i", path, "-an", "-vf", f"fps={fps},scale={width}:{h}",
                          "-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    raw = p.stdout.read()
    p.wait()
    n = len(raw) // (width * h * 3)
    frames = [np.frombuffer(raw[i * width * h * 3:(i + 1) * width * h * 3], np.uint8).reshape(h, width, 3) for i in range(n)]
    times = [i / fps for i in range(n)]
    return frames, times, info["duration"]


def _to_canvas_y(fy, aspect_src, canvas_aspect=9 / 16):
    # 录屏(更窄)按高度对齐：y 比例不变
    return fy


def analyze_reference(video=None, images=(), out_dir=None, canvas=(1080, 1920)):
    W, H = canvas
    rep = {"source": {"video": video, "images": list(images)}}
    over = {}
    frames, times = [], []
    if video and os.path.exists(video):
        frames, times, dur = _read_frames(video)
        rep["duration"] = dur
    imgs = [imread(p) for p in images if os.path.exists(p)]
    imgs = [cv2.resize(im, (270, int(im.shape[0] * 270 / im.shape[1]))) for im in imgs if im is not None]

    if len(frames) >= 6:
        h = frames[0].shape[0]
        g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
        # 1) 常驻层：时间方差小且边缘多
        sd = g.std(0)
        edges = np.mean([cv2.Canny(f.astype(np.uint8), 60, 160) > 0 for f in g[:: max(1, len(g) // 20)]], 0)
        static_edge = (sd < 10) & (edges > 0.5)
        prof = static_edge.mean(1)
        top_rows = np.where(prof[: int(h * 0.35)] > 0.04)[0]
        if len(top_rows):
            hb = (top_rows.max() + 1) / h
            rep["header_bottom_frac"] = round(float(hb), 3)
        low = np.where(prof[int(h * 0.6): int(h * 0.88)] > 0.05)[0]
        if len(low):
            rep["static_low_band_frac"] = [round(float((low.min() + int(h * 0.6)) / h), 3),
                                           round(float((low.max() + int(h * 0.6)) / h), 3)]
        # 2) 切换节奏（只看中部区域，避开常驻层）
        mid = g[:, int(h * 0.2): int(h * 0.65)]
        d = np.abs(np.diff(mid, axis=0)).mean((1, 2))
        thr = max(8.0, np.median(d) * 3.5)
        ev = [times[i + 1] for i in np.where(d > thr)[0]]
        merged = []
        for t in ev:
            if not merged or t - merged[-1] > 0.6:
                merged.append(t)
        rep["layout_change_times"] = [round(t, 2) for t in merged]
        if len(merged) >= 2:
            gaps = np.diff([0.0] + merged)
            rep["avg_block_sec"] = round(float(np.mean(gaps)), 2)
        # 3) 黑场
        lum = g.mean((1, 2))
        dips = [times[i] for i in range(1, len(lum) - 1) if lum[i] < 0.55 * np.median(lum) and lum[i] < lum[i - 1]]
        rep["dip_to_black_times"] = [round(t, 2) for t in dips]
        # 4) 标题栏主色
        hb = rep.get("header_bottom_frac", 0.15)
        px = np.concatenate([f[: int(h * hb)].reshape(-1, 3) for f in frames[:: max(1, len(frames) // 10)]]).astype(np.float32)
        px = px[np.random.RandomState(0).choice(len(px), min(len(px), 20000), replace=False)]
        _, lab, cen = cv2.kmeans(px, 6, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0), 3,
                                 cv2.KMEANS_PP_CENTERS)
        hsv = cv2.cvtColor(cen.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)[0]
        k = int(np.argmax(hsv[:, 1].astype(int) * (hsv[:, 2] > 60)))
        b, gg, r = cen[k]
        rep["header_main_color"] = "#%02x%02x%02x" % (int(r), int(gg), int(b))
        # 5) 字幕行：中下部白色像素多、且随时间变化
        white = np.stack([((f.min(2) > 215)) for f in frames]).astype(np.float32)
        wv = white.std(0).mean(1)
        lo, hi = int(h * 0.45), int(h * 0.8)
        y = lo + int(np.argmax(cv2.GaussianBlur(wv[lo:hi].reshape(-1, 1), (1, 9), 0).ravel()))
        rep["subtitle_y_frac"] = round(float(y / h), 3)

    # 6) 截图中的圆形画中画
    circles = []
    for im in imgs:
        gr = cv2.medianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 5)
        c = cv2.HoughCircles(gr, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80, param1=120, param2=45,
                             minRadius=int(im.shape[1] * 0.18), maxRadius=int(im.shape[1] * 0.45))
        if c is not None:
            x, yv, r = c[0][0]
            circles.append((x / im.shape[1], yv / im.shape[0], r / im.shape[1]))
    if circles:
        cx, cy, r = np.median(np.array(circles), 0)
        rep["pip_circle"] = {"cx_frac": round(float(cx), 3), "cy_frac": round(float(cy), 3), "r_frac_w": round(float(r), 3)}

    # 7) 视觉大模型（可选）
    if images:
        v = llm.analyze_images(list(images))
        if v:
            rep["vlm"] = v

    # ---- 转换为风格覆盖参数 ----
    if "avg_block_sec" in rep:
        b = float(np.clip(rep["avg_block_sec"], 2.5, 12))
        over.setdefault("pacing", {})["block_sec"] = {
            "full": [max(3.0, b * 1.2), max(5.0, b * 2.2)], "pip_round": [max(3.5, b * 1.4), max(6.0, b * 2.5)],
            "vignette": [max(3.0, b * 1.2), max(5.0, b * 2.0)], "pip_circle": [2.5, 6.5]}
    if "subtitle_y_frac" in rep and 0.55 < rep["subtitle_y_frac"] < 0.8:
        over.setdefault("subtitle", {})["y"] = int(rep["subtitle_y_frac"] * H)
    if "pip_circle" in rep:
        pc = rep["pip_circle"]
        # 截图为窄屏录屏：x 按居中裁切映射
        src_aspect = imgs[0].shape[1] / imgs[0].shape[0] if imgs else 9 / 16
        vis_w = H * src_aspect
        off = (W - vis_w) / 2
        r = pc["r_frac_w"] * vis_w
        cx = off + pc["cx_frac"] * vis_w
        cy = pc["cy_frac"] * H
        d = int(round(r * 2 * 1.04))
        over.setdefault("modes", {})["pip_circle"] = {"w": d, "h": d, "radius": d // 2,
                                                      "x": int(cx - d / 2), "y": int(cy - d / 2)}
    if "header_main_color" in rep:
        rep["note_header_color"] = "标题主色已测出，可在 header.title_grad 中替换"
    v = rep.get("vlm") or {}
    if v.get("disclaimer_text"):
        over.setdefault("brand", {})["disclaimer"] = [s for s in str(v["disclaimer_text"]).replace("，", " ").split("\n") if s][:2]
    if v.get("title_color"):
        over.setdefault("header", {})["title_grad"] = ["#e8f6fb", v["title_color"], v["title_color"]]
    rep["style_overrides"] = over
    if out_dir:
        save_json(rep, os.path.join(out_dir, "reference_analysis.json"))
    log("参考分析:", {k: rep[k] for k in rep if k not in ("source", "layout_change_times", "style_overrides", "vlm")})
    return rep, over
