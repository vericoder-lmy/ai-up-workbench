"""对标视频 → 风格库：分析一条对标（参考）视频，自动生成一份可直接用于剪辑的风格模板。

原理：系统的画面由一组参数化组件构成（标题栏/进度条/四种版式/字幕样式/弹字/卡片/节奏/剪切力度）。
本模块从对标视频中测量能量化的特征，选择最接近的基础模板，再把配色、版式组合、节奏等参数改成对标视频的样子。
这是“用现有组件近似复刻”，对标视频里系统没有的特效（如分屏、贴纸动画）不会凭空出现；
生成后可在网页里微调（主色、字体、版式开关、节奏），或让开发者新增组件。
"""
import colorsys
import copy
import glob
import json
import os
import re
import shutil
import time

import cv2
import numpy as np

from . import llm
from .analyze import _read_frames
from .utils import imread, log, save_json, imwrite

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLES = os.path.join(HERE, "styles")
BASES = os.path.join(STYLES, "_base")  # AI_UP：生成新风格用的底稿模板，不在风格列表里显示


# ---------------- 颜色工具 ----------------
def _hex(bgr):
    b, g, r = [int(max(0, min(255, v))) for v in bgr]
    return "#%02x%02x%02x" % (r, g, b)


def _shade(hex_color, light):
    """light>0 变亮（向白），<0 变暗（向黑），范围 -1~1。"""
    h = hex_color.lstrip("#")
    r, g, b = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    if light >= 0:
        r, g, b = [c + (255 - c) * light for c in (r, g, b)]
    else:
        r, g, b = [c * (1 + light) for c in (r, g, b)]
    return "#%02x%02x%02x" % (int(r), int(g), int(b))


def _sat(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return colorsys.rgb_to_hsv(r, g, b)


# ---------------- 特征测量 ----------------
def measure(video, images=()):
    frames, times, dur = _read_frames(video, fps=4.0, max_frames=360)
    if len(frames) < 8:
        raise ValueError("对标视频太短或无法读取（至少需要 2 秒）")
    h, w = frames[0].shape[:2]
    F = {"duration": round(dur, 1), "frames": len(frames), "aspect": round(w / h, 3)}
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames])

    # 1) 常驻包装层（时间稳定 + 边缘密集）
    sd = gray.std(0)
    edges = np.mean([cv2.Canny(f.astype(np.uint8), 60, 160) > 0 for f in gray[:: max(1, len(gray) // 30)]], 0)
    static_edge = (sd < 12) & (edges > 0.4)
    prof = static_edge.mean(1)
    mid_bg = float(prof[int(h * 0.35): int(h * 0.6)].mean())  # 固定机位时背景本身也是静止的，用中部作基准
    top = np.where(prof[: int(h * 0.35)] > max(0.035, mid_bg * 2.5))[0]
    hb = (top.max() + 1) / h if len(top) else 0.0
    F["header_frac"] = round(float(hb), 3)
    F["header_density"] = round(float(prof[: max(1, int(h * hb))].mean()) if hb else 0.0, 3)
    F["static_bg"] = round(mid_bg, 3)
    F["has_header"] = bool(hb >= 0.07 and F["header_density"] > 0.03 and F["header_density"] > mid_bg * 3)
    low = np.where(prof[int(h * 0.62): int(h * 0.9)] > 0.04)[0]
    F["has_footer_text"] = bool(len(low) >= 3)

    # 2) 明暗主题
    hh = max(4, int(h * max(hb, 0.12)))
    F["header_luma"] = round(float(gray[:, :hh].mean()), 1)
    F["luma"] = round(float(np.median(gray.mean((1, 2)))), 1)
    F["dark_header"] = bool(F["has_header"] and F["header_luma"] < 95)
    F["dark_theme"] = bool(F["luma"] < 85)

    # 3) 标题栏主色：常驻区域里的高饱和像素
    title_color = None
    if F["has_header"]:
        m = static_edge[:hh]
        cand = []
        for k in range(0, len(frames), max(1, len(frames) // 12)):
            px = frames[k][:hh][cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0]
            hs = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            cand.append(px[(hs[:, 1] > 70) & (hs[:, 2] > 70)])
        cand = np.concatenate(cand) if cand else np.zeros((0, 3))
        if len(cand) > 80:
            _, lab, cen = cv2.kmeans(cand.astype(np.float32), 3, None,
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2, cv2.KMEANS_PP_CENTERS)
            cnt = np.bincount(lab.ravel(), minlength=3)
            title_color = _hex(cen[int(np.argmax(cnt))])
    F["title_color"] = title_color

    # 4) 字幕：中下部随时间变化的亮色文字行
    bright = (hsv[..., 2] > 200)
    var_rows = bright.astype(np.float32).std(0).mean(1)
    lo, hi = int(h * 0.45), int(h * 0.85)
    sm = cv2.GaussianBlur(var_rows[lo:hi].reshape(-1, 1), (1, 9), 0).ravel()
    y = lo + int(np.argmax(sm))
    F["subtitle_y_frac"] = round(y / h, 3)
    band = slice(max(0, y - int(h * 0.025)), min(h, y + int(h * 0.025)))
    bpx = hsv[:, band][bright[:, band]]
    yellow = ((bpx[:, 0] >= 15) & (bpx[:, 0] <= 38) & (bpx[:, 1] > 90)).mean() if len(bpx) else 0
    F["subtitle_color"] = "yellow" if yellow > 0.35 else "white"
    rows_on = np.where(sm > sm.max() * 0.5)[0]
    F["subtitle_h_frac"] = round(float(len(rows_on) / h), 3)

    # 5) 画中画：同一帧里清晰区域 + 大片模糊区域；圆形画中画：位置固定、边缘完整的圆
    mixed = 0
    y0, y1 = int(h * max(hb, 0.12)), int(h * 0.8)
    found = []
    samples = 0
    for k in range(len(frames)):
        g = gray[k, y0:y1]
        lap = cv2.Laplacian(g, cv2.CV_32F)
        ch, cw = g.shape[0] // 8, g.shape[1] // 5
        vs = np.array([lap[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].var() for i in range(8) for j in range(5)])
        if (vs > 300).mean() > 0.18 and (vs < 25).mean() > 0.25:
            mixed += 1
        if k % 3 == 0:
            samples += 1
            gi = cv2.medianBlur(gray[k].astype(np.uint8), 5)
            ed = cv2.dilate(cv2.Canny(gi, 50, 150), np.ones((3, 3), np.uint8))
            c = cv2.HoughCircles(gi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80, param1=120, param2=40,
                                 minRadius=int(w * 0.16), maxRadius=int(w * 0.45))
            for x, yy, r in (c[0] if c is not None else []):
                ang = np.linspace(0, 2 * np.pi, 90, endpoint=False)
                px = (x + r * np.cos(ang)).astype(int)
                py = (yy + r * np.sin(ang)).astype(int)
                ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                cover = ed[py[ok], px[ok]].astype(bool).mean() if ok.sum() > 30 else 0
                if cover > 0.55 and ok.mean() > 0.6:  # 边框连续（画出来的圆），而不是头部等自然轮廓
                    found.append((x, yy, r))
    best = 0
    for a in found:  # 同一位置反复出现才算画中画
        best = max(best, sum(1 for b in found if abs(a[0] - b[0]) < 8 and abs(a[1] - b[1]) < 8 and abs(a[2] - b[2]) < 8))
    F["pip_ratio"] = round(mixed / len(frames), 3)
    F["circle_ratio"] = round(best / max(1, samples), 3) if best >= 2 else 0.0
    for im in images:
        img = imread(im) if isinstance(im, str) else im
        if img is None:
            continue
        g = cv2.medianBlur(cv2.cvtColor(cv2.resize(img, (270, int(img.shape[0] * 270 / img.shape[1]))), cv2.COLOR_BGR2GRAY), 5)
        c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80, param1=120, param2=45, minRadius=45, maxRadius=125)
        if c is not None:
            F["circle_ratio"] = max(F["circle_ratio"], 0.2)
            F["pip_ratio"] = max(F["pip_ratio"], 0.15)

    # 6) 关键词大字：画面中部突然出现的大块高饱和文字
    mid = slice(int(h * 0.35), int(h * 0.75))
    satm = (hsv[:, mid, :, 1] > 150) & (hsv[:, mid, :, 2] > 140)
    frac = satm.mean((1, 2))
    base = np.median(frac)
    on = frac > base + 0.012
    events, prev = 0, False
    for v in on:
        if v and not prev:
            events += 1
        prev = v
    F["pop_per_min"] = round(events / max(0.2, dur / 60), 1)
    pop_color = None
    if on.any():
        px = hsv[:, mid][on][satm[on]]
        if len(px) > 50:
            hue = int(np.median(px[:, 0]))
            bgr = cv2.cvtColor(np.uint8([[[hue, 220, 235]]]), cv2.COLOR_HSV2BGR)[0, 0]
            pop_color = _hex(bgr)
    F["pop_color"] = pop_color

    # 7) 节奏：中部画面突变（版式切换/跳切）
    midg = gray[:, int(h * 0.2): int(h * 0.7)]
    d = np.abs(np.diff(midg, axis=0)).mean((1, 2))
    thr = max(6.0, float(np.median(d)) * 2.5)
    ev = [times[i + 1] for i in np.where(d > thr)[0]]
    merged = []
    for t in ev:
        if not merged or t - merged[-1] > 0.6:
            merged.append(t)
    F["cuts_per_min"] = round(len(merged) / max(0.2, dur / 60), 1)
    F["avg_block_sec"] = round(float(np.mean(np.diff([0.0] + merged))) if len(merged) >= 2 else dur, 2)
    lum = gray.mean((1, 2))
    F["dips"] = int(sum(1 for i in range(1, len(lum) - 1) if lum[i] < 0.55 * np.median(lum) and lum[i] < lum[i - 1]))

    # 8) 说话节奏（停顿长短 → 剪切力度）
    try:
        from .audio import load_audio, vad
        x = load_audio(video, 16000, 1)
        _, segs, _, _ = vad(x, 16000)
        gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        F["pause_median"] = round(float(np.median(gaps)), 2) if gaps else None
    except Exception:
        F["pause_median"] = None

    # 代表帧（做预览图）：选字幕区最“热闹”的一帧
    k = int(np.argmax(bright[:, band].mean((1, 2)) + static_edge.mean() * 0)) if len(frames) else 0
    F["_preview_frame"] = frames[min(len(frames) - 1, max(0, k))]
    return F


# ---------------- 生成风格 ----------------
def _slug(name):
    s = re.sub(r"[^\w一-鿿]+", "_", name).strip("_")
    return ("ref_" + s)[:40] if s else "ref_" + time.strftime("%m%d%H%M%S")


def _build_ai_tech(F, name, vlm):
    """暗调、无标题栏的知识/科技博主：用 motion 风格做底（全屏信息卡 + 巨字 + 极简字幕），只按测量结果调节奏。"""
    with open(os.path.join(BASES, "ai_tech.json"), encoding="utf-8") as f:
        st = json.load(f)
    notes = ["暗调电影感", "全屏动态信息卡", "巨字动效", "极简宽字距字幕"]
    pc = st["pacing"]
    b = float(np.clip(F["avg_block_sec"], 1.5, 10.0))
    pc["block_sec"]["card"] = [float(np.clip(b * 1.2, 2.0, 3.2)), float(np.clip(b * 2.0, 3.0, 4.5))]
    pc["block_sec"]["full"] = [float(np.clip(b * 2.0, 3.0, 6.0)), float(np.clip(b * 4.0, 5.5, 9.0))]
    notes.append(f"平均 {F['avg_block_sec']}s 一次切换")
    kp = st["keyword_pop"]
    kp["max_per_min"] = int(np.clip(round(F["pop_per_min"]), 1, 5)) if F["pop_per_min"] >= 1 else 2
    if F["subtitle_color"] == "yellow":
        st["subtitle"]["color"] = "#ffe14d"
        notes.append("黄色字幕")
    if F.get("pause_median") is not None and F["pause_median"] < 0.3:
        st["cut"].update({"keep_gap": 0.1, "min_gap": 0.2})
        notes.append("紧凑剪切")
    if F.get("aspect", 0.56) > 1.05:
        notes.append("横屏对标")
    st.update({"name": None, "display_name": name, "generated": True, "base": "ai_tech",
               "orientation_hint": "landscape" if F.get("aspect", 0.56) > 1.05 else "portrait",
               "description": "对标生成：" + " · ".join(notes),
               "analysis": {k: v for k, v in F.items() if not k.startswith("_")},
               "created": time.strftime("%Y-%m-%d %H:%M")})
    return st


def build_style(F, name, vlm=None):
    vlm = vlm or {}
    if not F["has_header"] and F["dark_theme"]:
        return _build_ai_tech(F, name, vlm)
    if not F["has_header"]:
        base_id = "minimal_fast"
    elif F["dark_header"] or F["dark_theme"]:
        base_id = "tech_blue"
    else:
        base_id = "doctor_kepu"
    with open(os.path.join(BASES, base_id + ".json"), encoding="utf-8") as f:
        st = json.load(f)
    st = copy.deepcopy(st)
    notes = []
    # 配色
    accent = None
    for c in (vlm.get("title_color"), F.get("title_color"), F.get("pop_color")):
        if c and re.match(r"^#[0-9a-fA-F]{6}$", str(c)) and _sat(c)[1] > 0.25:
            accent = c.lower()
            break
    accent = accent or st.get("accent", "#22d3ee")
    pop = (vlm.get("accent_color") if vlm.get("accent_color") else None) or F.get("pop_color") or accent
    st["accent"] = accent
    hd, pg = st["header"], st["progress"]
    hd["show"] = F["has_header"]
    pg["show"] = F["has_header"] and F["header_frac"] > 0.1
    if F["has_header"]:
        hd["title_grad"] = [_shade(accent, 0.65), accent, _shade(accent, -0.35)]
        hd["title_stroke_color"] = _shade(accent, -0.8)
        pg["fill_color"] = accent
        pg["node_active_ring"] = accent
        pg["label_active_color"] = pop if pop != accent else _shade(accent, -0.2)
        notes.append("标题栏" + ("（深色）" if F["dark_header"] else "（浅色）") + (" + 章节进度条" if pg["show"] else ""))
    else:
        notes.append("无标题栏")
    # 版式
    modes = st["modes"]
    if "pip_round" not in modes:  # 极简模板也带全套版式定义（复制自医生模板）
        with open(os.path.join(BASES, "doctor_kepu.json"), encoding="utf-8") as f:
            modes.update(json.load(f)["modes"])
    pc = st["pacing"]
    if F["pip_ratio"] > 0.12:
        pc["cycle"] = ["full", "pip_round", "full", "vignette"] if F["dark_theme"] or F["pip_ratio"] > 0.3 else ["full", "pip_round"]
        pc["chapter_mode"] = "pip_circle" if F["circle_ratio"] > 0.05 else "pip_round"
        notes.append(f"画中画 {int(F['pip_ratio'] * 100)}%" + ("（含圆形）" if F["circle_ratio"] > 0.05 else ""))
    else:
        pc["cycle"] = ["full"]
        pc["chapter_mode"] = "full"
        notes.append("全程全屏")
    for m in ("pip_round", "pip_circle"):
        modes[m]["border_color"] = _shade(accent, -0.6) if not F["dark_theme"] else accent
    # 节奏
    b = float(np.clip(F["avg_block_sec"], 2.0, 12.0))
    pc["block_sec"] = {"full": [max(3.0, b * 1.2), max(5.0, b * 2.2)], "pip_round": [max(3.5, b * 1.4), max(6.0, b * 2.5)],
                       "vignette": [max(3.0, b * 1.2), max(5.0, b * 2.0)], "pip_circle": [2.5, 6.5]}
    if F["cuts_per_min"] > 18:
        pc["zoom_levels"] = [1.0, 1.15, 1.05, 1.22]
    notes.append(f"平均 {F['avg_block_sec']}s 一次切换")
    # 弹字
    kp = st["keyword_pop"]
    if F["pop_per_min"] >= 1:
        kp["color"] = pop
        kp["max_per_min"] = int(np.clip(round(F["pop_per_min"]), 2, 8))
        kp["min_gap"] = float(np.clip(60 / max(1, F["pop_per_min"]) * 0.8, 6, 20))
        notes.append(f"关键词弹字 约{F['pop_per_min']}次/分")
    else:
        kp["max_per_min"] = 0
        notes.append("无关键词弹字")
    # 字幕
    sc = st["subtitle"]
    sc["color"] = "#ffe14d" if F["subtitle_color"] == "yellow" else "#ffffff"
    if 0.55 < F["subtitle_y_frac"] < 0.85:
        sc["y"] = int(F["subtitle_y_frac"] * st["canvas"]["h"])
    if 0.02 < F["subtitle_h_frac"] < 0.07:
        sc["size"] = int(np.clip(F["subtitle_h_frac"] * st["canvas"]["h"] * 0.95, 64, 100))
    if str(vlm.get("subtitle_style", "")).lower() in ("pill", "box"):
        st["modes"]["full"]["subtitle"] = "pill"
    notes.append("黄色字幕" if F["subtitle_color"] == "yellow" else "白色字幕")
    # 剪切力度
    if F.get("pause_median") is not None and F["pause_median"] < 0.3:
        st["cut"].update({"keep_gap": 0.1, "min_gap": 0.2})
        notes.append("紧凑剪切")
    st["disclaimer"]["show"] = st["disclaimer"].get("show", True)
    if F["dark_theme"]:
        st["broll"]["card_colors"] = [[_shade(accent, -0.92), _shade(accent, -0.6)], ["#05070b", _shade(accent, -0.7)]]
    if F.get("aspect", 0.56) > 1.05:
        notes.append("横屏对标")
    st.update({"name": None, "display_name": name, "generated": True, "base": base_id,
               "orientation_hint": "landscape" if F.get("aspect", 0.56) > 1.05 else "portrait",
               "description": "对标生成：" + " · ".join(notes),
               "analysis": {k: v for k, v in F.items() if not k.startswith("_")},
               "created": time.strftime("%Y-%m-%d %H:%M")})
    return st


def generate(video, name, images=(), progress=None):
    """生成并保存风格。返回 (style_id, style dict)。"""
    def step(msg, frac):
        if progress:
            progress(msg, frac)
        log(f"[对标生成] {msg}")
    step("读取对标视频", 0.05)
    F = measure(video, images)
    step("分析配色与版式", 0.6)
    vlm = None
    if llm.enabled(vision=True):
        step("视觉大模型识别标题/特效", 0.7)
        tmp = []
        for i, k in enumerate((0.2, 0.5, 0.8)):
            p = os.path.join(STYLES, f"_vlm_{i}.jpg")
            imwrite(p, F["_preview_frame"])
            tmp.append(p)
        vlm = llm.analyze_images(list(images) + tmp)
        for p in tmp:
            os.remove(p)
    st = build_style(F, name, vlm)
    sid = _slug(name)
    k = 2
    while os.path.exists(os.path.join(STYLES, sid + ".json")):
        sid = f"{_slug(name)}_{k}"
        k += 1
    st["name"] = sid
    step("保存风格与对标素材", 0.9)
    ref_dir = os.path.join(HERE, "references", sid)
    os.makedirs(ref_dir, exist_ok=True)
    shutil.copy(video, os.path.join(ref_dir, "reference" + os.path.splitext(video)[1].lower()))
    for i, im in enumerate(images):
        shutil.copy(im, os.path.join(ref_dir, f"ref_{i + 1}{os.path.splitext(im)[1].lower()}"))
    os.makedirs(os.path.join(STYLES, "previews"), exist_ok=True)
    fr = F["_preview_frame"]
    th, tw = fr.shape[:2]
    tgt = 9 / 16
    if tw / th > tgt:  # 横屏 → 中间裁成竖屏
        nw = int(th * tgt)
        fr = fr[:, (tw - nw) // 2:(tw - nw) // 2 + nw]
    imwrite(os.path.join(STYLES, "previews", sid + ".jpg"), cv2.resize(fr, (360, 640)), [cv2.IMWRITE_JPEG_QUALITY, 88])
    st["preview_source"] = "reference"
    save_json(st, os.path.join(STYLES, sid + ".json"))
    step("完成", 1.0)
    return sid, st


# ---------------- 微调 ----------------
TWEAKABLE = ("display_name", "accent", "title_font", "subtitle_color", "show_header", "show_progress", "use_pip",
             "use_circle", "pace", "pops", "namecard")


def tweak(sid, t):
    p = os.path.join(STYLES, sid + ".json")
    with open(p, encoding="utf-8") as f:
        st = json.load(f)
    if t.get("display_name"):
        st["display_name"] = str(t["display_name"])[:30]
    if st.get("motion"):
        return _tweak_motion(sid, p, st, t)
    if t.get("accent") and re.match(r"^#[0-9a-fA-F]{6}$", t["accent"]):
        a = t["accent"].lower()
        st["accent"] = a
        st["header"]["title_grad"] = [_shade(a, 0.65), a, _shade(a, -0.35)]
        st["header"]["title_stroke_color"] = _shade(a, -0.8)
        st["progress"]["fill_color"] = a
        st["progress"]["node_active_ring"] = a
        st["keyword_pop"]["color"] = a if t.get("pop_follow_accent", True) else st["keyword_pop"]["color"]
    if t.get("title_font") in ("书法", "黑体"):
        cal = ["STXINGKA.TTF", "STKAITI.TTF", "simkai.ttf", "NotoSerifCJK-Black.ttc"]
        sans = ["NotoSansCJK-Black.ttc", "Noto Sans SC Bold (TrueType).otf", "HarmonyOS_Sans_SC_Bold.ttf", "msyhbd.ttc"]
        st["fonts"]["title"] = cal if t["title_font"] == "书法" else sans
        st["fonts"]["pop"] = cal if t["title_font"] == "书法" else sans
    if t.get("subtitle_color") in ("white", "yellow"):
        st["subtitle"]["color"] = "#ffe14d" if t["subtitle_color"] == "yellow" else "#ffffff"
    if "show_header" in t:
        st["header"]["show"] = bool(t["show_header"])
    if "show_progress" in t:
        st["progress"]["show"] = bool(t["show_progress"])
    if "use_pip" in t or "use_circle" in t:
        use_pip = bool(t.get("use_pip", st["pacing"]["cycle"] != ["full"]))
        use_circle = bool(t.get("use_circle", st["pacing"].get("chapter_mode") == "pip_circle"))
        st["pacing"]["cycle"] = ["full", "pip_round", "full", "vignette"] if use_pip else ["full"]
        st["pacing"]["chapter_mode"] = "pip_circle" if use_circle else ("pip_round" if use_pip else "full")
    if t.get("pace") in ("慢", "中", "快"):
        b = {"慢": 6.0, "中": 4.0, "快": 2.5}[t["pace"]]
        st["pacing"]["block_sec"].update({"full": [b * 1.2, b * 2.2], "pip_round": [b * 1.4, b * 2.5],
                                          "vignette": [b * 1.2, b * 2.0]})
    if "pops" in t:
        st["keyword_pop"]["max_per_min"] = 4 if t["pops"] else 0
    if "namecard" in t:
        st["namecard"]["show"] = bool(t["namecard"])
    save_json(st, p)
    pout = os.path.join(STYLES, "previews", sid + "_out.jpg")
    if os.path.exists(pout):
        os.remove(pout)  # 参数变了，成片效果图等下一条成片重新生成
    return st


def _tweak_motion(sid, p, st, t):
    """motion 类风格（AI博主等）的微调：主色→强调渐变与背景光晕；"画中画"开关→全屏信息卡开关。"""
    if t.get("accent") and re.match(r"^#[0-9a-fA-F]{6}$", t["accent"]):
        a = t["accent"].lower()
        st["accent"] = a
        st["motion"]["accent_grad"] = [_shade(a, 0.35), a, _shade(a, -0.25)]
        st["motion"]["glow"] = _shade(a, -0.45)
    if t.get("subtitle_color") in ("white", "yellow"):
        st["subtitle"]["color"] = "#ffe14d" if t["subtitle_color"] == "yellow" else "#f2f2f4"
    if "use_pip" in t:
        on = bool(t["use_pip"])
        st["pacing"]["cycle"] = ["full", "card", "full", "card"] if on else ["full"]
        st["pacing"]["chapter_mode"] = "card" if on else "full"
    if t.get("pace") in ("慢", "中", "快"):
        b = {"慢": 6.0, "中": 4.0, "快": 2.6}[t["pace"]]
        st["pacing"]["block_sec"]["full"] = [b * 1.1, b * 2.0]
        st["pacing"]["block_sec"]["card"] = [min(3.2, b * 0.75), min(4.5, b * 1.1)]
    if "pops" in t:
        st["keyword_pop"]["max_per_min"] = 3 if t["pops"] else 0
        st["keyword_pop"]["hook"] = bool(t["pops"])
    save_json(st, p)
    pout = os.path.join(STYLES, "previews", sid + "_out.jpg")
    if os.path.exists(pout):
        os.remove(pout)
    return st


def summary(st):
    """给网页显示的可调参数当前值。"""
    return {"display_name": st.get("display_name"), "accent": st.get("accent", "#22d3ee"),
            "title_font": "书法" if "STXINGKA" in " ".join(st["fonts"]["title"]) else "黑体",
            "subtitle_color": "yellow" if st["subtitle"].get("color", "").lower() == "#ffe14d" else "white",
            "show_header": st["header"].get("show", True), "show_progress": st["progress"].get("show", True),
            "use_pip": st["pacing"]["cycle"] != ["full"], "use_circle": st["pacing"].get("chapter_mode", "pip_circle") == "pip_circle",
            "pace": "快" if st["pacing"]["block_sec"]["full"][0] < 4 else ("慢" if st["pacing"]["block_sec"]["full"][0] > 6 else "中"),
            "pops": st["keyword_pop"].get("max_per_min", 4) > 0, "namecard": st["namecard"].get("show", True),
            "generated": bool(st.get("generated")), "description": st.get("description", ""),
            "motion": bool(st.get("motion"))} if not st.get("motion") else {
        "display_name": st.get("display_name"), "accent": st.get("accent", "#a23cff"), "title_font": "黑体",
        "subtitle_color": "yellow" if st["subtitle"].get("color", "").lower() == "#ffe14d" else "white",
        "show_header": False, "show_progress": False, "use_pip": "card" in st["pacing"]["cycle"], "use_circle": False,
        "pace": "快" if st["pacing"]["block_sec"]["full"][0] < 3.5 else ("慢" if st["pacing"]["block_sec"]["full"][0] > 6 else "中"),
        "pops": st["keyword_pop"].get("max_per_min", 3) > 0, "namecard": False,
        "generated": bool(st.get("generated")), "description": st.get("description", ""), "motion": True}


def delete(sid):
    p = os.path.join(STYLES, sid + ".json")
    with open(p, encoding="utf-8") as f:
        st = json.load(f)
    if not st.get("generated"):
        raise PermissionError("内置风格不能删除")
    os.remove(p)
    for q in (os.path.join(STYLES, "previews", sid + ".jpg"), os.path.join(STYLES, "previews", sid + "_out.jpg")):
        if os.path.exists(q):
            os.remove(q)
    shutil.rmtree(os.path.join(HERE, "references", sid), ignore_errors=True)
