"""AI_UP：手动调节的标题（封面 + 视频里的标题）。

参数（网页“标题与封面”面板编辑，存在 04_工程/<任务>/标题设置.json：{"video": {...}, "cover": {...}}）：
  lines        两行文字（第二行可留空）
  font1/font2  字体文件名（见 FONT_CHOICES）     size1/size2  字号（像素，按竖屏 1080 宽 / 横屏 1080 高）
  color1/color2  颜色 #RRGGBB；grad1/grad2 为 true 时做“亮→本色→暗”的竖向渐变
  stroke(描边粗细) stroke_color  shadow(投影浓度 0~1)  gap(行距)
  port/land    {"x": 横向位置%, "y": 文字块中心的纵向位置%, "align": "center|left", "scale": 缩放}
  封面专用：blur(景深：背景虚化 0~30) dark(整体压暗 0~0.8) shade(文字区渐变压暗 0~1) bar(短横线) seal(印章文字) name(律师名牌)
  视频专用：enable  mode("always" 全程 / "intro" 只在开场)  intro_sec
"""
import hashlib
import json
import os

import cv2
import numpy as np

from . import graphics as G
from .graphics import Layer, text_image

FONT_CHOICES = [
    ("SourceHanSansCN-Heavy.otf", "思源黑体 特粗"), ("SourceHanSansCN-Bold.otf", "思源黑体 粗"),
    ("SourceHanSansCN-Medium.otf", "思源黑体 中"), ("LawSerif-Black.otf", "宋体 特粗"), ("LawSerif-Bold.otf", "宋体 粗"),
    ("msyhbd.ttc", "微软雅黑 粗"), ("simhei.ttf", "黑体"), ("simkai.ttf", "楷体"), ("STKAITI.TTF", "华文楷体"),
    ("STXINGKA.TTF", "华文行楷"), ("FZSTK.TTF", "方正舒体"), ("STHUPO.TTF", "华文琥珀"), ("STZHONGS.TTF", "华文中宋"),
    ("STLITI.TTF", "华文隶书"), ("SIMLI.TTF", "隶书"), ("STXINWEI.TTF", "华文新魏"),
]


def available_fonts():
    from .utils import _index_fonts
    idx = _index_fonts()
    return [{"id": f, "name": n} for f, n in FONT_CHOICES if f.lower() in idx]


def _font(name, size):
    return G.font({"fonts": {"x": [name or "SourceHanSansCN-Heavy.otf", "SourceHanSansCN-Heavy.otf",
                                   "NotoSansCJK-Black.ttc", "msyhbd.ttc"]}}, "x", max(8, int(size)))


def _mix(c, to, k):
    a = np.array(G.hex2rgb(c), np.float32)
    b = np.array(G.hex2rgb(to), np.float32)
    v = (a * (1 - k) + b * k).astype(int)
    return "#%02x%02x%02x" % tuple(int(x) for x in v)


def _grad(c):
    return [_mix(c, "#ffffff", 0.55), c, _mix(c, "#000000", 0.28)]


# ---------------- 默认值（和自动版式长得一样，用户在此基础上调） ----------------
def defaults(tl, st, photo=False, src="video"):
    from .cover import short_lines, title_lines
    law = tl.get("law") or {}
    acc = (st.get("hud") or {}).get("accent") or st.get("accent") or "#ff8a2a"
    if law:
        lines = [x for x in (law.get("title") or title_lines(tl)) if x][:2]
        base = dict(lines=lines, font1="LawSerif-Black.otf", font2="LawSerif-Bold.otf", color1="#f3d27a",
                    color2="#fbefd0", grad1=True, grad2=True, stroke=2, stroke_color="#4a3110", shadow=0.8, gap=20)
        cover = dict(base, size1=150, size2=100, port={"x": 50, "y": 68, "align": "center", "scale": 1.0},
                     land={"x": 6, "y": 50, "align": "left", "scale": 1.15}, blur=0, dark=0.0, shade=0.88, bar=False,
                     seal=((law.get("hook") or {}).get("red") or ""), name=bool(law.get("name")))
        lc = (st.get("law") or {}).get("port") or {}
        video = dict(base, size1=lc.get("t1_size", 118), size2=lc.get("t2_size", 74),
                     port={"x": 50, "y": 17, "align": "center", "scale": 1.0},
                     land={"x": 50, "y": 15, "align": "center", "scale": 0.72}, enable=False, mode="always", intro_sec=4)
    else:
        lines = short_lines(tl, int((st.get("cover") or {}).get("max_chars", 6))) if photo else title_lines(tl)
        base = dict(lines=[x for x in lines if x][:2], font1="SourceHanSansCN-Heavy.otf",
                    font2="SourceHanSansCN-Heavy.otf", color1="#ffffff", color2=acc, grad1=False, grad2=True,
                    stroke=0, stroke_color="#000000", shadow=0.7, gap=18)
        cc = st.get("cover") or {}
        cover = dict(base, size1=176 if photo else 124, size2=200 if photo else 146,
                     port={"x": 50, "y": 70 if photo else 66, "align": "center", "scale": 1.0},
                     land={"x": 7, "y": 50, "align": "left", "scale": 1.05 if photo else 1.15},
                     blur=0, dark=float(cc.get("dark", 0.3)) if photo else 0.0, shade=0.85, bar=True, seal="", name=False)
        video = dict(base, size1=96, size2=112, port={"x": 50, "y": 14, "align": "center", "scale": 1.0},
                     land={"x": 50, "y": 13, "align": "center", "scale": 0.8}, enable=False, mode="intro", intro_sec=4)
    cover["src"] = src
    return {"cover": cover, "video": video}


# ---------------- 文字块 ----------------
_CACHE = {}


def build_block(p, land, maxw=1e9):
    """返回 [(Layer, w, h)]，带缓存。某行超过 maxw 时自动缩小这一行。"""
    o = p.get("land" if land else "port") or {}
    sc = float(o.get("scale", 1.0))
    maxw = int(maxw)
    key = hashlib.md5(json.dumps([p.get(k) for k in ("lines", "font1", "font2", "size1", "size2", "color1", "color2",
                                                      "grad1", "grad2", "stroke", "stroke_color", "shadow")] + [sc, maxw],
                                 ensure_ascii=False).encode()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    out = []
    for i, ln in enumerate([x for x in (p.get("lines") or []) if str(x).strip()][:2]):
        k = i + 1
        size = float(p.get(f"size{k}", 120)) * sc
        f = _font(p.get(f"font{k}"), size)
        w0 = f.getlength(str(ln)) + size * 0.03 * len(str(ln))
        if w0 > maxw:  # 太长：按比例缩小这一行
            size = max(16, size * maxw / w0)
            f = _font(p.get(f"font{k}"), size)
        col = p.get(f"color{k}", "#ffffff")
        sh = float(p.get("shadow", 0.7))
        img, pad = text_image(str(ln), f, fill=col, grad=_grad(col) if p.get(f"grad{k}") else None,
                              stroke=int(p.get("stroke", 0) * sc), stroke_color=p.get("stroke_color", "#000000"),
                              shadow=(0, int(size * 0.05), int(size * 0.08), sh) if sh > 0.01 else None,
                              spacing=int(size * 0.03))
        L = Layer.from_pil(img, pad, pad)
        out.append((L, L.w - 2 * L.ox, L.h - 2 * L.oy))
    if len(_CACHE) > 60:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def place(p, land, W, H):
    """文字块左上角与尺寸：(x0, y0, bw, bh, 每行 x)。"""
    o = p.get("land" if land else "port") or {}
    cx, cy = W * float(o.get("x", 50)) / 100, H * float(o.get("y", 50)) / 100
    left = o.get("align", "center") == "left"
    if left:
        maxw = (W * 0.56 - cx) if land else (W - cx) * 0.94
    else:
        maxw = min(cx, W - cx) * 2 * 0.94
    blk = build_block(p, land, max(W * 0.2, maxw))
    gap = float(p.get("gap", 18)) * float(o.get("scale", 1.0))
    bw = max([w for _, w, _ in blk] or [0])
    bh = sum(h for _, _, h in blk) + gap * max(0, len(blk) - 1)
    x0 = cx if left else cx - bw / 2
    y0 = cy - bh / 2
    xs = [x0 if left else cx - w / 2 for _, w, _ in blk]
    return x0, y0, bw, bh, xs, gap, blk


def draw_block(frame, p, land, op=1.0, dy=0.0):
    H, W = frame.shape[:2]
    x0, y0, bw, bh, xs, gap, blk = place(p, land, W, H)
    y = y0 + dy
    for (L, w, h), x in zip(blk, xs):
        L.draw(frame, x, y, op)
        y += h + gap
    return x0, y0, bw, bh


# ---------------- 封面 ----------------
def _depth(frame, blur, model=None):
    """景深：人物（脸 + 上半身）保持清晰，背景按 blur 虚化。"""
    if blur <= 0.3:
        return frame
    H, W = frame.shape[:2]
    from .cover import _faces
    det = None
    if model and os.path.exists(model) and hasattr(cv2, "FaceDetectorYN"):
        det = cv2.FaceDetectorYN.create(model, "", (320, 320), 0.6, 0.3, 5)
    f = _faces(det, frame) if det is not None else None
    m = np.zeros((H, W), np.float32)
    if f is not None:
        x, y, fw, fh = [float(v) for v in f[:4]]
        cx, cy = x + fw / 2, y + fh / 2
        cv2.ellipse(m, (int(cx), int(cy)), (int(fw * 0.75), int(fh * 0.8)), 0, 0, 360, 1.0, -1)
        cv2.ellipse(m, (int(cx), int(cy + fh * 2.6)), (int(fw * 1.9), int(fh * 2.3)), 0, 0, 360, 1.0, -1)
    else:
        cv2.ellipse(m, (W // 2, int(H * 0.55)), (int(W * 0.3), int(H * 0.45)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(W, H) * 0.03)[..., None]
    k = max(W, H) / 1920
    bl = cv2.GaussianBlur(frame, (0, 0), blur * k)
    return (frame.astype(np.float32) * m + bl.astype(np.float32) * (1 - m)).astype(np.uint8)


def render_cover(base, face, st, tl, p, land, photo=False, model=None, size=None):
    from .cover import _gradient, compose, compose_photo
    W, H = size or ((1920, 1080) if land else (1080, 1920))
    frame = compose_photo(base, W, H, land) if photo else compose(base, face, W, H, side="right" if land else None)
    frame = _depth(frame, float(p.get("blur", 0)), model)
    dark = float(p.get("dark", 0))
    if dark > 0:
        frame[:] = (frame.astype(np.float32) * (1 - dark)).astype(np.uint8)
    x0, y0, bw, bh, xs, gap, blk = place(p, land, W, H)
    shade = float(p.get("shade", 0))
    o = p.get("land" if land else "port") or {}
    if shade > 0:
        if o.get("align") == "left" and land:
            _gradient(frame, int(min(W, x0 + bw + W * 0.18)), int(max(0, x0 - W * 0.02)), 0.0, shade, axis=1)
        elif y0 + bh / 2 > H * 0.5:
            _gradient(frame, int(max(0, y0 - H * 0.28)), int(min(H, y0 + bh + H * 0.05)), 0.0, shade)
        else:
            _gradient(frame, int(min(H, y0 + bh + H * 0.25)), int(max(0, y0 - H * 0.05)), 0.0, shade)
    sc = float(o.get("scale", 1.0))
    left = o.get("align") == "left"
    cx = x0 if left else x0 + bw / 2
    col2 = p.get("color2", "#ff8a2a")
    seal = str(p.get("seal") or "").strip()
    if seal:
        f = _font("LawSerif-Black.otf", 62 * sc)
        ti, pad = text_image(seal, f, fill="#ffffff", spacing=6)
        tw, th = ti.width - 2 * pad, ti.height - 2 * pad
        bw2, bh2 = int(tw + 56 * sc), int(th + 26 * sc)
        m, _ = G.rrect_mask(bw2, bh2, 10, 0, 0)
        col = np.zeros((bh2, bw2, 4), np.uint8)
        col[..., :3] = G.hex2rgb("#c4161e")
        col[..., 3] = (m * 235).astype(np.uint8)
        from PIL import Image
        box = Image.fromarray(col, "RGBA")
        box.alpha_composite(ti, (int(28 * sc) - pad, int(13 * sc) - pad))
        SL = Layer.from_pil(box)
        SL.draw(frame, cx if left else cx - SL.w / 2, y0 - SL.h - 34 * sc)
    elif p.get("bar"):
        bc = G.hex2rgb(col2)[::-1]
        bx = int(cx if left else cx - 65 * sc)
        cv2.rectangle(frame, (bx, int(y0 - 46 * sc)), (int(bx + 130 * sc), int(y0 - 34 * sc)), bc, -1)
    draw_block(frame, p, land)
    law = tl.get("law") or {}
    if p.get("name") and law.get("name"):
        nf, rf = _font("SourceHanSansCN-Heavy.otf", 60 * sc), _font("SourceHanSansCN-Heavy.otf", 32 * sc)
        sh = (0, 3, 5, 0.7)
        i1, p1 = text_image(law["name"], nf, fill="#f6b31c", shadow=sh, skew=0.18, spacing=2)
        i2, p2 = text_image(" " + law.get("role", "律师"), rf, fill="#f6b31c", shadow=sh, skew=0.18)
        L1, L2 = Layer.from_pil(i1, p1, p1), Layer.from_pil(i2, p2, p2)
        firm = None
        if law.get("firm"):
            i3, p3 = text_image(law["firm"], rf, fill="#ffffff", shadow=sh)
            firm = Layer.from_pil(i3, p3, p3)
        w1, h1 = L1.w - 2 * L1.ox, L1.h - 2 * L1.oy
        w2, h2 = L2.w - 2 * L2.ox, L2.h - 2 * L2.oy
        tw = max(w1 + w2, (firm.w - 2 * firm.ox) if firm else 0)
        nx = cx if left else cx - tw / 2
        ny = y0 + bh + 30 * sc
        L1.draw(frame, nx, ny)
        L2.draw(frame, nx + w1, ny + (h1 - h2) * 0.8)
        if firm:
            firm.draw(frame, nx, ny + h1 + 12 * sc)
    return frame


# ---------------- 视频里的标题 ----------------
class VideoTitle:
    def __init__(self, p, W, H):
        self.p, self.W, self.H = p, W, H
        self.land = W > H

    def draw(self, frame, t):
        p = self.p
        q = min(1.0, max(0.0, (t - 0.15) / 0.4))
        if q <= 0:
            return
        e = 1 - (1 - q) ** 3
        op = e
        if p.get("mode") == "intro":
            end = float(p.get("intro_sec", 4))
            op *= min(1.0, max(0.0, (end - t) / 0.35))
            if op <= 0.003:
                return
        draw_block(frame, p, self.land, op, dy=(1 - e) * 24)
