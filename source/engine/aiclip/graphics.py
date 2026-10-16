"""图形层：文字（描边/渐变/阴影/斜体）、头部标题栏、章节进度条、名片、弹字、B-roll 知识卡片。
所有图层预先渲染为预乘 alpha 的 numpy 数组，逐帧只做 ROI 混合，保证渲染速度。"""
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .utils import find_font


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i: i + 2], 16) for i in (0, 2, 4))


# ---------------- 字体 ----------------
_FONTS = {}


def font(style, role, size):
    key = (tuple(style["fonts"][role]), int(size))  # 多风格混剪时不同风格的字体互不串用
    if key in _FONTS:
        return _FONTS[key]
    path = find_font(style["fonts"][role])
    idx = 0
    if path.lower().endswith(".ttc"):
        # 优先选择简体中文字面
        for i in range(10):
            try:
                f = ImageFont.truetype(path, int(size), index=i)
            except Exception:
                break
            name = " ".join(f.getname())
            if " SC" in name or "SC " in name or "简" in name:
                idx = i
                break
    f = ImageFont.truetype(path, int(size), index=idx)
    _FONTS[key] = f
    return f


# ---------------- 图层 ----------------
class Layer:
    """预乘 alpha 图层：c=(h,w,3) float32 BGR*alpha, a=(h,w,1) float32。
    额外预计算 uint8 版本（inv=255*(1-a), cu=c），不透明度为 1 时用 OpenCV SIMD 整数混合，速度约快 3 倍。"""
    __slots__ = ("c", "a", "w", "h", "ox", "oy", "inv", "cu")

    def __init__(self, c, a, ox=0, oy=0):
        self.c, self.a = c, a
        self.h, self.w = a.shape[:2]
        self.ox, self.oy = ox, oy  # 锚点偏移（文字图像的内边距）
        self.inv = np.ascontiguousarray(np.repeat(np.clip(255 * (1 - a) + 0.5, 0, 255).astype(np.uint8), 3, 2))
        self.cu = np.ascontiguousarray(np.clip(c + 0.5, 0, 255).astype(np.uint8))

    @staticmethod
    def from_pil(img, ox=0, oy=0):
        arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
        a = arr[:, :, 3:4] / 255.0
        c = arr[:, :, 2::-1] * a
        return Layer(np.ascontiguousarray(c), np.ascontiguousarray(a), ox, oy)

    def scaled(self, s):
        if abs(s - 1) < 1e-3:
            return self
        w, h = max(1, int(self.w * s)), max(1, int(self.h * s))
        c = cv2.resize(self.c, (w, h), interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(self.a, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
        return Layer(c, a, self.ox * s, self.oy * s)

    def draw(self, frame, x, y, opacity=1.0):
        """把图层文字原点放在 (x, y)（内部扣除 ox/oy 内边距）。frame: uint8 BGR，原地修改。"""
        if opacity <= 0.003:
            return
        x, y = int(round(x - self.ox)), int(round(y - self.oy))
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + self.w), min(H, y + self.h)
        if x1 <= x0 or y1 <= y0:
            return
        sx, sy = x0 - x, y0 - y
        sl = (slice(sy, sy + (y1 - y0)), slice(sx, sx + (x1 - x0)))
        roi = frame[y0:y1, x0:x1]
        if opacity >= 0.999:
            tmp = cv2.multiply(roi, self.inv[sl], scale=1 / 255.0)
            cv2.add(tmp, self.cu[sl], dst=roi)
        else:
            c, a = self.c[sl], self.a[sl]
            r = roi.astype(np.float32) * (1 - a * opacity) + c * opacity
            roi[:] = r.astype(np.uint8)

    def draw_center(self, frame, cx, cy, opacity=1.0, scale=1.0):
        L = self.scaled(scale) if scale != 1.0 else self
        L.draw(frame, cx - (L.w - 2 * L.ox) / 2, cy - (L.h - 2 * L.oy) / 2, opacity)


# ---------------- 文字 ----------------
def _grad_img(w, h, colors):
    cols = [np.array(hex2rgb(c), np.float32) for c in colors]
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    seg = len(cols) - 1
    out = np.zeros((h, 3), np.float32)
    for i in range(h):
        p = t[i, 0] * seg
        k = min(seg - 1, int(p))
        f = p - k
        out[i] = cols[k] * (1 - f) + cols[k + 1] * f
    img = np.repeat(out[:, None, :], w, 1).astype(np.uint8)
    return Image.fromarray(img, "RGB")


def text_image(text, fnt, fill="#ffffff", stroke=0, stroke_color="#000000", shadow=None, grad=None,
               skew=0.0, bold=0, pad=None, spacing=0):
    """返回 (PIL RGBA, pad)。shadow=(dx,dy,blur,alpha)。"""
    if pad is None:
        pad = int(stroke + (max(abs(shadow[0]), abs(shadow[1])) + shadow[2] * 2 if shadow else 0) + bold + 6)
    # 逐字排版以支持字距
    widths = [fnt.getlength(ch) for ch in text]
    tw = int(sum(widths) + spacing * max(0, len(text) - 1))
    asc, desc = fnt.getmetrics()
    th = asc + desc
    W, H = tw + 2 * pad, th + 2 * pad
    fill_m = Image.new("L", (W, H), 0)
    strk_m = Image.new("L", (W, H), 0)
    df, ds = ImageDraw.Draw(fill_m), ImageDraw.Draw(strk_m)
    x = pad
    for ch, wch in zip(text, widths):
        df.text((x, pad), ch, font=fnt, fill=255)
        if stroke + bold > 0:
            ds.text((x, pad), ch, font=fnt, fill=255, stroke_width=stroke + bold, stroke_fill=255)
        x += wch + spacing
    if bold:
        fill_m = fill_m.filter(ImageFilter.MaxFilter(bold * 2 + 1))
    if stroke + bold == 0:
        strk_m = fill_m
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if shadow:
        dx, dy, bl, al = shadow
        sh = Image.new("L", (W, H), 0)
        sh.paste(strk_m, (int(dx), int(dy)))
        if bl:
            sh = sh.filter(ImageFilter.GaussianBlur(bl))
        sh = sh.point(lambda v: int(v * al))
        out.paste(Image.new("RGBA", (W, H), (0, 0, 0, 255)), (0, 0), sh)
    if stroke:
        out.paste(Image.new("RGBA", (W, H), hex2rgb(stroke_color) + (255,)), (0, 0), strk_m)
    src = _grad_img(W, H, grad).convert("RGBA") if grad else Image.new("RGBA", (W, H), hex2rgb(fill) + (255,))
    out.paste(src, (0, 0), fill_m)
    if skew:
        out = out.transform((W + int(H * abs(skew)), H), Image.AFFINE, (1, skew, -H * skew if skew > 0 else 0, 0, 1, 0),
                            resample=Image.BICUBIC)
    return out, pad


def text_layer(text, fnt, **kw):
    img, pad = text_image(text, fnt, **kw)
    return Layer.from_pil(img, pad, pad)


# ---------------- 头部 ----------------
def build_header(style, brand, labels):
    """静态部分：logo、主标题、副标题。返回 Layer（整幅 1080 宽、头部高度）。"""
    hd = style["header"]
    W = style["canvas"]["w"]
    H = hd["haze_height"]
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    # logo：竖排姓名 + 红色印章
    if brand.get("logo_text") and hd.get("logo_show", True):
        f = font(style, "title", hd["logo_size"])
        y = hd["logo_y"]
        for ch in brand["logo_text"][:3]:
            ti, pad = text_image(ch, f, fill=hd["logo_color"], stroke=2, stroke_color="#ffffff",
                                 shadow=(2, 2, 2, 0.35))
            img.alpha_composite(ti, (hd["logo_x"] - pad, y - pad))
            y += int(hd["logo_size"] * 0.92)
        if brand.get("seal_text"):
            s = 30
            seal = Image.new("RGBA", (s, int(s * 1.35)), hex2rgb(hd["seal_color"]) + (235,))
            fs = font(style, "label", 13)
            d = ImageDraw.Draw(seal)
            for k, ch in enumerate(brand["seal_text"][:2]):
                d.text((8, 4 + k * 17), ch, font=fs, fill=(255, 240, 230))
            img.alpha_composite(seal, (hd["logo_x"] + hd["logo_size"] - 8, y - 18))
    # 主标题
    if brand.get("title"):
        size = hd["title_size"]
        f = font(style, "title", size)
        while f.getlength(brand["title"]) > W - 180 and size > 50:
            size -= 4
            f = font(style, "title", size)
        ti, pad = text_image(brand["title"], f, grad=hd["title_grad"], stroke=hd["title_stroke"],
                             stroke_color=hd["title_stroke_color"], shadow=(3, 5, 4, 0.55), bold=2, spacing=-2)
        x = int(hd["title_cx"] - ti.width / 2)
        x = max(110, min(W - ti.width + pad, x))
        img.alpha_composite(ti, (x, hd["title_y"] - pad + (hd["title_size"] - size) // 2))
    if brand.get("slogan"):
        f = font(style, "title", hd["slogan_size"])
        ti, pad = text_image(brand["slogan"], f, grad=hd["slogan_grad"], stroke=hd["slogan_stroke"],
                             stroke_color=hd["slogan_stroke_color"], shadow=(2, 4, 3, 0.5), spacing=3)
        img.alpha_composite(ti, (hd["slogan_right"] - ti.width + pad, hd["slogan_y"] - pad))
    return Layer.from_pil(img)


class Progress:
    """章节进度条：轨道 + 填充 + 节点 + 标签（当前章节红色高亮），按 2px 量化缓存。"""

    def __init__(self, style, labels):
        self.st = style
        self.p = style["progress"]
        self.W = style["canvas"]["w"]
        self.labels = labels
        n = max(1, len(labels))
        m = self.p["margin"]
        self.xs = [m + (self.W - 2 * m) * (i + 0.5) / n for i in range(n)]
        self.cache = {}
        size = self.p["label_size"]
        f = font(style, "label", size)
        maxw = (self.W - 2 * m) / n * 0.94
        while any(f.getlength(l) > maxw for l in labels) and size > 22:
            size -= 2
            f = font(style, "label", size)
        self.lab = {}
        for i, l in enumerate(labels):
            for act in (0, 1):
                self.lab[(i, act)] = text_layer(l, f, fill=self.p["label_active_color"] if act else self.p["label_color"],
                                                stroke=self.p["label_stroke"], stroke_color=self.p["label_stroke_color"],
                                                shadow=(1, 2, 2, 0.35))

    def node_x(self, i):
        return self.xs[max(0, min(len(self.xs) - 1, i))]

    def bar(self, fill_x, active, pulse):
        key = (int(fill_x) // 2, active, int(pulse * 10))
        L = self.cache.get(key)
        if L is not None:
            return L
        p = self.p
        r = p["node_r"]
        Hh = r * 2 + 16
        img = Image.new("RGBA", (self.W, Hh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cy = Hh // 2
        t = p["thickness"]
        m = p["margin"]
        d.rounded_rectangle([m, cy - t // 2, self.W - m, cy + t // 2], radius=t // 2, fill=hex2rgb(p["track_color"]) + (235,))
        d.rounded_rectangle([m, cy - t // 2, max(m + t, fill_x), cy + t // 2], radius=t // 2, fill=hex2rgb(p["fill_color"]) + (255,))
        sh = Image.new("RGBA", (self.W, Hh), (0, 0, 0, 0))
        ds = ImageDraw.Draw(sh)
        for i, x in enumerate(self.xs):
            rr = r * (1 + 0.18 * pulse) if i == active else r
            ds.ellipse([x - rr + 2, cy - rr + 3, x + rr + 2, cy + rr + 3], fill=(0, 0, 0, 90))
        sh = sh.filter(ImageFilter.GaussianBlur(3))
        img = Image.alpha_composite(img, sh)
        d = ImageDraw.Draw(img)
        for i, x in enumerate(self.xs):
            rr = r * (1 + 0.18 * pulse) if i == active else r
            d.ellipse([x - rr, cy - rr, x + rr, cy + rr], fill=hex2rgb(p["node_color"]) + (255,))
            if i == active:
                d.ellipse([x - rr, cy - rr, x + rr, cy + rr], outline=hex2rgb(p["node_active_ring"]) + (255,), width=5)
            elif x <= fill_x:
                d.ellipse([x - rr * 0.38, cy - rr * 0.38, x + rr * 0.38, cy + rr * 0.38],
                          fill=hex2rgb(p["fill_color"]) + (255,))
        L = Layer.from_pil(img, 0, cy)
        if len(self.cache) > 600:
            self.cache.clear()
        self.cache[key] = L
        return L

    def draw(self, frame, fill_x, active, pulse=0.0):
        self.bar(fill_x, active, pulse).draw(frame, 0, self.p["y"])
        for i, x in enumerate(self.xs):
            L = self.lab[(i, 1 if i == active else 0)]
            L.draw(frame, x - (L.w - 2 * L.ox) / 2, self.p["label_y"])


# ---------------- B-roll 知识卡片 ----------------
def make_card(style, label, w, h, seed=0):
    """无素材时自动生成的动态背景卡：深色渐变 + 光斑 + 网格 + 竖排大字。返回 BGR uint8（比画布大，供推拉）。"""
    rnd = random.Random(seed)
    cols = style["broll"]["card_colors"][seed % len(style["broll"]["card_colors"])]
    c0, c1 = np.array(hex2rgb(cols[0]), np.float32), np.array(hex2rgb(cols[1]), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.clip((xx / w * 0.6 + (1 - yy / h) * 0.4), 0, 1)[..., None]
    img = (c0 * (1 - t) + c1 * t)
    # 径向光
    cx, cy = w * 0.72, h * 0.42
    rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.6 * h)
    img += (np.clip(1 - rad, 0, 1) ** 2 * 70)[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img, "RGB").convert("RGBA")
    # 光斑
    bok = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    db = ImageDraw.Draw(bok)
    for _ in range(26):
        r = rnd.randint(18, 90)
        x, y = rnd.randint(0, w), rnd.randint(0, h)
        a = rnd.randint(18, 55)
        db.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, a))
    bok = bok.filter(ImageFilter.GaussianBlur(10))
    pil = Image.alpha_composite(pil, bok)
    d = ImageDraw.Draw(pil)
    # 细网格与十字
    for gx in range(0, w, 90):
        d.line([(gx, 0), (gx, h)], fill=(255, 255, 255, 14), width=1)
    for gy in range(0, h, 90):
        d.line([(0, gy), (w, gy)], fill=(255, 255, 255, 14), width=1)
    for _ in range(8):
        x, y, s = rnd.randint(0, w), rnd.randint(0, h), rnd.randint(10, 22)
        d.line([(x - s, y), (x + s, y)], fill=(255, 255, 255, 60), width=4)
        d.line([(x, y - s), (x, y + s)], fill=(255, 255, 255, 60), width=4)
    # 竖排大字（右侧）
    if label:
        n = len(label)
        size = int(min(260, (h * 0.62) / max(1, n)))
        f = font(style, "title", size)
        y0 = int(h * 0.5 - n * size * 0.98 / 2)
        x0 = int(w * 0.78 - size / 2)
        for k, ch in enumerate(label):
            ti, pad = text_image(ch, f, fill="#ffffff", stroke=0, shadow=(0, 0, 14, 0.5))
            ti.putalpha(ti.getchannel("A").point(lambda v: int(v * 0.30)))
            pil.alpha_composite(ti, (x0 - pad, y0 + int(k * size * 0.98) - pad))
        # 横向小标签
        f2 = font(style, "sub", 46)
        ti, pad = text_image("— " + label + " —", f2, fill="#ffffff", shadow=(0, 2, 4, 0.5))
        ti.putalpha(ti.getchannel("A").point(lambda v: int(v * 0.55)))
        pil.alpha_composite(ti, (int(w * 0.78 - ti.width / 2), max(0, int(y0 - 110))))
    return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


# ---------------- 形状遮罩 ----------------
def rrect_mask(w, h, r, feather=0.0, border=0.0):
    """圆角矩形 SDF 遮罩，返回 (alpha, border_alpha) float32 (h,w)。feather>0 时为柔边（暗角圆）。"""
    w, h = max(2, int(w)), max(2, int(h))
    r = min(r, w / 2, h / 2)
    ys = (np.arange(h, dtype=np.float32) + 0.5 - h / 2)[:, None]
    xs = (np.arange(w, dtype=np.float32) + 0.5 - w / 2)[None, :]
    qx = np.abs(xs) - (w / 2 - r)
    qy = np.abs(ys) - (h / 2 - r)
    d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - r
    if feather > 1:
        a = np.clip(-d / feather, 0, 1)
        a = a * a * (3 - 2 * a)
    else:
        a = np.clip(0.5 - d, 0, 1)
    b = None
    if border > 0.5:
        b = np.clip(0.5 - (np.abs(d + border / 2) - border / 2), 0, 1)
    return a.astype(np.float32), (b.astype(np.float32) if b is not None else None)
