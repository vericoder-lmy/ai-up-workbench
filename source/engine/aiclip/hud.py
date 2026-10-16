"""AI_UP 信息层（HUD）：人物始终全屏，信息以“眉标 + 大字/大数字 + 列表/卡片/数据”的形式叠在画面一侧，
一侧压暗保证可读。参考：对标视频的“暗场信息叠加”风格；配色只用 橙 / 黑 / 白。

timeline["graphics"] 每项：
  {"t0", "t1", "type": hero|headline|list|cards|stats,
   "kicker": ["ENGLISH KICKER", "中文小标"],
   "hero": {"pre": "$", "val": "100", "suf": "DEALS", "sub": "个商单"},   # type=hero
   "lines": ["视频内容", "必须利他"],                                       # type=headline（最后一行橙色）
   "en": "GIVE THE AUDIENCE A WIN", "note": "一行白色小字说明",
   "items": [{"n": "01", "en": "WORKFLOW", "zh": "工作流 + Skill 库"}]      # list / cards
           或 [{"label": "GROWTH · 涨粉", "val": "98.18%", "note": "+5,294"}] # stats
  }
"""
import math
import random
import re

import cv2
import numpy as np

from . import graphics as G
from .graphics import Layer, text_layer
from .utils import clamp, ease_out_back, ease_out_cubic

_NUM = re.compile(r"^([0-9][0-9,]*)(\.[0-9]+)?$")
_DECODE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/#+-"


def _smooth(e0, e1, x):
    p = np.clip((x - e0) / max(1e-6, e1 - e0), 0, 1)
    return p * p * (3 - 2 * p)


def _font(st, role, size):
    return G.font(st, role, size)


class _El:
    __slots__ = ("kind", "x", "y", "delay", "layer", "make", "w", "h", "data")

    def __init__(self, kind, x, y, delay=0.0, layer=None, make=None, w=0, h=0, data=None):
        self.kind, self.x, self.y, self.delay = kind, x, y, delay
        self.layer, self.make, self.w, self.h, self.data = layer, make, w, h, data


class HUD:
    def __init__(self, tl, style, W, H):
        self.st = style
        self.cfg = dict(style.get("hud") or {})
        self.W, self.H = W, H
        self.land = W > H
        self.items = sorted(tl.get("graphics") or [], key=lambda g: g["t0"])
        self.t0s = [g["t0"] for g in self.items]
        c = self.cfg
        self.accent = c.get("accent", "#ff8a2a")
        self.white = c.get("white", "#f7f1ea")
        self.dim = c.get("dim", "#b9ab9c")
        self.s = float(c.get("scale_land", 1.0) if self.land else c.get("scale_port", 1.12))
        self.in_sec, self.out_sec = c.get("in_sec", 0.45), c.get("out_sec", 0.28)
        self._built = {}
        self._txt = {}
        self.plus = bool(c.get("plus"))
        self._scrim = self._make_scrim()
        self._scrim_r = np.ascontiguousarray(self._scrim[:, ::-1]) if self.land else None
        self._scrim_cache = {}

    # ---------- 压暗 ----------
    def _make_scrim(self):
        W, H, A = self.W, self.H, float(self.cfg.get("scrim", 0.72))
        if self.land:
            xs = np.arange(W, dtype=np.float32)
            ys = np.arange(H, dtype=np.float32)
            ax = (1 - _smooth(W * 0.08, W * 0.62, xs))[None, :]
            ay = (0.75 + 0.25 * (1 - _smooth(0, H * 0.5, ys)))[:, None]
            a = A * ax * ay
            self._sx0, self._sy0 = 0, 0
            a = a[:, : int(W * 0.64)]
        else:
            ys = np.arange(H, dtype=np.float32)
            a = (A * _smooth(H * 0.40, H * 0.70, ys))[:, None] * np.ones((1, W), np.float32)
            y0 = int(H * 0.38)
            a = a[y0:]
            self._sx0, self._sy0 = 0, y0
        return np.ascontiguousarray(a.astype(np.float32))

    def layout(self, g):
        """side=一侧信息（默认）；bigword=人物移到一边、另一边超大字（橙光 Plus）；stage=人物缩到角落小窗、整屏信息卡。"""
        lay = g.get("layout") or "side"
        return lay if self.plus and lay in ("side", "bigword", "stage") else "side"

    def active(self, t, lay, in_sec=0.45, out_sec=0.35):
        """当前时刻某种版式的信息块和它的进出进度（0~1）。"""
        best = (None, 0.0)
        for g in self.items:
            if self.layout(g) != lay or not (g["t0"] - 0.01 <= t < g["t1"] + out_sec):
                continue
            p = min(clamp((t - g["t0"]) / in_sec), clamp((g["t1"] + out_sec - t) / out_sec))
            if p > best[1]:
                best = (g, p)
        return best

    def _apply_scrim(self, frame, k, right=False):
        q = int(round(clamp(k) * 12))
        if q <= 0:
            return
        src = self._scrim_r if (right and self._scrim_r is not None) else self._scrim
        mul = self._scrim_cache.get((q, right))
        if mul is None:
            m = np.clip((1 - src * (q / 12.0)) * 255 + 0.5, 0, 255).astype(np.uint8)
            mul = self._scrim_cache[(q, right)] = np.ascontiguousarray(np.repeat(m[..., None], 3, 2))
        h, w = mul.shape[:2]
        x0 = self.W - w if (right and self.land) else self._sx0
        roi = frame[self._sy0: self._sy0 + h, x0: x0 + w]
        cv2.multiply(roi, mul[: roi.shape[0], : roi.shape[1]], dst=roi, scale=1 / 255.0)

    # ---------- 文字 ----------
    def _t(self, text, role, size, color, spacing=0, shadow=(0, 3, 8, 0.55)):
        if role in ("latin", "num") and any(ord(c) > 0x2FFF for c in text):
            role = "label" if role == "latin" else "title"  # 英文字体没有中文字形
        key = (text, role, int(size), color, spacing)
        L = self._txt.get(key)
        if L is None:
            if len(self._txt) > 600:
                self._txt.clear()
            L = self._txt[key] = text_layer(text, _font(self.st, role, int(size)), fill=color,
                                            shadow=shadow, spacing=spacing)
        return L

    @staticmethod
    def _vw(L):
        return L.w - 2 * L.ox

    @staticmethod
    def _vh(L):
        return L.h - 2 * L.oy

    def _box(self, w, h, r=10, fill_a=0.58, border_a=0.45, left_bar=False):
        w, h = int(w), int(h)
        a, b = G.rrect_mask(w, h, r, 0, 2)
        acc = np.array(G.hex2rgb(self.accent)[::-1], np.float32)
        col = np.zeros((h, w, 3), np.float32)
        alpha = a * fill_a
        if b is not None:
            alpha = alpha * (1 - b * border_a) + b * border_a
            col += (b * border_a)[..., None] * acc
        if left_bar:
            bar = np.zeros((h, w), np.float32)
            bar[int(h * 0.22): int(h * 0.78), 0:4] = 1.0
            alpha = np.maximum(alpha, bar)
            col = col * (1 - bar[..., None]) + bar[..., None] * acc
        return Layer(np.ascontiguousarray(col), np.ascontiguousarray(alpha[..., None]))

    # ---------- 排版：把一个信息块拆成若干元素（相对坐标） ----------
    def _build(self, g):
        s = self.s
        A, Wt, Dm = self.accent, self.white, self.dim
        els = []
        y = 0.0
        lay = self.layout(g)
        max_w = (self.W * 0.46) if self.land else (self.W - 2 * self.cfg.get("margin_port", 64))
        big = 1.0
        if lay == "bigword":
            big = 1.75
            if self.land:
                max_w = self.W * 0.40
        elif lay == "stage":
            s *= 1.3 if self.land else 1.1  # 整屏卡片：字和卡片都放大
            if self.land:
                max_w = self.W * 0.6
        # 眉标：竖线 + 英文（逐字解码）+ 中文小字
        kk = g.get("kicker") or ["", ""]
        ken = str(kk[0] if kk else "").upper()
        kzh = str(kk[1] if len(kk) > 1 else "")
        if ken or kzh:
            size_en, size_zh = 24 * s, 22 * s
            hen = self._vh(self._t("A", "latin", size_en, A, int(7 * s))) if ken else 0
            hzh = self._vh(self._t("中", "label", size_zh, Wt)) if kzh else 0
            bar_h = hen + hzh + 6 * s
            k0 = 0.3 if lay == "stage" else 0.0
            els.append(_El("bar", 0, y, k0, w=max(3, int(4 * s)), h=int(bar_h)))
            if ken:
                els.append(_El("decode", 16 * s, y, k0 + 0.05, data=(ken, "latin", size_en, A, int(7 * s))))
            if kzh:
                els.append(_El("slide", 16 * s, y + hen + 4 * s, k0 + 0.12,
                               layer=self._t(kzh, "label", size_zh, Wt, int(3 * s))))
            y += bar_h + 34 * s
        typ = g.get("type", "headline")
        d = 0.18 + (0.3 if lay == "stage" else 0.12 if lay == "bigword" else 0)  # 等人物缩放到位再出字
        if typ == "hero" and g.get("hero"):
            h = g["hero"]
            val = str(h.get("val", ""))
            pre, suf, sub = str(h.get("pre", "")), str(h.get("suf", "")).upper(), str(h.get("sub", ""))
            is_num = bool(_NUM.match(re.sub(r"[%+ ]", "", val)))
            role = "num" if re.fullmatch(r"[0-9,.\-+%~→ ]+", val) else "title"

            def parts(vs):
                vL = self._t(val, role, vs, Wt, 0, shadow=(0, 8, 18, 0.6))
                pL = self._t(pre, "num", vs * 0.55, A, 0, shadow=(0, 6, 14, 0.5)) if pre else None
                sL = self._t(suf, "latin", max(22 * s, vs * 0.24), A, int(8 * s)) if suf else None
                bL = self._t(sub, "heavy", max(26 * s, vs * 0.19), Wt, 2) if sub else None
                side = max([self._vw(L) for L in (sL, bL) if L] or [0])
                wv = (self._vw(pL) + 10 * s if pL else 0) + self._vw(vL)
                return vL, pL, sL, bL, wv, side

            vsize = (260 * s if len(val) <= 4 else (210 * s if len(val) <= 6 else 170 * s)) * min(big, 1.3)
            below = False
            while True:
                vL, pL, sL, bL, wv, side = parts(vsize)
                if wv + (14 * s + side if side else 0) <= max_w:
                    break
                if wv <= max_w and side and vsize <= 200 * s:
                    below = True  # 后缀放到数字下面
                    break
                if vsize <= 110 * s:
                    below = bool(side)
                    break
                vsize -= 10 * s
            vh = self._vh(vL)
            x = 0.0
            if pL:
                els.append(_El("slide", x, y + vh - self._vh(pL) - vh * 0.08, d, layer=pL))
                x += self._vw(pL) + 10 * s
            els.append(_El("count" if is_num else "pop", x, y, d + 0.04, layer=vL, data=(val, role, vsize, Wt)))
            if below:
                sx, sy = 4 * s, y + vh + 6 * s
            else:
                sx, sy = x + self._vw(vL) + 14 * s, y + vh * 0.5
            if sL:
                els.append(_El("slide", sx, sy, d + 0.16, layer=sL))
                if below:
                    sx += self._vw(sL) + 16 * s
                else:
                    sy += self._vh(sL) + 8 * s
            if bL:
                els.append(_El("slide", sx, sy, d + 0.22, layer=bL))
            y += vh + 8 * s
            if below:
                y += max([self._vh(L) for L in (sL, bL) if L] or [0]) + 14 * s
        elif typ in ("headline", "hero"):
            lines = [str(l) for l in (g.get("lines") or [str((g.get("hero") or {}).get("val", ""))]) if l][:2]
            for k, line in enumerate(lines):
                size = 108 * s * big
                while size > 60 * s and _font(self.st, "title", int(size)).getlength(line) > max_w:
                    size -= 6 * s
                col = A if (k == len(lines) - 1 and len(lines) > 1) else Wt
                L = self._t(line, "title", size, col, 2, shadow=(0, 6, 16, 0.6))
                els.append(_El("rise", 0, y, d + 0.1 * k, layer=L))
                y += self._vh(L) * 0.98
            if len(lines) == 1:
                els.append(_El("line", 0, y + 6 * s, d + 0.2, w=int(min(max_w, 200 * s)), h=max(3, int(5 * s))))
                y += 14 * s
            y += 6 * s
        en = str(g.get("en", "")).upper()
        if en:
            L = self._t(en, "latin", 30 * s, A, int(9 * s))
            els.append(_El("slide", 2 * s, y + 6 * s, d + 0.25, layer=L))
            y += self._vh(L) + 22 * s
        note = str(g.get("note", ""))
        if note:
            L = self._t(note, "label", 30 * s, Wt, 2)
            els.append(_El("slide", 0, y + 8 * s, d + 0.32, layer=L))
            y += self._vh(L) + 26 * s
        items = g.get("items") or []
        if typ == "list" and items:
            rw = min(max_w, 860 * s) if self.land else max_w
            rh = 92 * s
            for k, it in enumerate(items[:4]):
                box = self._box(rw, rh, 10 * s)
                dd = max(d + 0.12 + 0.1 * k, float(it.get("at", 0) or 0))
                els.append(_El("slide", 0, y, dd, layer=box, data=("row", k, float(it.get("at", 0) or 0))))
                nL = self._t(str(it.get("n") or f"{k + 1:02d}"), "num", 46 * s, A, 0, shadow=None)
                els.append(_El("slide", 24 * s, y + (rh - self._vh(nL)) / 2, dd, layer=nL, data=("row", k, float(it.get("at", 0) or 0))))
                tx = 24 * s + max(self._vw(nL), 56 * s) + 26 * s
                eL = self._t(str(it.get("en", "")).upper(), "latin", 19 * s, A, int(5 * s), shadow=None) \
                    if it.get("en") else None
                zL = self._t(str(it.get("zh", "")), "heavy", 30 * s, Wt, 1, shadow=None)
                th = (self._vh(eL) + 4 * s if eL else 0) + self._vh(zL)
                ty = y + (rh - th) / 2
                if eL:
                    els.append(_El("slide", tx, ty, dd + 0.04, layer=eL, data=("row", k, float(it.get("at", 0) or 0))))
                    ty += self._vh(eL) + 4 * s
                els.append(_El("slide", tx, ty, dd + 0.06, layer=zL, data=("row", k, float(it.get("at", 0) or 0))))
                y += rh + 12 * s
        elif typ in ("cards", "stats") and items:
            n = min(3, len(items))
            gap = 14 * s
            cw = (min(max_w, 920 * s) - gap * (n - 1)) / n
            pad_x, pad_y = 18 * s, 18 * s
            inner = cw - 2 * pad_x

            def fit(text, role, size, col, sp, minsize):
                """单行放不下就缩小字号，保证不出框。"""
                L = self._t(text, role, size, col, sp, shadow=None)
                while self._vw(L) > inner and size > minsize:
                    size *= 0.92
                    L = self._t(text, role, size, col, sp, shadow=None)
                return L, size

            def fit2(text, role, size, col, sp, minsize, gap_after):
                """太长时拆成两行（每行不小于原字号的 78%），仍放不下再缩小。返回 rows 片段。"""
                L, sz = fit(text, role, size, col, sp, size * 0.78)
                if self._vw(L) <= inner or len(text) < 4:
                    return [(fit(text, role, size, col, sp, minsize)[0], gap_after, None)]
                cut = len(text) // 2
                for c in "，,、 ·":
                    k = text.find(c, max(1, len(text) // 3))
                    if 0 < k < len(text) * 2 // 3 + 1:
                        cut = k + 1
                        break
                a, b = text[:cut].strip(" ，,、·"), text[cut:].strip(" ，,、·")
                La, sa = fit(a, role, size, col, sp, minsize)
                Lb, sb = fit(b, role, sa, col, sp, minsize)
                if sb < sa:
                    La = fit(a, role, sb, col, sp, minsize)[0]
                return [(La, 2 * s, None), (Lb, gap_after, None)]

            # 先排每张卡里的文字，再按最高的那张定卡片高度（文字多少都不会溢出边框）
            cards = []
            for k, it in enumerate(items[:n]):
                rows = []  # (layer, dy_after, kind, data)
                if typ == "cards":
                    tag = str(it.get("tag") or f"NO. {str(it.get('n') or k + 1).zfill(2)}").upper()
                    rows.append((fit(tag, "latin", 17 * s, A, int(4 * s), 11 * s)[0], 8 * s, None))
                    if it.get("en"):
                        rows.append((fit(str(it["en"]).upper(), "latin", 23 * s, Wt, int(3 * s), 14 * s)[0], 6 * s, None))
                    rows += fit2(str(it.get("zh", "")), "heavy", 27 * s, Wt, 1, 17 * s, 0)
                else:
                    rows.append((fit(str(it.get("label", "")).upper(), "latin", 16 * s, A, int(4 * s), 11 * s)[0],
                                 6 * s, None))
                    val = str(it.get("val", ""))
                    num_ok = bool(_NUM.match(re.sub(r"[%+↑ ]", "", val)))
                    arrow = it.get("up", True) and num_ok
                    vL, vs = fit(val, "num", 58 * s, Wt, 0, 30 * s)
                    if arrow and self._vw(vL) + 34 * s > inner:
                        vL, vs = fit(val, "num", vs * 0.85, Wt, 0, 26 * s)
                    rows.append((vL, 8 * s, ("val", val, vs, num_ok, arrow)))
                    if it.get("note"):
                        rows += fit2(str(it["note"]), "label", 19 * s, Dm, 0, 13 * s, 0)
                h = sum(self._vh(L) + dy for L, dy, _ in rows)
                cards.append((it, rows, h))
            ch = max(h for _, _, h in cards) + 2 * pad_y
            for k, (it, rows, _) in enumerate(cards):
                cx = k * (cw + gap)
                dd = max(d + 0.12 + 0.1 * k, float(it.get("at", 0) or 0))
                els.append(_El("rise", cx, y, dd, layer=self._box(cw, ch, 10 * s, 0.6, 0.4)))
                yy = y + pad_y
                for r, (L, dy, info) in enumerate(rows):
                    if info and info[0] == "val":
                        _, val, vs, num_ok, arrow = info
                        els.append(_El("count" if num_ok else "rise", cx + pad_x, yy, dd, layer=L,
                                       data=(val, "num", vs, Wt)))
                        if arrow:
                            aL = self._t("↑", "label", vs * 0.75, A, 0, shadow=None)
                            els.append(_El("rise", cx + pad_x + self._vw(L) + 6 * s,
                                           yy + (self._vh(L) - self._vh(aL)) / 2, dd + 0.3, layer=aL))
                    else:
                        els.append(_El("rise", cx + pad_x, yy, dd, layer=L))  # 和卡片同步上升，过程中也不出框
                    yy += self._vh(L) + dy
            y += ch + 10 * s
        return els, y

    def _anchor(self, height, lay="side"):
        if lay == "stage":
            return (110, 96) if self.land else (self.cfg.get("margin_port", 64), 210)
        if lay == "bigword" and self.land:
            return self.W * 0.565, max(70, self.H * 0.44 - height / 2)
        if self.land:
            return self.cfg.get("x_land", 78), self.cfg.get("y_land", 64)
        bottom = self.cfg.get("bottom_port", 1450)
        return self.cfg.get("margin_port", 64), max(self.H * 0.40, bottom - height)

    # ---------- 动画 ----------
    def _num_text(self, val, p):
        m = _NUM.match(re.sub(r"[%+ ]", "", val))
        if not m or p >= 1:
            return val
        ip, dp = m.group(1), m.group(2) or ""
        target = float((ip + dp).replace(",", ""))
        cur = target * p
        dec = len(dp) - 1 if dp else 0
        s = f"{cur:,.{dec}f}" if "," in ip else f"{cur:.{dec}f}"
        return val.replace(m.group(1) + (m.group(2) or ""), s, 1)

    def _decode_text(self, text, p, seed):
        n = len(text)
        k = int(n * p)
        rnd = random.Random(seed)
        out = list(text[:k])
        for ch in text[k:]:
            out.append(ch if ch in " ·•/" else rnd.choice(_DECODE))
        return "".join(out)

    def draw(self, frame, t):
        if not self.items:
            return
        i = max(0, min(len(self.items) - 1, np.searchsorted(self.t0s, t, side="right") - 1))
        cands = [g for g in self.items[max(0, i - 1): i + 2] if g["t0"] - 0.01 <= t < g["t1"] + self.out_sec]
        if not cands:
            return
        # 压暗：取最强的那个（整屏卡片自带暗背景，不压暗；大字版式在大字那一侧压暗）
        k, kr = 0.0, 0.0
        for g in cands:
            lt, rem = t - g["t0"], g["t1"] + self.out_sec - t
            e = min(clamp(lt / 0.35), clamp(rem / self.out_sec))
            lay = self.layout(g)
            if lay == "bigword" and self.land:
                kr = max(kr, e)
            elif lay != "stage":
                k = max(k, e)
        self._apply_scrim(frame, k)
        if kr > 0:
            self._apply_scrim(frame, kr, right=True)
        for g in cands:
            self._draw_one(frame, g, t)

    def _draw_one(self, frame, g, t):
        key = id(g)
        b = self._built.get(key)
        if b is None:
            els, h = self._build(g)
            b = self._built[key] = (els, h, self._anchor(h, self.layout(g)))
        els, h, (ax, ay) = b
        lt = t - g["t0"]
        out = clamp((g["t1"] + self.out_sec - t) / self.out_sec)
        out_e = ease_out_cubic(out)
        dx_out = -36 * self.s * (1 - out_e)
        fps_seed = int(t * 30)
        rows = [e.data for e in els if isinstance(e.data, tuple) and e.data and e.data[0] == "row"]
        cur_row = max([r[1] for r in rows if r[2] <= lt] or [-1]) if any(r[2] > 0 for r in rows) else -1
        for e in els:
            p = clamp((lt - e.delay) / self.in_sec)
            if p <= 0:
                continue
            pe = ease_out_cubic(p)
            op = min(1.0, p * 1.6) * out_e
            if cur_row >= 0 and isinstance(e.data, tuple) and e.data and e.data[0] == "row" and e.data[1] < cur_row:
                op *= 0.5  # 已讲过的条目变暗，突出正在讲的这一条
            x, y = ax + e.x + dx_out, ay + e.y
            if e.kind == "bar":
                hh = max(1, int(e.h * pe))
                acc = G.hex2rgb(self.accent)[::-1]
                x0, y0 = int(x), int(y)
                roi = frame[max(0, y0): max(0, y0 + hh), max(0, x0): max(0, x0 + e.w)]
                if roi.size:
                    roi[:] = (roi * (1 - op) + np.array(acc, np.float32) * op).astype(np.uint8)
            elif e.kind == "line":
                ww = max(1, int(e.w * ease_out_cubic(clamp((lt - e.delay) / 0.6))))
                acc = G.hex2rgb(self.accent)[::-1]
                roi = frame[int(y): int(y) + e.h, int(x): int(x) + ww]
                if roi.size:
                    roi[:] = (roi * (1 - op) + np.array(acc, np.float32) * op).astype(np.uint8)
            elif e.kind == "decode":
                text, role, size, col, sp = e.data
                pd = clamp((lt - e.delay) / 0.55)
                shown = text if pd >= 1 else self._decode_text(text, pd, fps_seed // 2 + len(text))
                L = self._t(shown, role, size, col, sp) if pd >= 1 else \
                    text_layer(shown, _font(self.st, "label" if any(ord(c) > 0x2FFF for c in shown) else role, int(size)),
                               fill=col, shadow=(0, 3, 8, 0.55), spacing=sp)
                L.draw(frame, x, y, op)
            elif e.kind == "count":
                val, role, size, col = e.data
                pc = ease_out_cubic(clamp((lt - e.delay) / 0.8))
                txt = self._num_text(val, pc)
                L = e.layer if txt == val else self._t(txt, role, size, col, 0, shadow=(0, 8, 18, 0.6))
                L.draw(frame, x, y + 26 * self.s * (1 - pe), op)
            elif e.kind == "pop":
                sc = 1.0 + 0.16 * (1 - ease_out_back(p))
                L = e.layer
                cx, cy = x + self._vw(L) / 2, y + self._vh(L) / 2
                L.draw_center(frame, cx, cy, op, sc) if abs(sc - 1) > 0.004 else L.draw(frame, x, y, op)
            elif e.kind == "rise":
                e.layer.draw(frame, x, y + 26 * self.s * (1 - pe), op)
            else:  # slide
                e.layer.draw(frame, x - 40 * self.s * (1 - pe), y, op)

    def busy(self, t0, t1):
        """[t0, t1) 是否和信息层重叠（给素材小窗让位）。"""
        return any(g["t0"] - 0.5 < t1 and t0 < g["t1"] + 0.5 for g in self.items)
