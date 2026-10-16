"""Motion 动效层（AI博主 / 科技类风格）：
- 全屏动态信息卡：标题卡、数字滚动卡、章节列表卡、关键词卡、片头卡
- A-roll 上的巨字动效（kinetic typography）
- 电影感调色（对比、降饱和、暗角、颗粒）
所有元素都是预渲染图层 + 逐帧变换（位移/透明度/缩放/裁切），渲染速度与普通包装层相当。
坐标全部按画面比例计算，横屏、竖屏通用。"""
import math

import cv2
import numpy as np

from . import graphics as G
from .graphics import Layer, font, text_layer
from .utils import clamp, ease_out_back, ease_out_cubic, lerp


def _ease_out_expo(p):
    p = clamp(p)
    return 1.0 if p >= 1 else 1 - 2 ** (-10 * p)


def _hex(c):
    r, g, b = G.hex2rgb(c)
    return np.array([b, g, r], np.float32)


class Motion:
    def __init__(self, st, W, H, fps=30):
        self.st, self.W, self.H, self.fps = st, W, H, fps
        self.S = min(W, H)
        self.land = W > H
        m = st.get("motion", {})
        self.cfg = m
        self.c_bg0 = m.get("bg_top", "#0e0e13")
        self.c_bg1 = m.get("bg_bottom", "#07070a")
        self.c_glow = m.get("glow", "#b0179e")
        self.c_glow2 = m.get("glow2", "#3b1d8f")
        self.c_text = m.get("text", "#f4f4f6")
        self.c_dim = m.get("dim", "#8d8d9a")
        self.c_faint = m.get("faint", "#5a5a66")
        self.grad = m.get("accent_grad", ["#ff3d9a", "#a23cff", "#4f6bff"])
        self.mx = int(W * (0.085 if self.land else 0.08))
        self._bgs = None
        self._cache = {}
        self._bar = {}
        self._vig = None
        self._grain = None

    # ------------------------------------------------------------------ 背景
    def _make_bgs(self):
        W, H, S = self.W, self.H, self.S
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        t = (yy / H)[..., None]
        img = _hex(self.c_bg0) * (1 - t) + _hex(self.c_bg1) * t

        def glow(cx, cy, r, col, a):
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
            return (np.clip(1 - d, 0, 1) ** 2.2 * a)[..., None] * _hex(col)

        img = img + glow(W * 0.12, H * 1.02, S * 0.95, self.c_glow, 0.30)
        img = img + glow(W * 0.98, H * -0.05, S * 0.85, self.c_glow2, 0.22)
        # 暗角
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        img = img * (1 - 0.35 * np.clip(d - 0.55, 0, 1))[..., None]
        base = np.clip(img, 0, 255)
        rng = np.random.RandomState(7)
        out = []
        for _ in range(3):  # 三张颗粒交替，营造胶片感
            n = rng.normal(0, 2.6, (H // 2, W // 2, 1)).astype(np.float32)
            n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
            out.append(np.clip(base + n, 0, 255).astype(np.uint8))
        return out

    def background(self, t):
        if self._bgs is None:
            self._bgs = self._make_bgs()
        return self._bgs[int(t * 12) % 3].copy()

    # ------------------------------------------------------------------ 调色（A-roll）
    def grade(self, frame, t):
        """电影感调色：对比/压黑 → 降饱和 → 暗部偏色 → 暗角 → 颗粒。全部用 uint8 查表与 SIMD 运算，开销很小。"""
        g = self.st.get("grade") or {}
        if not g:
            return frame
        if self._vig is None:
            H, W = frame.shape[:2]
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
            v = 1 - g.get("vignette", 0.3) * np.clip((d - 0.5) / 0.9, 0, 1) ** 1.5
            self._vig = cv2.merge([(v * 255).astype(np.uint8)] * 3)
            c, lift = g.get("contrast", 1.06), g.get("lift", -4)
            x = np.arange(256, dtype=np.float32)
            self._lut = np.clip((x - 128) * c + 128 + lift, 0, 255).astype(np.uint8)
            self._tint_luts = None
            if g.get("shadow_tint"):
                col = _hex(g["shadow_tint"]) * g.get("shadow_tint_amount", 0.5)
                w = (1 - x / 255.0) ** 2
                self._tint_luts = [np.clip(w * col[k], 0, 255).astype(np.uint8) for k in range(3)]
            rng = np.random.RandomState(3)
            amt = g.get("grain", 3.0)
            self._grain = []
            for _ in range(3):
                n = cv2.resize(rng.normal(0, amt, (H // 2, W // 2)).astype(np.float32), (W, H))
                pos = cv2.merge([np.clip(n, 0, 255).astype(np.uint8)] * 3)
                neg = cv2.merge([np.clip(-n, 0, 255).astype(np.uint8)] * 3)
                self._grain.append((pos, neg))
        out = cv2.LUT(frame, self._lut)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        sat = g.get("saturation", 0.9)
        if abs(sat - 1) > 0.01:
            out = cv2.addWeighted(out, sat, cv2.merge([gray, gray, gray]), 1 - sat, 0)
        if self._tint_luts:
            out = cv2.add(out, cv2.merge([cv2.LUT(gray, lt) for lt in self._tint_luts]))
        out = cv2.multiply(out, self._vig, scale=1 / 255.0)
        pos, neg = self._grain[int(t * 12) % 3]
        out = cv2.subtract(cv2.add(out, pos), neg)
        return out

    # ------------------------------------------------------------------ 文字工具
    def _f(self, role, size):
        return font(self.st, role, int(size))

    def _txt(self, text, role, size, fill, shadow=None, spacing=0):
        key = ("t", text, role, int(size), fill, shadow, spacing)
        L = self._cache.get(key)
        if L is None:
            if len(self._cache) > 900:
                self._cache.clear()
            L = self._cache[key] = text_layer(text, self._f(role, size), fill=fill, shadow=shadow, spacing=spacing)
        return L

    def _chars(self, text, role, size, fill, shadow=None, spacing=0):
        """逐字图层 + 每个字的 x 偏移（用于逐字入场）。"""
        key = ("c", text, role, int(size), fill, shadow, spacing)
        v = self._cache.get(key)
        if v is None:
            f = self._f(role, size)
            xs, x = [], 0.0
            Ls = []
            for ch in text:
                Ls.append(text_layer(ch, f, fill=fill, shadow=shadow) if ch.strip() else None)
                xs.append(x)
                x += f.getlength(ch) + spacing
            v = self._cache[key] = (Ls, xs, x - spacing, f)
        return v

    @staticmethod
    def _vis_h(L):
        return L.h - 2 * L.oy

    def _fit(self, text, role, size, max_w, min_size):
        f = self._f(role, size)
        while f.getlength(text) > max_w and size > min_size:
            size -= 2
            f = self._f(role, size)
        return size

    def _draw_chars(self, frame, text, role, size, x, y, lt, delay=0.0, stagger=0.035, rise=None, fill=None,
                    opacity=1.0, shadow=None, spacing=0, dur_in=0.32):
        """逐字上浮淡入。(x, y) 为第一行左上角。返回文字宽度。"""
        fill = fill or self.c_text
        Ls, xs, tw, f = self._chars(text, role, size, fill, shadow, spacing)
        rise = self.S * 0.03 if rise is None else rise
        for L, cx, k in zip(Ls, xs, range(len(Ls))):
            if L is None:
                continue
            q = (lt - delay - k * stagger) / dur_in
            if q <= 0:
                continue
            e = ease_out_cubic(q)
            L.draw(frame, x + cx, y + rise * (1 - e), opacity * clamp(q * 1.6))
        return tw

    def _type_text(self, frame, text, role, size, x, y, lt, delay=0.0, cps=40.0, fill=None, opacity=1.0, spacing=2):
        """打字机：逐字出现 + 闪烁光标。"""
        n = int(max(0.0, lt - delay) * cps)
        if n <= 0:
            return
        shown = text[:n]
        L = self._txt(shown, role, size, fill or self.c_dim, None, spacing)
        L.draw(frame, x, y, opacity)
        if n < len(text) or int(lt * 3) % 2 == 0:
            cx = x + L.w - 2 * L.ox + 4
            cv2.rectangle(frame, (int(cx), int(y + size * 0.1)), (int(cx + max(2, size * 0.12)), int(y + size * 1.05)),
                          tuple(int(c * opacity) for c in _hex(self.c_dim)), -1)

    def _bar_img(self, w, h):
        key = (int(w), int(h))
        b = self._bar.get(key)
        if b is None:
            cols = [_hex(c) for c in self.grad]
            t = np.linspace(0, 1, int(w), dtype=np.float32)
            seg = len(cols) - 1
            row = np.zeros((int(w), 3), np.float32)
            for i, tv in enumerate(t):
                p = tv * seg
                k = min(seg - 1, int(p))
                row[i] = cols[k] * (1 - (p - k)) + cols[k + 1] * (p - k)
            b = self._bar[key] = np.repeat(row[None], int(h), 0).astype(np.uint8)
        return b

    def _draw_bar(self, frame, x, y, w, h, p, opacity=1.0, align="left"):
        """渐变强调线，p=0..1 为生长进度。"""
        p = clamp(p)
        if p <= 0.002:
            return
        bar = self._bar_img(w, h)
        ww = max(1, int(w * p))
        x0 = int(x) if align == "left" else int(x + w - ww)
        src = bar[:, :ww] if align == "left" else bar[:, -ww:]
        H, W = frame.shape[:2]
        y0 = int(y)
        if y0 < 0 or y0 + h > H or x0 < 0 or x0 + ww > W:
            return
        roi = frame[y0:y0 + int(h), x0:x0 + ww]
        frame[y0:y0 + int(h), x0:x0 + ww] = cv2.addWeighted(roi, 1 - opacity, src, opacity, 0)
        # 线头光点
        if 0.02 < p < 0.995:
            cx = x0 + ww if align == "left" else x0
            cv2.circle(frame, (int(cx), int(y0 + h / 2)), max(2, int(h * 1.1)), (255, 255, 255), -1, cv2.LINE_AA)

    def _hud(self, frame, card, lt, t_out):
        """四角的小号等宽字 HUD，增加"界面感"。"""
        sz = max(13, int(self.S * 0.0165))
        col = self.c_faint
        op = clamp(lt / 0.3) * 0.9
        top = int(self.H * 0.045)
        left = self._txt(card.get("hud_l", ""), "mono", sz, col, None, 2)
        left.draw(frame, self.mx, top, op)
        mm, ss = int(t_out // 60), int(t_out % 60)
        right = self._txt(f"{card.get('hud_r', 'REC')}  {mm:02d}:{ss:02d}", "mono", sz, col, None, 2)
        right.draw(frame, self.W - self.mx - (right.w - 2 * right.ox), top, op)
        # 细分隔线
        yl = int(top + sz * 2.0)
        w = int((self.W - 2 * self.mx) * ease_out_cubic(lt / 0.6))
        if w > 2:
            c = tuple(int(v) for v in _hex("#24242c"))
            cv2.line(frame, (self.mx, yl), (self.mx + w, yl), c, 1, cv2.LINE_AA)

    def _giant(self, frame, text, lt, x_right=True, y_frac=0.2, alpha=0.06):
        size = int(self.S * (0.36 if self.land else 0.42))
        L = self._txt(text, "num", size, "#ffffff")
        e = ease_out_cubic(lt / 0.8)
        vw = L.w - 2 * L.ox
        x = (self.W - self.mx * 0.6 - vw) if x_right else self.mx * 0.6
        x += self.W * 0.035 * (1 - e) - lt * self.S * 0.006  # 入场 + 缓慢漂移
        L.draw(frame, x, self.H * y_frac, alpha * e)

    # ------------------------------------------------------------------ 卡片
    def _particles(self, frame, t):
        """缓慢上浮的细小光点，让静态背景"活"起来。"""
        if not self.cfg.get("particles", True):
            return
        if not hasattr(self, "_pts"):
            rng = np.random.RandomState(11)
            n = 46
            self._pts = np.stack([rng.rand(n), rng.rand(n), rng.uniform(0.004, 0.02, n),
                                  rng.uniform(0.6, 2.2, n), rng.uniform(0.15, 0.55, n), rng.rand(n) * 6.28], 1)
        W, H = self.W, self.H
        for x, y, v, r, a, ph in self._pts:
            yy = (y - v * t) % 1.0
            xx = x + 0.006 * math.sin(t * 0.7 + ph)
            al = a * (0.6 + 0.4 * math.sin(t * 1.3 + ph))
            c = int(255 * al * 0.55)
            cv2.circle(frame, (int(xx * W), int(yy * H)), max(1, int(r * self.S / 1080)), (c, int(c * 0.85), c), -1,
                       cv2.LINE_AA)

    def render_card(self, card, lt, dur, t_out):
        frame = self.background(t_out)
        self._particles(frame, t_out)
        typ = card.get("type", "headline")
        getattr(self, "_card_" + typ, self._card_headline)(frame, card, lt, dur)
        self._hud(frame, card, lt, t_out)
        return frame

    def _card_headline(self, frame, c, lt, dur):
        S, W, H, mx = self.S, self.W, self.H, self.mx
        self._giant(frame, c.get("idx", "01"), lt, True, 0.18 if self.land else 0.14)
        tag = c.get("tag", "")
        ty = H * (0.36 if self.land else 0.4)
        tsz = max(14, int(S * 0.019))
        self._type_text(frame, tag, "mono", tsz, mx, ty, lt, 0.05, 45)
        text = c.get("text", "")
        size = self._fit(text, "heavy", S * (0.088 if self.land else 0.1), W - 2 * mx - S * 0.05, S * 0.05)
        hy = ty + tsz * 2.2
        tw = self._draw_chars(frame, text, "heavy", size, mx, hy, lt, 0.12, 0.04, spacing=int(size * 0.02))
        by = hy + size * 1.32
        self._draw_bar(frame, mx, by, max(S * 0.12, min(tw, S * 0.22)), max(3, int(S * 0.005)),
                       ease_out_expo_p(lt, 0.35, 0.55))
        sub = c.get("sub", "")
        if sub:
            ssz = int(S * (0.03 if self.land else 0.034))
            ssz = self._fit(sub, "medium", ssz, W - 2 * mx, S * 0.022)
            self._draw_chars(frame, sub, "medium", ssz, mx, by + S * 0.04, lt, 0.45, 0.012, rise=S * 0.012,
                             fill=self.c_dim, spacing=int(ssz * 0.06))

    def _card_number(self, frame, c, lt, dur):
        S, W, H, mx = self.S, self.W, self.H, self.mx
        right = bool(c.get("right"))
        self._giant(frame, c.get("idx", "01"), lt, not right, 0.16 if self.land else 0.12, 0.045)
        val, dec = float(c.get("value", 0)), int(c.get("decimals", 0))
        e = _ease_out_expo((lt - 0.15) / 1.1)
        cur = val * e
        s = f"{cur:.{dec}f}" if dec else f"{int(round(cur))}"
        nsz = int(S * (0.2 if self.land else 0.22))
        unit, plus = c.get("unit", ""), c.get("plus", False)
        NL = self._txt(s, "num", nsz, self.c_text, (0, 6, 16, 0.55))
        # 以最终数字宽度定位，滚动时不跳动
        final = f"{val:.{dec}f}" if dec else f"{int(round(val))}"
        fw = self._f("num", nsz).getlength(final)
        usz = int(nsz * 0.38)
        uw = self._f("heavy", usz).getlength(unit) if unit else 0
        pw = nsz * 0.32 if plus else 0
        total = fw + (S * 0.012 + uw if unit else 0) + pw
        x0 = (W - mx - total) if right else mx
        cy = H * (0.47 if self.land else 0.46)
        op = clamp((lt - 0.1) / 0.25)
        y_num = cy - nsz * 0.62
        NL.draw(frame, x0 + (fw - (NL.w - 2 * NL.ox)), y_num + S * 0.02 * (1 - op), op)
        xx = x0 + fw + S * 0.012
        if plus:
            P = self._txt("+", "num", int(nsz * 0.5), self.c_text)
            P.draw(frame, xx - S * 0.004, y_num + nsz * 0.48, op)
            xx += pw
        if unit:
            U = self._txt(unit, "heavy", usz, self.c_text)
            uq = clamp((lt - 0.9) / 0.3)
            U.draw(frame, xx, y_num + nsz * 0.9 - usz * 1.05 + S * 0.02 * (1 - uq), uq)
        tsz = max(14, int(S * 0.019))
        tag = c.get("tag", "")
        tag_w = self._f("mono", tsz).getlength(tag) + 2 * len(tag)
        self._type_text(frame, tag, "mono", tsz, (W - mx - tag_w) if right else mx, y_num - tsz * 1.2, lt, 0.05, 45)
        by = y_num + nsz * 1.2
        bw = S * 0.16
        self._draw_bar(frame, (W - mx - bw) if right else mx, by, bw, max(3, int(S * 0.005)),
                       ease_out_expo_p(lt, 0.3, 0.9), align="right" if right else "left")
        cap = c.get("caption", "")
        if cap:
            csz = self._fit(cap, "medium", int(S * 0.03), W - 2 * mx, S * 0.02)
            cw = self._f("medium", csz).getlength(cap) + len(cap) * int(csz * 0.06)
            self._draw_chars(frame, cap, "medium", csz, (W - mx - cw) if right else mx, by + S * 0.035, lt, 0.5, 0.012,
                             rise=S * 0.012, fill=self.c_dim, spacing=int(csz * 0.06))

    def _card_list(self, frame, c, lt, dur):
        S, W, H, mx = self.S, self.W, self.H, self.mx
        items = c.get("items", [])
        act = int(c.get("active", 0))
        self._giant(frame, f"{act + 1:02d}", lt, True, 0.22 if self.land else 0.6, 0.06)
        n = len(items)
        lo = max(0, min(act - 2, n - 6))
        vis = list(range(lo, min(n, lo + 6)))
        step = S * (0.095 if self.land else 0.1)
        y0 = H * 0.5 - step * len(vis) / 2 + step * 0.1
        tsz = max(14, int(S * 0.019))
        self._type_text(frame, c.get("tag", "CHAPTERS"), "mono", tsz, mx, y0 - step * 0.8, lt, 0.0, 45)
        for r, i in enumerate(vis):
            d = 0.08 + r * 0.07
            q = ease_out_cubic((lt - d) / 0.4)
            if q <= 0:
                continue
            on = i == act
            y = y0 + r * step
            x = mx + S * 0.03 * (1 - q)
            nsz = int(S * 0.026)
            NL = self._txt(f"{i + 1:02d}", "num", nsz, self.c_text if on else self.c_faint)
            NL.draw(frame, x, y + S * 0.012, q * (1 if on else 0.8))
            lsz = int(S * (0.056 if on else 0.04))
            col = self.c_text if on else self.c_dim
            L = self._txt(items[i], "heavy" if on else "medium", lsz, col)
            L.draw(frame, x + S * 0.06, y + (S * 0.044 - lsz) * 0.6, q * (1 if on else 0.55))
            if on:
                self._draw_bar(frame, mx - S * 0.025, y + S * 0.005, max(3, int(S * 0.004)), int(S * 0.05),
                               1.0, q)
                bw = (L.w - 2 * L.ox)
                self._draw_bar(frame, x + S * 0.06, y + lsz * 1.25, bw, max(2, int(S * 0.003)),
                               ease_out_expo_p(lt, d + 0.25, 0.6))

    def _card_keywords(self, frame, c, lt, dur):
        S, W, H, mx = self.S, self.W, self.H, self.mx
        words = c.get("words", [])[:3]
        self._giant(frame, c.get("idx", "01"), lt, True, 0.14 if self.land else 0.1, 0.045)
        size = int(S * (0.088 if self.land else 0.095))
        gap = S * 0.07
        f = self._f("heavy", size)
        ws = [f.getlength(w) for w in words]
        horizontal = sum(ws) + gap * (len(words) - 1) <= W - 2 * mx
        tsz = max(14, int(S * 0.019))
        if horizontal:
            x = (W - (sum(ws) + gap * (len(words) - 1))) / 2
            y = H * 0.44
            tag = c.get("tag", "")
            tw = self._f("mono", tsz).getlength(tag) + 2 * len(tag)
            self._type_text(frame, tag, "mono", tsz, (W - tw) / 2, y - S * 0.07, lt, 0.0, 45)
            for k, (w, wl) in enumerate(zip(words, ws)):
                d = 0.12 + k * 0.22
                self._draw_chars(frame, w, "heavy", size, x, y, lt, d, 0.05)
                q = clamp((lt - d - 0.1) / 0.3)
                L = self._txt(f"0{k + 1}", "mono", tsz, self.c_faint)
                L.draw(frame, x, y + size * 1.3, q)
                if k == len(words) - 1:
                    self._draw_bar(frame, x, y + size * 1.22, wl, max(3, int(S * 0.005)), ease_out_expo_p(lt, d + 0.3, 0.5))
                elif k < len(words) - 1:
                    cx = x + wl + gap / 2
                    cv2.circle(frame, (int(cx), int(y + size * 0.62)), max(2, int(S * 0.004)),
                               tuple(int(v * q) for v in _hex(self.c_faint)), -1, cv2.LINE_AA)
                x += wl + gap
        else:
            y = H * 0.36
            self._type_text(frame, c.get("tag", ""), "mono", tsz, mx, y - S * 0.06, lt, 0.0, 45)
            for k, (w, wl) in enumerate(zip(words, ws)):
                d = 0.12 + k * 0.22
                L = self._txt(f"0{k + 1}", "mono", tsz, self.c_faint)
                L.draw(frame, mx, y + size * 0.35, clamp((lt - d) / 0.3))
                self._draw_chars(frame, w, "heavy", size, mx + S * 0.07, y, lt, d, 0.05)
                if k == len(words) - 1:
                    self._draw_bar(frame, mx + S * 0.07, y + size * 1.25, wl, max(3, int(S * 0.005)),
                                   ease_out_expo_p(lt, d + 0.3, 0.5))
                y += size * 1.7

    def _card_title(self, frame, c, lt, dur):
        S, W, H, mx = self.S, self.W, self.H, self.mx
        self._giant(frame, c.get("idx", "01"), lt, True, 0.2 if self.land else 0.12, 0.07)
        tsz = max(14, int(S * 0.02))
        y = H * (0.38 if self.land else 0.4)
        self._type_text(frame, c.get("tag", ""), "mono", tsz, mx, y, lt, 0.0, 40)
        text = c.get("text", "")
        size = self._fit(text, "heavy", S * (0.11 if self.land else 0.12), W - 2 * mx, S * 0.06)
        y2 = y + tsz * 2.4
        tw = self._draw_chars(frame, text, "heavy", size, mx, y2, lt, 0.15, 0.06, rise=S * 0.05,
                              shadow=(0, 6, 18, 0.5), spacing=int(size * 0.02))
        self._draw_bar(frame, mx, y2 + size * 1.3, max(S * 0.1, tw * 0.35), max(4, int(S * 0.006)),
                       ease_out_expo_p(lt, 0.45, 0.7))
        sub = c.get("sub", "")
        if sub:
            ssz = int(S * 0.03)
            self._draw_chars(frame, sub, "medium", ssz, mx, y2 + size * 1.3 + S * 0.035, lt, 0.6, 0.02,
                             rise=S * 0.01, fill=self.c_dim, spacing=int(ssz * 0.15))

    # ------------------------------------------------------------------ A-roll 巨字
    def kinetic(self, frame, text, lt, dur, slot=0):
        """A-roll 上的大字：逐字砸入（放大→回弹），带柔和投影，结束时上移淡出。"""
        S, W, H = self.S, self.W, self.H
        kp = self.st.get("keyword_pop", {})
        base = S * (0.2 if self.land else 0.21)
        size = int(self._fit(text, "heavy", base, W * 0.9, S * 0.1))
        shadow = (0, int(size * 0.05), int(size * 0.09), 0.6)
        Ls, xs, tw, f = self._chars(text, "heavy", size, kp.get("color", "#ffffff"), shadow, int(size * 0.01))
        if slot % 2 == 0:
            x0, y0 = W * 0.045, H * (0.05 if self.land else 0.12)
        else:
            x0, y0 = W - W * 0.045 - tw, H * (0.05 if self.land else 0.12)
        out = clamp((lt - (dur - 0.22)) / 0.22)
        for k, (L, cx) in enumerate(zip(Ls, xs)):
            if L is None:
                continue
            q = (lt - k * 0.07) / 0.28
            if q <= 0:
                continue
            s = lerp(1.55, 1.0, ease_out_back(clamp(q), 1.8))
            op = clamp(q * 2.5) * (1 - out)
            Lw, Lh = L.w - 2 * L.ox, L.h - 2 * L.oy
            L.draw_center(frame, x0 + cx + Lw / 2, y0 - S * 0.04 * out + Lh / 2, op, s)


def ease_out_expo_p(lt, delay, dur):
    return _ease_out_expo((lt - delay) / dur)
