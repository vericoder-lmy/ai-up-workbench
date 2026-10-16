"""逐帧合成器：按 timeline.json 把源视频帧 + 背景 + 画中画 + 包装层合成为成片帧。
纯 numpy/OpenCV 实现，配合分段多进程渲染与硬件编码，速度远快于浏览器逐帧渲染。"""
import bisect
import math

import cv2
import numpy as np

from . import graphics as G
from .graphics import Layer, font, text_layer
from .textproc import wrap
from .utils import FFReader, clamp, ease_in_out_cubic, ease_out_back, ease_out_cubic, imread, lerp


def _blur_small(img, W, H, sigma=2.2, scale=10):
    s = cv2.resize(img, (max(8, W // scale), max(8, H // scale)), interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), sigma)
    return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)


class Compositor:
    def __init__(self, tl, style):
        self.tl, self.st = tl, style
        self.W, self.H = tl["canvas"]
        self.land = self.W > self.H
        self.fps = tl["fps"]
        self.sw, self.sh = tl["src"]["w"], tl["src"]["h"]
        f = tl["face"]
        self.fx, self.fy = f["cx"] * self.sw, f["cy"] * self.sh
        self.fw = float(f.get("fw", 0.3)) * self.sw  # 脸宽（像素）
        asp = self.W / self.H
        # 竖屏素材出横屏：人物按高度完整放在中间，两侧用模糊背景填充（不放大裁切）
        self.fit = self.sw / self.sh < asp * 0.7
        if self.sw / self.sh > asp:  # 横屏素材：取与画布同比例的竖条
            self.base_ch, self.base_cw = float(self.sh), self.sh * asp
        else:
            self.base_cw, self.base_ch = float(self.sw), self.sw / asp
        # 横屏素材出竖屏时，裁出的竖条里脸已经很大：推镜幅度相应减小，避免把脸裁掉
        ratio = self.fw / max(1.0, self.base_cw)
        self.zoom_k = float(clamp((0.62 - ratio) / 0.25, 0.3, 1.0))
        self.blocks = tl["blocks"]
        self.bstarts = [b["start"] for b in self.blocks]
        self.subs = tl["subs"]
        self.sstarts = [s["start"] for s in self.subs]
        self.zooms = tl["zooms"]
        self.ztimes = [z["t"] for z in self.zooms]
        self.chapters = tl["chapters"]
        self.cstarts = [c["start"] for c in self.chapters]
        brand = tl["brand"]
        self.header = G.build_header(style, brand, [c["label"] for c in self.chapters])
        self.progress = G.Progress(style, [c["label"] for c in self.chapters])
        self._mask_cache = {}
        self._card_cache = {}
        self._sub_cache = {}
        self._tw_cache = {}
        self._pop_cache = {}
        self._media = {}
        self.motion = None
        if style.get("motion") or style.get("grade") or any(b["mode"] == "card" for b in self.blocks) \
                or style.get("keyword_pop", {}).get("kind") == "kinetic":
            from .motion import Motion
            self.motion = Motion(style, self.W, self.H, self.fps)
        self.hud = None
        if (style.get("hud") or {}).get("enable") and tl.get("graphics"):
            from .hud import HUD
            self.hud = HUD(tl, style, self.W, self.H)
        self.vtitle = None
        if tl.get("title_fx"):  # 手动设置的视频标题
            from .titlefx import VideoTitle
            self.vtitle = VideoTitle(tl["title_fx"], self.W, self.H)
        self.law = None
        if tl.get("law"):
            from .law import LawFX
            self.law = LawFX(tl, style, self.W, self.H)
        self._build_static()

    # ---------- 静态元素 ----------
    def _build_static(self):
        st = self.st
        dc = st["disclaimer"]
        fd = font(st, "hand", dc["size"])
        self.disc = [text_layer(t, fd, fill=dc["color"], stroke=dc["stroke"], stroke_color=dc["stroke_color"],
                                shadow=tuple(dc["shadow"]), skew=dc["skew"], spacing=6)
                     for t in self.tl["brand"].get("disclaimer", [])[:2]]
        nc = self.tl.get("namecard")
        self.nc_layers = []
        if nc:
            c = st["namecard"]
            name = nc["name"]
            fl = font(st, "title", c["name_size"])
            self.nc_layers.append(text_layer(name, fl, fill=c["color"], stroke=c["stroke"],
                                             stroke_color=c["stroke_color"], shadow=(4, 6, 6, 0.6), spacing=10, bold=1))
            fh = font(st, "hand", c["line_size"])
            for line in nc["lines"][:6]:
                self.nc_layers.append(text_layer(line, fh, fill=c["color"], stroke=4, stroke_color=c["stroke_color"],
                                                 shadow=(3, 4, 4, 0.6)))
        p = st["panel"]
        m, b = G.rrect_mask(p["w"], p["h"], p["radius"], 0, 3)
        # 面板背后本身就是模糊背景，无需再模糊：预合成为常量图层（白色半透明 + 白边）
        a = m * p["tint_alpha"]
        a = a * (1 - b * p["border_alpha"]) + b * p["border_alpha"]
        col = np.array(G.hex2rgb(p["tint"])[::-1], np.float32)
        self.panel_layer = Layer(np.ascontiguousarray(a[..., None] * col), np.ascontiguousarray(a[..., None]))

    # ---------- 形状 ----------
    def _shape(self, block, t):
        mode = block["mode"]
        if mode == "full" and self.fit:
            z = self._zoom(t)
            cw, ch = self.sw / z, self.sh / z
            cx = clamp(self.fx, cw / 2, self.sw - cw / 2)
            cy = clamp(self.fy, ch / 2, self.sh - ch / 2)
            h = float(self.H)
            w = h * self.sw / self.sh
            return dict(x=(self.W - w) / 2, y=0.0, w=w, h=h, r=0.0, b=0.0, f=0.0,
                        cx=cx, cy=cy, cw=cw, ch=ch, full=0.0)
        if mode == "full":
            z = self._zoom(t)
            # 按画布比例裁切（兼容横屏/方形素材），以人脸为中心
            bw, bh = self.base_cw, self.base_ch
            cw, ch = bw / z, bh / z
            bx = clamp(self.fx, bw / 2, self.sw - bw / 2)
            by = clamp(self.fy, bh / 2, self.sh - bh / 2)
            cx = self.fx + (bx - self.fx) / z
            cy = self.fy + (by - self.fy) / z
            cx = clamp(cx, cw / 2, self.sw - cw / 2)
            cy = clamp(cy, ch / 2, self.sh - ch / 2)
            return dict(x=0.0, y=0.0, w=float(self.W), h=float(self.H), r=0.0, b=0.0, f=0.0,
                        cx=cx, cy=cy, cw=cw, ch=ch, full=1.0)
        m = self.st["modes"][mode]
        w, h = float(m["w"]), float(m["h"])
        cw = self.base_cw * m.get("crop_w", 0.9)
        # 画中画至少框住头肩（约 2 倍脸宽），横屏素材的竖条很窄，不加这条会只剩半张脸
        cw = min(float(self.sw), max(cw, self.fw * 2.0))
        ch = cw * h / w
        if ch > self.sh:
            ch = self.sh
            cw = ch * w / h
        cx = clamp(self.fx, cw / 2, self.sw - cw / 2)
        cy = clamp(self.fy + ch * 0.17, ch / 2, self.sh - ch / 2)
        return dict(x=float(m["x"]), y=float(m["y"]), w=w, h=h, r=float(m["radius"]), b=float(m["border"]),
                    f=float(m["feather"]), cx=cx, cy=cy, cw=cw, ch=ch, full=0.0)

    def _zoom(self, t):
        k = bisect.bisect_right(self.ztimes, t) - 1
        if k < 0:
            return 1.0
        z1 = self.zooms[k]["z"]
        if self.zooms[k].get("cut"):  # 换机位式的硬切：不做缓动
            return 1.0 + (z1 - 1.0) * self.zoom_k
        z0 = self.zooms[k - 1]["z"] if k > 0 else 1.0
        p = ease_out_cubic((t - self.zooms[k]["t"]) / self.st["pacing"]["zoom_ease_sec"])
        return 1.0 + (lerp(z0, z1, p) - 1.0) * self.zoom_k

    @staticmethod
    def _mix(a, b, p):
        return {k: lerp(a[k], b[k], p) for k in a}

    # ---------- 背景 ----------
    def _card(self, label, seed):
        key = (label, seed)
        if key not in self._card_cache:
            kb = 1 + self.st["broll"]["kenburns"] + 0.02
            self._card_cache[key] = G.make_card(self.st, label, int(self.W * kb), int(self.H * kb), seed)
        return self._card_cache[key]

    def _media_frame(self, path, tt):
        kb = 1 + self.st["broll"]["kenburns"] + 0.02
        tw, th = int(self.W * kb), int(self.H * kb)
        if path.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi")):
            rd = self._media.get(path)
            if rd is None:
                rd = self._media[path] = FFReader(path, self.fps, (tw, th))
            return rd.get(max(0, int(tt * self.fps)))
        img = self._media.get(path)
        if img is None:
            src = imread(path)
            if src is None:
                return None
            s = max(tw / src.shape[1], th / src.shape[0])
            r = cv2.resize(src, (int(src.shape[1] * s) + 1, int(src.shape[0] * s) + 1), interpolation=cv2.INTER_AREA)
            y0, x0 = (r.shape[0] - th) // 2, (r.shape[1] - tw) // 2
            img = self._media[path] = np.ascontiguousarray(r[y0: y0 + th, x0: x0 + tw])
        return img

    def _background(self, bg, block, t, src):
        W, H = self.W, self.H
        if bg is None:
            bg = {"type": "blur"}
        if bg["type"] == "blur":
            x0 = int(clamp(self.fx - self.base_cw / 2, 0, self.sw - self.base_cw))
            y0 = int(clamp(self.fy - self.base_ch / 2, 0, self.sh - self.base_ch))
            img = _blur_small(src[y0: y0 + int(self.base_ch), x0: x0 + int(self.base_cw)], W, H)
            if bg.get("light"):
                img = cv2.addWeighted(img, 0.72, img, 0, 255 * 0.28)
            if bg.get("dim"):
                img = cv2.convertScaleAbs(img, alpha=bg["dim"])
            return img
        if bg["type"] == "media":
            base = self._media_frame(bg["path"], t - block["start"])
            if base is None:
                base = self._card("", 0)
        else:
            base = self._card(bg.get("label", ""), int(bg.get("seed", 0)))
        # Ken Burns 缓慢推近
        dur = max(1.0, block["end"] - block["start"])
        p = clamp((t - block["start"]) / dur)
        kb = self.st["broll"]["kenburns"]
        s = 1 + kb * p
        bh, bw = base.shape[:2]
        cw, ch = bw / s, bh / s
        x0 = (bw - cw) / 2 + (bw - W * 1.0) * 0.1 * (p - 0.5)
        y0 = (bh - ch) / 2
        M = np.float32([[W / cw, 0, -x0 * W / cw], [0, H / ch, -y0 * H / ch]])
        img = cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if bg.get("dark"):
            img = cv2.convertScaleAbs(img, alpha=self.st["broll"]["dark"] if bg["type"] == "media" else 0.85)
        return img

    def _bg_of(self, b):
        if b["mode"] == "full":
            return {"type": "blur", "dim": 0.62}
        return b["bg"]

    # ---------- 画中画 ----------
    def _mask(self, w, h, r, f, b):
        key = (int(w), int(h), int(r), int(f), int(b))
        m = self._mask_cache.get(key)
        if m is None:
            if len(self._mask_cache) > 64:
                self._mask_cache.clear()
            m = self._mask_cache[key] = G.rrect_mask(w, h, r, f, b)
        return m

    def _draw_pip(self, frame, src, s, border_color):
        w, h = int(round(s["w"])), int(round(s["h"]))
        if w < 4 or h < 4:
            return
        sx, sy = w / s["cw"], h / s["ch"]
        M = np.float32([[sx, 0, -(s["cx"] - s["cw"] / 2) * sx], [0, sy, -(s["cy"] - s["ch"] / 2) * sy]])
        pip = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a, bm = self._mask(w, h, s["r"], s["f"], s["b"])
        x, y = int(round(s["x"])), int(round(s["y"]))
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        px0, py0 = x0 - x, y0 - y
        sl = (slice(py0, py0 + y1 - y0), slice(px0, px0 + x1 - x0))
        pa = a[sl][..., None]
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        out = roi * (1 - pa) + pip[sl].astype(np.float32) * pa
        if bm is not None:
            bb = bm[sl][..., None]
            out = out * (1 - bb) + np.array(G.hex2rgb(border_color)[::-1], np.float32) * bb
        frame[y0:y1, x0:x1] = out.astype(np.uint8)

    # ---------- 磨砂 ----------
    def _tint_layer(self, w, h, r, tint, alpha, border_alpha):
        key = ("tint", int(w), int(h), int(r), tint, alpha, border_alpha)
        L = self._mask_cache.get(key)
        if L is None:
            m, b = G.rrect_mask(w, h, r, 0, 3)
            a = m * alpha
            a = a * (1 - b * border_alpha) + b * border_alpha
            col = np.array(G.hex2rgb(tint)[::-1], np.float32)
            L = self._mask_cache[key] = Layer(np.ascontiguousarray(a[..., None] * col), np.ascontiguousarray(a[..., None]))
        return L

    def _frost(self, frame, x, y, w, h, mask, border, tint, alpha, border_alpha, blur=12):
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0 or alpha <= 0:
            return
        roi = frame[y0:y1, x0:x1]
        bl = _blur_small(roi, x1 - x0, y1 - y0, sigma=blur / 6, scale=6).astype(np.float32)
        tc = np.array(G.hex2rgb(tint)[::-1], np.float32)
        fro = bl * (1 - alpha) + tc * alpha
        sl = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
        m = mask[sl][..., None]
        out = roi.astype(np.float32) * (1 - m) + fro * m
        if border is not None and border_alpha > 0:
            bb = border[sl][..., None] * border_alpha
            out = out * (1 - bb) + 255 * bb
        frame[y0:y1, x0:x1] = out.astype(np.uint8)

    # ---------- 字幕 ----------
    def _fit_lines(self, text, role, size, max_w, max_lines=2, min_size=56):
        f = font(self.st, role, size)
        while f.getlength(text) > max_w and size > min_size:
            size -= 4
            f = font(self.st, role, size)
        if f.getlength(text) <= max_w:
            return [text], f
        per = max(2, int(max_w / (size * 1.0)))
        return wrap(text, per)[:max_lines], f

    def _sub_layers(self, idx, style_kind):
        key = (idx, style_kind)
        if key in self._sub_cache:
            return self._sub_cache[key]
        sc = self.st["subtitle"]
        text = self.subs[idx]["text"]
        lines, f = self._fit_lines(text, "sub", sc["size"], sc["max_width"] - (90 if style_kind == "pill" else 0))
        if style_kind == "minimal":  # 极简：无描边、宽字距、柔和投影
            sp = int(sc.get("spacing", 4))
            if sp and len(lines) == 1 and f.getlength(text) + sp * len(text) > sc["max_width"]:
                lines = wrap(text, max(2, (len(text) + 1) // 2))[:2]
            Ls = [text_layer(l, f, fill=sc["color"], stroke=int(sc.get("stroke", 0)), stroke_color=sc["stroke_color"],
                             shadow=tuple(sc["shadow"]), spacing=sp) for l in lines]
        elif style_kind == "pill":
            pc = sc["pill"]
            Ls = [text_layer(l, f, fill=sc["color"], stroke=pc["text_stroke"], stroke_color=pc["text_stroke_color"],
                             shadow=tuple(pc["shadow"])) for l in lines]
        else:
            Ls = [text_layer(l, f, fill=sc["color"], stroke=sc["stroke"], stroke_color=sc["stroke_color"],
                             shadow=tuple(sc["shadow"])) for l in lines]
        if len(self._sub_cache) > 300:
            self._sub_cache.clear()
        self._sub_cache[key] = (Ls, f.size)
        return self._sub_cache[key]

    def _draw_subtitle(self, frame, t, mode, block):
        k = bisect.bisect_right(self.sstarts, t) - 1
        if k < 0 or t >= self.subs[k]["end"]:
            return
        s = self.subs[k]
        sc = self.st["subtitle"]
        kind = self.st["modes"].get(mode, self.st["modes"]["full"])["subtitle"]
        if kind == "typewriter":
            return self._draw_typewriter(frame, t, k)
        Ls, size = self._sub_layers(k, kind)
        p = clamp((t - s["start"]) / sc["pop_in_sec"])
        sc_ = 1.0 if kind == "minimal" else lerp(0.9, 1.0, ease_out_back(p))
        op = ease_out_cubic(p)
        lh = size * (1.45 if kind == "minimal" else 1.22)
        y0 = sc["y"] - lh * len(Ls) / 2
        if kind == "pill":
            pc = sc["pill"]
            vw = max(L.w - 2 * L.ox for L in Ls)
            pw, ph = int(vw + 2 * pc["pad_x"]), int(lh * len(Ls) + 2 * pc["pad_y"])
            px, py = int(self.W / 2 - pw / 2), int(sc["y"] - ph / 2)
            m, b = self._mask(pw, ph, min(pc["radius"], ph / 2), 0, 3)
            self._frost(frame, px, py, pw, ph, m * op, b, pc["tint"], pc["tint_alpha"], pc["border_alpha"] * op, pc["blur"])
        if kind == "minimal" and sc.get("backdrop", 0) > 0 and block.get("mode") != "card":
            # 柔和暗底：字幕落在白衣服、亮背景上也看得清，又不像字幕条那样生硬
            vw = max(L.w - 2 * L.ox for L in Ls)
            pw, ph = int(vw + size * 2.4), int(lh * len(Ls) + size * 1.2)
            key = ("bd", pw, ph)
            BL = self._mask_cache.get(key)
            if BL is None:
                m, _ = G.rrect_mask(pw, ph, ph / 2, ph * 0.5, 0)
                a = np.ascontiguousarray((m * sc["backdrop"])[..., None])
                BL = self._mask_cache[key] = Layer(np.zeros((ph, pw, 3), np.float32), a)
            BL.draw(frame, self.W / 2 - pw / 2, sc["y"] - ph / 2, op)
        for i, L in enumerate(Ls):
            L.draw_center(frame, self.W / 2, y0 + lh * (i + 0.5), op, sc_)
        ec = sc.get("en") or {}
        if ec.get("show") and s.get("en"):
            key = ("en", k)
            EL = self._sub_cache.get(key)
            if EL is None:
                esz = int(size * ec.get("scale", 0.56))
                fe = font(self.st, ec.get("role", "medium"), esz)
                txt = s["en"]
                while fe.getlength(txt) > sc["max_width"] and esz > 16:
                    esz -= 2
                    fe = font(self.st, ec.get("role", "medium"), esz)
                EL = self._sub_cache[key] = text_layer(txt, fe, fill=ec.get("color", sc["color"]),
                                                       shadow=tuple(sc["shadow"]), spacing=int(ec.get("spacing", 1)))
            EL.draw_center(frame, self.W / 2, y0 + lh * len(Ls) + (EL.h - 2 * EL.oy) / 2 + ec.get("gap", 4), op * ec.get("alpha", 0.92))

    def _draw_typewriter(self, frame, t, k):
        s = self.subs[k]
        tw = self.st["typewriter"]
        text = s["text"]
        lines, f = self._fit_lines(text, "sub", tw["size"], tw.get("max_w", 860), 2, min(88, tw["size"]))
        n_total = sum(len(l) for l in lines)
        dur = max(0.3, s["end"] - s["start"])
        cps = max(tw["chars_per_sec"], n_total / (dur * 0.75))
        n = min(n_total, int((t - s["start"]) * cps) + 1)
        key = (k, n)
        if key not in self._tw_cache:
            if len(self._tw_cache) > 400:
                self._tw_cache.clear()
            shown, rem = [], n
            for l in lines:
                shown.append(l[:max(0, rem)])
                rem -= len(l)
            Ls = [text_layer(x, f, fill="#ffffff", stroke=3, stroke_color="#4a4a4a", shadow=(3, 5, 5, 0.55))
                  if x else None for x in shown]
            full_w = max(f.getlength(l) for l in lines)
            self._tw_cache[key] = (Ls, full_w, shown)
        Ls, full_w, shown = self._tw_cache[key]
        lh = f.size * 1.2
        ph = int(lh * len(lines) + 2 * tw["pill_pad_y"])
        pw = int(full_w + 2 * tw["pill_pad_x"])
        cx, cy = tw.get("x", self.W / 2), tw["y"]
        px, py = int(cx - pw / 2), int(cy - ph / 2)
        self._tint_layer(pw, ph, min(60, ph / 2), "#ffffff", 0.22, 0.30).draw(frame, px, py)
        x_left = cx - full_w / 2
        last_end = (x_left, cy - lh * len(lines) / 2 + lh / 2)
        for i, L in enumerate(Ls):
            yc = cy - lh * len(lines) / 2 + lh * (i + 0.5)
            if L is not None:
                L.draw(frame, x_left, yc - (L.h - 2 * L.oy) / 2)
                last_end = (x_left + (L.w - 2 * L.ox), yc)
        if tw.get("cursor") and (n < n_total or int(t * 2.4) % 2 == 0):
            x, yc = last_end
            cv2.rectangle(frame, (int(x + 6), int(yc - f.size * 0.48)), (int(x + 11), int(yc + f.size * 0.5)),
                          (255, 255, 255), -1)

    # ---------- 弹字 ----------
    def _draw_pops(self, frame, t):
        kp = self.st["keyword_pop"]
        for i, p in enumerate(self.tl["pops"]):
            if not (p["t"] <= t < p["t"] + p["dur"]):
                continue
            if kp.get("kind") == "kinetic" and self.motion is not None:
                self.motion.kinetic(frame, p["text"], t - p["t"], p["dur"], slot=i)
                continue
            L = self._pop_cache.get(i)
            if L is None:
                size = kp["size"]
                f = font(self.st, "pop", size)
                while f.getlength(p["text"]) > self.W * 0.9 and size > 90:
                    size -= 8
                    f = font(self.st, "pop", size)
                L = self._pop_cache[i] = text_layer(p["text"], f, fill=kp["color"], stroke=kp["stroke"],
                                                    stroke_color=kp["stroke_color"], shadow=tuple(kp["shadow"]),
                                                    bold=3, spacing=6)
            lt = t - p["t"]
            if lt < kp["in_sec"]:
                q = lt / kp["in_sec"]
                s, op = lerp(1.7, 1.0, ease_out_back(q, 2.2)), ease_out_cubic(q * 1.5)
            elif lt > p["dur"] - kp["out_sec"]:
                q = (lt - (p["dur"] - kp["out_sec"])) / kp["out_sec"]
                s, op = lerp(1.0, 1.08, q), 1 - q
            else:
                s, op = 1.0 + 0.015 * math.sin(lt * 6), 1.0
            L.draw_center(frame, self.W / 2, kp["y"], op, s)

    # ---------- 名片 ----------
    def _draw_namecard(self, frame, t):
        nc = self.tl.get("namecard")
        if not nc or not self.nc_layers:
            return
        lt = t - nc["start"]
        if lt < 0 or lt > nc["dur"]:
            return
        c = self.st["namecard"]
        out = clamp((nc["dur"] - lt) / 0.4)
        y = c["y"]
        for i, L in enumerate(self.nc_layers):
            q = ease_out_cubic((lt - i * 0.1) / 0.4)
            if q <= 0:
                break
            x = c["x"] - 60 * (1 - q)
            L.draw(frame, x, y, q * out)
            y += (L.h - 2 * L.oy) + c["line_gap"]

    # ---------- 头部 ----------
    def _draw_header(self, frame, t):
        hd = self.st["header"]
        if hd.get("show", True):
            self._draw_haze_title(frame)
        if not self.st["progress"].get("show", True) or len(self.chapters) < 2:
            return  # 只有一个章节时进度条没有意义
        self._draw_progress(frame, t)

    def _draw_haze_title(self, frame):
        hd = self.st["header"]
        hh = hd["haze_height"]
        roi = frame[:hh]
        bl = _blur_small(roi, self.W, hh, sigma=hd["haze_blur"] / 8, scale=8).astype(np.float32)
        lin = np.clip(np.linspace(1, 0, hh, dtype=np.float32) * hd.get("haze_plateau", 1.0), 0, 1)
        g = (lin ** 1.4 * hd["haze_alpha_top"])[:, None, None]
        tint = np.array(G.hex2rgb(hd.get("haze_tint", "#ececec"))[::-1], np.float32)
        mix = hd.get("haze_mix", 0.5)
        fro = bl * (1 - mix) + tint * mix
        out = roi.astype(np.float32) * (1 - g) + fro * g
        frame[:hh] = out.astype(np.uint8)
        self.header.draw(frame, 0, 0)

    def _draw_progress(self, frame, t):
        k = max(0, bisect.bisect_right(self.cstarts, t) - 1)
        ch = self.chapters[k]
        frac = clamp((t - ch["start"]) / max(0.1, ch["end"] - ch["start"]))
        x0 = self.progress.node_x(k)
        x1 = self.progress.node_x(k + 1) if k + 1 < len(self.chapters) else self.W - self.st["progress"]["margin"]
        fill = x0 + (x1 - x0) * frac * 0.9
        pulse = clamp(1 - (t - ch["start"]) / self.st["progress"]["anim_sec"]) if k > 0 else 0.0
        if pulse > 0:
            prev = self.progress.node_x(k - 1)
            fill = lerp(prev + (x0 - prev) * 0.9, x0, ease_out_cubic(1 - pulse))
        self.progress.draw(frame, fill, k, pulse)

    # ---------- 主渲染 ----------
    def _zoom_mix(self, a, b, p):
        """推拉模糊转场：旧画面放大并虚化淡出，新画面从略大缩回清晰。"""
        W, H = self.W, self.H

        def zoom(img, z, blur):
            if z != 1.0:
                M = np.float32([[z, 0, W / 2 * (1 - z)], [0, z, H / 2 * (1 - z)]])
                img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if blur > 0.6:
                img = _blur_small(img, W, H, sigma=blur, scale=4)
            return img
        a2 = zoom(a, 1 + 0.08 * p, 3.0 * p)
        b2 = zoom(b, 1 + 0.05 * (1 - p), 3.0 * (1 - p))
        return cv2.addWeighted(a2, 1 - p, b2, p, 0)

    def _card_frame(self, b, t):
        return self.motion.render_card(b["bg"]["card"], t - b["start"], b["end"] - b["start"], t)

    def render(self, src, t):
        bi = max(0, bisect.bisect_right(self.bstarts, t) - 1)
        blk = self.blocks[bi]
        prev = self.blocks[bi - 1] if bi > 0 else None
        T = self.st["pacing"]["transition_sec"]
        p = ease_in_out_cubic((t - blk["start"]) / T) if prev else 1.0
        card_cur = blk["mode"] == "card" and self.motion is not None
        card_prev = prev is not None and prev["mode"] == "card" and self.motion is not None
        if card_cur:
            frame = self._card_frame(blk, t)
            if prev is not None and p < 1:
                other = self._card_frame(prev, t) if card_prev else self._aroll(src, t, prev, None, 1.0)
                frame = self._zoom_mix(other, frame, p)
        else:
            frame = self._aroll(src, t, blk, None if card_prev else prev, 1.0 if card_prev else p)
            if card_prev and p < 1:
                frame = self._zoom_mix(self._card_frame(prev, t), frame, p)
        return self._overlays(frame, t, blk)

    def _aroll(self, src, t, blk, prev, p):
        """人物画面（全屏/画中画及其转场），返回底图。"""
        W, H = self.W, self.H
        if blk["mode"] == "card":
            blk = dict(blk, mode="full")
        if prev is not None and prev["mode"] == "card":
            prev, p = None, 1.0
        cur_s = self._shape(blk, t)
        if prev is not None and p < 1:
            prev_s = self._shape(prev, t)
            s = self._mix(prev_s, cur_s, p)
        else:
            s = cur_s
        cur_m = self.st["modes"].get(blk["mode"], {})
        if s["full"] >= 0.999:
            sx = W / s["cw"]
            M = np.float32([[sx, 0, -(s["cx"] - s["cw"] / 2) * sx], [0, H / s["ch"], -(s["cy"] - s["ch"] / 2) * H / s["ch"]]])
            up = self.st.get("upscale") or {}
            interp = cv2.INTER_CUBIC if up.get("cubic", False) and sx > 1.05 else cv2.INTER_LINEAR
            frame = cv2.warpAffine(src, M, (W, H), flags=interp, borderMode=cv2.BORDER_REPLICATE)
            amt = float(up.get("sharpen", 0)) * clamp((sx - 1.0) / 0.5)
            if amt > 0.01:  # 小素材放大后轻微锐化，抵消插值带来的发软
                blur = cv2.GaussianBlur(frame, (0, 0), up.get("radius", 1.2))
                frame = cv2.addWeighted(frame, 1 + amt, blur, -amt, 0)
        else:
            if blk["mode"] == "full" and not self.fit:
                bg_img = self._background(prev["bg"], prev, t, src)
            else:
                bg_img = self._background(self._bg_of(blk), blk, t, src)
                if prev is not None and (prev["mode"] != "full" or self.fit) and p < 1 \
                        and self._bg_of(prev) != self._bg_of(blk):
                    bg_prev = self._background(self._bg_of(prev), prev, t, src)
                    bg_img = cv2.addWeighted(bg_prev, 1 - p, bg_img, p, 0)
            frame = bg_img
            # 面板（圆形画中画模式）
            pan_op = 0.0
            if blk["mode"] == "pip_circle":
                pan_op = p
            elif prev is not None and prev["mode"] == "pip_circle" and p < 1:
                pan_op = 1 - p
            if pan_op > 0.01:
                pc = self.st["panel"]
                self.panel_layer.draw(frame, pc["x"], pc["y"], pan_op)
            if blk["mode"] != "full":
                bc = cur_m.get("border_color", "#111111")
            elif prev is not None and p < 1:
                bc = self.st["modes"].get(prev["mode"], {}).get("border_color", "#111111")
            else:
                bc = "#000000"
            self._draw_pip(frame, src, s, bc)
        if self.motion is not None and self.st.get("grade"):
            frame = self.motion.grade(frame, t)
        return frame

    # ---------- AI_UP：素材小窗 ----------
    def _overlay_frame(self, path, tt, w, h):
        key = ("ov", path, w, h)
        if path.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi")):
            rd = self._media.get(key)
            if rd is None:
                rd = self._media[key] = FFReader(path, self.fps, (w, h))
            return rd.get(max(0, int(tt * self.fps)))
        img = self._media.get(key)
        if img is None:
            try:
                src = imread(path)
            except Exception:  # noqa  素材被删/改名时跳过，不让整条视频失败
                src = None
            if src is None:
                return None
            sc = max(w / src.shape[1], h / src.shape[0])
            r = cv2.resize(src, (int(src.shape[1] * sc) + 1, int(src.shape[0] * sc) + 1), interpolation=cv2.INTER_AREA)
            y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
            img = self._media[key] = np.ascontiguousarray(r[y0: y0 + h, x0: x0 + w])
        return img

    def _draw_media_pip(self, frame, t, blk):
        ov = blk.get("overlay")
        if not ov or not (ov["start"] <= t < ov["end"]):
            return
        c = self.st.get("media_pip") or {}
        W, H = self.W, self.H
        x, y, w, h = c.get("x", 60), c.get("y", 300), c.get("w", W - 120), c.get("h", int((W - 120) * 9 / 16))
        r, bw = c.get("radius", 28), c.get("border", 4)
        ins, outs = c.get("in_sec", 0.35), c.get("out_sec", 0.3)
        lt, rem = t - ov["start"], ov["end"] - t
        p_in = ease_out_back(clamp(lt / ins)) if ins > 0 else 1.0
        p_out = clamp(rem / outs) if outs > 0 else 1.0
        alpha = clamp(min(lt / ins if ins > 0 else 1, p_out))
        z0 = float(c.get("zoom_from", 0.86))
        sc = z0 + (1 - z0) * min(p_in, 1.04)
        img = self._overlay_frame(ov["path"], lt + ov.get("offset", 0), w, h)
        if img is None:
            return
        push = float(c.get("push", 0.0)) * clamp(lt / max(1.0, ov["end"] - ov["start"]))
        if push > 0.001:  # 素材在小窗里缓慢推近
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 + push)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sw, sh = max(4, int(w * sc)), max(4, int(h * sc))
        if (sw, sh) != (w, h):
            img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        ox, oy = int(x + (w - sw) / 2), int(y + (h - sh) / 2 + (1 - p_out) * 40)
        # 阴影 + 发光边框
        sh_a = c.get("shadow", 0.55) * alpha
        if sh_a > 0:
            pad = 40
            m = np.zeros((sh + pad * 2, sw + pad * 2), np.float32)
            cv2.rectangle(m, (pad, pad + 14), (pad + sw, pad + sh + 14), 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), 18) * sh_a
            x0, y0 = ox - pad, oy - pad
            fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + m.shape[1]), min(H, y0 + m.shape[0])
            if fx1 > fx0 and fy1 > fy0:
                mm = m[fy0 - y0: fy1 - y0, fx0 - x0: fx1 - x0][..., None]
                roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
                frame[fy0:fy1, fx0:fx1] = (roi * (1 - mm)).astype(np.uint8)
        a, bm = self._mask(sw, sh, r, 0, bw)
        fx0, fy0, fx1, fy1 = max(0, ox), max(0, oy), min(W, ox + sw), min(H, oy + sh)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        sl = (slice(fy0 - oy, fy1 - oy), slice(fx0 - ox, fx1 - ox))
        pa = a[sl][..., None] * alpha
        roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        out = roi * (1 - pa) + img[sl].astype(np.float32) * pa
        if bm is not None:
            bb = bm[sl][..., None] * alpha
            out = out * (1 - bb) + np.array(G.hex2rgb(c.get("border_color", "#ff8a2a"))[::-1], np.float32) * bb
        frame[fy0:fy1, fx0:fx1] = out.astype(np.uint8)

    # ---------- 橙光 Plus：人物移位放大（大字）/ 人物缩到角落小窗（整屏卡片） ----------
    def _reframe(self, frame, p):
        """人物放大并移到画面一侧（横屏移到左侧 ~36%），另一侧留给超大字。"""
        W, H = self.W, self.H
        e = ease_in_out_cubic(clamp(p))
        cfg = (self.st.get("hud") or {}).get("reframe") or {}
        if self.land:
            z = 1 + (cfg.get("zoom", 1.3) - 1) * e
            fx, fy = float(self.tl["face"]["cx"]) * W, float(self.tl["face"]["cy"]) * H
            tx = (cfg.get("face_x", 0.36) * W - z * fx) * e + (1 - e) * (fx - z * fx)
        else:
            z = 1 + (cfg.get("zoom_port", 1.12) - 1) * e
            fx, fy = W / 2, H * 0.3
            tx = fx - z * fx
        ty = fy - z * fy
        tx = min(0.0, max(W - z * W, tx))
        ty = min(0.0, max(H - z * H, ty))
        M = np.float32([[z, 0, tx], [0, z, ty]])
        return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _stage_bg(self):
        bg = getattr(self, "_stage_bg_img", None)
        if bg is None:
            W, H = self.W, self.H
            mc = self.st.get("motion") or {}
            top = np.array(G.hex2rgb(mc.get("bg_top", "#130d09"))[::-1], np.float32)
            bot = np.array(G.hex2rgb(mc.get("bg_bottom", "#070504"))[::-1], np.float32)
            glow = np.array(G.hex2rgb(mc.get("glow", "#ff6a1a"))[::-1], np.float32)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            v = (yy / H)[..., None]
            img = top * (1 - v) + bot * v
            r = np.sqrt(((xx - W * 0.82) / (W * 0.55)) ** 2 + ((yy - H * 0.12) / (H * 0.6)) ** 2)
            img += glow * (np.clip(1 - r, 0, 1) ** 2 * 0.22)[..., None]
            r2 = np.sqrt(((xx - W * 0.1) / (W * 0.5)) ** 2 + ((yy - H * 0.95) / (H * 0.5)) ** 2)
            img += glow * (np.clip(1 - r2, 0, 1) ** 2 * 0.08)[..., None]
            noise = np.random.default_rng(3).normal(0, 2.0, (H, W, 1)).astype(np.float32)
            bg = self._stage_bg_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return bg

    def _stage(self, frame, t, p, g):
        """整屏卡片：口播画面从全屏平滑缩到右下角小窗（圆角 + 橙边），背后换成暗色橙光背景。"""
        W, H = self.W, self.H
        e = ease_in_out_cubic(clamp(p))
        if e <= 0.001:
            return frame
        if self.land:
            ww = int(W * 0.27)
            wh = int(ww * H / W)
            dx, dy = W - ww - 64, H - wh - 150
            sx0, sy0, sw, sh = 0.0, 0.0, float(W), float(H)  # 横屏：整幅缩小
        else:
            ww, wh = int(W * 0.40), int(W * 0.40 * 1.25)
            dx, dy = W - ww - 48, 1480 - wh
            sw = W * 0.82
            sh = sw * 1.25
            sx0, sy0 = (W - sw) / 2, max(0.0, H * 0.31 - sh / 2)  # 竖屏：只取脸部那一块
        # 源区域、目标区域都从“整屏”插值过去
        cx0, cy0, cw, ch = sx0 * e, sy0 * e, W + (sw - W) * e, H + (sh - H) * e
        tx0, ty0, tw, th = dx * e, dy * e, W + (ww - W) * e, H + (wh - H) * e
        tw, th = max(8, int(round(tw))), max(8, int(round(th)))
        crop = frame[int(cy0): int(cy0 + ch), int(cx0): int(cx0 + cw)]
        win = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        out = self._stage_bg().copy()
        r = 22 * e
        a, bm = self._mask(tw, th, r, 0, 3 if e > 0.3 else 0)
        x0, y0 = int(round(tx0)), int(round(ty0))
        x1, y1 = min(W, x0 + tw), min(H, y0 + th)
        sub_a = a[: y1 - y0, : x1 - x0][..., None]
        roi = out[y0:y1, x0:x1].astype(np.float32)
        res = roi * (1 - sub_a) + win[: y1 - y0, : x1 - x0].astype(np.float32) * sub_a
        if bm is not None:
            bb = bm[: y1 - y0, : x1 - x0][..., None] * clamp((e - 0.3) / 0.7) * 0.9
            res = res * (1 - bb) + np.array(G.hex2rgb(self.st.get("accent", "#ff8a2a"))[::-1], np.float32) * bb
        out[y0:y1, x0:x1] = res.astype(np.uint8)
        if g.get("type") == "media" and g.get("media"):
            self._stage_media(out, t, g, e)
        return out

    def _stage_media(self, frame, t, g, e):
        """整屏卡片里的大号素材：从小放大弹出，播放中缓慢推近。"""
        W, H = self.W, self.H
        if self.land:
            x, y, w = 110, 200, int(W * 0.56)
        else:
            x, y, w = 64, 330, W - 128
        h = int(w * 9 / 16)
        lt = t - g["t0"]
        pin = clamp((lt - 0.3) / 0.5)
        if pin <= 0:
            return
        out_p = clamp((g["t1"] + 0.35 - t) / 0.35)
        sc = (0.55 + 0.45 * ease_out_back(pin)) * (0.92 + 0.08 * out_p)
        push = 1.0 + 0.06 * clamp(lt / max(1.0, g["t1"] - g["t0"]))
        img = self._overlay_frame(g["media"], lt + g.get("offset", 0), w, h)
        if img is None:
            return
        if push > 1.001:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, push)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sw, sh = max(4, int(w * sc)), max(4, int(h * sc))
        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        ox, oy = int(x + (w - sw) / 2), int(y + (h - sh) / 2)
        a, bm = self._mask(sw, sh, 18, 0, 3)
        al = min(1.0, pin * 1.8) * out_p
        x1, y1 = min(W, ox + sw), min(H, oy + sh)
        if x1 <= ox or y1 <= oy:
            return
        pa = a[: y1 - oy, : x1 - ox][..., None] * al
        roi = frame[oy:y1, ox:x1].astype(np.float32)
        res = roi * (1 - pa) + img[: y1 - oy, : x1 - ox].astype(np.float32) * pa
        if bm is not None:
            bb = bm[: y1 - oy, : x1 - ox][..., None] * al
            res = res * (1 - bb) + np.array(G.hex2rgb(self.st.get("accent", "#ff8a2a"))[::-1], np.float32) * bb
        frame[oy:y1, ox:x1] = res.astype(np.uint8)

    def _overlays(self, frame, t, blk):
        if self.hud is not None and self.hud.plus:
            g, p = self.hud.active(t, "bigword", 0.45, 0.35)
            if g is not None and p > 0.001:
                frame = self._reframe(frame, p)
            g, p = self.hud.active(t, "stage", 0.5, 0.4)
            if g is not None and p > 0.001:
                frame = self._stage(frame, t, p, g)
        self._draw_media_pip(frame, t, blk)
        for ov in self.tl.get("overlays") or []:  # 信息层模式：素材时段不受版式块边界限制
            if ov["start"] <= t < ov["end"]:
                self._draw_media_pip(frame, t, {"overlay": ov})
        if self.hud is not None:
            self.hud.draw(frame, t)
        if self.law is not None:  # 律师动效库：标题/名牌/开场钩子/重点词字幕/免责声明
            k = bisect.bisect_right(self.sstarts, t) - 1
            self.law.draw(frame, t, k if k >= 0 and t < self.subs[k]["end"] else None)
            if self.vtitle is not None:
                self.vtitle.draw(frame, t)
            return frame
        self._draw_header(frame, t)
        self._draw_namecard(frame, t)
        if blk["mode"] == "full":
            self._draw_pops(frame, t)
        self._draw_subtitle(frame, t, blk["mode"], blk)
        if self.vtitle is not None:
            self.vtitle.draw(frame, t)
        dc = self.st["disclaimer"]
        if dc.get("show", True):
            for L, y in zip(self.disc, (dc["y1"], dc["y2"])):
                L.draw(frame, dc["right"] - (L.w - 2 * L.ox), y, dc["alpha"])
        # 黑场转场
        half = self.st["pacing"]["dip_sec"] / 2
        for d in self.tl["dips"]:
            if abs(t - d) < half:
                a = 0.85 * (1 - abs(t - d) / half)
                frame = cv2.convertScaleAbs(frame, alpha=1 - a)
                break
        return frame


class MultiCompositor:
    """一条视频混用多个风格：按版式块上的 style 序号把每一帧交给对应风格的合成器。"""

    def __init__(self, tl, styles):
        self.comps = [Compositor(tl, s) for s in styles]
        self.blocks = tl["blocks"]
        self.bstarts = [b["start"] for b in self.blocks]

    def render(self, src, t):
        bi = max(0, bisect.bisect_right(self.bstarts, t) - 1)
        k = int(self.blocks[bi].get("style", 0))
        return self.comps[max(0, min(k, len(self.comps) - 1))].render(src, t)


def make_compositor(tl, style):
    """style 可以是单个风格 dict，也可以是风格列表（多风格混剪）。"""
    if isinstance(style, list):
        return MultiCompositor(tl, style) if len(style) > 1 else Compositor(tl, style[0])
    return Compositor(tl, style)
