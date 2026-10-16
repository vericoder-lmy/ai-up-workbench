"""AI_UP：律师口播动效库（风格 law_gold「律政金」）。参考：律师坐在办公室窗前讲解的竖屏口播。

画面元素（全部叠在人物全屏画面上，人物不缩放）：
1. 封面帧：开头 0.3 秒，标题两行放在半透明深色圆角框里（抖音封面截取用）；
2. 顶部常驻金色标题：两行宋体，第一行大、第二行小，金色渐变 + 深色投影；
3. 律师名牌：姓名（金色粗斜体）+「律师」+ 律所（白色），从左到右擦除式出现后常驻；
4. 开场钩子：第一句用金色宋体在字幕位置“虚焦→清晰”出现；随后一个红色宋体大字（如“深刻感悟”）
   从大到小、从模糊到清晰砸进来，金字退到后面；第二句用打字机效果（带光标）打在红字上方；
5. 正文字幕：白色粗黑体，每句一个重点词用红色粗斜体 + 白描边，并轻微弹出；
6. 免责声明：底部一行小字（可在风格里关掉）。
素材小窗出现时，名牌自动淡出让位。
"""
import hashlib
import json
import math
import os
import re

import cv2
import numpy as np
from PIL import Image

from . import graphics as G
from .graphics import Layer, font, text_image
from .utils import clamp, ease_out_back, ease_out_cubic, lerp, log

NUM_HEAD = re.compile(r"^(第[一二三四五六七八九十0-9]+[条点个步]?|首先|其次|再次|最后|另外|还有)[，,、：:]?")
SEP = re.compile(r"\s*[|｜/／]\s*|\s+|[，,]\s*")


# ======================= 规划（planner 调用） =======================
def _vis(s):
    return sum(0.55 if re.match(r"[A-Za-z0-9]", c) else 1.0 for c in str(s))


def _title_lines(title, raw):
    """标题拆两行：用户标题里有空格/逗号/竖线就按它拆；否则用大模型给的两行；再不行按安全断点对半。"""
    from .hudplan import split_title
    t = str(title or "").strip()
    two = split_title(t)
    if two:
        return two
    lines = [str(x).strip() for x in (raw or {}).get("title") or [] if str(x).strip()]
    if len(lines) >= 2 and _vis(lines[0]) <= 9 and _vis(lines[1]) <= 11:
        return lines[:2]
    if not t:
        return lines[:2] if lines else []
    from .hudplan import _split2
    return _split2(t, per=8)


LEX = ("诉讼时效 举证责任 刑事责任 民事责任 连带责任 违约金 定金 订金 借条 欠条 收据 证据 转账记录 聊天记录 录音 合同 协议 "
       "担保 抵押 起诉 立案 判决 执行 强制执行 失信 拘留 坐牢 判刑 有期徒刑 缓刑 罚款 罚金 赔偿 补偿 违法 犯罪 诈骗 "
       "非法集资 洗钱 帮信 赌博 侵权 离婚 抚养权 财产分割 彩礼 婚前财产 遗嘱 继承 工伤 社保 加班费 经济补偿 仲裁 "
       "风险 陷阱 后果 无效 撤销 时效 利息 高利贷 本金 保证人 律师 法院 公安 报警 自首 取保候审 不起诉 "
       "现金 转账 备注 还款 期限 借款 欠款 签字 按手印 公证 律师函 起诉状 证人 违约 定罪 量刑").split()


def _rule_kw(text, pops, terms=()):
    m = NUM_HEAD.match(text)
    if m:
        rest = text[m.end():].strip()
        if 2 <= len(rest) <= 8:
            return rest
    hits = [w for w in LEX if w in text and w != text]
    if hits:
        return max(hits, key=len)
    return ""  # 自动统计的高频片段常常是半个词（如“现金交”），规则版不用它们


def plan(subs, brand, style, pops, terms):
    """返回 timeline 里的 law 字段。"""
    raw = None
    sig = hashlib.sha1(("\n".join(s["text"] for s in subs) + "|" + str(brand.get("title", ""))).encode("utf-8")).hexdigest()[:16]
    cache = os.environ.get("AICLIP_HUD_CACHE")
    cache = os.path.join(os.path.dirname(cache), "律师包装规划.json") if cache else ""
    if cache and os.path.exists(cache):
        try:
            c = json.load(open(cache, encoding="utf-8"))
            if c.get("sig") == sig:
                raw = c.get("raw")
                log("律师包装：复用上次的规划")
        except Exception:  # noqa
            raw = None
    if raw is None:
        from . import llm
        if llm.enabled():
            raw = llm.plan_law(subs, brand.get("title", ""))
            if raw and cache:
                try:
                    json.dump({"sig": sig, "raw": raw}, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                except OSError:
                    pass
    raw = raw or {}
    title = _title_lines(brand.get("title", ""), raw)
    # 每句重点词：必须是原句里的连续片段，不能是整句
    kw = {}
    got = raw.get("kw") or {}
    if isinstance(got, list):
        got = {str(x.get("i")): x.get("w") for x in got if isinstance(x, dict)}
    for i, s in enumerate(subs):
        w = str(got.get(str(i)) or "").strip()
        if not (w and w in s["text"] and 2 <= len(w) <= 8 and (w != s["text"] or len(w) <= 4)):
            w = "" if got else _rule_kw(s["text"], pops)
        if w:
            kw[str(i)] = w
    # 开场钩子
    hook = None
    if subs:
        red = str(raw.get("hook") or "").strip()
        head = "".join(title) + subs[0]["text"] + (subs[1]["text"] if len(subs) > 1 else "")
        if not (2 <= len(red) <= 6 and red in head):
            # 规则版：优先标题第二行开头 4 个字（如“深刻感悟”），其次法律词表里出现在开头的词
            cand = [title[1][:4]] if len(title) > 1 and len(title[1]) >= 4 else []
            cand += [w for w in LEX if 2 <= len(w) <= 5 and w in head]
            red = cand[0] if cand else (title[0][:4] if title else "")
        s0 = subs[0]
        tw = [1] if len(subs) > 1 and subs[1]["start"] - s0["end"] < 1.0 and subs[1]["end"] - s0["start"] < 7.5 else []
        end = subs[tw[-1]]["end"] if tw else s0["end"]
        d0 = s0["end"] - s0["start"]
        hook = {"gold": 0, "tw": tw, "red": red, "t0": round(s0["start"], 3),
                "t_red": round(s0["start"] + max(0.5, min(d0 * 0.75, 1.4)), 3) if red else None,
                "end": round(end, 3)}
        for i in [0] + tw:
            kw.pop(str(i), None)
    lc = style.get("law") or {}
    out = {"title": title, "name": brand.get("name", ""), "role": brand.get("role", "律师") or "律师",
           "firm": brand.get("firm", ""), "kw": kw, "hook": hook,
           "disclaimer": brand.get("law_disclaimer") or lc.get("disclaimer", "")}
    log(f"律师包装：标题 {title}，钩子大字「{hook and hook['red']}」，重点词 {len(kw)}/{len(subs)} 句，"
        f"名牌 {out['name'] or '（未填姓名，不显示）'}")
    return out


# ======================= 渲染（compositor 调用） =======================
def _L(img_pad):
    img, pad = img_pad
    return Layer.from_pil(img, pad, pad)


def _blur_layer(L, sigma):
    if sigma < 0.4:
        return L
    k = int(sigma * 3) * 2 + 1
    c = cv2.GaussianBlur(L.c, (k, k), sigma)
    a = cv2.GaussianBlur(L.a[..., 0], (k, k), sigma)[..., None]
    return Layer(np.ascontiguousarray(c), np.ascontiguousarray(a), L.ox, L.oy)


def _hstack(parts):
    """把多段文字（PIL 图, pad）横向拼成一行，竖直居中。返回 (PIL, 每段在行内的 x 偏移, 行宽, 行高)。"""
    ws = [im.width - 2 * pad for im, pad in parts]
    hs = [im.height for im, pad in parts]
    P = max(pad for _, pad in parts)
    H = max(hs)
    W = sum(ws) + 2 * P
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    x, xs = P, []
    for (im, pad), w in zip(parts, ws):
        out.alpha_composite(im, (int(x - pad), int((H - im.height) / 2)))
        xs.append(x - P)
        x += w
    return out, P, xs, sum(ws)


class LawFX:
    def __init__(self, tl, style, W, H):
        self.tl, self.st, self.W, self.H = tl, style, W, H
        self.d = tl.get("law") or {}
        c = style.get("law") or {}
        self.land = W > H
        self.c = dict(c.get("port") or {}, **(c.get("land") or {} if self.land else {}))
        self.cfg = c
        self.k = self.c.get("scale", 1.0)
        self.subs = tl["subs"]
        self._cache = {}
        self._blur = {}
        ov = [(o["start"], o["end"]) for o in tl.get("overlays") or []]
        ov += [(b["overlay"]["start"], b["overlay"]["end"]) for b in tl["blocks"] if b.get("overlay")]
        self.ovs = ov
        self._build()

    # ---------- 静态图层 ----------
    def _build(self):
        st, k, c = self.st, self.k, self.c
        gold = self.cfg.get("gold_grad", ["#fff5d6", "#f3d27a", "#c9952f"])
        gold2 = self.cfg.get("gold_grad2", ["#ffffff", "#fbefd0", "#e6bf68"])
        t = self.d.get("title") or []
        self.title = []
        maxw = self.W * (0.86 if not self.land else 0.6)
        for i, line in enumerate(t[:2]):
            size = int((c.get("t1_size", 118) if i == 0 else c.get("t2_size", 74)) * k)
            f = font(st, "serif_black" if i == 0 else "serif", size)
            while f.getlength(line) + 4 * len(line) > maxw and size > 30:
                size -= 4
                f = font(st, "serif_black" if i == 0 else "serif", size)
            self.title.append(_L(text_image(line, f, grad=gold if i == 0 else gold2, stroke=2,
                                            stroke_color="#4a3110", shadow=(0, int(5 * k), int(7 * k), 0.75),
                                            spacing=int(4 * k))))
        # 封面框里的白字
        self.cover = []
        for i, line in enumerate(t[:2]):
            size = int((76 if i == 0 else 54) * k)
            f = font(st, "heavy", size)
            while f.getlength(line) > self.W * 0.8 and size > 28:
                size -= 4
                f = font(st, "heavy", size)
            self.cover.append(_L(text_image(line, f, fill="#ffffff", shadow=(0, 3, 6, 0.6), spacing=int(3 * k))))
        # 名牌
        self.name = None
        if self.d.get("name"):
            nf = font(st, "heavy", int(c.get("name_size", 66) * k))
            rf = font(st, "heavy", int(c.get("name_size", 66) * 0.5 * k))
            ff = font(st, "heavy", int(c.get("firm_size", 34) * k))
            sh = (0, int(3 * k), int(5 * k), 0.7)
            row1, P, _, w1 = _hstack([text_image(self.d["name"], nf, fill="#f6b31c", shadow=sh, skew=0.18, spacing=2),
                                      text_image(" " + self.d.get("role", "律师"), rf, fill="#f6b31c", shadow=sh,
                                                 skew=0.18)])
            rows = [(row1, P)]
            if self.d.get("firm"):
                rows.append(text_image(self.d["firm"], ff, fill="#ffffff", shadow=sh, spacing=1))
            Wd = max(im.width for im, _ in rows)
            Hd = sum(im.height - 2 * p for im, p in rows) + int(10 * k) + 2 * P
            canvas = Image.new("RGBA", (Wd + 2 * P, Hd), (0, 0, 0, 0))
            y = P
            for im, p in rows:
                canvas.alpha_composite(im, (P - p, y - p))
                y += im.height - 2 * p + int(10 * k)
            self.name = Layer.from_pil(canvas, P, P)
        # 免责声明
        self.disc = None
        if self.d.get("disclaimer") and self.cfg.get("show_disclaimer", True):
            f = font(st, "medium", int(c.get("disc_size", 26) * k))
            self.disc = _L(text_image(self.d["disclaimer"], f, fill="#ffffff", shadow=(0, 2, 4, 0.8), spacing=2))
        self.name_x = self.W / 2 - 0.35 * min(self.W, self.H * 9 / 16)

    # ---------- 工具 ----------
    def _blurred(self, key, L, sigma):
        q = round(sigma * 2) / 2
        kk = (key, q)
        B = self._blur.get(kk)
        if B is None:
            if len(self._blur) > 120:
                self._blur.clear()
            B = self._blur[kk] = _blur_layer(L, q)
        return B

    def _in_overlay(self, t):
        a = 0.0
        for s, e in self.ovs:
            if s - 0.3 <= t <= e + 0.3:
                a = max(a, min(clamp((t - s + 0.3) / 0.3), clamp((e + 0.3 - t) / 0.3)))
        return a

    # ---------- 字幕：白字 + 红色重点词 ----------
    def _sub(self, i):
        key = ("sub", i)
        if key in self._cache:
            return self._cache[key]
        st, c, k = self.st, self.c, self.k
        text = self.subs[i]["text"]
        w = (self.d.get("kw") or {}).get(str(i), "")
        size = int(c.get("sub_size", 64) * k)
        maxw = self.W * (0.9 if not self.land else 0.7)

        def segs(txt):
            if w and w in txt:
                a = txt.find(w)
                return [(txt[:a], 0), (w, 1), (txt[a + len(w):], 0)]
            return [(txt, 0)]

        def width(txt, sz):
            f, fk = font(st, "heavy", sz), font(st, "heavy", int(sz * 1.04))
            return sum((fk.getlength(s) * 1.1 + sz * 0.1) if kk else f.getlength(s) + 2 * len(s) for s, kk in segs(txt))

        lines = [text]
        while width(text, size) > maxw and size > c.get("sub_size", 64) * k * 0.8:
            size -= 2
        if width(text, size) > maxw:
            # 两行：断点不落在重点词中间，尽量居中
            best, bi = 1e9, len(text) // 2
            a = text.find(w) if w else -1
            for j in range(2, len(text) - 1):
                if a >= 0 and a < j < a + len(w):
                    continue
                d = abs(j - len(text) / 2) + (0 if text[j - 1] in "，,的了是" else 0.6)
                if d < best:
                    best, bi = d, j
            lines = [text[:bi].rstrip("，, "), text[bi:].lstrip("，, ")]
        f, fk = font(st, "heavy", size), font(st, "heavy", int(size * 1.04))
        sh = (0, int(3 * k), int(6 * k), 0.85)
        out = []
        for ln in lines:
            base_parts, kw_pos = [], None
            parts = segs(ln)
            for s, kk in parts:
                if not s:
                    continue
                if kk:
                    base_parts.append(text_image(s, fk, fill="#ffffff", stroke=0, shadow=None, skew=0.2, spacing=2,
                                                 pad=int(10 * k)))
                    kw_pos = len(base_parts) - 1
                else:
                    base_parts.append(text_image(s, f, fill="#ffffff", stroke=2, stroke_color="#2a2a2a", shadow=sh,
                                                 spacing=2))
            img, P, xs, tw = _hstack(base_parts)
            KL = None
            if kw_pos is not None:
                # 重点词单独一层（要做弹出动画），底图上对应位置留空
                kim, kp = text_image(w, fk, fill=self.cfg.get("kw_color", "#ee1c25"), stroke=int(5 * k),
                                     stroke_color="#ffffff", shadow=(0, int(3 * k), int(5 * k), 0.7), skew=0.2,
                                     spacing=2, bold=1)
                KL = (Layer.from_pil(kim, kp, kp), xs[kw_pos] + (base_parts[kw_pos][0].width - 2 * base_parts[kw_pos][1]) / 2)
                blank = Image.new("RGBA", base_parts[kw_pos][0].size, (0, 0, 0, 0))
                base_parts[kw_pos] = (blank, base_parts[kw_pos][1])
                img, P, xs, tw = _hstack(base_parts)
            out.append((Layer.from_pil(img, P, P), tw, KL))
        res = (out, size)
        if len(self._cache) > 400:
            self._cache.clear()
        self._cache[key] = res
        return res

    def _draw_sub(self, frame, t, i):
        s = self.subs[i]
        lines, size = self._sub(i)
        lt = t - s["start"]
        op = clamp(lt / 0.08)
        lh = size * 1.32
        y0 = self.c.get("sub_y", 1430) * (self.H / (1080 if self.land else 1920)) - lh * len(lines) / 2
        for j, (L, tw, KL) in enumerate(lines):
            yc = y0 + lh * (j + 0.5)
            x0 = self.W / 2 - tw / 2
            L.draw(frame, x0, yc - (L.h - 2 * L.oy) / 2, op)
            if KL:
                K, cx = KL
                q = clamp((lt - 0.05) / 0.22)
                sc = lerp(1.45, 1.0, ease_out_back(q, 2.0)) if q < 1 else 1.0
                K.draw_center(frame, x0 + cx, yc, ease_out_cubic(q * 1.6) * op, sc)

    # ---------- 开场钩子 ----------
    def _hook_layers(self):
        if "hook" in self._cache:
            return self._cache["hook"]
        h, st, k, c = self.d["hook"], self.st, self.k, self.c
        txt = self.subs[h["gold"]]["text"]
        size = int(c.get("gold_size", 80) * k)
        f = font(st, "serif", size)
        while f.getlength(txt) + 10 * k * len(txt) > self.W * (0.88 if not self.land else 0.6) and size > 36:
            size -= 4
            f = font(st, "serif", size)
        gold = _L(text_image(txt, f, grad=["#fff1c2", "#f5c64e", "#d99a1e"], shadow=(0, int(4 * k), int(6 * k), 0.8),
                             spacing=int(10 * k), stroke=1, stroke_color="#5a3a08"))
        red = None
        if h.get("red"):
            size = int(c.get("red_size", 168) * k)
            f = font(st, "serif_black", size)
            while f.getlength(h["red"]) > self.W * 0.86 and size > 60:
                size -= 6
                f = font(st, "serif_black", size)
            red = _L(text_image(h["red"], f, grad=["#e8323a", "#c4161e", "#8c0c12"],
                                shadow=(0, int(6 * k), int(12 * k), 0.65), spacing=int(4 * k)))
        self._cache["hook"] = (gold, red)
        return gold, red

    def _draw_hook(self, frame, t):
        h = self.d["hook"]
        gold, red = self._hook_layers()
        sy = self.c.get("sub_y", 1430) * (self.H / (1080 if self.land else 1920))
        lt = t - h["t0"]
        fade_out = clamp((h["end"] + 0.25 - t) / 0.25)
        tr = h.get("t_red")
        # 金色第一句：虚焦放大 → 清晰；红字砸进来后退到后面变暗
        q = clamp(lt / 0.35)
        sc = lerp(1.22, 1.0, ease_out_cubic(q))
        dim = 1.0
        if tr is not None and t >= tr:
            dim = lerp(1.0, 0.45, ease_out_cubic(clamp((t - tr) / 0.3)))
        if h["tw"] and t >= self.subs[h["tw"][0]]["start"]:
            dim *= clamp(1 - (t - self.subs[h["tw"][0]]["start"]) / 0.3)
        if dim * fade_out > 0.01:
            B = self._blurred("gold", gold, 9 * (1 - ease_out_cubic(q)))
            B.draw_center(frame, self.W / 2, sy - 18 * self.k * (1 if red is not None and tr is not None and t >= tr else 0),
                          ease_out_cubic(q) * dim * fade_out, sc)
        # 红色大字
        if red is not None and tr is not None and t >= tr:
            q2 = clamp((t - tr) / 0.32)
            e = ease_out_cubic(q2)
            sc2 = lerp(1.7, 1.0, e)
            B = self._blurred("red", red, 12 * (1 - e))
            B.draw_center(frame, self.W / 2, sy + 22 * self.k, clamp(q2 * 2.2) * fade_out, sc2)
        # 打字机
        for i in h["tw"]:
            s = self.subs[i]
            if not (s["start"] <= t < h["end"] + 0.25):
                continue
            self._typewriter(frame, t, i, sy - (82 if red is not None else 0) * self.k, fade_out)

    def _typewriter(self, frame, t, i, cy, op):
        s = self.subs[i]
        text = s["text"]
        size = int(self.c.get("sub_size", 64) * self.k)
        f = font(self.st, "heavy", size)
        while f.getlength(text) > self.W * 0.88 and size > 30:
            size -= 2
            f = font(self.st, "heavy", size)
        dur = max(0.3, s["end"] - s["start"])
        cps = max(7.0, len(text) / (dur * 0.7))
        n = min(len(text), int((t - s["start"]) * cps) + 1)
        key = ("tw", i, n)
        L = self._cache.get(key)
        if L is None:
            L = self._cache[key] = _L(text_image(text[:n], f, fill="#ffffff", stroke=2, stroke_color="#222222",
                                                 shadow=(0, 3, 6, 0.85), spacing=2))
        full = f.getlength(text) + 2 * len(text)
        x0 = self.W / 2 - full / 2
        L.draw(frame, x0, cy - (L.h - 2 * L.oy) / 2, op)
        if n < len(text) or int(t * 2.4) % 2 == 0:
            x = x0 + (L.w - 2 * L.ox) + 6
            a = op
            if a > 0.05:
                cv2.rectangle(frame, (int(x), int(cy - size * 0.5)), (int(x + max(3, 4 * self.k)), int(cy + size * 0.5)),
                              (255, 255, 255), -1)

    # ---------- 主入口 ----------
    def draw(self, frame, t, sub_k):
        H, W, k = self.H, self.W, self.k
        cover = self.cfg.get("cover_sec", 0.3)
        if t < cover and self.cover:
            # 封面：半透明深色圆角框 + 白色标题
            ws = [L.w - 2 * L.ox for L in self.cover]
            hs = [L.h - 2 * L.oy for L in self.cover]
            bw, bh = int(max(ws) + 90 * k), int(sum(hs) + 24 * k * (len(hs) - 1) + 70 * k)
            cy = H * self.c.get("cover_y", 0.7)
            x0, y0 = int(W / 2 - bw / 2), int(cy - bh / 2)
            m, _ = G.rrect_mask(bw, bh, 22 * k, 0, 0)
            roi = frame[max(0, y0):y0 + bh, max(0, x0):x0 + bw]
            mm = m[: roi.shape[0], : roi.shape[1], None] * 0.55
            roi[:] = (roi.astype(np.float32) * (1 - mm)).astype(np.uint8)
            y = y0 + 35 * k
            for L, hh in zip(self.cover, hs):
                L.draw(frame, W / 2 - (L.w - 2 * L.ox) / 2, y)
                y += hh + 24 * k
            return
        # 顶部标题（手动设置了视频标题时由 titlefx 画，这里跳过）
        q = ease_out_cubic(clamp((t - cover) / 0.3))
        title = [] if self.tl.get("title_fx") else self.title
        y = self.c.get("title_y", 205) * (H / (1080 if self.land else 1920))
        for j, L in enumerate(title):
            hh = L.h - 2 * L.oy
            L.draw_center(frame, W / 2, y + hh / 2 + (1 - q) * 30 * k, q, lerp(1.06, 1.0, q))
            y += hh + self.c.get("title_gap", 22) * k
        # 名牌：左→右擦除出现
        if self.name is not None:
            a = 1 - self._in_overlay(t)
            lt = t - cover - 0.1
            if lt > 0 and a > 0.01:
                p = ease_out_cubic(clamp(lt / 0.5))
                L = self.name
                ny = self.c.get("name_y", 1080) * (H / (1080 if self.land else 1920))
                if p >= 1:
                    L.draw(frame, self.name_x, ny, a)
                else:
                    edge = int(L.w * p)
                    ramp = max(1, int(40 * k))
                    xs = np.arange(L.w, dtype=np.float32)
                    m = np.clip((edge - xs) / ramp, 0, 1)[None, :, None]
                    Lp = Layer(np.ascontiguousarray(L.c * m), np.ascontiguousarray(L.a * m), L.ox, L.oy)
                    Lp.draw(frame, self.name_x, ny, a)
        # 字幕 / 开场钩子
        h = self.d.get("hook")
        if h and t < h["end"] + 0.25 and t >= h["t0"]:
            self._draw_hook(frame, t)
            hook_idx = {h["gold"], *h["tw"]}
        else:
            hook_idx = set()
        if sub_k is not None and sub_k not in hook_idx:
            self._draw_sub(frame, t, sub_k)
        if self.disc is not None:
            L = self.disc
            dy = self.c.get("disc_y", 1830) * (H / (1080 if self.land else 1920))
            L.draw(frame, W / 2 - (L.w - 2 * L.ox) / 2, dy, self.cfg.get("disc_alpha", 0.7))
