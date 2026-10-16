"""AI_UP：成片封面（竖屏 1080x1920 + 横屏 1920x1080），封面上写本期主题。

1. 从干净的口播视频（没有字幕/包装的对口型高清版）里挑一帧：人脸清楚、画面锐利、人脸够大；
2. 按画幅构图：比例接近就直接按人脸位置裁切；竖素材出横屏时人物放右侧、背景用同一帧放大虚化；
3. 按风格写主题：
   - 橙光系列：黑色渐变压暗 + 白色第一行 + 橙色渐变第二行（重点），上方一条橙色短线；
   - 律师系列：金色宋体两行标题 + 红色钩子词印章 + 律师名牌。
主题文字来自 timeline.json（律师风格用 law.title / hook，其他用 brand.title 拆两行）。
"""
import json
import os
import re
import subprocess

import cv2
import numpy as np
from PIL import Image

from . import graphics as G
from .graphics import Layer, font, text_image
from .utils import log


# ---------------- 选帧 ----------------
def _ffmpeg():
    from .utils import ffmpeg_bin
    return ffmpeg_bin()


def _grab(video, t):
    r = subprocess.run([_ffmpeg(), "-v", "error", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
                        "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    if not r.stdout:
        return None
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)


def _faces(det, img):
    if det is None:
        return None
    h, w = img.shape[:2]
    k = 640 / w if w > 640 else 1.0
    small = cv2.resize(img, (int(w * k), int(h * k))) if k < 1 else img
    det.setInputSize((small.shape[1], small.shape[0]))
    _, f = det.detect(small)
    if f is None or not len(f):
        return None
    f = max(f, key=lambda r: r[2] * r[3]) / k
    return f


def pick_frame(video, duration, model=None, n=14):
    det = None
    if model and os.path.exists(model) and hasattr(cv2, "FaceDetectorYN"):
        det = cv2.FaceDetectorYN.create(model, "", (320, 320), 0.6, 0.3, 5)
    best, best_s = None, -1e9
    for i in range(n):
        t = duration * (0.08 + 0.72 * i / max(1, n - 1))
        img = _grab(video, t)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = _faces(det, img)
        if f is not None:
            x, y, fw, fh = [float(v) for v in f[:4]]
            roi = g[int(max(0, y)):int(y + fh), int(max(0, x)):int(x + fw)]
            sharp = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size else 0
            # 张嘴程度：两嘴角之间、嘴角线下方一小块里“暗像素”的比例（张嘴露出口腔会变暗），封面要闭嘴
            mr, ml = f[10:12], f[12:14]
            mw = max(4.0, abs(ml[0] - mr[0]))
            cy = (mr[1] + ml[1]) / 2
            box = g[int(cy - mw * 0.12):int(cy + mw * 0.35), int(min(mr[0], ml[0]) + mw * 0.15):int(max(mr[0], ml[0]) - mw * 0.15)]
            skin = np.median(roi) if roi.size else 128
            openness = float((box < skin * 0.45).mean()) if box.size else 0.0
            s = float(f[14]) * 3 + np.log1p(sharp) + fw / img.shape[1] * 4 - openness * 14
            face = ((x + fw / 2) / img.shape[1], (y + fh / 2) / img.shape[0], fw / img.shape[1])
        else:
            s = np.log1p(cv2.Laplacian(g, cv2.CV_64F).var()) - 5
            face = None
        if s > best_s:
            best, best_s = (img, face, t), s
    return best


# ---------------- 构图 ----------------
def _fill(img, W, H, fx, fy, place=(0.5, 0.42), zoom=1.0):
    """按画布比例裁切：尽量让人脸落在画布的 place 位置。"""
    h, w = img.shape[:2]
    s = max(W / w, H / h) * zoom
    cw, ch = W / s, H / s
    x0 = min(max(fx * w - place[0] * cw, 0), w - cw)
    y0 = min(max(fy * h - place[1] * ch, 0), h - ch)
    M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def compose(img, face, W, H, side=None):
    """side='right'：横屏封面人物放在右侧，左边留给大字。"""
    h, w = img.shape[:2]
    fx, fy, fw = face or (0.5, 0.38, 0.2)
    fill_zoom = max(W / w, H / h) / min(W / w, H / h)
    place = (0.68 if side == "right" else 0.5, 0.36 if H > W else 0.42)
    if fill_zoom < 1.9 or w / h >= W / H:  # 横素材出竖屏：直接按人脸裁中间一条
        # 人脸占画面太小就稍微推近一点
        z = float(np.clip(0.2 / max(0.05, fw * w * max(W / w, H / h) / W), 1.0, 1.35)) if H > W else 1.0
        if side == "right" and w / h >= W / H * 0.95:
            z = max(z, 1.3)  # 横素材出横屏封面：推近一点，才有空间把人挪到右边
        return _fill(img, W, H, fx, fy, place, z)
    # 比例差太多（竖素材出横屏）：背景虚化铺满，人物按高度放进去
    bg = _fill(img, W, H, fx, fy, (0.5, 0.5))
    bg = cv2.GaussianBlur(cv2.resize(bg, (W // 8, H // 8)), (0, 0), 3)
    bg = (cv2.resize(bg, (W, H)) * 0.55).astype(np.uint8)
    s = H / h
    fg = cv2.resize(img, (int(w * s), H), interpolation=cv2.INTER_CUBIC)
    x = int(place[0] * W - fx * fg.shape[1])
    x = min(max(x, 0), W - fg.shape[1])
    feather = int(fg.shape[1] * 0.12)
    m = np.ones(fg.shape[1], np.float32)
    ramp = np.linspace(0, 1, feather)
    m[:feather], m[-feather:] = ramp, ramp[::-1]
    m = m[None, :, None]
    roi = bg[:, x:x + fg.shape[1]].astype(np.float32)
    bg[:, x:x + fg.shape[1]] = (roi * (1 - m) + fg * m).astype(np.uint8)
    return bg


def _sharp_resize(img, w, h):
    out = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if w > img.shape[1] * 1.1:  # 放大后轻微锐化
        bl = cv2.GaussianBlur(out, (0, 0), 1.4)
        out = cv2.addWeighted(out, 1.45, bl, -0.45, 0)
    return out


def compose_photo(img, W, H, land):
    """固定封面照（如 3:4 半身照）：竖屏放上方按宽铺满、下方用虚化延伸；横屏放右侧按高铺满、左侧虚化延伸。"""
    h, w = img.shape[:2]
    bg = cv2.resize(img, (max(8, W // 10), max(8, H // 10)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), 2.5), (W, H), interpolation=cv2.INTER_LINEAR)
    bg = (bg * 0.45).astype(np.uint8)
    if land:
        s = H / h
        fw = int(w * s)
        fg = _sharp_resize(img, fw, H)
        x = W - fw
        m = np.ones(fw, np.float32)
        f = int(fw * 0.22)
        m[:f] = np.linspace(0, 1, f) ** 1.3
        m = m[None, :, None]
        roi = bg[:, x:].astype(np.float32)
        bg[:, x:] = (roi * (1 - m) + fg * m).astype(np.uint8)
    else:
        s = W / w
        fh = min(H, int(h * s))
        fg = _sharp_resize(img, W, int(h * s))[:fh]
        m = np.ones(fh, np.float32)
        f = int(fh * 0.18)
        m[-f:] = np.linspace(1, 0, f) ** 1.3
        m = m[:, None, None]
        roi = bg[:fh].astype(np.float32)
        bg[:fh] = (roi * (1 - m) + fg * m).astype(np.uint8)
    return bg


def short_lines(tl, n=7):
    """封面大字：短。标题够短就用标题，否则用信息层第一个大字标题，再不行按标题前半截。"""
    from .hudplan import _split2, _vis, _clip
    from . import llm
    lines = title_lines(tl)
    title = "".join(lines)
    got = llm.cover_text(title, tl.get("subs") or [], n) if llm.enabled() else None
    if got and all(_vis(x) <= n + 1.5 for x in got):
        return got
    if all(_vis(x) <= n + 1 for x in lines):
        return lines
    for g in tl.get("graphics") or []:
        ls = [x for x in g.get("lines") or [] if x]
        if g.get("type") == "headline" and ls and all(_vis(x) <= n + 1 for x in ls):
            return ls[:2]
    t = "".join(lines)
    t = re.split(r"[，,。！!？?]", t)[0] or t
    return [_clip(x, n + 1) for x in _split2(t, per=n)][:2]


def _gradient(frame, y0, y1, a0, a1, axis=0, color=(0, 0, 0)):
    H, W = frame.shape[:2]
    n = (H if axis == 0 else W)
    d = (y1 - y0) if abs(y1 - y0) >= 1 else 1
    t = np.clip((np.arange(n, dtype=np.float32) - y0) / d, 0, 1)
    a = a0 + (a1 - a0) * t * t * (3 - 2 * t)
    a = a[:, None, None] if axis == 0 else a[None, :, None]
    frame[:] = (frame.astype(np.float32) * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


def _layer(img_pad):
    img, pad = img_pad
    return Layer.from_pil(img, pad, pad)


def _fit(st, role, text, size, maxw, sp=0, min_size=40):
    f = font(st, role, size)
    while f.getlength(text) + sp * len(text) > maxw and size > min_size:
        size -= 4
        f = font(st, role, size)
    return f


def _vw(L):
    return L.w - 2 * L.ox


def _vh(L):
    return L.h - 2 * L.oy


# ---------------- 文字 ----------------
def title_lines(tl):
    law = tl.get("law") or {}
    if law.get("title"):
        return [x for x in law["title"] if x][:2]
    t = str((tl.get("brand") or {}).get("title") or "").strip()
    if not t and tl.get("subs"):
        t = tl["subs"][0]["text"]
    from .hudplan import _split2, split_title
    two = split_title(t)
    if two:
        return two
    return _split2(t, per=7)[:2]


def _orange(frame, st, lines, land, big=False, dark=0.0):
    H, W = frame.shape[:2]
    acc = (st.get("motion") or {}).get("accent_grad") or ["#ffc46b", "#ff8a2a", "#ff4d1f"]
    if dark > 0:  # 整体压暗（保留一点暖色）
        frame[:] = (frame.astype(np.float32) * (1 - dark)).astype(np.uint8)
    if big and land:
        _gradient(frame, int(W * 0.66), int(W * 0.12), 0.0, 0.8, axis=1)
        maxw, x = W * 0.54, int(W * 0.07)
        sizes = (210, 236)
    elif big:
        short = H < W * 1.5  # 3:4 竖封面：字放在下半部分
        _gradient(frame, int(H * (0.35 if short else 0.42)), int(H * (0.8 if short else 0.78)), 0.0, 0.85)
        maxw, x = W * 0.9, None
        sizes = (200, 226)
    elif land:
        _gradient(frame, int(W * 0.62), int(W * 0.08), 0.0, 0.88, axis=1)
        _gradient(frame, int(H * 0.55), H, 0.0, 0.5)
        maxw, x = W * 0.5, int(W * 0.065)
        sizes = (150, 172)
    else:
        _gradient(frame, int(H * 0.38), int(H * 0.82), 0.0, 0.86)
        maxw, x = W * 0.88, None
        sizes = (124, 146)
    Ls = []
    for i, ln in enumerate(lines):
        f = _fit(st, "heavy", ln, sizes[min(i, 1)] if len(lines) > 1 else (220 if big else 150), maxw, 4)
        if i == len(lines) - 1 and len(lines) > 1:
            Ls.append(_layer(text_image(ln, f, grad=acc, shadow=(0, 8, 14, 0.7), spacing=4)))
        else:
            Ls.append(_layer(text_image(ln, f, fill="#ffffff", shadow=(0, 8, 14, 0.7), spacing=4)))
    gap = 18
    tot = sum(_vh(L) for L in Ls) + gap * (len(Ls) - 1)
    y = (H - tot) / 2 if land else H * ((0.7 if H < W * 1.5 else 0.71) if big else 0.66) - tot / 2
    bar_c = G.hex2rgb(acc[1])[::-1]
    if land:
        cv2.rectangle(frame, (x, int(y - 46)), (x + 130, int(y - 34)), bar_c, -1)
    else:
        cv2.rectangle(frame, (int(W / 2 - 65), int(y - 46)), (int(W / 2 + 65), int(y - 34)), bar_c, -1)
    for L in Ls:
        L.draw(frame, x if land else W / 2 - _vw(L) / 2, y)
        y += _vh(L) + gap


def _law(frame, st, tl, lines, land):
    H, W = frame.shape[:2]
    law = tl.get("law") or {}
    gold = (st.get("law") or {}).get("gold_grad", ["#fff5d6", "#f3d27a", "#c9952f"])
    gold2 = (st.get("law") or {}).get("gold_grad2", ["#ffffff", "#fbefd0", "#e6bf68"])
    if land:
        _gradient(frame, int(W * 0.64), int(W * 0.1), 0.0, 0.88, axis=1)
        maxw, x = W * 0.52, int(W * 0.06)
        sizes = (172, 118)
    else:
        _gradient(frame, int(H * 0.4), int(H * 0.86), 0.0, 0.88)
        _gradient(frame, int(H * 0.2), 0, 0.0, 0.45)
        maxw, x = W * 0.9, None
        sizes = (150, 100)
    Ls = []
    for i, ln in enumerate(lines):
        role = "serif_black" if i == 0 else "serif"
        f = _fit(st, role, ln, sizes[min(i, 1)], maxw, 6)
        Ls.append(_layer(text_image(ln, f, grad=gold if i == 0 else gold2, stroke=2, stroke_color="#4a3110",
                                    shadow=(0, 8, 12, 0.8), spacing=6)))
    # 红色钩子词印章
    seal = None
    red = ((law.get("hook") or {}).get("red") or "").strip()
    if red:
        f = font(st, "serif_black", 64 if not land else 60)
        ti, pad = text_image(red, f, fill="#ffffff", spacing=6)
        tw, th = ti.width - 2 * pad, ti.height - 2 * pad
        bw, bh = tw + 56, th + 26
        box = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        m, _ = G.rrect_mask(bw, bh, 10, 0, 0)
        col = np.zeros((bh, bw, 4), np.uint8)
        col[..., :3] = G.hex2rgb("#c4161e")
        col[..., 3] = (m * 235).astype(np.uint8)
        box = Image.fromarray(col, "RGBA")
        box.alpha_composite(ti, (28 - pad, 13 - pad))
        seal = Layer.from_pil(box)
    # 名牌
    name = None
    if law.get("name"):
        nf, rf, ff = font(st, "heavy", 60), font(st, "heavy", 32), font(st, "heavy", 32)
        r1 = text_image(law["name"], nf, fill="#f6b31c", shadow=(0, 3, 5, 0.7), skew=0.18, spacing=2)
        r2 = text_image(" " + law.get("role", "律师"), rf, fill="#f6b31c", shadow=(0, 3, 5, 0.7), skew=0.18)
        parts = [_layer(r1), _layer(r2)]
        firm = _layer(text_image(law["firm"], ff, fill="#ffffff", shadow=(0, 3, 5, 0.7))) if law.get("firm") else None
        name = (parts, firm)
    gap = 22
    tot = sum(_vh(L) for L in Ls) + gap * (len(Ls) - 1)
    y = (H - tot) / 2 if land else H * 0.68 - tot / 2
    if seal is not None:
        seal.draw(frame, x if land else W / 2 - seal.w / 2, y - seal.h - 34)
    for L in Ls:
        L.draw(frame, x if land else W / 2 - _vw(L) / 2, y)
        y += _vh(L) + gap
    if name:
        parts, firm = name
        y += 30
        nw = sum(_vw(p) for p in parts)
        nx = x if land else W / 2 - max(nw, _vw(firm) if firm else 0) / 2
        xx = nx
        for p in parts:
            p.draw(frame, xx, y + (_vh(parts[0]) - _vh(p)) * 0.8)
            xx += _vw(p)
        if firm:
            firm.draw(frame, nx, y + _vh(parts[0]) + 12)


SHARED_PHOTO = "assets/covers/my_cover.jpg"  # 我的封面照（橙光系列统一用；放一张自己的 3:4 半身照，没有就从视频里挑帧）


def _abs(p):
    if p and not os.path.isabs(p):
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p)
    return p


def photo_path(st, src=None, base_dir=None):
    """封面照片路径；src：photo=统一封面照，upload=这条任务上传的照片，video=从视频挑帧（返回空），None=按风格默认。"""
    if src == "video":
        return ""
    if src == "upload":
        p = os.path.join(base_dir or "", "封面照片.jpg")
        return p if base_dir and os.path.exists(p) else ""
    cc = st.get("cover") or {}
    p = _abs(cc.get("image") or (SHARED_PHOTO if src == "photo" else ""))
    if not (p and os.path.exists(p)) and src == "photo":
        p = _abs(SHARED_PHOTO)
    return p if p and os.path.exists(p) else ""


def cover_size(st, land, photo=None):
    if land:
        return 1920, 1080
    if photo is None:
        photo = bool(photo_path(st))
    cc = st.get("cover") or {}
    return tuple(cc.get("port_size") or (1080, 1440)) if photo else (1080, 1920)  # 用照片时竖屏 3:4


def get_base(video, tl, st, model=None, base_dir=None, src=None):
    """封面底图：照片（统一封面照 / 上传的），或从视频挑的一帧（挑过就缓存成 base_dir/封面底图.jpg，下次直接用）。
    返回 (图, 人脸(cx,cy,fw) 或 None, 是否照片, 说明)。"""
    photo = photo_path(st, src, base_dir)
    if photo:
        img = cv2.imdecode(np.fromfile(photo, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img, None, True, f"封面照片 {os.path.basename(photo)}"
    sig = ""
    if video and os.path.exists(video):
        stt = os.stat(video)
        sig = f"{os.path.basename(video)}|{stt.st_size}|{int(stt.st_mtime)}"
    if base_dir:
        bj, bp = os.path.join(base_dir, "封面底图.json"), os.path.join(base_dir, "封面底图.jpg")
        try:
            meta = json.load(open(bj, encoding="utf-8"))
            if os.path.exists(bp) and (not sig or meta.get("sig") == sig):
                img = cv2.imdecode(np.fromfile(bp, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img, tuple(meta["face"]) if meta.get("face") else None, False, f"缓存的底图（{meta.get('t', 0):.1f}s）"
        except Exception:  # noqa
            pass
    from .utils import probe
    dur = float(probe(video).get("duration") or 10)
    picked = pick_frame(video, dur, model)
    if picked is None:
        raise RuntimeError("封面：读不到视频画面")
    img, face, t = picked
    if face is None and tl.get("face"):
        f = tl["face"]
        face = (f.get("cx", 0.5), f.get("cy", 0.38), f.get("fw", 0.2))
    if base_dir:
        try:
            cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(os.path.join(base_dir, "封面底图.jpg"))
            json.dump({"sig": sig, "face": list(face) if face else None, "t": t},
                      open(os.path.join(base_dir, "封面底图.json"), "w", encoding="utf-8"))
        except Exception:  # noqa
            pass
    return img, face, False, f"视频 {t:.1f}s 的画面"


def make_covers(video, tl, st, outs, model=None, base_dir=None, custom=None):
    """outs: {"portrait": path, "landscape": path}；custom：手动设置的封面参数（titlefx）。返回 {方向: 路径}。"""
    img, face, photo, how = get_base(video, tl, st, model, base_dir, (custom or {}).get("src"))
    cc = st.get("cover") or {}
    is_law = bool(tl.get("law"))
    if custom:
        log(f"封面：{how}，手动设置 {custom.get('lines')}")
    done = {}
    for o, path in outs.items():
        land = o == "landscape"
        W, H = cover_size(st, land, photo)
        if custom:
            from .titlefx import render_cover
            frame = render_cover(img, face, st, tl, custom, land, photo, model, (W, H))
        elif photo:
            lines = short_lines(tl, int(cc.get("max_chars", 7)))
            frame = compose_photo(img, W, H, land)
            _orange(frame, st, lines, land, big=True, dark=float(cc.get("dark", 0.3)))
        else:
            lines = title_lines(tl)
            frame = compose(img, face, W, H, side="right" if land else None)
            if is_law:
                _law(frame, st, tl, lines, land)
            else:
                _orange(frame, st, lines, land)
        if not custom and o == next(iter(outs)):
            log(f"封面：{how}，主题 {lines}，{'律师' if is_law else '橙光'}版式")
        cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tofile(path)
        done[o] = path
    return done
