"""网页“标题与封面”面板（AI_UP）：手动调节封面和视频标题的文字、字体、位置、景深、颜色。
设置存在 04_工程/<任务>/标题设置.json：{"video": {...}, "cover": {...}}，参数含义见 engine/aiclip/titlefx.py。
- 封面：保存后立即重画两张封面（不用重新剪辑）；
- 视频标题：保存后重新剪辑这条任务（配音、对口型复用），剪辑时 run.py 通过环境变量 AICLIP_TITLE_FX 读取。"""
import base64
import glob
import importlib
import json
import os
import sys
import threading

from .pipeline import ENGINE

_lock = threading.Lock()
_bases = {}  # jid -> (img, face, photo)
MODEL = os.path.join(ENGINE, "models", "face_detection_yunet_2023mar.onnx")


def _eng():
    if ENGINE not in sys.path:
        sys.path.insert(0, ENGINE)
    from aiclip import cover, titlefx
    # 工作台进程会一直开着：每次都重新载入引擎里的封面代码，更新后不用重启工作台
    from aiclip import hudplan
    for m in (hudplan, cover, titlefx):
        importlib.reload(m)
    return cover, titlefx


def _ctx(j):
    d = j["project_dir"]
    tl = None
    for o in ("portrait", "landscape"):
        p = os.path.join(d, f"_work_{o}", "timeline.json")
        if os.path.exists(p):
            tl = json.load(open(p, encoding="utf-8"))
            break
    if tl is None:  # 还没剪过：用任务标题凑一个最小时间线
        tl = {"brand": {"title": j["title"]}, "subs": [], "law": None, "graphics": []}
    st = json.load(open(os.path.join(ENGINE, "styles", j["style"] + ".json"), encoding="utf-8"))
    lips = [p for p in glob.glob(os.path.join(d, "对口型_*.mp4")) if ".part" not in p]
    lips.sort(key=lambda p: ("_高清" not in p, -os.path.getmtime(p)))
    return d, tl, st, (lips[0] if lips else "")


def settings_path(j):
    return os.path.join(j["project_dir"], "标题设置.json")


def load(j):
    cover, titlefx = _eng()
    d, tl, st, lip = _ctx(j)
    law = bool(tl.get("law") or (st.get("law") or {}).get("enable"))
    src = "video" if law else "photo"  # 默认：橙光系列用统一封面照，律师用视频画面
    df = titlefx.defaults(tl, st, src == "photo", src)
    saved = {}
    try:
        saved = json.load(open(settings_path(j), encoding="utf-8"))
    except Exception:  # noqa
        pass
    out = {k: dict(df[k], **(saved.get(k) or {})) for k in ("cover", "video")}
    return {"settings": out, "defaults": df, "fonts": titlefx.available_fonts(), "law": bool(tl.get("law")),
            "saved": {k: bool(saved.get(k)) for k in ("cover", "video")}, "has_video": bool(lip),
            "has_upload": os.path.exists(os.path.join(d, "封面照片.jpg"))}


def _base(j, src):
    d = j["project_dir"]
    up = os.path.join(d, "封面照片.jpg")
    key = (j["id"], src, os.path.getmtime(up) if src == "upload" and os.path.exists(up) else 0)
    with _lock:
        if key in _bases:
            return _bases[key]
    cover, _ = _eng()
    d, tl, st, lip = _ctx(j)
    if src == "upload" and not os.path.exists(up):
        raise RuntimeError("还没上传照片：点“上传照片”选一张")
    if not cover.photo_path(st, src, d) and not lip:
        raise RuntimeError("这条任务还没有对口型视频，做完对口型后才能预览")
    img, face, photo, _ = cover.get_base(lip, tl, st, MODEL, d, src)
    with _lock:
        _bases[key] = (img, face, photo)
    return img, face, photo


def upload(j, data):
    """这条任务专用的封面照片：存成 04_工程/<任务>/封面照片.jpg。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("读不了这张图片，请用 jpg / png")
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(os.path.join(j["project_dir"], "封面照片.jpg"))
    h, w = img.shape[:2]
    return {"ok": True, "size": [w, h]}


def preview(j, kind, orient, p):
    import cv2
    cover, titlefx = _eng()
    importlib.reload(titlefx)
    d, tl, st, lip = _ctx(j)
    land = orient == "landscape"
    if kind == "cover":
        img, face, photo = _base(j, p.get("src") or "video")
        W, H = cover.cover_size(st, land, photo)
        frame = titlefx.render_cover(img, face, st, tl, p, land, photo, MODEL, (W, H))
    else:  # 视频标题：在这条视频的人物画面上示意（成片里人物取景可能略有不同）
        img, face, photo = _base(j, "video")
        W, H = (1920, 1080) if land else (1080, 1920)
        frame = cover.compose(img, face, W, H)
        titlefx.draw_block(frame, p, land)
    k = 720 / max(W, H)
    small = cv2.resize(frame, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def save(j, settings, apply, Q):
    """apply: cover / video。返回 {"covers": {...}} 或 {"requeued": True}。"""
    path = settings_path(j)
    try:
        cur = json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa
        cur = {}
    cur[apply] = settings
    json.dump(cur, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if apply == "video":
        Q.retry(j["id"])
        return {"requeued": True}
    cover, titlefx = _eng()
    importlib.reload(titlefx)
    d, tl, st, lip = _ctx(j)
    files = json.loads(j.get("files") or "{}")
    if not files:
        return {"covers": {}, "note": "成片还没出来，设置已保存，剪完会按这个生成封面"}
    first = next(iter(files.values()))
    import datetime  # noqa
    base = os.path.splitext(first["video"])[0].rsplit("_", 1)[0]
    outs = {"portrait": first.get("cover_port") or base + "_封面_竖屏.jpg",
            "landscape": first.get("cover_land") or base + "_封面_横屏.jpg"}
    done = cover.make_covers(lip, tl, st, outs, MODEL, base_dir=d, custom=settings)
    for o, f in files.items():
        f.update(cover=done.get(o, f.get("cover", "")), cover_port=done.get("portrait", ""),
                 cover_land=done.get("landscape", ""))
    Q.update(j["id"], files=files)
    from .pipeline import ROOT
    rel = {o: os.path.relpath(f, ROOT).replace("\\", "/") for o, f in done.items()}
    return {"covers": done, "covers_rel": rel}
