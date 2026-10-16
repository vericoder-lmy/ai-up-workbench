"""通用素材库：00_素材库/通用素材/ 下的图片和视频，剪辑时以画中画小窗插进口播段落。
materials.json 记录每个素材的关键词、是否通用、是否启用；引擎（engine/aiclip/planner.py）直接读它。"""
import json
import os
import re
import shutil
import threading

from .pipeline import ROOT

DIR = os.path.join(ROOT, "00_素材库", "通用素材")
INDEX = os.path.join(DIR, "materials.json")
THUMBS = os.path.join(DIR, "_thumbs")
TRASH = os.path.join(DIR, "_回收站")
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".m4v")
VIDEO = (".mp4", ".mov", ".m4v")
_lock = threading.Lock()


def _rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def _load():
    try:
        return json.load(open(INDEX, encoding="utf-8"))
    except Exception:
        return []


def _save(items):
    os.makedirs(DIR, exist_ok=True)
    tmp = INDEX + ".tmp"
    json.dump(items, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    os.replace(tmp, INDEX)


def _thumb(name):
    src = os.path.join(DIR, name)
    dst = os.path.join(THUMBS, os.path.splitext(name)[0] + ".jpg")
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        try:
            from . import media
            os.makedirs(THUMBS, exist_ok=True)
            pre = ["-ss", "1"] if name.lower().endswith(VIDEO) else []
            media.run(pre + ["-i", src, "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", "-y", dst])
        except Exception:
            return ""
    return _rel(dst)


def sync():
    """把文件夹里新放进来的文件补进索引，删掉已经不存在的。返回带缩略图的列表。"""
    with _lock:
        os.makedirs(DIR, exist_ok=True)
        items = [m for m in _load() if os.path.exists(os.path.join(DIR, m.get("file", "")))]
        have = {m["file"] for m in items}
        for f in sorted(os.listdir(DIR)):
            if f.lower().endswith(EXTS) and not f.startswith("_") and f not in have:
                items.append({"file": f, "keywords": [], "generic": True, "enabled": True})
        _save(items)
    out = []
    for m in items:
        p = os.path.join(DIR, m["file"])
        out.append(dict(m, url=_rel(p), thumb=_thumb(m["file"]),
                        kind="video" if m["file"].lower().endswith(VIDEO) else "image",
                        size=os.path.getsize(p)))
    return out


def has_any():
    return any(m.get("enabled", True) for m in sync())


def safe_name(name):
    name = re.sub(r'[\\/:*?"<>|]+', "_", os.path.basename(name)).strip() or "素材.mp4"
    if not name.lower().endswith(EXTS):
        raise ValueError("只支持 jpg/png/webp 图片和 mp4/mov 视频")
    base, ext = os.path.splitext(name)
    k, cand = 2, name
    while os.path.exists(os.path.join(DIR, cand)):
        cand = f"{base}_{k}{ext}"
        k += 1
    return cand


def add(name, data):
    os.makedirs(DIR, exist_ok=True)
    name = safe_name(name)
    with open(os.path.join(DIR, name), "wb") as f:
        f.write(data)
    return sync()


def update(file, keywords=None, generic=None, enabled=None):
    with _lock:
        items = _load()
        for m in items:
            if m.get("file") == file:
                if keywords is not None:
                    if isinstance(keywords, str):
                        keywords = re.split(r"[,，、\s]+", keywords)
                    m["keywords"] = [k.strip() for k in keywords if k.strip()]
                if generic is not None:
                    m["generic"] = bool(generic)
                if enabled is not None:
                    m["enabled"] = bool(enabled)
        _save(items)
    return sync()


def remove(file):
    """不直接删除：挪到 _回收站，后悔了可以挪回来。"""
    src = os.path.join(DIR, os.path.basename(file))
    if os.path.exists(src):
        os.makedirs(TRASH, exist_ok=True)
        dst = os.path.join(TRASH, os.path.basename(file))
        k = 2
        while os.path.exists(dst):
            b, e = os.path.splitext(os.path.basename(file))
            dst = os.path.join(TRASH, f"{b}_{k}{e}")
            k += 1
        shutil.move(src, dst)
    return sync()
