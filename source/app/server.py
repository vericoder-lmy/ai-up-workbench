"""AI_UP 网页工作台：只用 Python 标准库。启动：python -m app.server（或双击 启动工作台.bat）。"""
import glob
import json
import mimetypes
import os
import re
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .clients.heygem import _http, HeyGemError
from .clients.qwen_voice import QwenVoice
from .jobs import Queue
from . import materials
from .pipeline import ROOT, ENGINE, get_voice_id, load_cfg, tts_sentence
from .text import split_sentences

VERSION = "0.3.0"


def _build():
    """代码指纹：app/ 和 web/ 里任何文件改过都会变，启动器据此判断开着的工作台是不是旧版本。"""
    import hashlib
    h = hashlib.sha1()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in sorted(glob.glob(os.path.join(root, "app", "**", "*.py"), recursive=True) +
                    [os.path.join(root, "web", "index.html")]):
        try:
            h.update(f"{os.path.relpath(p, root)}:{os.path.getmtime(p):.0f}:{os.path.getsize(p)}".encode())
        except OSError:
            pass
    return h.hexdigest()[:10]


BUILD = _build()
PORT = 8790
Q = None
_voice_lock = threading.Lock()
_voice_cache = {}


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def meta():
    lib = os.path.join(ROOT, "00_素材库")
    avatars, seen = [], set()
    for p in sorted(glob.glob(os.path.join(lib, "数字人", "*.mp4")), key=lambda x: "_原片" not in x):
        name = os.path.splitext(os.path.basename(p))[0].replace("_原片", "")
        if name not in seen:
            seen.add(name)
            thumb = os.path.join(lib, "数字人", "_prepared", name + "_封面.jpg")
            if not os.path.exists(thumb):  # 原片可能是 HEVC，浏览器放不了，所以用截图当封面
                try:
                    from . import media
                    os.makedirs(os.path.dirname(thumb), exist_ok=True)
                    media.run(["-ss", "3", "-i", p, "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", thumb])
                except Exception:
                    thumb = ""
            avatars.append({"name": name, "file": rel(p), "thumb": rel(thumb) if thumb else ""})
    from . import avatar_mix
    for gname, mem in sorted(avatar_mix.groups().items()):
        first = next((a for a in avatars if a["name"] == mem[0]), None)
        avatars.insert(0, {"name": avatar_mix.GROUP + gname, "label": f"{gname} · {len(mem)} 条随机混排",
                           "file": "", "thumb": first["thumb"] if first else "", "group": mem})
    voices = sorted({os.path.splitext(os.path.basename(p))[0]
                     for ext in ("m4a", "wav", "mp3", "aac", "flac")
                     for p in glob.glob(os.path.join(lib, "声音", "*." + ext))})
    styles = []
    for p in sorted(glob.glob(os.path.join(ENGINE, "styles", "*.json"))):
        try:
            s = json.load(open(p, encoding="utf-8"))
            sid = os.path.splitext(os.path.basename(p))[0]
            pv = os.path.join(ENGINE, "styles", "previews", sid + ".jpg")
            styles.append({"id": sid, "name": s.get("display_name") or s.get("name"), "desc": s.get("description", ""),
                           "accent": s.get("accent") or (s.get("motion") or {}).get("glow", ""),
                           "law": bool((s.get("law") or {}).get("enable")),
                           "preview": rel(pv) if os.path.exists(pv) else ""})
        except Exception:
            pass
    default_style = cfg_style = load_cfg().get("style_default", "aiup_orange")
    styles.sort(key=lambda s: s["id"] != default_style)
    cfg = load_cfg()
    try:
        _http("GET", cfg["heygem"]["video_url"] + "/", timeout=2, raw=True)
        heygem = True
    except HeyGemError:
        heygem = False
    return {"version": VERSION, "build": BUILD, "avatars": avatars, "voices": voices, "styles": styles,
            "default_avatar": cfg.get("avatar_default", {}).get("name"), "default_style": default_style,
            "emotions": list(__import__("app.audiofix", fromlist=["EMOTIONS"]).EMOTIONS),
            "materials": materials.sync(), "default_voice": cfg.get("voice_default"),
            "services": {"heygem": heygem, "llm_model": cfg.get("llm", {}).get("model", "")}}


def preview_tts(voice, text, instruction="", rate=1.0):
    cfg = load_cfg()
    with _voice_lock:
        if voice not in _voice_cache:
            qv = QwenVoice(cfg["voice"], os.path.join(ROOT, "00_素材库", "声音", "音色库.json"), log=lambda m: None)
            _voice_cache[voice] = (qv, get_voice_id(qv, voice))
    qv, vid = _voice_cache[voice]
    return rel(tts_sentence(qv, vid, text, instruction, rate)), getattr(qv, "fallback_reason", None)


def _errlog(path, e):
    """接口出错时记一笔到 06_日报/工作台日志.txt（不含密钥），方便排查。"""
    import datetime
    import traceback
    try:
        os.makedirs(os.path.join(ROOT, "06_日报"), exist_ok=True)
        with open(os.path.join(ROOT, "06_日报", "工作台日志.txt"), "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now():%m-%d %H:%M:%S}] {path.split('?')[0]}\n{traceback.format_exc()[-1500:]}\n")
    except Exception:
        pass


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, o, code=200):
        b = json.dumps(o, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _file(self, path, download=None):
        if not os.path.isfile(path):
            return self._json({"error": "文件不存在"}, 404)
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if a and b else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        if download:  # 浏览器直接下载（中文文件名按 RFC 5987 编码）
            self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + urllib.parse.quote(download))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (ConnectionError, OSError):
                pass

    def _fix(self, u):
        """音频纠正：上传原音频/视频、生成纠正后的一句、替换回原音频/视频。"""
        import importlib
        from . import audiofix
        importlib.reload(audiofix)  # 改了 audiofix.py 不用重启工作台
        if u.path == "/api/fix/upload":
            name = urllib.parse.unquote(self.headers.get("X-Filename", "audio.mp3"))
            return self._json(audiofix.create(name, self._body()))
        d = json.loads(self._body() or b"{}")
        if u.path == "/api/fix/generate":
            try:
                it = audiofix.generate(d["id"], d["start"], d["end"], d.get("text", ""), d.get("emotion", "keep"),
                                       d.get("custom", ""), bool(d.get("align", True)), d.get("ref", "near"))
            except (ValueError, FileNotFoundError) as e:
                return self._json({"error": str(e)}, 400)
            return self._json(it)
        if u.path == "/api/fix/replace":
            return self._json({"file": audiofix.replace(d["id"], d["rid"], bool(d.get("video")))})
        return self._json({"error": "not found"}, 404)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                return self._file(os.path.join(ROOT, "web", "index.html"))
            if u.path == "/api/meta":
                return self._json(meta())
            if u.path == "/api/jobs":
                return self._json({"jobs": Q.list()})
            if u.path == "/api/materials":
                return self._json({"materials": materials.sync()})
            if u.path.startswith("/media/"):
                p = os.path.normpath(os.path.join(ROOT, urllib.parse.unquote(u.path[7:])))
                if not p.startswith(os.path.normpath(ROOT) + os.sep):
                    return self._json({"error": "路径不允许"}, 403)
                dl = urllib.parse.parse_qs(u.query).get("dl", [""])[0]
                return self._file(p, download=(dl or os.path.basename(p)) if dl else None)
            if u.path == "/api/pron":  # 读音纠正词典
                from . import pron
                return self._json({"text": pron.text(), "n": len(pron.pairs())})
            if u.path == "/api/titleset":  # 标题与封面面板：读设置（没保存过就是默认值）
                from . import titleset
                j = Q.get(urllib.parse.parse_qs(u.query).get("id", [""])[0])
                if not j:
                    return self._json({"error": "任务不存在"}, 404)
                return self._json(titleset.load(j))
            if u.path == "/api/fix/sessions":
                from . import audiofix
                return self._json({"sessions": audiofix.sessions(), "emotions": list(audiofix.EMOTIONS)})
            m = re.match(r"^/api/fix/s/([^/]+)$", u.path)
            if m:
                from . import audiofix
                return self._json(audiofix.get(urllib.parse.unquote(m.group(1))))
            self._json({"error": "not found"}, 404)
        except Exception as e:
            _errlog(self.path, e)
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            if u.path == "/api/shutdown":  # 启动器发现旧版本时用来关掉旧进程
                self._json({"ok": True})
                threading.Timer(0.3, lambda: os._exit(0)).start()
                return
            if u.path == "/api/sentences":
                return self._json({"sentences": split_sentences(json.loads(self._body())["script"])})
            if u.path == "/api/tts":
                d = json.loads(self._body())
                from .audiofix import instruction_for
                path, note = preview_tts(d["voice"], d["text"], instruction_for(d.get("emotion"), d.get("custom", "")),
                                         float(d.get("rate") or 1.0))
                return self._json({"url": "/media/" + path, "note": note or ""})
            if u.path == "/api/jobs":
                d = json.loads(self._body())
                if not d.get("script", "").strip():
                    return self._json({"error": "文案是空的"}, 400)
                return self._json({"id": Q.create(d)})
            m = re.match(r"^/api/jobs/(\w+)/(cancel|retry|open)$", u.path)
            if m:
                jid, act = m.groups()
                if act == "cancel":
                    Q.cancel(jid)
                elif act == "retry":
                    Q.retry(jid)
                else:
                    j = Q.get(jid)
                    files = json.loads(j["files"] or "{}") if j else {}
                    target = os.path.dirname(next(iter(files.values()))["video"]) if files else (j or {}).get("project_dir")
                    if target and hasattr(os, "startfile"):
                        os.startfile(target)
                return self._json({"ok": True})
            if u.path in ("/api/materials/update", "/api/materials/delete"):
                d = json.loads(self._body())
                if u.path.endswith("delete"):
                    return self._json({"materials": materials.remove(d["file"])})
                return self._json({"materials": materials.update(d["file"], d.get("keywords"), d.get("generic"), d.get("enabled"))})
            if u.path == "/api/emotions":  # 按文案自动识别每句的情绪和语速
                import importlib
                from . import emotion_ai
                importlib.reload(emotion_ai)
                d = json.loads(self._body())
                items, src = emotion_ai.detect(d.get("sentences") or [], d.get("title", ""))
                return self._json({"items": items, "source": src})
            if u.path == "/api/pron/add":  # 把 AI 发现的多音字加进读音词典
                from . import pron
                d = json.loads(self._body())
                return self._json({"n": pron.add([(str(a), str(b)) for a, b in d.get("pairs") or []])})
            if u.path == "/api/pron":  # 保存读音纠正词典
                from . import pron
                d = json.loads(self._body())
                n = pron.save(str(d.get("text") or ""))
                if d.get("test"):
                    return self._json({"n": n, "out": pron.apply(d["test"])})
                return self._json({"n": n})
            if u.path == "/api/titleset/upload":  # 这条任务专用的封面照片
                from . import titleset
                j = Q.get(urllib.parse.parse_qs(u.query).get("id", [""])[0])
                if not j:
                    return self._json({"error": "任务不存在"}, 404)
                return self._json(titleset.upload(j, self._body()))
            if u.path in ("/api/titleset/preview", "/api/titleset/save"):
                import importlib
                from . import titleset
                importlib.reload(titleset)
                d = json.loads(self._body())
                j = Q.get(d.get("id", ""))
                if not j:
                    return self._json({"error": "任务不存在"}, 404)
                if u.path.endswith("preview"):
                    return self._json({"img": titleset.preview(j, d.get("kind", "cover"), d.get("orient", "portrait"),
                                                               d.get("params") or {})})
                if d.get("kind") == "video" and j["state"] in ("处理中", "排队中"):
                    return self._json({"error": "这条任务正在处理，等它完成再改视频标题"}, 400)
                return self._json(titleset.save(j, d.get("settings") or {}, d.get("kind", "cover"), Q))
            if u.path == "/api/title":  # 按文案内容让 AI 提炼标题
                import importlib
                from . import title_ai
                importlib.reload(title_ai)
                d = json.loads(self._body())
                t, src = title_ai.suggest(d.get("script") or "", d.get("style", ""))
                return self._json({"title": t, "source": src})
            if u.path.startswith("/api/fix/"):
                return self._fix(u)
            if u.path == "/api/upload" and urllib.parse.parse_qs(u.query).get("kind", [""])[0] == "material":
                name = urllib.parse.unquote(self.headers.get("X-Filename", "素材.mp4"))
                return self._json({"materials": materials.add(name, self._body())})
            if u.path == "/api/upload":
                kind = urllib.parse.parse_qs(u.query).get("kind", ["avatar"])[0]
                name = urllib.parse.unquote(self.headers.get("X-Filename", "upload"))
                name = re.sub(r'[\\/:*?"<>|]+', "_", os.path.basename(name))
                sub = "数字人" if kind == "avatar" else "声音"
                dst = os.path.join(ROOT, "00_素材库", sub, name)
                with open(dst, "wb") as f:
                    f.write(self._body())
                return self._json({"ok": True, "file": rel(dst)})
            self._json({"error": "not found"}, 404)
        except Exception as e:
            _errlog(self.path, e)
            self._json({"error": str(e)}, 500)


def main():
    global Q
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    Q = Queue()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}"
    print(f"AI_UP 工作台已启动：{url}\n关闭这个窗口就会停止工作台（正在跑的任务下次打开会继续）。")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
