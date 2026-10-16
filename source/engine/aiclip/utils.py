"""通用工具：日志、ffmpeg 调用、字体查找、缓动函数。"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

_T0 = time.time()


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def ffmpeg_bin():
    p = os.environ.get("AICLIP_FFMPEG") or shutil.which("ffmpeg")
    if p:
        return p
    try:  # pip install imageio-ffmpeg 自带静态 ffmpeg
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def ffprobe_bin():
    p = os.environ.get("AICLIP_FFPROBE")
    if p:
        return p
    f = ffmpeg_bin()
    cand = os.path.join(os.path.dirname(f), "ffprobe" + (".exe" if f.lower().endswith(".exe") else ""))
    return cand if os.path.exists(cand) else (shutil.which("ffprobe") or "ffprobe")


def run(cmd, check=True, quiet=True):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and r.returncode != 0:
        err = r.stderr.decode("utf-8", "ignore").strip()[-800:]
        raise RuntimeError("命令失败: " + " ".join(map(str, cmd[:6])) + " ...\n" + err)
    return r


def _probe_by_ffmpeg(path):
    """没有 ffprobe 时：解析 ffmpeg -i 的输出。"""
    import re
    r = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    txt = r.stderr.decode("utf-8", "ignore")
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", txt)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0
    v = re.search(r"Stream.*Video:.*?(\d{2,5})x(\d{2,5})", txt)
    f = re.search(r"([\d.]+) fps", txt)
    rot = re.search(r"rotate\s*:\s*(-?\d+)|rotation of (-?[\d.]+)", txt)
    w, h = (int(v.group(1)), int(v.group(2))) if v else (0, 0)
    if rot and abs(int(float(rot.group(1) or rot.group(2)))) in (90, 270):
        w, h = h, w
    return {"duration": dur, "has_audio": "Audio:" in txt, "w": w, "h": h, "fps": float(f.group(1)) if f else 30.0}


def probe(path):
    if not (shutil.which(ffprobe_bin()) or os.path.exists(ffprobe_bin())):
        return _probe_by_ffmpeg(path)
    r = run([ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    info = json.loads(r.stdout.decode("utf-8", "ignore"))
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    out = {"duration": float(info["format"].get("duration", 0)), "has_audio": a is not None}
    if v:
        w, h = int(v["width"]), int(v["height"])
        rot = 0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(sd["rotation"])
        if "tags" in v and "rotate" in v["tags"]:
            rot = int(v["tags"]["rotate"])
        if abs(rot) in (90, 270):
            w, h = h, w
        num, den = v.get("avg_frame_rate", "30/1").split("/")
        fps = float(num) / float(den) if float(den) else 30.0
        out.update({"w": w, "h": h, "fps": fps})
    return out


# ---------------- 字体 ----------------
def font_dirs():
    dirs = []
    env = os.environ.get("AICLIP_FONT_DIRS")
    if env:
        dirs += [d for d in env.split(os.pathsep) if d]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs.append(os.path.join(here, "fonts"))
    if os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(os.path.join(windir, "Fonts"))
        la = os.environ.get("LOCALAPPDATA")
        if la:
            dirs.append(os.path.join(la, "Microsoft", "Windows", "Fonts"))
    else:
        dirs += ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
                 os.path.expanduser("~/Library/Fonts"), "/Library/Fonts", "/System/Library/Fonts"]
    return [d for d in dirs if os.path.isdir(d)]


_FONT_INDEX = None


def _index_fonts():
    global _FONT_INDEX
    if _FONT_INDEX is None:
        _FONT_INDEX = {}
        for d in font_dirs():
            for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
                if p.lower().endswith((".ttf", ".ttc", ".otf")):
                    _FONT_INDEX.setdefault(os.path.basename(p).lower(), p)
    return _FONT_INDEX


def find_font(candidates):
    """candidates: 文件名或绝对路径列表，返回第一个存在的。"""
    idx = _index_fonts()
    for c in candidates:
        if os.path.isabs(c) and os.path.exists(c):
            return c
        p = idx.get(os.path.basename(c).lower())
        if p:
            return p
    raise FileNotFoundError("找不到字体，候选: %s；可设置环境变量 AICLIP_FONT_DIRS 或把字体放进 fonts/ 目录" % candidates)


# ---------------- 缓动 ----------------
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease_out_cubic(p):
    p = clamp(p)
    return 1 - (1 - p) ** 3


def ease_in_out_cubic(p):
    p = clamp(p)
    return 4 * p * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_out_back(p, s=1.70158):
    p = clamp(p)
    p -= 1
    return p * p * ((s + 1) * p + s) + 1


def lerp(a, b, p):
    return a + (b - a) * p


def deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------- 图像/视频读取（兼容 Windows 中文路径） ----------------
def imread(path):
    import cv2
    import numpy as np
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img, params=None):
    """cv2.imwrite 在 Windows 中文路径下会写出乱码文件名，统一用这个。"""
    import cv2
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img, params or [])
    if ok:
        buf.tofile(path)
    return ok


def read_frames(path, times, width=None):
    """用 ffmpeg 精确抽取若干时间点的帧（BGR）。width 指定则等比缩放。"""
    import numpy as np
    info = probe(path)
    w, h = info["w"], info["h"]
    if width:
        h = int(round(h * width / w / 2)) * 2
        w = width
    out = []
    for t in times:
        r = subprocess.run([ffmpeg_bin(), "-v", "error", "-ss", f"{max(0, t):.3f}", "-i", path, "-frames:v", "1",
                            "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if len(r.stdout) >= w * h * 3:
            out.append(np.frombuffer(r.stdout[: w * h * 3], np.uint8).reshape(h, w, 3).copy())
    return out


class FFReader:
    """顺序读取视频帧（ffmpeg 管道），可循环。"""

    def __init__(self, path, fps, size):
        self.path, self.fps, self.size = path, fps, size
        self.p = None
        self.idx = -1
        self.frame = None

    def _open(self):
        if self.p:
            self.p.kill()
        w, h = self.size
        self.p = subprocess.Popen([ffmpeg_bin(), "-v", "error", "-stream_loop", "-1", "-i", self.path, "-an",
                                   "-vf", f"fps={self.fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
                                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
        self.idx = -1

    def get(self, k):
        import numpy as np
        if self.p is None or k < self.idx:
            self._open()
        w, h = self.size
        while self.idx < k:
            buf = self.p.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            self.frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            self.idx += 1
        return self.frame
