"""渲染：分段多进程合成 + 硬件编码（NVENC 自动检测）+ 音频处理 + 封装。"""
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import audio as A
from . import sfx as S
from .utils import ffmpeg_bin, load_json, log, run


def detect_encoder(pref="auto"):
    if pref in ("x264", "cpu"):
        return "x264"
    if pref == "auto" and (os.cpu_count() or 2) >= 8:
        # 瓶颈在逐帧合成（CPU）。多核机器用 x264 可以开 6 个并行分段；NVENC 受显存限制只能开 2 路，反而更慢
        return "x264"
    for enc in (["h264_nvenc"] if pref in ("auto", "nvenc") else []):
        r = subprocess.run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x240:d=0.2",
                            "-c:v", enc, "-f", "null", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode == 0:
            return "nvenc"
    return "x264"


def encoder_args(kind, st):
    e = st["encode"]
    if kind == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(e["nvenc_cq"]),
                "-b:v", "0", "-maxrate", "20M", "-bufsize", "40M", "-g", "60", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", e["x264_preset"], "-crf", str(e["crf"]), "-g", "60",
            "-pix_fmt", "yuv420p", "-threads", "2"]


def _render_chunk(job):
    # 子进程入口（Windows spawn 兼容：只传路径与数值）
    from .compositor import make_compositor
    from .planner import TimeMap
    tl = load_json(job["timeline"])
    st = load_json(job["style"])
    comp = make_compositor(tl, st)
    tm = TimeMap(tl["segments"])
    fps = tl["fps"]
    W, H = tl["canvas"]
    sw, sh = tl["src"]["w"], tl["src"]["h"]
    f0, f1 = job["f0"], job["f1"]
    ss = max(0.0, tm.out2src(f0 / fps) - 0.5)
    rd = subprocess.Popen([ffmpeg_bin(), "-v", "error", "-ss", f"{ss:.3f}", "-i", job["src"], "-an",
                           "-vf", f"fps={fps},scale={sw}:{sh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=sw * sh * 3 * 4)
    wr = subprocess.Popen([ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"] + job["enc"] + [job["out"]],
                          stdin=subprocess.PIPE, bufsize=W * H * 3 * 2)
    nbytes = sw * sh * 3
    cur, frame = -1, None
    t0 = time.time()
    prog = os.path.join(os.path.dirname(job["out"]), f"prog_chunk_{job['idx']:02d}.txt")
    for f in range(f0, f1):
        if (f - f0) % 15 == 0:
            try:
                with open(prog, "w") as pf:
                    pf.write(str(f - f0))
            except OSError:
                pass
        t = f / fps
        k = int(round((tm.out2src(t) - ss) * fps))
        while cur < k:
            buf = rd.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
            cur += 1
        if frame is None:
            frame = np.zeros((sh, sw, 3), np.uint8)
        out = comp.render(frame, t)
        wr.stdin.write(out.tobytes())
    wr.stdin.close()
    wr.wait()
    rd.kill()
    try:
        with open(prog, "w") as pf:
            pf.write(str(f1 - f0))
    except OSError:
        pass
    if wr.returncode != 0:
        raise RuntimeError(f"分段 {job['idx']} 编码失败（若使用 NVENC，可减少 --workers 或改用 --encoder x264）")
    return {"chunk": job["idx"], "frames": f1 - f0, "sec": time.time() - t0}


def render_video(tl_path, style_path, src, work, workers, enc_kind, st, frames=None):
    tl = load_json(tl_path)
    total = frames or tl["frames"]
    workers = max(1, workers)
    if enc_kind == "nvenc":
        workers = min(workers, 2)  # 笔记本显卡显存小，4 路 NVENC 同时开会 CUDA out of memory
    n_chunks = workers if total > 300 else 1
    bounds = [int(total * i / n_chunks) for i in range(n_chunks + 1)]
    enc = encoder_args(enc_kind, st)
    jobs = [{"idx": i, "timeline": tl_path, "style": style_path, "src": src, "f0": bounds[i], "f1": bounds[i + 1],
             "out": os.path.join(work, f"chunk_{i:02d}.mp4"), "enc": enc} for i in range(n_chunks)]
    log(f"渲染 {total} 帧（{total / tl['fps']:.1f}s），{n_chunks} 个分段并行，编码器={enc_kind}")
    import glob as _g
    import time as _t
    from . import progress as P
    for fp in _g.glob(os.path.join(work, "prog_chunk_*.txt")):
        os.remove(fp)
    P.report(work, "渲染画面", 0, frames_total=total, render_start=_t.time(), encoder=enc_kind)
    t0 = time.time()
    try:
        if n_chunks == 1:
            res = [_render_chunk(jobs[0])]
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(_render_chunk, jobs))
    except Exception as e:  # NVENC 开不起来时子进程报 BrokenPipe 等各种异常，统一在本进程内改用 x264，不必整条重跑
        if enc_kind != "nvenc":
            raise
        log("NVENC 编码失败，自动改用 x264 继续：", str(e).strip().splitlines()[-1] if str(e).strip() else repr(e))
        for j in jobs:
            j["enc"] = encoder_args("x264", st)
        w2 = max(1, min(6, (os.cpu_count() or 2) - 1))
        P.report(work, "渲染画面", 0, frames_total=total, render_start=_t.time(), encoder="x264")
        for fp in _g.glob(os.path.join(work, "prog_chunk_*.txt")):
            os.remove(fp)
        with ProcessPoolExecutor(max_workers=w2) as ex:
            res = list(ex.map(_render_chunk, jobs))
    dt = time.time() - t0
    log(f"画面渲染完成：{dt:.1f}s，平均 {total / dt:.1f} fps")
    lst = os.path.join(work, "chunks.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write("file '%s'\n" % os.path.abspath(j["out"]).replace("\\", "/"))
    return lst, total / tl["fps"]


def build_audio(tl, src, st, work, duration, has_audio=True):
    sr = 48000
    n = int(round(duration * sr))
    if has_audio:
        x = A.load_audio(src, sr, 2)
        y = A.cut_concat(x, sr, tl["segments"])
        y = A.normalize_speech(y, st["audio"]["target_db"])
    else:
        y = np.zeros((n, 2), np.float32)
    if len(y) < n:
        y = np.concatenate([y, np.zeros((n - len(y), 2), np.float32)])
    y = y[:n].copy()
    if st["audio"].get("sfx", True):
        g = 10 ** (st["audio"]["sfx_gain_db"] / 20)
        gains = {"whoosh": 0.8, "pop": 0.9, "ding": 0.55}
        for e in tl["sfx"]:
            if e["t"] < duration:
                A.mix_at(y, S.get(e["kind"]), e["t"], sr, g * gains.get(e["kind"], 1.0))
    bgm = st["audio"].get("bgm")
    if bgm and os.path.exists(bgm):
        b = A.load_audio(bgm, sr, 2)
        reps = int(np.ceil(n / max(1, len(b))))
        b = np.tile(b, (reps, 1))[:n] * (10 ** (st["audio"]["bgm_gain_db"] / 20))
        fade = min(n, sr * 2)
        b[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
        y = y + b
    y = np.tanh(y / 0.95) * 0.95
    p = os.path.join(work, "audio.wav")
    A.save_wav(y, sr, p)
    return p


def mux(chunk_list, audio_wav, out, st):
    run([ffmpeg_bin(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", chunk_list, "-i", audio_wav,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", st["encode"]["audio_bitrate"],
         "-shortest", "-movflags", "+faststart", out])
