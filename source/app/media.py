"""FFmpeg 小工具：找 ffmpeg、转码配音和数字人素材。"""
import os
import re
import shutil
import subprocess

# 送进 HeyGem 的视频短边最多 720：4GB 显存跑得动、速度快；清晰度由“画质还原”从原片补回来
LIPSYNC_SIDE = 720


def ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # AICLIP 的 setup.bat 装过它
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("找不到 ffmpeg：请把 ffmpeg 加入 PATH，或运行 pip install imageio-ffmpeg")


def run(args):
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error"] + args
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("ffmpeg 失败：" + (r.stderr or "").strip()[-800:])


def info(path):
    """(宽, 高, 时长秒)，考虑手机视频的旋转标记。"""
    r = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", r.stderr)
    d = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    if not m or not d:
        raise RuntimeError(f"读不出视频信息：{path}")
    w, h = int(m.group(1)), int(m.group(2))
    if re.search(r"rotate\s*:\s*-?(90|270)|rotation of -?(90|270)", r.stderr):
        w, h = h, w
    return w, h, int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))


def small_scale(side=LIPSYNC_SIDE):
    """短边缩到 side 以内（只缩不放）的 scale 滤镜。"""
    return (f"scale='if(gte(iw,ih),-2,min(iw,{side}))':'if(gte(iw,ih),min(ih,{side}),-2)':flags=lanczos")


def prep_voice(src, dst):
    """参考音频 → 24kHz 单声道 wav，响度拉到 -16 LUFS（原录音偏小声）。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    run(["-i", src, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", dst])


def prep_lipsync_audio(src, dst):
    """给 HeyGem 的配音：16kHz 单声道（HeyGem 推荐格式，也更省显存）。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    run(["-i", src, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst])


def prep_avatar(src, dst, start=None, end=None, side=LIPSYNC_SIDE):
    """数字人素材 → 送 HeyGem 用的 H.264 25fps 小视频（短边 ≤720）。
    HeyGem 固定输出 25fps，预处理就压到 25fps；分辨率小显存占用低、速度快；成片清晰度由画质还原补回。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    args = []
    if start is not None:
        args += ["-ss", str(start)]
    if end is not None:
        args += ["-to", str(end)]
    args += ["-i", src, "-vf", f"fps=25,{small_scale(side)}", "-c:v", "libx264",
             "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p", "-threads", "4", "-an", dst]
    run(args)
