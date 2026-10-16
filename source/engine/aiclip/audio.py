"""音频读取、能量 VAD、剪切拼接、混音与响度处理（纯 numpy，速度快）。"""
import subprocess

import numpy as np

from .utils import ffmpeg_bin, run


def load_audio(path, sr=16000, channels=1):
    cmd = [ffmpeg_bin(), "-v", "error", "-i", path, "-vn", "-ac", str(channels), "-ar", str(sr),
           "-f", "s16le", "-"]
    r = run(cmd)
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels)
    return x


def energy_db(x, sr, hop=0.01):
    h = int(sr * hop)
    n = len(x) // h
    e = np.sqrt((x[: n * h].reshape(n, h) ** 2).mean(1) + 1e-12)
    return 20 * np.log10(e + 1e-6)


def vad(x, sr, hop=0.01, rel=0.35, min_speech=0.08, merge_gap=0.12):
    """返回 (逐帧布尔数组, 语音区间列表[秒])。阈值相对噪声底/语音电平自适应。"""
    db = energy_db(x, sr, hop)
    lo, hi = np.percentile(db, 10), np.percentile(db, 90)
    thr = lo + rel * (hi - lo)
    act = db > thr
    segs, s = [], None
    for i, b in enumerate(act):
        if b and s is None:
            s = i
        elif not b and s is not None:
            segs.append([s * hop, i * hop])
            s = None
    if s is not None:
        segs.append([s * hop, len(act) * hop])
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] < merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    merged = [m for m in merged if m[1] - m[0] >= min_speech]
    return act, merged, db, thr


def cut_concat(x, sr, intervals, fade=0.008):
    """按区间切片并拼接，每段做极短淡入淡出避免爆音。x: (n,) 或 (n,c)。"""
    nf = max(1, int(fade * sr))
    parts = []
    for a, b in intervals:
        seg = x[int(round(a * sr)): int(round(b * sr))].copy()
        if len(seg) > 2 * nf:
            ramp = np.linspace(0, 1, nf, dtype=np.float32)
            if seg.ndim == 2:
                ramp = ramp[:, None]
            seg[:nf] *= ramp
            seg[-nf:] *= ramp[::-1]
        parts.append(seg)
    return np.concatenate(parts) if parts else x[:0]


def normalize_speech(x, target_db=-17.0, peak=0.89):
    mono = x.mean(1) if x.ndim == 2 else x
    db = energy_db(mono, 48000, 0.05)
    loud = db[db > np.percentile(db, 40)]
    cur = float(np.mean(loud)) if len(loud) else -30
    g = 10 ** ((target_db - cur) / 20)
    g = float(np.clip(g, 0.3, 8.0))
    y = x * g
    # 软限幅
    y = np.tanh(y / peak) * peak
    return y.astype(np.float32)


def mix_at(track, clip, t, sr, gain=1.0):
    i = int(t * sr)
    if i >= len(track):
        return
    n = min(len(clip), len(track) - i)
    if track.ndim == 2 and clip.ndim == 1:
        clip = np.repeat(clip[:, None], track.shape[1], 1)
    track[i: i + n] += clip[:n] * gain


def save_wav(x, sr, path):
    y = np.clip(np.nan_to_num(x), -1, 1)
    y = (y * 32767).astype(np.int16)
    ch = 1 if y.ndim == 1 else y.shape[1]
    p = subprocess.Popen([ffmpeg_bin(), "-v", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", str(ch),
                          "-i", "-", path], stdin=subprocess.PIPE)
    p.communicate(y.tobytes())
    if p.returncode:
        raise RuntimeError("写音频失败")
