"""程序合成音效（无需素材库）：转场 whoosh、弹字 pop、章节 ding。
如需替换为自己的音效，在 assets/sfx/ 下放 whoosh.wav / pop.wav / ding.wav 即可。"""
import os

import numpy as np

from .audio import load_audio

SR = 48000
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _whoosh(d=0.42, seed=1):
    rng = np.random.RandomState(seed)
    n = int(SR * d)
    x = rng.randn(n).astype(np.float32)
    # 扫频带通：一阶低通/高通级联，截止频率随时间上扬再回落
    t = np.linspace(0, 1, n, dtype=np.float32)
    fc = 300 + 3500 * np.sin(np.pi * t) ** 2
    y = np.zeros(n, np.float32)
    lp = hp = prev = 0.0
    for i in range(n):
        a = 2 * np.pi * fc[i] / SR
        a = a / (a + 1)
        lp = lp + a * (x[i] - lp)
        hp = 0.96 * (hp + lp - prev)
        prev = lp
        y[i] = hp
    env = np.clip(np.sin(np.pi * t), 0, 1) ** 1.6
    y = y * env
    return y / (np.abs(y).max() + 1e-6) * 0.8


def _pop(d=0.16):
    n = int(SR * d)
    t = np.arange(n, dtype=np.float32) / SR
    f = 900 * np.exp(-t * 22) + 220
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t * 28)
    return (y * 0.9).astype(np.float32)


def _ding(d=0.9):
    n = int(SR * d)
    t = np.arange(n, dtype=np.float32) / SR
    y = (np.sin(2 * np.pi * 1318.5 * t) * 0.6 + np.sin(2 * np.pi * 2637 * t) * 0.25 +
         np.sin(2 * np.pi * 1975.5 * t) * 0.15) * np.exp(-t * 5.5)
    y[: int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))
    return (y * 0.7).astype(np.float32)


_CACHE = {}


def get(kind):
    if kind in _CACHE:
        return _CACHE[kind]
    p = os.path.join(_HERE, "assets", "sfx", kind + ".wav")
    if os.path.exists(p):
        y = load_audio(p, SR, 1)
    else:
        y = {"whoosh": _whoosh, "pop": _pop, "ding": _ding}[kind]()
    _CACHE[kind] = y
    return y
