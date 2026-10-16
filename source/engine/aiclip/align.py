"""字幕与音频自动对齐。

很多 SRT 来自剪映/其他工具导出的版本，与原视频存在累计漂移（例如原视频停顿更长）。
方法：
  1) 能量 VAD 得到真实语音活动序列；
  2) SRT 生成"字幕活动"序列；
  3) 带斜率约束的 DTW 求单调时间映射（首尾锚定）；
  4) 每条字幕边界吸附到附近的语音起止点。
默认优先：本地 sherpa-onnx 语音识别逐字对齐（见 asr.py），DTW 作为兜底。
"""
import numpy as np

from .utils import log


def _agreement(subs, act, hop):
    sa = np.zeros(len(act), bool)
    for s in subs:
        sa[int(s["start"] / hop): int(s["end"] / hop)] = True
    n = min(len(sa), len(act))
    inter = (sa[:n] & act[:n]).sum()
    return inter / max(1, act[:n].sum()), inter / max(1, sa[:n].sum())


def dtw_align(subs, act, hop=0.01, res=0.1):
    R = int(round(res / hop))
    K = len(act) // R
    va = act[: K * R].reshape(K, R).mean(1).astype(np.float32)
    T = max(s["end"] for s in subs)
    N = int(T / res) + 2
    sa = np.zeros(N, np.float32)
    for s in subs:
        sa[int(s["start"] / res): int(s["end"] / res)] = 1
    on = np.where(va > 0.3)[0]
    j_first, j_last = (on[0], on[-1]) if len(on) else (0, K - 1)
    j_end = min(K - 1, j_last + 1)
    INF = np.float32(1e9)
    C = np.abs(sa[:, None] - va[None, :]).astype(np.float32)
    D = np.full((N, K), INF, np.float32)
    # 起点允许在视频前 3 秒内自由选择
    D[0, : max(1, min(K, int(3 / res)))] = C[0, : max(1, min(K, int(3 / res)))]
    for i in range(1, N):
        a = np.full(K, INF, np.float32)
        a[1:] = D[i - 1, :-1] + C[i, 1:]
        a[2:] = np.minimum(a[2:], D[i - 1, :-2] + 2 * C[i, 2:])
        if i >= 2:
            a[1:] = np.minimum(a[1:], D[i - 2, :-1] + 2 * C[i, 1:])
        D[i] = a
    i_end = min(N - 1, int(T / res))
    i, j = i_end, j_end
    if D[i, j] >= INF:
        j = int(np.argmin(D[i]))
    path = [(i, j)]
    while i > 0 and j > 0:
        c = [(D[i - 1, j - 1], i - 1, j - 1)]
        if j >= 2:
            c.append((D[i - 1, j - 2], i - 1, j - 2))
        if i >= 2:
            c.append((D[i - 2, j - 1], i - 2, j - 1))
        _, i, j = min(c)
        path.append((i, j))
    path = np.array(path[::-1], np.float64) * res
    out = []
    for s in subs:
        a = float(np.interp(s["start"], path[:, 0], path[:, 1]))
        b = float(np.interp(s["end"], path[:, 0], path[:, 1]))
        out.append(dict(s, start=a, end=max(b, a + 0.2)))
    return out


def snap(subs, segs, win=0.35):
    starts = np.array([a for a, b in segs]) if segs else np.zeros(0)
    ends = np.array([b for a, b in segs]) if segs else np.zeros(0)
    out = []
    for s in subs:
        a, b = s["start"], s["end"]
        if len(starts):
            k = np.argmin(np.abs(starts - a))
            if abs(starts[k] - a) < win:
                a = float(starts[k])
            k = np.argmin(np.abs(ends - b))
            if abs(ends[k] - b) < win:
                b = float(ends[k])
        out.append(dict(s, start=a, end=max(b, a + 0.2)))
    # 去重叠
    for i in range(1, len(out)):
        if out[i]["start"] < out[i - 1]["end"]:
            mid = (out[i]["start"] + out[i - 1]["end"]) / 2
            out[i - 1]["end"] = max(out[i - 1]["start"] + 0.15, mid)
            out[i]["start"] = max(mid, out[i - 1]["end"])
    return out


def align_subs(subs, act, segs, hop=0.01, mode="auto", audio_path=None, work_dir="."):
    """mode: auto（有本地 ASR 用 ASR，否则 DTW）| asr | dtw | none"""
    rec, prec = _agreement(subs, act, hop)
    speech_end = segs[-1][1] if segs else 0
    drift = abs(subs[-1]["end"] - speech_end)
    log(f"原始字幕-语音一致性: recall={rec:.2f} precision={prec:.2f} 尾部偏差={drift:.1f}s")
    if mode == "none":
        return subs, {"method": "none"}
    # DTW 结果始终计算，作为 ASR 低置信度句子的兜底
    if drift < 1.0 and rec > 0.8 and prec > 0.8:
        base, base_m = snap(subs, segs, 0.25), "snap_only"
    else:
        base, base_m = snap(dtw_align(subs, act, hop), segs), "dtw"
    if mode in ("auto", "asr") and audio_path:
        from . import asr
        if asr.available():
            try:
                chars = asr.transcribe(audio_path, work_dir)
                out, info = asr.align_with_asr(subs, chars, segs)
                if out is not None:
                    # 低置信度句子：用 DTW 结果按邻句平移修正
                    fixed = 0
                    for i, s in enumerate(out):
                        if s.get("conf", 1) < 0.35:
                            d = base[i]
                            s["start"], s["end"] = d["start"], d["end"]
                            fixed += 1
                    for i in range(1, len(out)):
                        if out[i]["start"] < out[i - 1]["end"]:
                            out[i - 1]["end"] = max(out[i - 1]["start"] + 0.2, out[i]["start"])
                        if out[i]["end"] <= out[i]["start"]:
                            out[i]["end"] = out[i]["start"] + 0.3
                    rec2, prec2 = _agreement(out, act, hop)
                    info.update({"fallback_subs": fixed, "recall": round(rec2, 3), "precision": round(prec2, 3)})
                    log(f"ASR 逐字对齐：匹配率 {info['match_ratio']:.0%}，CTC 延迟校准 {info['lag']:.2f}s，"
                        f"低置信度 {fixed} 句用 DTW 兜底；对齐后 recall={rec2:.2f} precision={prec2:.2f}")
                    return out, info
                log("ASR 匹配率过低（字幕与语音内容差异大），改用 DTW：", info)
            except Exception as e:
                log("本地 ASR 不可用，改用 DTW：", e)
        elif mode == "asr":
            log("未找到 models/sherpa-onnx，改用 DTW")
    rec2, prec2 = _agreement(base, act, hop)
    log(f"{base_m} 对齐后: recall={rec2:.2f} precision={prec2:.2f}")
    return base, {"method": base_m, "drift_before": drift, "recall": rec2, "precision": prec2}
