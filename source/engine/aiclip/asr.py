"""本地语音识别对齐（sherpa-onnx，离线、无需联网、无需 GPU、无需 pip 安装）。

用 ASR 得到“每个字的发音时间”，再把 SRT 文本逐字对上去，从而得到准确的字幕起止时间。
即使 ASR 有个别错字（同音字），也只影响那几个字，靠前后已匹配的字插值，整体仍然准确。

需要的文件（已随项目放在 models/sherpa-onnx/ 下）：
  bin/sherpa-onnx(.exe) + onnxruntime 动态库
  model/model.int8.onnx + model/tokens.txt   （streaming-zipformer-small-ctc-zh-int8-2025-04-01，约 26MB）
来源：https://github.com/k2-fsa/sherpa-onnx （Apache-2.0）
"""
import difflib
import json
import os
import re
import subprocess

import numpy as np

from .utils import ffmpeg_bin, log

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUNCT = re.compile(r"[\s，。！？、,.!?；;：:“”\"'‘’（）()《》<>【】\[\]…—\-~·]")


def _env(root, exe):
    env = dict(os.environ)
    if os.name == "nt":
        # onnxruntime.dll 依赖 VC++ 运行库（msvcp140/vcruntime140）；未安装运行库时借用 Python/OpenCV 自带的副本
        import sys
        extra = [os.path.dirname(exe), sys.base_prefix, os.path.join(sys.base_prefix, "DLLs")]
        try:
            import cv2
            extra.append(os.path.dirname(cv2.__file__))
        except Exception:
            pass
        env["PATH"] = os.pathsep.join(extra) + os.pathsep + env.get("PATH", "")
    else:
        env["LD_LIBRARY_PATH"] = os.path.join(root, "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def sherpa_paths():
    root = os.environ.get("AICLIP_SHERPA_DIR") or os.path.join(HERE, "models", "sherpa-onnx")
    exe = os.path.join(root, "bin", "sherpa-onnx.exe" if os.name == "nt" else "sherpa-onnx")
    model = os.path.join(root, "model", "model.int8.onnx")
    tokens = os.path.join(root, "model", "tokens.txt")
    if all(os.path.exists(p) for p in (exe, model, tokens)):
        return root, exe, model, tokens
    return None


def available():
    return sherpa_paths() is not None


def selfcheck():
    """doctor 用：识别 1 秒静音，确认程序能启动。返回 (ok, 信息)。"""
    import tempfile
    import wave
    p = sherpa_paths()
    if not p:
        return False, "未找到 models/sherpa-onnx"
    root, exe, model, tokens = p
    d = tempfile.mkdtemp(prefix="aiclip_chk_")
    fn = os.path.join(d, "s.wav")
    with wave.open(fn, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    env = _env(root, exe)
    try:
        r = subprocess.run([exe, f"--tokens={tokens}", f"--zipformer2-ctc-model={model}", fn],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=60)
        ok = r.returncode == 0 and '"tokens"' in r.stdout.decode("utf-8", "ignore")
        return ok, ("正常" if ok else r.stdout.decode("utf-8", "ignore")[-300:])
    except Exception as e:
        return False, str(e)


def _chunks(x, sr, max_len=18.0, min_len=4.0):
    """在停顿处把长音频切成 <=18 秒的小段。流式模型对长音频的时间戳会累计漂移（实测 4 分钟漂 1.5 秒），
    分段识别后每段误差 <0.1 秒。"""
    from .audio import vad
    _, segs, _, _ = vad(x, sr)
    cuts = [0.0]
    pauses = [((segs[i][1] + segs[i + 1][0]) / 2, segs[i + 1][0] - segs[i][1]) for i in range(len(segs) - 1)]
    total = len(x) / sr
    while total - cuts[-1] > max_len:
        cand = [(p, d) for p, d in pauses if cuts[-1] + min_len < p <= cuts[-1] + max_len]
        nxt = max(cand, key=lambda pd: (pd[1] > 0.25, pd[0]))[0] if cand else cuts[-1] + max_len
        cuts.append(nxt)
    cuts.append(total)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def transcribe(video_path, work_dir, threads=None):
    """返回 [(字, 时间秒)]（时间为 CTC 峰值时刻，略晚于实际起音，后续会校准）。"""
    import wave
    from .audio import load_audio
    p = sherpa_paths()
    if not p:
        raise FileNotFoundError("未找到 models/sherpa-onnx（本地语音识别组件）")
    root, exe, model, tokens = p
    sr = 16000
    x = load_audio(video_path, sr, 1)
    # 识别程序对非英文路径支持不好：分段音频放在纯英文路径的临时目录
    import tempfile
    base = tempfile.gettempdir()
    if not base.isascii():
        base = os.path.join(root, "tmp")
    os.makedirs(base, exist_ok=True)
    cdir = tempfile.mkdtemp(prefix="aiclip_asr_", dir=base)
    chunks = _chunks(x, sr)
    files = []
    for k, (a, b) in enumerate(chunks):
        fn = os.path.join(cdir, f"c{k:04d}.wav")
        seg = (np.clip(x[int(a * sr): int(b * sr)], -1, 1) * 32767).astype(np.int16)
        with wave.open(fn, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(seg.tobytes())
        files.append(fn)
    env = _env(root, exe)
    threads = threads or max(1, min(4, (os.cpu_count() or 2)))
    r = subprocess.run([exe, f"--tokens={tokens}", f"--zipformer2-ctc-model={model}", f"--num-threads={threads}"] + files,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    out = r.stdout.decode("utf-8", "ignore")
    if r.returncode != 0:
        raise RuntimeError("语音识别失败：" + out[-500:])
    res, cur = {}, None
    names = {os.path.basename(f): k for k, f in enumerate(files)}
    for line in out.splitlines():
        line = line.strip()
        base = os.path.basename(line.replace("\\", "/"))
        if base in names:
            cur = names[base]
        elif line.startswith("{") and '"tokens"' in line and cur is not None:
            res[cur] = json.loads(line)
    if not res:
        raise RuntimeError("语音识别无输出：" + out[-500:])
    chars = []
    for k, (a, b) in enumerate(chunks):
        j = res.get(k)
        if j:
            chars += [(c, t + a) for c, t in _tokens_to_chars(j["text"], j["tokens"], j["timestamps"])]
    import shutil
    shutil.rmtree(cdir, ignore_errors=True)
    return chars


def _tokens_to_chars(text, tokens, stamps):
    """字节级 BPE：每个符号对应 1 个 UTF-8 字节。按字节偏移把 token 时间映射到字符。"""
    tb = []
    pos = 0
    for tk, ts in zip(tokens, stamps):
        n = len(tk.replace("▁", ""))
        tb.append((pos, ts))
        pos += n
    starts = [b for b, _ in tb]
    res = []
    off = 0
    for ch in text:
        k = max(0, np.searchsorted(starts, off, side="right") - 1) if starts else 0
        t = tb[k][1] if tb else 0.0
        if ch.strip():
            res.append((ch, float(t)))
        off += len(ch.encode("utf-8"))
    return res


def _calibrate_lag(chars, vad_segs):
    """CTC 峰值通常比实际起音晚 0.2~0.4s：用“语音段起点 ↔ 段内第一个字”的中位差校准。"""
    if not chars or not vad_segs:
        return 0.25
    ts = np.array([t for _, t in chars])
    d = []
    for a, b in vad_segs:
        if b - a < 0.3:
            continue
        k = np.searchsorted(ts, a - 0.05)
        if k < len(ts) and ts[k] - a < 0.9:
            d.append(ts[k] - a)
    return float(np.clip(np.median(d), 0.0, 0.6)) if len(d) >= 5 else 0.25


def align_with_asr(subs, chars, vad_segs, char_dur=0.2):
    """把 SRT 逐字对齐到 ASR 字时间。返回 (新字幕, 统计信息)。"""
    lag = _calibrate_lag(chars, vad_segs)
    hyp = "".join(c for c, _ in chars)
    htime = np.array([t - lag for _, t in chars], np.float64)
    ref_chars, owner = [], []
    for i, s in enumerate(subs):
        for ch in PUNCT.sub("", s["text"]):
            ref_chars.append(ch)
            owner.append(i)
    ref = "".join(ref_chars)
    rtime = np.full(len(ref), np.nan)
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    for blk in sm.get_matching_blocks():
        rtime[blk.a: blk.a + blk.size] = htime[blk.b: blk.b + blk.size]
    matched = ~np.isnan(rtime)
    # 单调性修正：去掉时间倒退的匹配
    last = -1e9
    for k in range(len(rtime)):
        if matched[k]:
            if rtime[k] < last:
                matched[k] = False
                rtime[k] = np.nan
            else:
                last = rtime[k]
    idx = np.where(matched)[0]
    if len(idx) < max(5, len(ref) * 0.2):
        return None, {"method": "asr_failed", "match_ratio": float(len(idx) / max(1, len(ref)))}
    # 未匹配字：在相邻已匹配字之间线性插值
    filled = np.interp(np.arange(len(ref)), idx, rtime[idx])
    # 未匹配段跨越停顿时，线性插值会把句首/句尾拉进停顿里：
    # 属于后一句的字从下一个匹配字往前按正常语速倒推，属于前一句的字从上一个匹配字往后顺推
    own = np.array(owner)
    rate = 0.2
    for k in range(len(ref)):
        if matched[k]:
            continue
        p0 = idx[idx < k]
        p1 = idx[idx > k]
        if len(p1) and own[p1[0]] == own[k]:
            filled[k] = max(filled[k], rtime[p1[0]] - (p1[0] - k) * rate)
        elif len(p0) and own[p0[-1]] == own[k]:
            filled[k] = min(filled[k], rtime[p0[-1]] + (k - p0[-1]) * rate)
    starts_v = np.array([a for a, b in vad_segs]) if vad_segs else np.zeros(0)
    ends_v = np.array([b for a, b in vad_segs]) if vad_segs else np.zeros(0)
    out, confs = [], []
    owner = np.array(owner)
    prev_last = -1e9
    for i, s in enumerate(subs):
        ks = np.where(owner == i)[0]
        if len(ks) == 0:
            out.append(dict(s))
            confs.append(0)
            continue
        conf = float(matched[ks].mean())
        # 稳健估计：CTC 单字时间有 ±0.3s 抖动，用句首/句尾各 4 个已匹配字按语速回推后取中位数
        hm = [k for k in ks[:4] if matched[k]]
        a = float(np.median([rtime[k] - (k - ks[0]) * rate for k in hm])) if len(hm) >= 2 else float(filled[ks[0]])
        tm_ = [k for k in ks[-4:] if matched[k]]
        b = (float(np.median([rtime[k] + (ks[-1] - k) * rate for k in tm_])) if len(tm_) >= 2
             else float(filled[ks[-1]])) + char_dur
        a = max(a, prev_last + 0.02)
        # 起点吸附到附近语音起点，但不能早于上一句最后一个字（避免句子在停顿处被拆开时提前出字幕）
        if len(starts_v):
            # 句首通常紧跟一个停顿：在“上一句最后一个字之后、本句首字估计时间 +0.3s 之前”找语音起点，取最接近的
            lo = max(prev_last + 0.08, a - 0.6)
            cand = starts_v[(starts_v > lo) & (starts_v < a + 0.3)]
            if len(cand):
                a = float(cand[np.argmin(np.abs(cand - a))])
            j = np.argmin(np.abs(ends_v - b))
            if abs(ends_v[j] - b) < 0.3:
                b = float(ends_v[j])
        prev_last = max(prev_last, min(float(filled[ks[-1]]), b - char_dur))
        out.append(dict(s, start=a, end=max(b, a + 0.25), conf=round(conf, 2)))
        confs.append(conf)
    # 去重叠 + 单调
    for i in range(1, len(out)):
        if out[i]["start"] < out[i - 1]["start"]:
            out[i]["start"] = out[i - 1]["start"] + 0.05
        if out[i]["start"] < out[i - 1]["end"]:
            out[i - 1]["end"] = max(out[i - 1]["start"] + 0.2, out[i]["start"])
        out[i]["end"] = max(out[i]["end"], out[i]["start"] + 0.25)
    info = {"method": "asr", "lag": round(lag, 3), "match_ratio": round(float(matched.mean()), 3),
            "low_conf_subs": int(sum(c < 0.4 for c in confs))}
    return out, info


def check_sync(video_path, subs_out, work_dir):
    """成片质检：对成片重新做一次语音识别，比较成片字幕与实际发音的时间差。"""
    from .audio import load_audio, vad
    x = load_audio(video_path, 16000, 1)
    _, segs, _, _ = vad(x, 16000)
    chars = transcribe(video_path, work_dir)
    ref, info = align_with_asr(subs_out, chars, segs)
    if ref is None:
        return {"ok": False, "reason": "成片识别匹配率过低", **info}
    d = np.array([abs(a["start"] - b["start"]) for a, b in zip(subs_out, ref) if b.get("conf", 0) >= 0.5])
    if not len(d):
        return {"ok": False, "reason": "无可比较的句子"}
    res = {"ok": bool(np.percentile(d, 90) < 0.5 and np.median(d) < 0.15), "median_err": round(float(np.median(d)), 3),
           "p90_err": round(float(np.percentile(d, 90)), 3), "max_err": round(float(d.max()), 3),
           "over_0.5s": int((d > 0.5).sum()), "checked": int(len(d)), "match_ratio": info["match_ratio"]}
    return res
