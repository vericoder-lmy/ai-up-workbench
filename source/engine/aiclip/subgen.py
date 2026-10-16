"""自动生成字幕（不需要人工提供 SRT）。

流程：SenseVoice 离线识别（带标点、数字规整）→ 去语气词 → 替换表 → 拼音近似纠错（医学词库/账号专有词）
     → 按标点和停顿断句（每行 ≤ max_chars 字）→ 输出 SRT。之后照常走 ASR 逐字对齐，时间轴再精修一遍。
可选：配置大模型（llm.py）后再做一次上下文纠错（只改错字，不改句子结构）。

模型：sherpa-onnx SenseVoice int8（中/粤/英/日/韩），models/sensevoice/，约 240MB。
"""
import glob
import json
import os
import re
import subprocess
import tempfile
import wave

import numpy as np

from .asr import _calibrate_lag, _chunks, _env, sherpa_paths
from .utils import log

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SV_DIR = os.path.join(HERE, "models", "sensevoice")
SV_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
          "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2")
PUNCT_END = set("，。！？、；：,.!?;:")
FILLER_CHARS = set("哎呃嗯唉")
CUE_WORDS = {"开始", "开始了", "好", "好了", "好的", "准备", "来", "一二三", "OK", "ok", "可以了", "停"}
FILLER_CLAUSES = {"哎", "哎呀", "哎哟", "呃", "嗯", "额", "啊", "哦", "唉", "嗯嗯", "呃呃", "那个", "就是说", "然后呢"}


# ---------------- 模型 ----------------
def _assemble():
    """模型按 <20MB 分片随项目分发（便于同步），首次使用时合并。"""
    target = os.path.join(SV_DIR, "model.int8.onnx")
    if os.path.exists(target):
        return target
    parts = sorted(glob.glob(os.path.join(SV_DIR, "model.int8.onnx.part*")))
    if parts:
        tmp = target + ".tmp"
        with open(tmp, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    out.write(f.read())
        if os.path.getsize(tmp) != sum(os.path.getsize(p) for p in parts):
            os.remove(tmp)
            return None
        os.replace(tmp, target)
        for p in parts:  # 合并成功后删除分片，节省空间
            try:
                os.remove(p)
            except OSError:
                pass
        log(f"已合并 SenseVoice 模型分片 {len(parts)} 个")
        return target
    return None


def download_model():
    """兜底：从 GitHub 下载（可设置 AICLIP_GH_MIRROR，如 https://ghfast.top/ 作为前缀加速）。"""
    import bz2
    import io
    import tarfile
    import urllib.request
    os.makedirs(SV_DIR, exist_ok=True)
    url = os.environ.get("AICLIP_GH_MIRROR", "") + SV_URL
    log("下载 SenseVoice 模型（约 160MB）：", url)
    data = urllib.request.urlopen(url, timeout=600).read()
    with tarfile.open(fileobj=io.BytesIO(bz2.decompress(data))) as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if name in ("model.int8.onnx", "tokens.txt"):
                with tf.extractfile(m) as src, open(os.path.join(SV_DIR, name), "wb") as dst:
                    dst.write(src.read())


def model_paths(auto_download=True):
    model = _assemble()
    tokens = os.path.join(SV_DIR, "tokens.txt")
    if (not model or not os.path.exists(tokens)) and auto_download:
        try:
            download_model()
            model = _assemble()
        except Exception as e:
            log("SenseVoice 模型下载失败：", e)
    if model and os.path.exists(tokens):
        return model, tokens
    return None


def offline_exe():
    p = sherpa_paths()
    if not p:
        return None
    root, exe, _, _ = p
    off = os.path.join(os.path.dirname(exe), "sherpa-onnx-offline" + (".exe" if os.name == "nt" else ""))
    return (root, off) if os.path.exists(off) else None


def available():
    return offline_exe() is not None and (
        os.path.exists(os.path.join(SV_DIR, "model.int8.onnx")) or glob.glob(os.path.join(SV_DIR, "*.part*")))


# ---------------- 识别 ----------------
def recognize(video_path, threads=None):
    """返回 [(token, 时间秒)]，token 为单字或标点（SenseVoice 带标点与数字规整）。"""
    from .audio import load_audio, vad
    mp = model_paths()
    oe = offline_exe()
    if not mp or not oe:
        raise FileNotFoundError("缺少 SenseVoice 模型或 sherpa-onnx-offline 程序（models/）")
    model, tokens = mp
    root, exe = oe
    sr = 16000
    x = load_audio(video_path, sr, 1)
    _, segs, _, _ = vad(x, sr)
    chunks = _chunks(x, sr, max_len=25.0)
    base = tempfile.gettempdir()
    if not base.isascii():
        base = os.path.join(root, "tmp")
    os.makedirs(base, exist_ok=True)
    d = tempfile.mkdtemp(prefix="aiclip_sv_", dir=base)
    files = []
    for k, (a, b) in enumerate(chunks):
        fn = os.path.join(d, f"c{k:04d}.wav")
        with wave.open(fn, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(x[int(a * sr): int(b * sr)], -1, 1) * 32767).astype(np.int16).tobytes())
        files.append(fn)
    threads = threads or max(1, min(4, os.cpu_count() or 2))
    r = subprocess.run([exe, f"--tokens={tokens}", f"--sense-voice-model={model}", "--sense-voice-use-itn=1",
                        f"--num-threads={threads}"] + files, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       env=_env(root, exe))
    out = r.stdout.decode("utf-8", "ignore")
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    if r.returncode != 0:
        raise RuntimeError("SenseVoice 识别失败：" + out[-500:])
    names = {os.path.basename(f): k for k, f in enumerate(files)}
    res, cur = {}, None
    for line in out.splitlines():
        line = line.strip()
        bn = os.path.basename(line.replace("\\", "/"))
        if bn in names:
            cur = names[bn]
        elif line.startswith("{") and '"tokens"' in line and cur is not None:
            res[cur] = json.loads(line)
    units = []
    for k, (a, b) in enumerate(chunks):
        j = res.get(k)
        if not j:
            continue
        for tok, t in zip(j["tokens"], j["timestamps"]):
            if tok.startswith("▁") and tok[1:].isascii() and units and units[-1][0].isascii():
                units.append([" ", float(t) + a])
            for ch in tok.replace("▁", ""):  # 拆成单字，保证“字符位置 = 单元位置”
                if ch.strip():
                    units.append([ch, float(t) + a])
        if units and units[-1][0] not in PUNCT_END:
            units.append(["。", units[-1][1]])  # 分段边界视为句末
    lag = _calibrate_lag([(u, t) for u, t in units if u not in PUNCT_END and u.strip()], segs)
    for u in units:
        u[1] = max(0.0, u[1] - lag)
    return units, segs


# ---------------- 纠错 ----------------
_PY = None
_FUZZY_INI = [{"z", "zh"}, {"c", "ch"}, {"s", "sh"}, {"n", "l"}, {"f", "h"}, {"r", "l"}]
_FUZZY_FIN = [{"an", "ang"}, {"en", "eng"}, {"in", "ing"}, {"ian", "iang"}, {"uan", "uang"}, {"on", "ong"}]


_WORDS = None


def _words():
    global _WORDS
    if _WORDS is None:
        with open(os.path.join(HERE, "data", "words.txt"), encoding="utf-8") as f:
            _WORDS = set(f.read().split())
    return _WORDS


def _py():
    global _PY
    if _PY is None:
        _PY = {}
        with open(os.path.join(HERE, "data", "pinyin.tsv"), encoding="utf-8") as f:
            for line in f:
                c, p = line.rstrip("\n").split("\t")
                ini, fin = p.split(" ")
                _PY[c] = (ini, fin)
    return _PY


def _syl(c):
    return _py().get(c, (c, c))


def _syl_sim(a, b):
    if a == b:
        return 1.0
    (ia, fa), (ib, fb) = a, b
    ini_ok = ia == ib or any(ia in g and ib in g for g in _FUZZY_INI)
    fin_ok = fa == fb or any(fa in g and fb in g for g in _FUZZY_FIN)
    if ini_ok and fin_ok:
        return 0.85
    if fa == fb:
        return 0.5
    if ia == ib and ia:
        return 0.3
    return 0.0


def _one_off(w, t):
    """等长、≥3 字、只有一个音节不同且该音节声母相同或近似（左氨酶→转氨酶）。"""
    if len(w) != len(t) or len(t) < 3:
        return False
    diff = [(a, b) for a, b in zip(w, t) if _syl(a) != _syl(b)]
    if len(diff) != 1 or _syl(w[-1]) != _syl(t[-1]):
        return False  # 只允许非末字不同（末字通常是词的核心，不同则多半是另一个词）
    (ia, fa), (ib, fb) = _syl(diff[0][0]), _syl(diff[0][1])
    vow = set("aeiouv")
    if not (set(fa) & set(fb) & vow):
        return False  # 韵母毫无共同元音（yi/yan、zhi/zheng）不算近似
    return ia == ib or any(ia in g and ib in g for g in _FUZZY_INI)


def _seq_sim(w, t):
    """拼音序列相似度（带插入/删除的编辑距离）。"""
    A = [_syl(c) for c in w]
    B = [_syl(c) for c in t]
    n, m = len(A), len(B)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1 - _syl_sim(A[i - 1], B[j - 1]))
    return 1 - D[n, m] / max(n, m)


def load_terms(extra=()):
    terms = set(t for t in extra if t)
    for fn in ("医学词库.txt", "自定义词库.txt"):
        p = os.path.join(HERE, "dict", fn)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("#"):
                        continue
                    terms.update(w for w in line.split() if len(w) >= 2 and re.match(r"^[一-鿿]+$", w))
    return sorted(terms, key=len, reverse=True)


def load_terms_file(fn):
    p = os.path.join(HERE, "dict", fn)
    out = []
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip().startswith("#"):
                    out += [w for w in line.split() if len(w) >= 2]
    return out


def load_replacements(extra=None):
    rep = {}
    p = os.path.join(HERE, "dict", "替换表.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    a, b = line.split("=", 1)
                    if a.strip():
                        rep[a.strip()] = b.strip()
    rep.update(extra or {})
    return rep


def _replace_in_units(units, i0, i1, new):
    """把 units[i0:i1] 替换为 new 字符串（时间线性分配）。"""
    t0 = units[i0][1]
    t1 = units[i1 - 1][1] if i1 - 1 >= i0 else t0
    n = len(new)
    seg = [[ch, t0 + (t1 - t0) * k / max(1, n - 1)] for k, ch in enumerate(new)]
    return units[:i0] + seg + units[i1:]


TWO_CHAR_OK = {"乙肝", "丙肝", "甲肝", "戊肝"}  # 内置词里允许做两字拼音纠错的（且附近要有同类词）


def correct(units, terms, replacements, name=None, user_terms=()):
    """在识别结果上做替换与拼音纠错。返回 (units, 改动列表)。"""
    changes = []
    # 1) 替换表（精确）
    for wrong, right in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
        p = 0
        while True:
            s = "".join(u[0] for u in units)
            k = s.find(wrong, p)
            if k < 0:
                break
            if s[k: k + len(right)] == right:  # 已经是正确写法（如 转氨→转氨酶）
                p = k + len(right)
                continue
            units = _replace_in_units(units, k, k + len(wrong), right)
            changes.append((wrong, right))
            p = k + len(right)
    # 2) 拼音近似纠错（只在纯汉字片段内）
    s = "".join(u[0] for u in units)
    termset = set(terms)
    # 已经正确出现的词所在位置，不参与纠错（避免 肝炎只→干眼症）
    protected = np.zeros(len(s) + 1, bool)
    for t in terms:
        for m in re.finditer(re.escape(t), s):
            protected[m.start(): m.end()] = True
    cands = []
    for t in terms:
        n = len(t)
        for L in (n - 1, n, n + 1):
            if L < 2:
                continue
            for k in range(0, len(s) - L + 1):
                w = s[k: k + L]
                if w == t or w in termset or not re.match(r"^[\u4e00-\u9fff]+$", w) or protected[k: k + L].any():
                    continue
                one = _one_off(w, t) and not any(w[i:j] in _words() for i in range(L) for j in range(i + 2, L + 1))
                if _syl_sim(_syl(w[0]), _syl(t[0])) < 0.8 or _syl_sim(_syl(w[-1]), _syl(t[-1])) < 0.8:
                    if not one:
                        continue
                if n == 2:
                    # 两字词误纠风险高（如 是其→湿气）：只纠用户专有词，或内置肝炎类词且附近有同类词
                    near = s[max(0, k - 10): k + L + 10]
                    if not (t in user_terms or (t in TWO_CHAR_OK and any(o in near for o in TWO_CHAR_OK if o != t))):
                        continue
                    if L != 2 or [_syl(c) for c in w] != [_syl(c) for c in t]:
                        continue
                    cands.append((1.0, n, k, L, t, w))
                    continue
                if w in _words() and t not in user_terms:
                    continue  # 本身是常用词（部分、只要…）不纠，避免误改
                if L == n - 1:
                    # 少一个字（转没→转氨酶）：首尾音节必须完全一致
                    if _syl(w[0]) != _syl(t[0]) or _syl(w[-1]) != _syl(t[-1]):
                        continue
                    need = 0.66
                else:
                    need = 0.8 if L != n else 0.75
                sim = 0.8 if one else _seq_sim(w, t)
                if sim >= need:
                    cands.append((sim, n, k, L, t, w))
    cands.sort(key=lambda c: (-c[0], -c[1]))
    used = np.zeros(len(s) + 1, bool)
    chosen = []
    for c in cands:
        sim, _, k, L, t, w = c
        if used[k: k + L].any():
            continue
        used[k: k + L] = True
        chosen.append(c)
    for sim, _, k, L, t, w in sorted(chosen, key=lambda c: -c[2]):
        units = _replace_in_units(units, k, k + L, t)
        changes.append((w, t))
    # 3) 自我介绍里的人名
    if name:
        s = "".join(u[0] for u in units)
        for m in re.finditer(r"我是([一-鿿]{2,4}?)(医生|大夫|主任|老师)", s):
            got = m.group(1)
            if got != name and len(got) == len(name):
                units = _replace_in_units(units, m.start(1), m.end(1), name)
                changes.append((got, name))
    return units, changes


# ---------------- 断句 ----------------
def segment(units, max_chars=14, min_chars=4, pause=0.55):
    """按标点、停顿断句；超长句在最大停顿处拆分；过短句与后句合并。"""
    # 开头的口令（“开始”“好”）：后面有明显停顿且很短，直接去掉
    chars = [(u, t) for u, t in units if u not in PUNCT_END]
    for k in range(1, min(4, len(chars))):
        if chars[k][1] - chars[k - 1][1] > 0.6:
            head = "".join(u for u, _ in chars[:k])
            if head in CUE_WORDS or len(head) <= 2:
                t_cut = chars[k][1]
                units = [x for x in units if x[1] >= t_cut - 1e-6]
            break
    clauses, cur = [], []
    for k, (u, t) in enumerate(units):
        if u in PUNCT_END:
            if cur:
                clauses.append(cur)
            cur = []
            continue
        if cur and t - cur[-1][1] > pause and len(cur) >= 6:
            clauses.append(cur)
            cur = []
        cur.append((u, t))
    if cur:
        clauses.append(cur)
    clauses = [c for c in clauses if "".join(u for u, _ in c) not in FILLER_CLAUSES]

    def split(c):
        if len(c) <= max_chars:
            return [c]
        n = (len(c) + max_chars - 1) // max_chars
        best, bk = -1, len(c) // 2
        lo, hi = max(min_chars, len(c) // n - 3), min(len(c) - min_chars, len(c) // n + 3)
        for k in range(lo, hi + 1):
            g = c[k][1] - c[k - 1][1]
            if g > best:
                best, bk = g, k
        return split(c[:bk]) + split(c[bk:])

    lines = []
    for c in clauses:
        lines += split(c)
    merged = []
    for c in lines:
        if merged and (len(c) < min_chars or len(merged[-1]) < min_chars) and \
                len(merged[-1]) + len(c) <= max_chars and c[0][1] - merged[-1][-1][1] < 1.0:
            merged[-1] = merged[-1] + c
        else:
            merged.append(c)
    # 去掉行首/行尾的语气词，以及开头/结尾的口令（“开始”“好了”）
    def trim(c):
        while c and c[0][0] in FILLER_CHARS:
            c = c[1:]
        while c and c[-1][0] in FILLER_CHARS:
            c = c[:-1]
        return c
    merged = [trim(c) for c in merged]
    merged = [c for c in merged if c]
    while merged and "".join(u for u, _ in merged[0]) in CUE_WORDS:
        merged = merged[1:]
    while merged and "".join(u for u, _ in merged[-1]) in CUE_WORDS:
        merged = merged[:-1]
    subs = []
    for c in merged:
        text = "".join(u for u, _ in c)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        a = c[0][1]
        b = c[-1][1] + 0.25
        subs.append({"start": round(a, 3), "end": round(b, 3), "text": text})
    for i in range(len(subs) - 1):
        subs[i]["end"] = min(subs[i]["end"], subs[i + 1]["start"])
        if subs[i]["end"] <= subs[i]["start"]:
            subs[i]["end"] = subs[i]["start"] + 0.3
    return subs


def generate_srt(video_path, out_srt, brand=None, max_chars=14, use_llm=True):
    """一键：视频 → SRT。返回 (字幕列表, 信息)。"""
    from . import llm
    from .srt import write_srt
    brand = brand or {}
    units, segs = recognize(video_path)
    raw = "".join(u for u, _ in units)
    user_terms = set(t for t in list(brand.get("terms", [])) + [brand.get("name", "")] if t) | set(load_terms_file("自定义词库.txt"))
    terms = load_terms(user_terms)
    reps = load_replacements(brand.get("replace"))
    name = brand.get("name") or None
    units, changes = correct(units, terms, reps, name, user_terms)
    subs = segment(units, max_chars=max_chars)
    info = {"chars": len(raw), "corrections": [f"{a}→{b}" for a, b in changes], "lines": len(subs)}
    if use_llm and llm.enabled():
        fixed = llm.fix_subtitles([s["text"] for s in subs], terms[:200])
        if fixed and len(fixed) == len(subs):
            n = 0
            for s, t in zip(subs, fixed):
                if t and t != s["text"] and abs(len(t) - len(s["text"])) <= 3:
                    s["text"] = t
                    n += 1
            info["llm_fixed_lines"] = n
    write_srt(subs, out_srt)
    with open(os.path.splitext(out_srt)[0] + ".识别原文.txt", "w", encoding="utf-8") as f:
        f.write(raw + "\n\n# 自动纠错：\n" + "\n".join(info["corrections"]))
    log(f"自动字幕：{len(subs)} 行，纠错 {len(changes)} 处" + (f"（{'，'.join(info['corrections'][:8])}…）" if changes else ""))
    return subs, info


def selfcheck():
    """doctor 用：确认 SenseVoice 模型可合并、程序可运行。"""
    import shutil
    oe = offline_exe()
    if not oe:
        return False, "缺少 models/sherpa-onnx/bin/sherpa-onnx-offline 程序"
    mp = model_paths(auto_download=False)
    if not mp:
        return False, "缺少 models/sensevoice 模型（可设置 AICLIP_GH_MIRROR 后运行一次自动下载）"
    root, exe = oe
    d = tempfile.mkdtemp(prefix="aiclip_svchk_")
    fn = os.path.join(d, "s.wav")
    with wave.open(fn, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    try:
        r = subprocess.run([exe, f"--tokens={mp[1]}", f"--sense-voice-model={mp[0]}", fn], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, env=_env(root, exe), timeout=120)
        ok = r.returncode == 0 and '"tokens"' in r.stdout.decode("utf-8", "ignore")
        return ok, "正常" if ok else r.stdout.decode("utf-8", "ignore")[-300:]
    except Exception as e:
        return False, str(e)
    finally:
        shutil.rmtree(d, ignore_errors=True)
