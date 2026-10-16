"""音频纠正（AI_UP）：AI 视频里个别字读错了，不用整条重做——
上传原音频/视频 → 在波形上框选读错的那一句 → 用这一句附近的原声克隆音色 → 输入正确文字 → 生成新的一句
（保留原情绪或指定情绪，可自动对齐原句时长）→ 试听、下载 MP3，或一键替换回原音频/原视频。

两个合成引擎，自动选择：
- 情绪指令引擎（qwen-audio-3.0-tts-flash + voice-enrollment）：支持“用开心的语气说”这类情绪指令；
- 原声克隆引擎（qwen3-tts-vc，和配音用的是同一套，已跑通）：不支持情绪指令，靠“用原句当参考音”来保留情绪。
情绪指令引擎不可用（接口/账号未开通等）时，自动退回原声克隆引擎，并在结果里写明。
"""
import base64
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave

import numpy as np

from . import media
from .clients import qwen_voice as QV
from .pipeline import ROOT, load_cfg

DIR = os.path.join(ROOT, "07_音频纠正")
SR = 48000
INSTRUCT_MODEL = "qwen-audio-3.0-tts-flash"
ENROLL_MODEL = "voice-enrollment"

EMOTIONS = {
    "平静": "用平静、自然的语气说",
    "开心": "用开心、愉快的语气说，带一点笑意",
    "激动": "用激动、兴奋的语气说，语速稍快",
    "温柔": "用温柔、亲切的语气说，语速稍慢",
    "严肃": "用严肃、认真的语气说",
    "悲伤": "用低落、伤感的语气说",
    "惊讶": "用惊讶的语气说",
    "生气": "用生气、不满的语气说",
}


SPEEDS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]


def instruction_for(emo, custom=""):
    """单句情绪 → 情绪指令：default/空=不加指令（自然朗读），预设=EMOTIONS，custom=自定义文字。"""
    if not emo or emo == "default":
        return ""
    if emo == "custom":
        return (custom or "").strip()[:60]
    return EMOTIONS.get(emo, "")


# ---------------- 基础 ----------------
def _sid():
    return uuid.uuid4().hex[:6]


def _sess_dir(sid):
    if not re.fullmatch(r"[\w\-一-鿿]+", sid or ""):
        raise ValueError("会话编号不对")
    d = os.path.join(DIR, sid)
    if not os.path.isdir(d):
        raise FileNotFoundError("找不到这段音频，可能已被删除")
    return d


def _rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def _read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
    return x.reshape(-1, ch), sr


def _write_wav(path, x, sr=SR):
    x = np.clip(x, -1, 1)
    if x.ndim == 1:
        x = x[:, None]
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def _to_wav(src, dst, sr=SR, mono=True):
    media.run(["-i", src, "-vn", "-ac", "1" if mono else "2", "-ar", str(sr), "-c:a", "pcm_s16le", dst])


def _to_mp3(src, dst):
    media.run(["-i", src, "-c:a", "libmp3lame", "-q:a", "2", dst])


def _has_video(path):
    r = subprocess.run([media.ffmpeg_bin(), "-hide_banner", "-i", path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return bool(re.search(r"Stream #.*Video:", r.stderr)) and not re.search(r"Video:.*(png|mjpeg).*attached", r.stderr)


def _db(x):
    return 20 * np.log10(max(1e-6, float(np.sqrt(np.mean(np.square(x)))) if len(x) else 1e-6))


def _voiced(x, sr):
    """按 20ms 帧取有声部分（能量高于峰值帧 -35dB）。"""
    n = int(sr * 0.02)
    if len(x) < n:
        return x
    fr = x[: len(x) // n * n].reshape(-1, n)
    e = np.sqrt(np.mean(fr ** 2, 1)) + 1e-9
    keep = e > e.max() * 10 ** (-35 / 20)
    return fr[keep].reshape(-1) if keep.any() else x


# ---------------- 会话：上传 / 波形 ----------------
def create(name, data):
    os.makedirs(DIR, exist_ok=True)
    stem = re.sub(r'[\\/:*?"<>|\s]+', "_", os.path.splitext(os.path.basename(name))[0])[:24] or "音频"
    sid = f"{datetime.date.today():%m%d}_{stem}_{_sid()}"
    d = os.path.join(DIR, sid)
    os.makedirs(d)
    ext = os.path.splitext(name)[1].lower() or ".bin"
    src = os.path.join(d, "原始" + ext)
    open(src, "wb").write(data)
    try:
        _to_wav(src, os.path.join(d, "原声.wav"), mono=False)
        _to_wav(src, os.path.join(d, "mono.wav"))
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        raise RuntimeError(f"读不出音频：{e}")
    x, _ = _read_wav(os.path.join(d, "mono.wav"))
    meta = {"name": name, "source": os.path.basename(src), "video": _has_video(src),
            "duration": round(len(x) / SR, 3), "created": time.time()}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return get(sid)


def _peaks(x, n=1600):
    x = x[:, 0] if x.ndim > 1 else x
    if len(x) < n:
        return [round(float(abs(v)), 3) for v in x]
    k = len(x) // n
    p = np.abs(x[: k * n]).reshape(n, k).max(1)
    p = p / max(1e-6, float(p.max()))
    return [round(float(v), 3) for v in p]


def get(sid):
    d = _sess_dir(sid)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    pk = os.path.join(d, "peaks.json")
    if not os.path.exists(pk):
        x, _ = _read_wav(os.path.join(d, "mono.wav"))
        json.dump(_peaks(x), open(pk, "w"))
    res = []
    rj = os.path.join(d, "results.json")
    if os.path.exists(rj):
        res = json.load(open(rj, encoding="utf-8"))
    return dict(meta, id=sid, peaks=json.load(open(pk)), audio=_rel(os.path.join(d, "原声.wav")), results=res)


def sessions(limit=20):
    if not os.path.isdir(DIR):
        return []
    out = []
    for sid in sorted(os.listdir(DIR), key=lambda s: -os.path.getmtime(os.path.join(DIR, s)))[:limit]:
        try:
            m = json.load(open(os.path.join(DIR, sid, "meta.json"), encoding="utf-8"))
            out.append({"id": sid, "name": m["name"], "duration": m["duration"]})
        except Exception:
            pass
    return out


# ---------------- 原句分析：情绪 / 语速 ----------------
def _f0(x, sr):
    """简易基频：40ms 帧自相关，返回有声帧的 F0 列表（Hz）。"""
    n, hop = int(sr * 0.04), int(sr * 0.01)
    lo, hi = int(sr / 400), int(sr / 70)
    e_all = np.sqrt(np.mean(x ** 2)) + 1e-9
    f = []
    for s in range(0, len(x) - n, hop):
        fr = x[s: s + n] - x[s: s + n].mean()
        if np.sqrt(np.mean(fr ** 2)) < e_all * 0.5:
            continue
        ac = np.correlate(fr, fr, "full")[n - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        k = int(np.argmax(seg)) + lo
        if ac[k] / ac[0] > 0.45:
            f.append(sr / k)
    return np.array(f, np.float32)


def analyze(x, sr, text=""):
    v = _voiced(x, sr)
    f0 = _f0(x[:: max(1, sr // 16000)], sr // max(1, sr // 16000))
    chars = len(re.sub(r"[\s，。,.!！?？、：:；;“”\"'（）()]", "", text))
    vd = len(v) / sr
    return {"duration": round(len(x) / sr, 3), "voiced": round(vd, 3), "db": round(_db(v), 1),
            "f0": round(float(np.median(f0)), 1) if len(f0) else 0.0,
            "f0_var": round(float(np.std(np.log(f0 + 1e-6))) * 12, 2) if len(f0) > 3 else 0.0,  # 半音
            "cps": round(chars / vd, 2) if vd > 0.2 and chars else 0.0}


def describe(a):
    """把原句的语速/音高/起伏/音量翻译成一句情绪指令。"""
    parts = []
    cps = a.get("cps", 0)
    if cps:
        parts.append("语速偏快" if cps > 5.2 else "语速偏慢" if cps < 3.6 else "语速适中")
    var = a.get("f0_var", 0)
    if var:
        parts.append("语调起伏明显、情绪饱满" if var > 3.2 else "语调平稳克制" if var < 1.6 else "语气自然")
    db = a.get("db", -30)
    parts.append("声音有力" if db > -18 else "声音轻柔" if db < -30 else "")
    parts = [p for p in parts if p]
    return "保持原来说话的情绪和节奏：" + "，".join(parts) if parts else "保持原来说话的情绪和节奏"


# ---------------- 合成引擎 ----------------
def _post(url, payload, key, headers=None, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers=dict({"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                              **(headers or {})))
    try:
        with QV._open(req, timeout) as r:
            body, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        body, status = e.read(), e.code
    except urllib.error.URLError as e:
        raise QV.VoiceError(f"连不上百炼接口：{e.reason}")
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        raise QV.VoiceError(f"百炼接口返回的不是 JSON（HTTP {status}）")
    if status >= 400 or data.get("code") and not data.get("output"):
        raise QV.VoiceError(f"{data.get('code', '')} {data.get('message', '')}（HTTP {status}）".strip())
    return data


def _download(url, dst):
    with QV._open(urllib.request.Request(url), 120) as r:
        open(dst, "wb").write(r.read())


class _Lib:
    """已注册音色缓存：同一段参考音 + 同一个引擎只注册一次。"""

    def __init__(self):
        self.path = os.path.join(DIR, "音色库.json")

    def get(self, k):
        try:
            return json.load(open(self.path, encoding="utf-8")).get(k)
        except Exception:
            return None

    def put(self, k, v):
        try:
            d = json.load(open(self.path, encoding="utf-8"))
        except Exception:
            d = {}
        d[k] = v
        os.makedirs(DIR, exist_ok=True)
        json.dump(d, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


class InstructEngine:
    """qwen-audio-3.0-tts-flash：支持情绪指令（instruction）。参考音先传到百炼临时存储（oss://，48 小时有效）。"""
    name = "情绪指令引擎"

    def __init__(self, cfg, log, model=None):
        self.key = QV.load_key(cfg["voice"])
        self.base = (cfg["voice"].get("api_url") or QV.DEFAULT_API_URL).rstrip("/")
        self.model = model or cfg.get("audiofix", {}).get("instruct_model", INSTRUCT_MODEL)
        self.log = log
        self.lib = _Lib()

    def _upload(self, path):
        q = urllib.parse.urlencode({"action": "getPolicy", "model": ENROLL_MODEL})
        req = urllib.request.Request(f"{self.base}/uploads?{q}", headers={"Authorization": f"Bearer {self.key}"})
        try:
            with QV._open(req, 30) as r:
                pol = json.loads(r.read().decode("utf-8"))["data"]
        except urllib.error.HTTPError as e:
            raise QV.VoiceError(f"获取上传凭证失败（HTTP {e.code}）")
        key = f"{pol['upload_dir']}/{os.path.basename(path)}"
        fields = [("OSSAccessKeyId", pol["oss_access_key_id"]), ("Signature", pol["signature"]),
                  ("policy", pol["policy"]), ("x-oss-object-acl", pol["x_oss_object_acl"]),
                  ("x-oss-forbid-overwrite", pol["x_oss_forbid_overwrite"]), ("key", key),
                  ("success_action_status", "200")]
        bd = "----aiup" + uuid.uuid4().hex
        body = b""
        for k, v in fields:
            body += f'--{bd}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8")
        body += (f'--{bd}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(path)}"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n").encode("utf-8") + open(path, "rb").read() + f"\r\n--{bd}--\r\n".encode()
        req = urllib.request.Request(pol["upload_host"], data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={bd}"})
        try:
            with QV._open(req, 60) as r:
                r.read()
        except urllib.error.HTTPError as e:
            raise QV.VoiceError(f"参考音上传失败（HTTP {e.code}）")
        return "oss://" + key

    def voice(self, ref_wav):
        dig = hashlib.sha256(open(ref_wav, "rb").read()).hexdigest()[:20]
        k = f"{self.model}:{dig}"
        v = self.lib.get(k)
        if v and time.time() - v.get("t", 0) < 6 * 24 * 3600:
            return v["voice"]
        url = self._upload(ref_wav)
        res = _post(f"{self.base}/services/audio/tts/customization",
                    {"model": ENROLL_MODEL, "input": {"action": "create_voice", "target_model": self.model,
                                                      "prefix": "aiupfix", "url": url}},
                    self.key, headers={"X-DashScope-OssResourceResolve": "enable"})
        out = res.get("output", {})
        vid = str(out.get("voice_id") or out.get("voice") or "").strip()
        if not vid:
            raise QV.VoiceError("音色注册没有返回 voice_id")
        # 复刻音色需要部署一会儿：轮询到 OK 再用（最多 90 秒）
        for _ in range(30):
            try:
                st = _post(f"{self.base}/services/audio/tts/customization",
                           {"model": ENROLL_MODEL, "input": {"action": "query_voice", "voice_id": vid}}, self.key)
                status = str(st.get("output", {}).get("status", "OK")).upper()
            except QV.VoiceError:
                status = "OK"  # 查询接口不可用时直接尝试合成
            if status == "OK":
                break
            if status in ("UNDEPLOYED", "FAILED"):
                raise QV.VoiceError(f"音色复刻失败（{status}）：参考音可能太短、太吵或有多人说话")
            time.sleep(3)
        self.lib.put(k, {"voice": vid, "t": time.time()})
        return vid

    def synth(self, ref_wav, text, instruction, out_wav, rate=1.0):
        vid = self.voice(ref_wav)
        inp = {"text": text, "voice": vid, "format": "wav", "sample_rate": 24000}
        if instruction:
            inp["instruction"] = instruction[:100]
        rate = float(rate or 1.0)
        local_rate = 1.0
        if abs(rate - 1.0) > 0.01:
            inp["rate"] = round(min(2.0, max(0.5, rate)), 2)  # 接口原生语速（0.5~2.0），比后期变速更自然
        try:
            res = _post(f"{self.base}/services/audio/tts/SpeechSynthesizer", {"model": self.model, "input": inp}, self.key)
        except QV.VoiceError:
            if "rate" not in inp:
                raise
            inp.pop("rate")  # 接口不认 rate 参数时：正常语速合成，再本地变速不变调
            local_rate = rate
            res = _post(f"{self.base}/services/audio/tts/SpeechSynthesizer", {"model": self.model, "input": inp}, self.key)
        audio = res.get("output", {}).get("audio", {}) or {}
        if audio.get("url"):
            _download(audio["url"], out_wav)
        elif audio.get("data"):
            open(out_wav, "wb").write(base64.b64decode(audio["data"]))
        else:
            raise QV.VoiceError("合成成功但没有返回音频")
        if abs(local_rate - 1.0) > 0.01:
            tmp = out_wav + ".r.wav"
            media.run(["-i", out_wav, "-af", f"atempo={min(2.0, max(0.5, local_rate)):.3f}", "-c:a", "pcm_s16le", tmp])
            os.replace(tmp, out_wav)
        return out_wav


class CloneEngine:
    """qwen3-tts-vc：和数字人配音同一套接口（已跑通），不支持情绪指令，靠原句参考音保留情绪。"""
    name = "原声克隆引擎"

    def __init__(self, cfg, log):
        self.qv = QV.QwenVoice(cfg["voice"], os.path.join(DIR, "音色库_克隆.json"), log=lambda m: None)

    def synth(self, ref_wav, text, instruction, out_wav):
        vid = self.qv.enroll(ref_wav, "audiofix")
        return self.qv.tts(vid, text, out_wav)


# ---------------- 生成 ----------------
def _reference(d, start, end, mode):
    """克隆用的参考音：默认取选区附近一共 ~15 秒（含选区本身，情绪最接近）；mode=all 取整段（最多 60 秒）。"""
    x, _ = _read_wav(os.path.join(d, "mono.wav"))
    x = x[:, 0]
    dur = len(x) / SR
    if mode == "all":
        a, b = 0.0, min(dur, 60.0)
        if end > 60:
            a, b = max(0.0, end - 60), end
    else:
        want = max(end - start, 15.0)
        c = (start + end) / 2
        a, b = max(0.0, c - want / 2), min(dur, c + want / 2)
        if b - a < want:
            a, b = max(0.0, b - want), min(dur, a + want)
    if b - a < 3.0:
        raise ValueError("参考音不到 3 秒：原音频太短，克隆不了音色")
    ref = os.path.join(d, f"参考_{int(a * 1000)}_{int(b * 1000)}.wav")
    if not os.path.exists(ref):
        seg = x[int(a * SR): int(b * SR)]
        tmp = ref + ".48k.wav"
        _write_wav(tmp, seg)
        media.run(["-i", tmp, "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", ref])
        os.remove(tmp)
    return ref


def _trim(y, sr=SR, thr_db=-42):
    n = int(sr * 0.01)
    if len(y) < n * 3:
        return y
    e = np.sqrt(np.mean(y[: len(y) // n * n].reshape(-1, n) ** 2, 1)) + 1e-9
    on = np.where(e > e.max() * 10 ** (thr_db / 20))[0]
    if not len(on):
        return y
    a, b = max(0, (on[0] - 3) * n), min(len(y), (on[-1] + 4) * n)
    return y[a:b]


def _atempo(src, dst, factor):
    """变速不变调；ffmpeg 的 atempo 单级只支持 0.5~2，这里串起来。"""
    chain, f = [], factor
    while f > 2.0:
        chain.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        chain.append("atempo=0.5")
        f /= 0.5
    chain.append(f"atempo={f:.5f}")
    media.run(["-i", src, "-af", ",".join(chain), "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", dst])


def _fade(y, ms=12):
    n = min(len(y) // 4, int(SR * ms / 1000))
    if n > 1:
        r = np.linspace(0, 1, n, dtype=np.float32)
        y[:n] *= r
        y[-n:] *= r[::-1]
    return y


def generate(sid, start, end, text, emotion="keep", custom="", align=True, ref_mode="near", log=print):
    d = _sess_dir(sid)
    text = (text or "").strip()
    if not text:
        raise ValueError("请先输入正确的文字")
    if len(text) > 300:
        raise ValueError("一次最多 300 字：请只填读错的那一句")
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    start, end = max(0.0, float(start)), min(meta["duration"], float(end))
    if end - start < 0.2:
        raise ValueError("请先在波形上框选读错的那一句")
    x, _ = _read_wav(os.path.join(d, "mono.wav"))
    seg = x[int(start * SR): int(end * SR), 0]
    an = analyze(seg, SR, text)
    if emotion == "keep":
        instruction = describe(an)
    elif emotion == "custom":
        instruction = (custom or "").strip()[:60] or describe(an)
    else:
        instruction = EMOTIONS.get(emotion, describe(an))
    cfg = load_cfg()
    ref = _reference(d, start, end, ref_mode)
    rid = datetime.datetime.now().strftime("%H%M%S") + "_" + _sid()[:3]
    raw = os.path.join(d, f"生成_{rid}_raw.wav")
    pref = cfg.get("audiofix", {}).get("engine", "auto")
    engines = [CloneEngine] if pref == "clone" else [InstructEngine]  # qwen3-tts-vc 已下线，不再作为备用
    notes, used = [], None
    for E in engines:
        try:
            eng = E(cfg, log)
            from . import pron
            eng.synth(ref, pron.apply(text), instruction, raw)  # 读音纠正词典
            used = eng.name
            break
        except Exception as e:  # noqa
            notes.append(f"{E.name}不可用：{str(e)[:160]}")
            log(notes[-1])
    if used is None:
        raise RuntimeError("；".join(notes))
    if used == CloneEngine.name and emotion not in ("keep",):
        notes.append("指定的情绪没有生效（原声克隆引擎不支持情绪指令），语气跟随原句")
    # 后期：去首尾静音 → 音量对齐原句 → 时长对齐 → 淡入淡出
    tmp = raw.replace("_raw.wav", "_48k.wav")
    _to_wav(raw, tmp)
    y, _ = _read_wav(tmp)
    y = _trim(y[:, 0])
    y *= 10 ** ((an["db"] - _db(_voiced(y, SR))) / 20)
    target = end - start
    natural = len(y) / SR
    if align and natural > 0.1:
        f = natural / target
        if not 0.6 <= f <= 1.7:
            notes.append(f"新句子和原句长度差太多（{natural:.1f}s vs {target:.1f}s），只做了部分对齐，建议调整文字")
            f = min(1.7, max(0.6, f))
        _write_wav(tmp, y)
        _atempo(tmp, tmp + ".t.wav", f)
        y, _ = _read_wav(tmp + ".t.wav")
        y = y[:, 0]
        os.remove(tmp + ".t.wav")
        n = int(round(target * SR))
        y = y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y), np.float32)])
    y = _fade(y.copy())
    out_wav = os.path.join(d, f"纠正_{rid}.wav")
    _write_wav(out_wav, y)
    out_mp3 = os.path.join(d, f"纠正_{rid}.mp3")
    try:
        _to_mp3(out_wav, out_mp3)
    except Exception as e:  # noqa
        out_mp3 = ""
        notes.append(f"MP3 编码失败，只提供 WAV：{str(e)[:80]}")
    for p in (raw, tmp):
        try:
            os.remove(p)
        except OSError:
            pass
    item = {"id": rid, "text": text, "start": round(start, 3), "end": round(end, 3),
            "emotion": emotion if emotion != "custom" else (custom or "自定义"), "instruction": instruction,
            "engine": used, "align": bool(align), "natural": round(natural, 2), "duration": round(len(y) / SR, 2),
            "wav": _rel(out_wav), "mp3": _rel(out_mp3) if out_mp3 else "", "notes": notes, "analysis": an,
            "time": time.time()}
    rj = os.path.join(d, "results.json")
    res = json.load(open(rj, encoding="utf-8")) if os.path.exists(rj) else []
    res.insert(0, item)
    json.dump(res, open(rj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return item


# ---------------- 替换回原音频 / 原视频 ----------------
def replace(sid, rid, video=False):
    """把这一句替换进原音频（两端 15ms 交叉淡化）；原文件是视频时可直接出替换音轨后的视频（画面不重新压缩）。"""
    d = _sess_dir(sid)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    res = json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))
    it = next((r for r in res if r["id"] == rid), None)
    if not it:
        raise FileNotFoundError("找不到这条生成结果")
    x, _ = _read_wav(os.path.join(d, "原声.wav"))
    y, _ = _read_wav(os.path.join(ROOT, it["wav"]))
    y = np.repeat(y[:, :1], x.shape[1], 1)
    a, b = int(it["start"] * SR), int(it["end"] * SR)
    xf = int(SR * 0.015)
    head, tail = x[:a].copy(), x[b:].copy()
    mid = y.copy()
    if len(head) > xf and len(mid) > xf * 2:  # 交界处交叉淡化，避免“咔哒”声
        r = np.linspace(0, 1, xf, dtype=np.float32)[:, None]
        mid[:xf] = x[a: a + xf] * (1 - r) + mid[:xf] * r if a + xf <= len(x) else mid[:xf]
    if len(tail) > xf and len(mid) > xf * 2:
        r = np.linspace(1, 0, xf, dtype=np.float32)[:, None]
        mid[-xf:] = mid[-xf:] * r + x[max(0, b - xf): b] * (1 - r) if b - xf >= 0 else mid[-xf:]
    full = np.concatenate([head, mid, tail])
    out_wav = os.path.join(d, f"替换后_{rid}.wav")
    _write_wav(out_wav, full)
    if video and meta.get("video"):
        src = os.path.join(d, meta["source"])
        out = os.path.join(d, f"替换后_{rid}.mp4")
        media.run(["-i", src, "-i", out_wav, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                   "-b:a", "192k", "-shortest", "-movflags", "+faststart", out])
        os.remove(out_wav)
        return _rel(out)
    out = os.path.join(d, f"替换后_{rid}.mp3")
    _to_mp3(out_wav, out)
    os.remove(out_wav)
    return _rel(out)
