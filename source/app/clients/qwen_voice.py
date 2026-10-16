"""阿里百炼 Qwen 声音复刻：注册一次音色，之后用文字合成你的声音。
接口：qwen-voice-enrollment（注册音色）+ qwen-audio TTS（合成，支持情绪/语速指令）。
API Key 放在 config.local.json 的 voice.api_key（或环境变量 DASHSCOPE_API_KEY），日志里不打印。"""
import base64
import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request
import wave

DEFAULT_API_URL = "https://dashscope.aliyuncs.com/api/v1"
ENROLL_MODEL = "qwen-voice-enrollment"
TARGET_MODEL = "qwen3-tts-vc-2026-01-22"


class VoiceError(RuntimeError):
    pass


def load_key(cfg_voice):
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    key = str(cfg_voice.get("api_key") or "").strip()  # config.local.json 里的 voice.api_key
    if key:
        return key
    path = cfg_voice.get("api_key_file", "")
    field = cfg_voice.get("api_key_field", "asr_api_key")
    if not path:
        raise VoiceError("没有配置阿里百炼 API Key：在 config.local.json 的 voice.api_key 里填写（或设置环境变量 DASHSCOPE_API_KEY）")
    try:
        key = str(json.load(open(path, encoding="utf-8")).get(field, "")).strip()
    except Exception as e:
        raise VoiceError(f"读不到百炼 API Key（{path} 的 {field}）：{e}")
    if not key:
        raise VoiceError(f"{path} 里的 {field} 是空的")
    return key


_ORIG_KEYLOG = os.environ.get("SSLKEYLOGFILE")


def _ssl_context():
    """自建 HTTPS 上下文：不写 SSL 密钥日志（环境变量 SSLKEYLOGFILE 指向不存在的路径时，
    握手会报 FileNotFoundError），证书优先用 certifi，其次用系统证书。"""
    os.environ.pop("SSLKEYLOGFILE", None)  # 只影响本程序；指向不存在的路径时 HTTPS 握手会失败
    cafile = None
    try:
        import certifi
        cafile = certifi.where()
    except Exception:
        pass
    return ssl.create_default_context(cafile=cafile)


def _system_proxies():
    """Windows 的系统代理（如 Clash 的 127.0.0.1:7890）会被 urllib 读成 https://…，
    导致和代理做 TLS 握手而失败（SSLEOFError）。这里统一改成 http:// 代理。"""
    out = {}
    for k, v in urllib.request.getproxies().items():
        if k in ("http", "https") and v:
            out[k] = "http://" + v.split("://", 1)[-1]
    return out


_CTX = _ssl_context()
# 百炼是国内服务，默认直连不走代理；直连失败再试系统代理
_OPENERS = [("直连", urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=_CTX)))]
if _system_proxies():
    _OPENERS.append(("系统代理", urllib.request.build_opener(urllib.request.ProxyHandler(_system_proxies()),
                                                         urllib.request.HTTPSHandler(context=_CTX))))


def _open(req, timeout, log=None):
    errs = []
    for name, op in _OPENERS:
        try:
            return op.open(req, timeout=timeout)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError) as e:
            errs.append(f"{name}：{getattr(e, 'reason', e)}")
            if log:
                log(f"百炼接口{name}失败：{getattr(e, 'reason', e)}")
    raise urllib.error.URLError("；".join(errs))


def ssl_env_note():
    """测试报告用：列出可能影响 HTTPS 的环境变量（只列是否设置和路径是否存在）。"""
    out = []
    if _ORIG_KEYLOG:
        out.append(f"SSLKEYLOGFILE={_ORIG_KEYLOG}（{'存在' if os.path.exists(_ORIG_KEYLOG) else '不存在'}，本程序已忽略）")
    sp = urllib.request.getproxies()
    if sp:
        out.append("系统代理=" + json.dumps(sp, ensure_ascii=False))
    for k in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"):
        v = os.environ.get(k)
        if v:
            exists = os.path.exists(v) if "PROXY" not in k else None
            out.append(f"{k}={v}" + ("" if exists is None else f"（{'存在' if exists else '不存在'}）"))
    return "；".join(out) or "无"


def _post(url, payload, key, timeout=240, log=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with _open(req, timeout, log) as r:
            body, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        body, status = e.read(), e.code
    except urllib.error.URLError as e:
        raise VoiceError(f"连不上百炼接口：{e.reason}")
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        raise VoiceError(f"百炼接口返回的不是 JSON（HTTP {status}）")
    if log:  # 只记录不含密钥和音频数据的部分
        brief = {k: v for k, v in data.items() if k != "output"}
        out = data.get("output", {})
        if isinstance(out, dict):
            brief["output_keys"] = list(out.keys())
            if "voice" in out:
                brief["voice"] = out["voice"]
        log(f"POST {url} → HTTP {status} {json.dumps(brief, ensure_ascii=False)[:400]}")
    if status >= 400:
        raise VoiceError(f"百炼接口失败：{data.get('message', data)}（request_id: {data.get('request_id', '')}）")
    return data


def wav_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


class QwenVoice:
    def __init__(self, cfg_voice, voices_json, log=print):
        self.key = load_key(cfg_voice)
        self.url = (cfg_voice.get("api_url") or DEFAULT_API_URL).rstrip("/")
        self.voices_json = voices_json
        self.log = log

    def _voices(self):
        try:
            return json.load(open(self.voices_json, encoding="utf-8"))
        except Exception:
            return {}

    def enroll(self, ref_wav, name):
        """同一段参考音频只注册一次，voice_id 记在 00_素材库/声音/音色库.json。"""
        if wav_seconds(ref_wav) < 3:
            raise VoiceError("参考录音不到 3 秒")
        digest = hashlib.sha256(open(ref_wav, "rb").read()).hexdigest()[:20]
        voices = self._voices()
        if digest in voices:
            self.log(f"复用已注册的音色：{voices[digest]['voice_id']}")
            return voices[digest]["voice_id"]
        data_uri = "data:audio/wav;base64," + base64.b64encode(open(ref_wav, "rb").read()).decode("ascii")
        res = _post(f"{self.url}/services/audio/tts/customization",
                    {"model": ENROLL_MODEL, "input": {"action": "create", "target_model": TARGET_MODEL,
                                                      "preferred_name": "aiup", "audio": {"data": data_uri}}},
                    self.key, log=self.log)
        vid = str(res.get("output", {}).get("voice", "")).strip()
        if not vid:
            raise VoiceError("音色注册成功但没有返回 voice ID")
        voices[digest] = {"voice_id": vid, "name": name, "model": TARGET_MODEL}
        os.makedirs(os.path.dirname(self.voices_json), exist_ok=True)
        json.dump(voices, open(self.voices_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return vid

    def tts(self, voice_id, text, out_wav):
        res = _post(f"{self.url}/services/aigc/multimodal-generation/generation",
                    {"model": TARGET_MODEL, "input": {"text": text, "voice": voice_id}}, self.key, log=self.log)
        audio = res.get("output", {}).get("audio", {}) or {}
        url, data = str(audio.get("url", "")).strip(), str(audio.get("data", "")).strip()
        os.makedirs(os.path.dirname(out_wav), exist_ok=True)
        if url:
            with _open(urllib.request.Request(url), 120, self.log) as r:
                open(out_wav, "wb").write(r.read())
        elif data:
            open(out_wav, "wb").write(base64.b64decode(data.split(",", 1)[-1]))
        else:
            raise VoiceError("合成成功但没有返回音频")
        if open(out_wav, "rb").read(4) != b"RIFF":
            raise VoiceError("返回的音频不是有效 WAV")
        return out_wav
