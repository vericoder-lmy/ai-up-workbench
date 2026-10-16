"""一条视频的完整流程：声音合成 → 对口型 → 字幕 → 用复制来的 AICLIP 引擎剪辑（竖屏/横屏）。
每一步的产物都存在工程目录里，重跑时已有的直接复用（改动效、断电重启都不用重做对口型）。"""
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import wave

from . import media
from .clients.heygem import HeyGem
from .clients.qwen_voice import QwenVoice, wav_seconds
from .text import split_sentences, subtitle_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engine")
STAGES = ["声音合成", "对口型", "字幕对齐", "编排", "动效渲染", "画面合成", "混音", "质检"]
ORIENT_NAMES = {"portrait": "竖屏", "landscape": "横屏"}
GAP_SEC = 0.18  # 句子之间的停顿


class Cancelled(Exception):
    pass


class MemoryHungry(Exception):
    """渲染时内存不足，可以降低并行进程数重试。"""


def load_cfg():
    cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    local = os.path.join(ROOT, "config.local.json")
    if os.path.exists(local):
        try:
            loc = json.load(open(local, encoding="utf-8"))
            for k, v in loc.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except Exception:
            pass
    return cfg


def safe_name(s, n=40):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', "", s).strip() or "未命名"
    return s[:n]


def sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------- 素材 ----------------

def voice_file(name):
    for ext in (".m4a", ".wav", ".mp3", ".aac", ".flac"):
        p = os.path.join(ROOT, "00_素材库", "声音", name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"找不到声音素材：{name}")


def avatar_file(name):
    d = os.path.join(ROOT, "00_素材库", "数字人")
    for cand in (name + "_原片.mp4", name + ".mp4", name + ".mov"):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"找不到数字人素材：{name}")


def avatar_settings(name, cfg):
    """数字人截取范围：00_素材库/数字人/<名>.json，或 config.json 的 avatar_default。"""
    p = os.path.join(ROOT, "00_素材库", "数字人", name + ".json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    d = cfg.get("avatar_default", {})
    if d.get("name") == name:
        return {"use_from": d.get("use_from"), "use_to": d.get("use_to")}
    return {}


def prepared_avatar(name, cfg, side=None):
    side = int(side or media.LIPSYNC_SIDE)
    src = avatar_file(name)
    st = avatar_settings(name, cfg)
    key = sha(name, os.path.getsize(src), os.path.getmtime(src), st.get("use_from"), st.get("use_to"),
              "side", side)
    out = os.path.join(ROOT, "00_素材库", "数字人", "_prepared", f"{safe_name(name)}_{key}.mp4")
    if not os.path.exists(out):
        media.prep_avatar(src, out + ".part.mp4", st.get("use_from"), st.get("use_to"), side=side)
        os.replace(out + ".part.mp4", out)
    return out


def get_voice_id(qv, voice_name):
    ref = os.path.join(ROOT, "00_素材库", "声音", "_prepared", safe_name(voice_name) + ".wav")
    src = voice_file(voice_name)
    if not os.path.exists(ref) or os.path.getmtime(ref) < os.path.getmtime(src):
        media.prep_voice(src, ref)
    qv.ref_wav = ref  # 备用引擎要用同一段参考录音
    hifi = os.path.join(ROOT, "00_素材库", "声音", "_prepared", safe_name(voice_name) + "_48k.wav")
    if not os.path.exists(hifi) or os.path.getmtime(hifi) < os.path.getmtime(src):
        # 给 qwen-audio 复刻用的高保真参考：原始录音 48k 单声道、不做响度处理，最长 20 秒
        media.run(["-i", src, "-t", "20", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", hifi])
    qv.ref_hifi = hifi
    if load_cfg().get("voice", {}).get("engine") == "qwen_audio":
        qv.use_audio = True  # 默认配音引擎：qwen-audio（qwen3-tts-vc 已下线，接口报 Model not exist）
        return "qa:" + safe_name(voice_name)
    try:
        return qv.enroll(ref, voice_name)
    except Exception as e:  # noqa  原声克隆接口不可用：直接走备用引擎
        qv.fallback_reason = str(e)[:200]
        return "fallback:" + safe_name(voice_name)


# ---------------- 配音 ----------------

def tts_sentence(qv, voice_id, text, instruction="", rate=1.0):
    """单句合成（带缓存）：同一音色、同一句话、同一情绪只合成一次。返回 24k 单声道 wav 路径。
    instruction 是情绪指令（只有 qwen-audio 引擎支持；旧的原声克隆引擎会忽略）。"""
    from . import pron
    text = pron.apply(text)  # 读音纠正：容易读错的词换成读法（字幕仍用原文）
    cache = os.path.join(ROOT, "00_素材库", "声音", "_cache")
    out = os.path.join(cache, sha(voice_id, text) + ".wav")
    use_audio = getattr(qv, "use_audio", False)
    if os.path.exists(out) and not getattr(qv, "fallback_reason", None) and not use_audio:
        return out
    rate = round(float(rate or 1.0), 2)
    amodel = load_cfg().get("voice", {}).get("audio_model", "qwen-audio-3.0-tts-flash")
    ref = getattr(qv, "ref_hifi", None) or getattr(qv, "ref_wav", "")
    fb = os.path.join(cache, sha("fallback", ref, text, *([instruction] if instruction else []),
                                 *([f"rate{rate}"] if rate != 1.0 else []),
                                 *([amodel] if amodel != "qwen-audio-3.0-tts-flash" else [])) + ".wav")
    if (getattr(qv, "fallback_reason", None) or use_audio) and os.path.exists(fb):
        return fb
    raw = out + f".{uuid.uuid4().hex[:6]}.raw.wav"  # 每次唯一，避免同一句同时试听两次互相删文件
    if not getattr(qv, "fallback_reason", None) and not use_audio and not str(voice_id).startswith("fallback:"):
        try:
            qv.tts(voice_id, text, raw)
            media.run(["-i", raw, "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", out])
            _rm(raw)
            return out
        except Exception as e:  # noqa
            qv.fallback_reason = str(e)[:200]
    # 备用：情绪指令引擎（qwen-audio）用同一段参考录音克隆，同一个 qv 之后的句子都走备用，保证整条音色一致
    from .audiofix import InstructEngine
    eng = getattr(qv, "_fb_engine", None)
    if eng is None:
        eng = qv._fb_engine = InstructEngine(load_cfg(), log=lambda m: None, model=amodel)
        if hasattr(qv, "log") and not use_audio:
            qv.log(f"原声克隆接口失败（{qv.fallback_reason}），改用备用配音引擎 {eng.model}")
    raw = fb + f".{uuid.uuid4().hex[:6]}.raw.wav"
    os.makedirs(cache, exist_ok=True)
    eng.synth(ref, text, instruction, raw, rate)
    media.run(["-i", raw, "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", fb])
    _rm(raw)
    return fb


def _rm(p):
    try:
        os.remove(p)
    except OSError:
        pass


def concat_wavs(paths, out, gap=GAP_SEC):
    """拼接成整段配音，返回每句的 (start, end)。"""
    spans, t = [], 0.0
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        for i, p in enumerate(paths):
            with wave.open(p, "rb") as r:
                frames = r.readframes(r.getnframes())
            n = len(frames) // 2
            spans.append((t, t + n / 24000))
            w.writeframes(frames)
            t += n / 24000
            if i < len(paths) - 1:
                w.writeframes(b"\x00\x00" * int(gap * 24000))
                t += gap
    return spans


def fmt_ts(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def build_srt(sentences, spans, out):
    """每句的时长按字数分给它的几行字幕。"""
    idx, rows = 1, []
    for s, (a, b) in zip(sentences, spans):
        lines = subtitle_lines(s) or [s]
        weights = [max(1, len(l)) for l in lines]
        tot, t = sum(weights), a
        for l, wgt in zip(lines, weights):
            d = (b - a) * wgt / tot
            rows.append(f"{idx}\n{fmt_ts(t)} --> {fmt_ts(t + d)}\n{l}\n")
            idx += 1
            t += d
    open(out, "w", encoding="utf-8").write("\n".join(rows))


# ---------------- 主流程 ----------------

class Job:
    """job: dict（来自任务表）；report(**fields) 更新任务进度；cancelled() 返回是否已取消。"""

    def __init__(self, job, report, cancelled, log):
        self.j, self.report, self.cancelled, self.log = job, report, cancelled, log
        self.cfg = load_cfg()
        self.dir = job["project_dir"]
        os.makedirs(self.dir, exist_ok=True)

    def check(self):
        if self.cancelled():
            raise Cancelled()

    def p(self, *a):
        return os.path.join(self.dir, *a)

    def run(self):
        j = self.j
        sentences = split_sentences(j["script"])
        if not sentences:
            raise ValueError("文案是空的")
        if j.get("title_auto") or self._title_is_first_sentence():
            # 没填标题：让 AI 根据整条文案提炼（不是照抄第一句）；失败就保持原样
            self.report(detail="AI 提炼标题")
            try:
                from . import title_ai
                t = title_ai.for_job(self.dir, j["script"], j.get("style", ""), self.log)
                if t and t != j["title"]:
                    j["title"] = t
                    self.report(title=t)
            except Exception as e:  # noqa
                self.log(f"AI 提炼标题失败，沿用原标题：{e}")
        json.dump({k: j[k] for k in ("title", "script", "avatar", "voice", "style", "outputs")},
                  open(self.p("project.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # 1 声音合成
        self.report(stage=0, stage_pct=0, detail="注册/读取音色")
        qv = QwenVoice(self.cfg["voice"], os.path.join(ROOT, "00_素材库", "声音", "音色库.json"), log=self.log)
        vid = get_voice_id(qv, j["voice"])
        from .audiofix import instruction_for
        try:
            emos = json.loads(j.get("emotions") or "{}")  # {句子原文: {"emo": "开心", "custom": ""}}
        except Exception:  # noqa
            emos = {}
        wavs = []
        for i, s in enumerate(sentences):
            self.check()
            self.report(stage=0, stage_pct=100 * i / len(sentences), detail=f"逐句合成 {i + 1}/{len(sentences)}")
            e = emos.get(s) or {}
            ins = instruction_for(e.get("emo"), e.get("custom", ""))
            if ins:
                self.log(f"第 {i + 1} 句情绪：{ins}")
            rate = float(e.get("rate") or 1.0)
            if abs(rate - 1.0) > 0.01:
                self.log(f"第 {i + 1} 句语速：{rate}×")
            wavs.append(tts_sentence(qv, vid, s, ins, rate))
        tts = self.p("配音.wav")
        spans = concat_wavs(wavs, tts)
        json.dump([{"text": s, "start": a, "end": b} for s, (a, b) in zip(sentences, spans)],
                  open(self.p("sentences.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        self.log(f"配音完成：{len(sentences)} 句，{spans[-1][1]:.1f} 秒")

        # 2 对口型（按 配音 + 数字人 的内容缓存，改字幕/动效/画幅不会重做）
        self.check()
        from . import avatar_mix
        self.mix_src = None
        self._spans = spans
        side = 1080 if str(j.get("lipsync_res") or "720") == "1080" else 720  # 对口型清晰度：720p 快 / 1080p 清楚
        if side != 720:
            self.log(f"对口型清晰度：{side}p（素材短边超过 {side} 才会缩小）")
        if avatar_mix.is_group(j["avatar"]):
            # 一组素材随机首尾相接，拼得比配音长：HeyGem 不用循环，就不会出现倒放
            self.report(stage=1, stage_pct=0, detail="拼接数字人素材（随机排列，不倒放）")
            av, hq, segs_ = avatar_mix.build(j["avatar"], spans, spans[-1][1], j["id"], self.cfg, self.p(""), log=self.log,
                                             side=side)
            self.mix_src = hq  # 高清原片拼接：画质还原用它把 HeyGem 的 720p 结果补回原分辨率
            cuts, acc = [], 0.0
            for _, a_, b_ in segs_[:-1]:
                acc += b_ - a_
                cuts.append(round(acc, 3))
            json.dump({"cuts": cuts}, open(self.p("数字人切换点.json"), "w", encoding="utf-8"))  # 剪辑时在这些点换景别
        else:
            av = prepared_avatar(j["avatar"], self.cfg, side)
        lip = self._lipsync(tts, av, side, j)
        lip = self.restore_quality(lip, j["avatar"])
        return self._after_lipsync(j, sentences, spans, lip)

    def _lipsync(self, tts, av, side, j):
        """HeyGem 对口型（按配音+数字人缓存）。1080p 失败（多半是显存不够）时自动退回 720p 再试一次。"""
        try:
            return self._lipsync_once(tts, av)
        except Cancelled:
            raise
        except Exception as e:  # noqa
            if side == 720:
                raise
            self.log(f"1080p 对口型失败（{str(e)[:120]}），可能是显存不够，自动改用 720p 重试")
            self.report(detail="1080p 对口型失败，改用 720p 重试")
            from . import avatar_mix
            if avatar_mix.is_group(j["avatar"]):
                spans = getattr(self, "_spans", None)
                av = avatar_mix.build(j["avatar"], spans, spans[-1][1], j["id"], self.cfg, self.p(""),
                                      log=self.log, side=720)[0]
            else:
                av = prepared_avatar(j["avatar"], self.cfg, 720)
            return self._lipsync_once(tts, av)

    def _lipsync_once(self, tts, av):
        key = sha(open(tts, "rb").read(), os.path.basename(av))
        lip = self.p(f"对口型_{key}.mp4")
        if not os.path.exists(lip):
            self.report(stage=1, stage_pct=0, detail="提交 HeyGem")
            hg = HeyGem(self.cfg, log=self.log)
            a16 = self.p("配音_16k.wav")
            media.prep_lipsync_audio(tts, a16)

            def prog(pct, el, size=0, info=None):
                self.check()
                info = info or {}
                parts = [f"HeyGem {pct or 0}%"]
                if info.get("msg"):
                    parts.append(info["msg"])
                fr, tot = info.get("frame") or 0, info.get("total") or 0
                stage_pct = pct or 0
                if fr and tot:
                    parts.append(f"帧 {fr}/{tot}")
                    stage_pct = 20 + 60 * min(1, fr / tot)  # 20%~80% 是逐帧处理阶段
                if info.get("spf"):
                    parts.append(f"每帧 {info['spf']:.1f} 秒")
                if info.get("eta"):
                    parts.append(f"剩余约 {int(info['eta'] // 60)} 分 {int(info['eta'] % 60)} 秒")
                parts.append(f"已用 {int(el // 60)} 分 {int(el % 60)} 秒")
                self.report(stage=1, stage_pct=max(stage_pct, pct or 0), detail=" · ".join(parts))

            try:
                hg.lipsync(a16, av, lip + ".part.mp4", prog)
            except Cancelled:
                hg.restart_container()  # HeyGem 没有取消接口，只能重启容器停掉它
                raise
            os.replace(lip + ".part.mp4", lip)
        else:
            self.log("复用已有的对口型视频")
        return lip

    def _after_lipsync(self, j, sentences, spans, lip):
        # 3 字幕
        self.report(stage=2, stage_pct=50, detail="按配音时间生成字幕")
        srt = self.p("字幕.srt")
        build_srt(sentences, spans, srt)

        # 4-8 剪辑（竖屏、横屏依次）
        outs = [o for o in j["outputs"] if o in ORIENT_NAMES] or ["portrait"]
        day = datetime.date.today().isoformat()
        done_dir = os.path.join(ROOT, "05_成片", day)
        os.makedirs(done_dir, exist_ok=True)
        results = {}
        for k, o in enumerate(outs):
            self.check()
            out = os.path.join(done_dir, f"{safe_name(j['title'])}_{ORIENT_NAMES[o]}.mp4")
            workers = int(self.cfg.get("render", {}).get("workers", 2))
            try:
                self.render(lip, srt, o, out, k, len(outs), workers)
            except MemoryHungry:
                # 内存不够（Docker/HeyGem 也占着内存）：降到单进程再试一次，慢一些但稳
                self.log(f"{ORIENT_NAMES[o]}渲染内存不足，改用 1 个进程重试")
                self.report(detail=f"{ORIENT_NAMES[o]} · 内存不足，改用单进程重试")
                self.render(lip, srt, o, out, k, len(outs), 1)
            results[o] = {"video": out, "srt": os.path.splitext(out)[0] + ".srt", "cover": ""}
        # 封面：横竖两张都出，写本期主题（橙光系列 / 律师系列各自的版式）
        self.check()
        self.report(detail="生成封面（竖屏 + 横屏）")
        base = os.path.join(done_dir, safe_name(j["title"]))
        covers = self.make_covers(lip, outs[0], {"portrait": base + "_封面_竖屏.jpg", "landscape": base + "_封面_横屏.jpg"})
        for o in outs:
            results[o].update(cover=covers.get(o, ""), cover_port=covers.get("portrait", ""),
                              cover_land=covers.get("landscape", ""))
            if not results[o]["cover"]:  # 封面生成失败：退回截成片第 1.5 秒
                c = os.path.splitext(results[o]["video"])[0] + "_封面.jpg"
                try:
                    media.run(["-ss", "1.5", "-i", results[o]["video"], "-frames:v", "1", "-q:v", "3", c])
                    results[o]["cover"] = c
                except Exception:
                    pass
        self.report(stage=8, stage_pct=100, detail="全部完成", files=results)
        return results

    def restore_quality(self, lip, avatar):
        """HeyGem 会把整张脸重画一遍（眼睛、眼镜、头发都会糊）：只保留它生成的嘴部，其余换回原片。"""
        if not self.cfg.get("lipsync", {}).get("restore", True):
            return lip
        hq = os.path.splitext(lip)[0] + "_高清.mp4"
        if os.path.exists(hq):
            self.log("复用已有的画质还原视频")
            return hq
        try:
            from .restore import restore
            if getattr(self, "mix_src", None):
                src, st = self.mix_src, {}
            else:
                src, st = avatar_file(avatar), avatar_settings(avatar, self.cfg)
            model = os.path.join(ENGINE, "models", "face_detection_yunet_2023mar.onnx")

            def prog(p):
                self.check()
                self.report(stage=1, stage_pct=100, detail=f"画质还原（只保留 HeyGem 的嘴部，其余用原片）{p:.0f}%")
            restore(lip, src, hq + ".part.mp4", model, log=self.log,
                    start=st.get("use_from"), end=st.get("use_to"), on_progress=prog)
            os.replace(hq + ".part.mp4", hq)
            return hq
        except Cancelled:
            raise
        except Exception as e:  # noqa  还原失败不影响出片，退回 HeyGem 原输出
            self.log(f"画质还原跳过：{e}")
            try:
                os.remove(hq + ".part.mp4")
            except OSError:
                pass
            return lip

    def make_covers(self, lip, orient, outs):
        """调用 engine/cover.py：从干净的口播画面挑一帧，按风格写上主题。失败返回 {}，不影响出片。"""
        tl = os.path.join(self.p(f"_work_{orient}"), "timeline.json")
        style = os.path.join(ENGINE, "styles", self.j["style"] + ".json")
        if not os.path.exists(tl):
            self.log("封面跳过：找不到 timeline.json")
            return {}
        cmd = [sys.executable, os.path.join(ENGINE, "cover.py"), "--video", lip, "--timeline", tl, "--style", style,
               "--portrait", outs["portrait"], "--landscape", outs["landscape"], "--base-dir", self.dir]
        if os.path.exists(self.p("标题设置.json")):  # 网页里手动调过的封面
            cmd += ["--settings", self.p("标题设置.json")]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        llm = self.cfg.get("llm", {})
        if llm.get("base_url") and llm.get("api_key"):  # 封面大字可让大模型压缩成短句
            env.update(AICLIP_LLM_BASE_URL=llm["base_url"], AICLIP_LLM_API_KEY=llm["api_key"],
                       AICLIP_LLM_MODEL=llm.get("model", ""), AICLIP_LLM_FORMAT=llm.get("api_format", "openai"))
            env.pop("SSLKEYLOGFILE", None)
            px = llm.get("proxy", "direct")
            if px == "direct":
                env["NO_PROXY"] = "*"
            elif px:
                env["HTTPS_PROXY"] = env["HTTP_PROXY"] = px
        try:
            r = subprocess.run(cmd, cwd=ENGINE, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=300, env=env,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for ln in (r.stdout or "").splitlines():
                if "封面" in ln:
                    self.log(ln.strip())
            if r.returncode != 0:
                self.log("封面生成失败：" + (r.stderr or "").strip()[-300:])
                return {}
            return {k: v for k, v in outs.items() if os.path.exists(v)}
        except Exception as e:  # noqa
            self.log(f"封面生成失败：{e}")
            return {}

    def _title_is_first_sentence(self):
        """老任务：没填标题时存的是文案开头（第一句或前 16 个字）。"""
        t, sc = self.j["title"].replace(" ", ""), self.j["script"].replace(" ", "")
        return len(t) >= 6 and sc.startswith(t)

    def display_title(self):
        """开场大字用的标题：老任务没填标题时存的是文案前 16 个字（会截断半句），这里换成第一句。"""
        t, sc = self.j["title"], self.j["script"]
        if len(t) >= 14 and sc.replace(" ", "").startswith(t.replace(" ", "")):
            first = (split_sentences(sc) or [t])[0]
            return first[:24]
        return t

    def render(self, lip, srt, orient, out, k, n, workers=2):
        """调用 engine/run.py（复制自 AICLIP），读它的 progress.json 转成本系统的阶段。"""
        work = self.p(f"_work_{orient}")
        style = os.path.join(ENGINE, "styles", self.j["style"] + ".json")
        cmd = [sys.executable, os.path.join(ENGINE, "run.py"), "--video", lip, "--srt", srt, "--style", style,
               "--orientation", orient, "--align", "none", "--workers", str(workers),
               "--encoder", self.cfg.get("render", {}).get("encoder", "auto"), "--title", self.display_title(), "--out", out, "--work", work]
        try:
            brand = json.loads(self.j.get("brand") or "{}")
        except ValueError:
            brand = {}
        if any(brand.values()):  # 律师动效库：名牌信息
            bp = self.p("名牌信息.json")
            json.dump(brand, open(bp, "w", encoding="utf-8"), ensure_ascii=False)
            cmd += ["--brand", bp]
        mat_dir = os.path.join(ROOT, "00_素材库", "通用素材")
        if self.j.get("use_materials", 1) and os.path.exists(os.path.join(mat_dir, "materials.json")):
            cmd += ["--broll", mat_dir]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", AICLIP_HUD_CACHE=self.p("信息层规划.json"))
        if os.path.exists(self.p("标题设置.json")):  # 网页里手动调过的视频标题
            env["AICLIP_TITLE_FX"] = self.p("标题设置.json")
        if os.path.exists(self.p("数字人切换点.json")):  # 数字人素材切换点：剪辑时配景别变化
            env["AICLIP_CUTS"] = self.p("数字人切换点.json")
        llm = self.cfg.get("llm", {})
        if llm.get("base_url") and llm.get("api_key"):
            env.update(AICLIP_LLM_BASE_URL=llm["base_url"], AICLIP_LLM_API_KEY=llm["api_key"],
                       AICLIP_LLM_MODEL=llm.get("model", ""), AICLIP_VLM_MODEL=llm.get("model", ""))
            env.pop("SSLKEYLOGFILE", None)
            env["AICLIP_LLM_FORMAT"] = llm.get("api_format", "openai")
            px = llm.get("proxy", "direct")
            if px == "direct":  # 测试大模型.bat 测出来直连可用：子进程不走系统代理
                env["NO_PROXY"] = "*"
            elif px:  # 走代理：显式给 http:// 代理，避开 Windows 系统代理被当成 https:// 的问题
                env["HTTPS_PROXY"] = env["HTTP_PROXY"] = px
        logf = open(self.p(f"剪辑日志_{ORIENT_NAMES[orient]}.txt"), "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ENGINE, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        sys.path.insert(0, ENGINE)
        from aiclip import progress as aprog  # noqa
        stage_map = {"准备": 3, "参考分析": 3, "识别字幕": 3, "字幕对齐": 3, "剪辑编排": 3,
                     "渲染画面": 5, "混音合成": 6, "成片质检": 7, "完成": 7}
        while proc.poll() is None:
            if self.cancelled():
                proc.kill()
                raise Cancelled()
            d = aprog.read(work) or {}
            st = stage_map.get(d.get("stage"), 3)
            part = d.get("pct", 0) / 100
            fields = {"stage": st, "stage_pct": min(99, part * 100), f"{orient[:4]}_pct": round(part * 100)}
            if d.get("frames_total"):
                fields["detail"] = (f"{ORIENT_NAMES[orient]} · 帧 {d.get('frames_done', 0)}/{d['frames_total']}"
                                    + (f" · {d['fps']} fps" if d.get("fps") else "")
                                    + (f" · 剩余约 {int(d['eta'])} 秒" if d.get("eta") else ""))
            else:
                fields["detail"] = f"{ORIENT_NAMES[orient]} · {d.get('stage', '准备')}"
            self.report(**fields)
            time.sleep(1)
        logf.close()
        if proc.returncode != 0 or not os.path.exists(out):
            full = open(self.p(f"剪辑日志_{ORIENT_NAMES[orient]}.txt"), encoding="utf-8", errors="replace").read()
            if workers > 1 and ("Insufficient memory" in full or "MemoryError" in full or "OutOfMemory" in full):
                raise MemoryHungry()
            tail = full[-1500:]
            raise RuntimeError(f"{ORIENT_NAMES[orient]}剪辑失败（退出码 {proc.returncode}）：{tail}")
        self.report(**{f"{orient[:4]}_pct": 100})
