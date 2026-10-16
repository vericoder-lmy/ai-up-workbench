"""一期第 1 步自测：克隆声音（阿里百炼）→ 合成一句话 → HeyGem 对口型。
双击根目录的「一键测试HeyGem.bat」运行；结果和报告在 04_工程/_HeyGem测试/。"""
import datetime
import json
import os
import platform
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import media  # noqa: E402
from app.clients import heygem as hgmod  # noqa: E402
from app.clients.heygem import HeyGem, HeyGemError, _http  # noqa: E402
from app.clients.qwen_voice import QwenVoice, wav_seconds  # noqa: E402

WORK = os.path.join(ROOT, "04_工程", "_HeyGem测试")
REPORT = os.path.join(WORK, "测试报告.txt")
TEXT = "大家好，这是我的 AI 数字人第一次开口说话。以后我的视频，都会由它帮我录。"
os.makedirs(WORK, exist_ok=True)
_lines = []


def first_voice(cfg):
    """config.json 没指定 voice_default 时，用 00_素材库/声音/ 里的第一个录音。"""
    if cfg.get("voice_default"):
        return cfg["voice_default"]
    d = os.path.join(ROOT, "00_素材库", "声音")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                   if f.lower().endswith((".m4a", ".wav", ".mp3", ".aac", ".flac")))
    if not names:
        raise RuntimeError("00_素材库/声音/ 里还没有录音：放一段 15~20 秒的本人录音进去")
    return names[0]


def first_avatar(cfg):
    """config.json 没指定 avatar_default 时，用 00_素材库/数字人/ 里的第一条视频。"""
    av = dict(cfg.get("avatar_default") or {})
    name = av.get("name") or ""
    if name and not name.startswith("group:"):
        return name, av
    d = os.path.join(ROOT, "00_素材库", "数字人")
    names = sorted(os.path.splitext(f)[0].replace("_原片", "") for f in os.listdir(d)
                   if f.lower().endswith((".mp4", ".mov")))
    if not names:
        raise RuntimeError("00_素材库/数字人/ 里还没有视频：放一条正脸说话的视频进去")
    return names[0], {}


def log(msg):
    _lines.append(f"[{datetime.datetime.now():%H:%M:%S}] {msg}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")


def say(msg):
    print(msg, flush=True)
    log(msg)


def step(no, title, fn):
    say(f"\n[{no}/5] {title} …")
    t0 = time.time()
    try:
        r = fn()
        say(f"      ✓ 完成（{time.time() - t0:.1f} 秒）")
        return r
    except Exception as e:
        say(f"      ✗ 失败：{e}")
        log(traceback.format_exc())
        say(f"\n测试停在第 {no} 步。请把这个文件发给 Claude：\n  {REPORT}")
        input("\n按回车键关闭…")
        sys.exit(1)


def main():
    cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    say(f"AI_UP · 声音 + 对口型自测　{datetime.datetime.now():%Y-%m-%d %H:%M}")
    log(f"Python {sys.version.split()[0]} · {platform.platform()}")

    def env():
        log(f"ffmpeg：{media.ffmpeg_bin()}")
        cs = hgmod.docker_containers()
        if not cs:
            say(f"      提示：没读到 Docker 容器信息（{hgmod.DOCKER_NOTE}）")
        for c in cs:
            log(f"容器 {c['name']}（{c['image']}）挂载：{c['mounts']}")
        hg = HeyGem(cfg, log=log)
        say(f"      对口型数据目录：{hg.vid_host or '未找到'}（容器内 {hg.vid_ctr}）")
        _http("GET", hg.video_url + "/", timeout=10, raw=True)
        say(f"      HeyGem 对口型服务 {hg.video_url} 可以连上")
        if not hg.vid_host:
            raise RuntimeError("找不到对口型数据目录，请在 config.json 填 heygem.video_data_dir（通常是 D:/heygem_data/face2face）")
        qv = QwenVoice(cfg["voice"], os.path.join(ROOT, "00_素材库", "声音", "音色库.json"), log=log)
        say("      已读到阿里百炼 API Key")
        from app.clients.qwen_voice import ssl_env_note
        log(f"HTTPS 相关环境变量：{ssl_env_note()}")
        return hg, qv

    hg, qv = step(1, "检查环境（ffmpeg、HeyGem 对口型、百炼 Key）", env)

    from app.pipeline import avatar_file, get_voice_id, tts_sentence  # 和工作台走同一套配音流程
    vname = first_voice(cfg)

    def clone():
        vid = get_voice_id(qv, vname)
        say(f"      声音：{vname}（用录音前 20 秒做复刻参考）")
        return vid

    voice_id = step(2, "克隆声音（阿里百炼 Qwen 声音复刻）", clone)

    tts_wav = os.path.join(WORK, "测试配音.wav")

    def synth():
        import shutil
        shutil.copyfile(tts_sentence(qv, voice_id, TEXT), tts_wav)
        say(f"      配音 {wav_seconds(tts_wav):.1f} 秒：{tts_wav}（可以先双击听听像不像你）")

    step(3, f"合成一句话：{TEXT}", synth)

    av_name, av = first_avatar(cfg)
    av_src = avatar_file(av_name)
    os.makedirs(os.path.join(ROOT, "00_素材库", "数字人", "_prepared"), exist_ok=True)
    av_ready = os.path.join(ROOT, "00_素材库", "数字人", "_prepared", av_name + "_test.mp4")

    def prep_avatar():
        if not os.path.exists(av_ready):
            media.prep_avatar(av_src, av_ready, av.get("use_from"), av.get("use_to"))
        say(f"      数字人：{av_name}，转成 H.264 25fps")

    step(4, "准备数字人素材", prep_avatar)

    out = os.path.join(WORK, "对口型测试.mp4")

    def progress(pct, el, size=0):
        extra = f" · 已生成画面 {size / 1048576:.1f} MB" if size else " · 准备中"
        print(f"\r      进度 {pct}%{extra} · 已用 {int(el // 60)} 分 {int(el % 60)} 秒      ", end="", flush=True)

    def lipsync():
        a16 = os.path.join(WORK, "测试配音_16k.wav")
        media.prep_lipsync_audio(tts_wav, a16)
        hg.lipsync(a16, av_ready, out, progress)
        print()

    step(5, "HeyGem 对口型（大约要几分钟，可以去倒杯水）", lipsync)
    say(f"\n全部通过！对口型结果：{out}")
    try:
        os.startfile(WORK)
    except Exception:
        pass
    input("\n按回车键关闭…")


if __name__ == "__main__":
    main()
