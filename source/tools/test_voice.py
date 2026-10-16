"""声音克隆对比测试：同一句话分别用几个接口合成，放到 06_日报/声音对比/ 里，你听一下哪个最像。
不会打印 API Key。"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("SSLKEYLOGFILE", None)

from app import media  # noqa: E402
from app.audiofix import InstructEngine  # noqa: E402
from app.clients import qwen_voice as QV  # noqa: E402
from app.pipeline import load_cfg, safe_name, voice_file  # noqa: E402

TEXT = "大家好，我是一个程序员，最近在用 AI 做视频。今天给你们看一个我自己写的小工具。"


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    cfg = load_cfg()
    name = cfg.get("voice_default") or sorted(
        os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "00_素材库", "声音"))
        if f.lower().endswith((".m4a", ".wav", ".mp3", ".aac", ".flac")))[0]
    out = os.path.join(ROOT, "06_日报", "声音对比")
    os.makedirs(out, exist_ok=True)
    src = voice_file(name)
    ref24 = os.path.join(ROOT, "00_素材库", "声音", "_prepared", safe_name(name) + ".wav")
    hifi = os.path.join(out, "参考_48k.wav")
    media.run(["-i", src, "-t", "20", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", hifi])
    media.run(["-i", src, "-t", "12", "-c:a", "pcm_s16le", os.path.join(out, "0_你的原声.wav")])
    print(f"测试句子：{TEXT}\n参考录音：{os.path.basename(src)}\n")
    rows = []

    # 1 旧接口 qwen3-tts-vc（之前效果最像的那个）
    qv = QV.QwenVoice(cfg["voice"], os.path.join(ROOT, "00_素材库", "声音", "音色库.json"), log=lambda m: None)
    for model in ("qwen3-tts-vc-2026-01-22", "qwen3-tts-vc"):
        t = time.time()
        try:
            QV.TARGET_MODEL = model
            vid = qv.enroll(ref24, name)
            qv.tts(vid, TEXT, os.path.join(out, f"1_旧接口_{model}.wav"))
            rows.append((f"1_旧接口_{model}.wav", "成功", time.time() - t))
        except Exception as e:  # noqa
            rows.append((f"旧接口 {model}", f"失败：{str(e)[:120]}", time.time() - t))

    # 2 新接口 qwen-audio（flash / plus），高保真参考
    for i, model in enumerate(("qwen-audio-3.0-tts-flash", "qwen-audio-3.0-tts-plus", "qwen-audio-3.1-tts-flash"), 2):
        t = time.time()
        f = os.path.join(out, f"{i}_新接口_{model}.wav")
        try:
            InstructEngine(cfg, print, model=model).synth(hifi, TEXT, "", f)
            rows.append((os.path.basename(f), "成功", time.time() - t))
        except Exception as e:  # noqa
            rows.append((model, f"失败：{str(e)[:120]}", time.time() - t))

    print("结果：")
    for n, st, sec in rows:
        print(f"  {n:<45} {st}（{sec:.0f} 秒）")
    print(f"\n声音文件在：{out}\n请先听 0_你的原声.wav，再挨个听其他几个，告诉 Claude 哪个最像。")
    json.dump([{"file": n, "status": st} for n, st, _ in rows], open(os.path.join(out, "结果.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if os.name == "nt":
        os.startfile(out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
