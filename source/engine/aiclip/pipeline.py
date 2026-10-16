"""总流程编排（单条任务）。批量/监听见 batch.py。"""
import glob
import os
import time

from . import analyze, layout, planner, render
from .progress import report as _prog
from .align import align_subs
from .audio import load_audio, vad
from .face import face_track
from .srt import parse_srt, write_srt
from .utils import deep_update, load_json, log, probe, save_json


class JobError(Exception):
    """输入问题导致的失败（重试无意义）。"""


def _jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _check_inputs(a):
    if not a.video or not os.path.isfile(a.video):
        raise JobError(f"找不到输入视频: {a.video}")
    if a.srt and not os.path.isfile(a.srt):
        raise JobError(f"找不到字幕文件: {a.srt}")
    paths = a.style if isinstance(a.style, (list, tuple)) else [a.style]
    paths = [q.strip() for p in paths for q in str(p).split(",") if q.strip()]
    for p in paths:
        if not os.path.isfile(p):
            raise JobError(f"找不到风格模板: {p}")
    a.styles = paths
    if a.ref and not os.path.isfile(a.ref):
        log(f"警告：参考视频不存在，忽略: {a.ref}")
        a.ref = None
    a.ref_images = [p for p in (a.ref_images or []) if os.path.isfile(p)]


def run_pipeline(a):
    t_all = time.time()
    _check_inputs(a)
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    style = load_json(a.styles[0])
    extra = [load_json(p) for p in a.styles[1:]]
    _prog(a.work, "准备")
    try:
        info = probe(a.video)
    except Exception as e:
        raise JobError(f"视频无法读取（文件损坏或未复制完整）：{e}")
    if not info.get("w"):
        raise JobError("无法读取视频画面，文件可能损坏或格式不支持")
    # 1) 参考分析
    if a.ref or a.ref_images:
        _prog(a.work, "参考分析")
        try:
            rep, over = analyze.analyze_reference(a.ref, a.ref_images, a.work,
                                                  (style["canvas"]["w"], style["canvas"]["h"]))
            if not getattr(a, "no_ref_override", False):
                deep_update(style, over)
        except Exception as e:  # 参考分析失败不影响出片
            log("参考分析失败，使用模板默认参数：", e)
    if getattr(a, "brand", None) and os.path.exists(a.brand):
        b = load_json(a.brand)
        for s_ in [style] + extra:
            deep_update(s_["brand"], b)
    brand_cli = {"title": getattr(a, "title", None), "slogan": getattr(a, "slogan", None)}
    workers = a.workers or (max(1, min(6, (os.cpu_count() or 2) - 1)) if (os.cpu_count() or 2) > 2 else 2)
    if getattr(a, "bgm", None):
        style["audio"]["bgm"] = a.bgm
    if getattr(a, "no_sfx", False):
        style["audio"]["sfx"] = False
    # 画幅：跟随原片 / 竖屏 / 横屏
    orient = layout.resolve(getattr(a, "orientation", "auto"), info["w"], info["h"])
    style = layout.apply(style, orient)
    extra = [layout.apply(s_, orient) for s_ in extra]
    for s_ in extra:
        s_["canvas"] = dict(style["canvas"])
        s_["audio"] = style["audio"]
    styles = [style] + extra
    log(f"输出画幅：{layout.NAMES[orient]} {style['canvas']['w']}x{style['canvas']['h']}"
        + (f"，混用 {len(styles)} 个风格：" + "、".join(s_.get("display_name") or s_.get("name", "") for s_ in styles)
           if len(styles) > 1 else ""))
    style_path = os.path.join(a.work, "style.resolved.json")
    save_json(style, style_path)
    render_style_path = style_path
    multi_path = os.path.join(a.work, "styles.resolved.json")
    if len(styles) > 1:
        save_json(styles, multi_path)
        render_style_path = multi_path
    elif os.path.exists(multi_path):
        os.remove(multi_path)

    tl_path = os.path.join(a.work, "timeline.json")
    log(f"源视频: {info['w']}x{info['h']} {info['fps']:.2f}fps {info['duration']:.1f}s 音频={'有' if info['has_audio'] else '无'}")
    if getattr(a, "from_timeline", None):
        tl = load_json(a.from_timeline)
        if list(tl["canvas"]) != [style["canvas"]["w"], style["canvas"]["h"]]:
            tl["canvas"] = [style["canvas"]["w"], style["canvas"]["h"]]  # 换了画幅：沿用剪辑决策，按新画幅重渲染
        if len(styles) == 1:
            for b in tl["blocks"]:
                b["style"] = 0
        save_json(tl, tl_path)
    else:
        if not a.srt:
            # 没给字幕：本地语音识别自动生成
            if not info["has_audio"]:
                raise JobError("视频没有声音，无法自动生成字幕")
            from . import subgen
            brand_all = dict(style["brand"])
            brand_all.update({k: v for k, v in brand_cli.items() if v})
            a.srt = os.path.join(a.work, "自动字幕.srt")
            _prog(a.work, "识别字幕")
            try:
                _, ginfo = subgen.generate_srt(a.video, a.srt, brand_all, style.get("subtitle", {}).get("max_chars", 14))
            except FileNotFoundError as e:
                raise JobError(f"无法自动生成字幕：{e}")
            save_json(ginfo, os.path.join(a.work, "自动字幕_信息.json"))
            if getattr(a, "subtitle_only", False):
                log("已生成字幕（--subtitle-only）：", a.srt)
                return {"ok": True, "subtitle_only": True, "srt": a.srt}
        subs = parse_srt(a.srt)
        if not subs:
            raise JobError("字幕文件为空或格式无法解析（需要标准 SRT）")
        _prog(a.work, "字幕对齐")
        # 2) 音频 VAD + 字幕对齐
        if info["has_audio"]:
            x = load_audio(a.video, 16000, 1)
            act, segs, _, _ = vad(x, 16000)
            log(f"字幕 {len(subs)} 条，语音段 {len(segs)} 个")
            subs_al, meta = align_subs(subs, act, segs, mode=a.align, audio_path=a.video, work_dir=a.work)
        else:
            log("素材无音轨：跳过对齐与去停顿")
            segs, subs_al, meta = [], subs, {"method": "none"}
            for s_ in styles:
                s_["cut"]["remove_silence"] = False
        subs_al = [s for s in subs_al if s["start"] < info["duration"]]
        write_srt(subs_al, os.path.join(a.work, "aligned_source.srt"))
        _prog(a.work, "剪辑编排")
        # 3) 主体定位
        face = face_track(a.video, info["duration"])
        # 4) 编排
        tl = planner.build_timeline(subs_al, segs, info, style, brand_cli, face, a.work, getattr(a, "broll", "") or "",
                                    styles=styles)
        tl["align"] = meta
        tl = _jsonable(tl)
        save_json(tl, tl_path)
    srt_out = os.path.splitext(a.out)[0] + ".srt"
    write_srt(tl["subs"], srt_out)
    if getattr(a, "plan_only", False):
        log("已输出 timeline.json（--plan-only），可手动修改后用 --from-timeline 渲染")
        return {"ok": True, "plan_only": True, "timeline": tl_path}
    # 5) 渲染
    enc = render.detect_encoder(a.encoder)
    frames = int(a.preview * tl["fps"]) if getattr(a, "preview", 0) else None
    if frames:
        frames = min(frames, tl["frames"])
    lst, dur = render.render_video(tl_path, render_style_path, a.video, a.work, workers, enc, style, frames)
    _prog(a.work, "混音合成")
    wav = render.build_audio(tl, a.video, style, a.work, dur, has_audio=info["has_audio"])
    tmp_out = os.path.splitext(a.out)[0] + ".part.mp4"
    render.mux(lst, wav, tmp_out, style)
    os.replace(tmp_out, a.out)  # 先写临时文件再改名，避免半成品被当成成片
    if not getattr(a, "keep_work", False):
        for f in glob.glob(os.path.join(a.work, "chunk_*.mp4")) + [wav, lst]:
            try:
                os.remove(f)
            except OSError:
                pass
    sync = None
    _prog(a.work, "成片质检")
    if info["has_audio"] and not frames and not getattr(a, "no_check", False):
        try:
            from . import asr
            if asr.available():
                sync = asr.check_sync(a.out, tl["subs"], a.work)
                log(f"成片音画字幕质检：中位误差 {sync.get('median_err')}s，90% 句子误差 < {sync.get('p90_err')}s，"
                    f"超过 0.5s 的 {sync.get('over_0.5s')} 句 → {'通过' if sync.get('ok') else '请人工检查'}")
        except Exception as e:
            log("成片质检跳过：", e)
    res = {"ok": True, "output": os.path.abspath(a.out), "sync_check": sync, "align": tl.get("align"), "srt": os.path.abspath(srt_out), "duration": round(dur, 2),
           "source_duration": round(info["duration"], 2), "encoder": enc, "workers": workers,
           "chapters": [c["label"] for c in tl["chapters"]],
           "orientation": orient, "canvas": tl["canvas"], "styles": [s_.get("name") for s_ in styles], "seconds": round(time.time() - t_all, 1)}
    save_json(res, os.path.join(a.work, "result.json"))
    _prog(a.work, "完成", 1)
    log(f"完成：{a.out}  成片 {dur:.1f}s，总耗时 {res['seconds']}s")
    return res
