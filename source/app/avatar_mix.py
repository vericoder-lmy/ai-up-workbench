"""数字人混排（AI_UP）：同一组数字人素材（如 橙光口播-1/2/3）随机排列、首尾相接，拼成比配音更长的一条，
再送进 HeyGem。这样 HeyGem 不会把短素材“正放-倒放”来回循环，画面里也就不会出现倒放。
换素材的位置尽量落在句子之间（配音的句间停顿处），看起来像正常的剪辑跳切。

产出两条：
- 高清原片拼接（组内最大分辨率，例如 1080p）：给“画质还原”当原片；
- 小尺寸拼接（短边 720）：送 HeyGem 对口型，快、省显存。"""
import glob
import json
import os
import random
import re

GROUP = "group:"


def is_group(name):
    return str(name).startswith(GROUP)


def _lib():
    from .pipeline import ROOT
    return os.path.join(ROOT, "00_素材库", "数字人")


def groups():
    """按名字归组：橙光口播-1 / 橙光口播-2 … → {"橙光口播": ["橙光口播-1", ...]}，至少 2 条才成组。"""
    names = set()
    for p in glob.glob(os.path.join(_lib(), "*.mp4")) + glob.glob(os.path.join(_lib(), "*.mov")):
        names.add(os.path.splitext(os.path.basename(p))[0].replace("_原片", ""))
    out = {}
    for n in names:
        m = re.match(r"^(.*?)[-_ ]?(\d+)$", n)
        if m and m.group(1):
            out.setdefault(m.group(1), []).append(n)
    return {k: sorted(v, key=lambda x: int(re.search(r"(\d+)$", x).group(1))) for k, v in out.items() if len(v) >= 2}


def members(name):
    g = groups().get(name[len(GROUP):] if is_group(name) else name)
    if not g:
        raise FileNotFoundError(f"找不到数字人素材组：{name}")
    return g


def plan(clips, spans, total, seed):
    """clips: [(名字, 时长)]；spans: 每句配音 (开始, 结束)。返回 [(名字, 入点, 出点)]，总长覆盖配音。"""
    rnd = random.Random(seed)
    bounds = sorted({round((spans[k][1] + spans[k + 1][0]) / 2, 3) for k in range(len(spans) - 1)})
    need = total + 0.5
    order, segs, t, last, used = [], [], 0.0, None, {}
    long = len(clips) >= 2 and sum(d for _, d in clips) >= 120.0 and min(d for _, d in clips) >= 15.0
    while t < need - 0.05:
        if not order:
            order = [c for c in clips]
            rnd.shuffle(order)
            if last and len(order) > 1 and order[0][0] == last:
                order.append(order.pop(0))  # 不让同一条连着出现
        name, dur = order.pop(0)
        k = used.get(name, 0)  # 第二次用到同一条时换个入点，避免一模一样的动作重复
        if long:
            # 长素材（如律师的整段口播）：每段连续播 10~18 秒，在句间停顿处切到另一条的随机位置，
            # 剪辑时这些切换点会配一次景别变化（近景↔中景），看起来像正常的换机位
            want = rnd.uniform(10.0, 18.0)
            start = round(rnd.uniform(0, max(0.0, dur - want - 4.0)), 2) if (segs or k) else 0.0
            avail = dur - start - 0.15
            cands = [b for b in bounds if t + 7.0 <= b <= t + min(avail, want + 6.0)]
            if need - t <= min(avail, want + 6.0):
                end_t = need
            elif cands:
                end_t = min(cands, key=lambda b: abs(b - (t + want)))
            else:
                cands = [b for b in bounds if t + 2.5 <= b <= t + avail]
                end_t = cands[-1] if cands else t + avail
            segs.append((name, start, round(start + end_t - t, 3)))
            used[name] = k + 1
            t, last = end_t, name
            continue
        start = 0.0 if k == 0 else round(rnd.uniform(0, max(0.0, min(3.0, dur * 0.25))), 2)
        avail = dur - start - 0.15
        cands = [b for b in bounds if t + 2.5 <= b <= t + avail]
        if need - t <= avail:
            end_t = need
        elif cands:
            end_t = cands[-1]  # 在这条素材能撑到的最后一个句间停顿处换
        else:
            end_t = t + avail  # 一句话比素材还长，只能句中换
        segs.append((name, start, round(start + end_t - t, 3)))
        used[name] = k + 1
        t, last = end_t, name
    return segs


def build(group_name, spans, total, seed, cfg, out_dir, log=print, side=720):
    """返回 (送 HeyGem 的小视频, 高清原片拼接, 分段)。"""
    from . import media
    from .pipeline import avatar_file, avatar_settings, sha
    names = members(group_name)
    src = {n: avatar_file(n) for n in names}
    st = {n: avatar_settings(n, cfg) for n in names}
    meta = {n: media.info(src[n]) for n in names}
    clips = []
    for n in names:
        a = st[n].get("use_from") or 0.0
        b = st[n].get("use_to") or meta[n][2]
        clips.append((n, max(1.0, min(b, meta[n][2]) - a)))
    segs = plan(clips, spans, total, seed)
    # 统一成组内最大的尺寸（不同手机/导出设置的素材也能拼在一起）
    W = max(meta[n][0] for n in names) // 2 * 2
    H = max(meta[n][1] for n in names) // 2 * 2
    key = sha(json.dumps(segs), W, H, *[(os.path.getsize(src[n]), os.path.getmtime(src[n])) for n in names])
    hq = os.path.join(out_dir, f"数字人混排_{key}_原片.mp4")
    small = os.path.join(out_dir, f"数字人混排_{key}" + ("" if side == 720 else f"_{side}p") + ".mp4")
    if not os.path.exists(hq):
        # 逐段单独转码再无损拼接：一次只解码一段，内存占用很小。
        # （以前用一个 filter_complex 同时 trim 很多段，高清 HEVC 素材会在内存里缓存大量帧，
        #   长文案时会把内存/页面文件吃光：WinError 1455 / Cannot allocate memory）
        parts = []
        tmpd = os.path.join(out_dir, f"_混排分段_{key}")
        os.makedirs(tmpd, exist_ok=True)
        for k, (n, a, b) in enumerate(segs):
            off = st[n].get("use_from") or 0.0
            part = os.path.join(tmpd, f"{k:03d}.mp4")
            nfr = max(1, int(round((b - a) * 25)))
            if not os.path.exists(part):
                media.run(["-ss", f"{a + off:.3f}", "-i", src[n], "-t", f"{b - a + 0.5:.3f}",
                           "-vf", f"fps=25,scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
                                  f"crop={W}:{H},setsar=1",
                           "-frames:v", str(nfr), "-c:v", "libx264", "-preset", "fast", "-crf", "12",
                           "-pix_fmt", "yuv420p", "-threads", "4", "-an", part + ".part.mp4"])
                os.replace(part + ".part.mp4", part)
            parts.append(part)
            log(f"数字人混排：第 {k + 1}/{len(segs)} 段")
        lst = os.path.join(tmpd, "list.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for p_ in parts:
                f.write("file '" + p_.replace("\\", "/").replace("'", "'\\''") + "'\n")
        media.run(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-an", hq + ".part.mp4"])
        os.replace(hq + ".part.mp4", hq)
        import shutil
        shutil.rmtree(tmpd, ignore_errors=True)
    if not os.path.exists(small):
        media.run(["-i", hq, "-vf", media.small_scale(side), "-c:v", "libx264", "-preset", "medium", "-crf", "12",
                   "-pix_fmt", "yuv420p", "-threads", "4", "-an", small + ".part.mp4"])
        os.replace(small + ".part.mp4", small)
    log(f"数字人混排（{W}x{H}）：" + " → ".join(f"{n}[{a:.1f}-{b:.1f}s]" for n, a, b in segs))
    return small, hq, segs
