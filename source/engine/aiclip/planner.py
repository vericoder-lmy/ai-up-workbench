"""剪辑编排：去停顿剪切 → 时间映射 → 章节 → 版式分块 → 推镜头/弹字/音效/黑场 → timeline.json。
决策全部落到 timeline.json，渲染器只按时间线执行，便于人工微调后重渲染。"""
import os
import re

from . import llm, textproc as T
from .utils import log


# ---------------- 剪切与时间映射 ----------------
def plan_cuts(subs, vad_segs, duration, cfg):
    if not cfg.get("remove_silence", True):
        return [[0.0, duration]]
    speech = []
    first = min(s["start"] for s in subs)
    last = max(s["end"] for s in subs)
    for a, b in vad_segs:
        if b < first - 0.3 or a > last + 0.3:
            continue  # 第一句之前/最后一句之后、字幕里没有的说话（如"开始""好了"）直接剪掉
        if b - a >= 0.5 or any(s["start"] - 0.3 < b and a < s["end"] + 0.3 for s in subs):
            speech.append([a, b])
    # 以真实语音(VAD)为准切停顿（含句内停顿）；只有 VAD 完全没检测到语音的字幕才用字幕时间兜底，防止轻声字被剪
    for s in subs:
        if not any(a < s["end"] and s["start"] < b for a, b in speech):
            speech.append([s["start"], s["end"]])
    speech.sort()
    merged = []
    for a, b in speech:
        if merged and a - merged[-1][1] < cfg["min_gap"]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    half = cfg["keep_gap"] / 2
    out = []
    for k, (a, b) in enumerate(merged):
        pa = cfg["lead"] if k == 0 else max(half, cfg["pad"])
        pb = cfg["tail"] if k == len(merged) - 1 else max(half, cfg["pad"])
        a2, b2 = max(0.0, a - pa), min(duration, b + pb)
        if out and a2 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b2)
        else:
            out.append([a2, b2])
    return out


class TimeMap:
    def __init__(self, segs):
        self.segs = []
        o = 0.0
        for a, b in segs:
            self.segs.append((a, b, o))
            o += b - a
        self.duration = o

    def src2out(self, t):
        for a, b, o in self.segs:
            if t < a:
                return o
            if t <= b:
                return o + (t - a)
        return self.duration

    def out2src(self, t):
        for a, b, o in self.segs:
            if t < o + (b - a):
                return a + max(0.0, t - o)
        a, b, o = self.segs[-1]
        return b


# ---------------- 动态信息卡（motion 风格）----------------
_CN_DIG = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?|[一二两三四五六七八九十百千]+)\s*(多|余|几)?\s*(万|亿|千|百)?\s*"
                     r"(次|个|条|天|年|月|周|小时|分钟|秒|倍|元|块|人|家|款|套|步|种|项|篇|期|万|亿|%|％)")


def _cn2num(t):
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return float(t)
    total, cur = 0, 0
    for ch in t:
        if ch in _CN_DIG:
            cur = _CN_DIG[ch]
        elif ch == "十":
            total += (cur or 1) * 10
            cur = 0
        elif ch == "百":
            total += (cur or 1) * 100
            cur = 0
        elif ch == "千":
            total += (cur or 1) * 1000
            cur = 0
    return float(total + cur)


def _find_number(text):
    """从一句话里找"可视化的数字"，如 200多次、75.4亿、3个、半个月 → (值, 小数位, 单位, 是否加号)。"""
    for m in _NUM_RE.finditer(text):
        raw, more, big, unit = m.groups()
        try:
            v = _cn2num(raw)
        except Exception:  # noqa
            continue
        if v <= 0 or (v < 2 and not big):
            continue
        if raw in ("一", "两") and not big:  # "一个""两个"太常见，不当数字展示
            continue
        dec = len(raw.split(".")[1]) if "." in raw else 0
        u = (big or "") + (unit if unit not in ("万", "亿") or not big else "")
        return v, dec, u, bool(more)
    return None


def _short(text, n):
    text = re.sub(r"[，。,.!！?？、：:；;\s]+", " ", text).strip()
    return text if len(text) <= n else text[:n]


def make_card(kind, sub_texts, label, terms, pops, chapters, ch, n_card, brand_title=""):
    """根据这一段口播内容，自动决定卡片类型与文字。"""
    txt = "".join(sub_texts)
    first = sub_texts[0] if sub_texts else ""
    hud_l = f"SCENE {n_card:02d} / CH.{ch + 1:02d}"
    base = {"idx": f"{n_card:02d}", "hud_l": hud_l, "hud_r": "REC"}
    if kind == "title":
        return dict(base, type="title", text=brand_title, sub=_short(first, 20), tag="AICLIP  /  EPISODE", idx="01")
    good_labels = len(chapters) >= 3 and all(len(c["label"]) >= 2 and not c["label"].startswith("要点")
                                             for c in chapters)
    if kind == "list" and good_labels:
        return dict(base, type="list", items=[c["label"] for c in chapters][:9], active=ch,
                    tag=f"CHAPTER {ch + 1:02d} / {len(chapters):02d}")
    num = _find_number(txt)
    if num:
        v, dec, u, more = num
        cap = next((t for t in sub_texts if _find_number(t)), first)
        return dict(base, type="number", value=v, decimals=dec, unit=u, plus=more, caption=_short(cap, 18),
                    tag="DATA  /  " + f"{n_card:02d}", right=n_card % 2 == 0)
    weak = set("直接 造成 多数 数值 超标 出现 发现 可能 需要 进行 应该 其实 只是 一定 非常 比较 特别 主要 不是 偏高 "
               "开始 最后 一样 不同 很多 一种 这种 那种 各种 所有 任何 目前 以后 之前 然后 而且 结果 项目".split())
    hits = [g for g in terms if 2 <= len(g) <= 5 and g in txt and g not in weak and g not in T.GENERIC
            and not any(ch in T.STOP for ch in g)]
    uniq = []
    def _overlap(a, b):
        return any(a[i:i + 2] in b for i in range(len(a) - 1))
    for g in sorted(hits, key=lambda x: (-len(x) if False else txt.find(x))):
        if not any(g in u or u in g or _overlap(g, u) for u in uniq):
            uniq.append(g)
    if len(uniq) >= 3 or (len(uniq) == 2 and n_card % 3 == 2):
        return dict(base, type="keywords", words=uniq[:3], tag="KEYWORDS")
    # 标题：优先用这段里"短而实"的一句（3~9 字、少虚词），口播里这类短句通常就是要点本身
    def score(t):
        c = re.sub(r"[，。,.!！?？、\s]", "", t)
        if not 3 <= len(c) <= 9:
            return -1
        bad = sum(ch in "我你他她它们这那就也都会的了啊呢吧吗么" for ch in c)
        return 10 - bad * 2 - abs(len(c) - 5) * 0.5
    cands = sorted(((score(t), i) for i, t in enumerate(sub_texts)), reverse=True)
    if cands and cands[0][0] >= 6:
        hi = cands[0][1]
        head = _short(sub_texts[hi], 9)
        rest = [t for i, t in enumerate(sub_texts) if i != hi]
        sub = _short(max(rest, key=len), 22) if rest else ""
    elif 4 <= len(_short(first, 99).replace(" ", "")) <= 12:
        head, sub = _short(first, 12), _short(sub_texts[1], 22) if len(sub_texts) > 1 else ""
    else:
        head = next((p for p in pops if p in txt and len(p) >= 2 and p not in T.GENERIC), None) \
            or (uniq[0] if uniq else None) or _short(first, 8)
        sub = _short(first if head not in first or len(first) > len(head) + 2 else
                     (sub_texts[1] if len(sub_texts) > 1 else ""), 22)
    if sub == head:
        sub = ""
    return dict(base, type="headline", text=head, sub=sub, tag=f"KEY POINT  /  {n_card:02d}")


# ---------------- AI_UP：信息层模式下的画中画时段 ----------------
def _plan_pip_slots(subs, graphics, blocks, broll_dir, style, D):
    """有启用的素材就一定要用上：按“每 ~18 秒一个、关键词句优先、尽量避开逐条出现的列表”挑时段，
    信息层在这些时段让位。返回 (overlays, graphics)。"""
    cfg = style.get("media_pip") or {}
    mats = _load_materials(broll_dir) if cfg.get("enable", True) else []
    if not mats or not subs:
        return [], graphics
    lo, hi = cfg.get("dur", [2.5, 5.0])
    every = float(cfg.get("every_sec", 18.0))
    gap = max(float(cfg.get("min_gap", 8.0)), 6.0)
    want = max(1, min(int(D // every) + 1, len(mats) * 2))
    outro = T.outro_start(subs)
    first_end = blocks[0]["end"] if blocks else 3.0

    def cost(t0, t1):
        c = 0.0
        for g in graphics:
            if g["t0"] < t1 and t0 < g["t1"]:
                prog = any(it.get("at", 0) > 0 for it in g.get("items") or [])
                c += {"list": 3, "cards": 3, "stats": 2, "hero": 2}.get(g["type"], 1) + (4 if prog else 0)
        return c

    cands = []
    for i, s in enumerate(subs):
        if s["start"] < 0.8 or (outro is not None and i >= outro):
            continue
        early = s["start"] < max(2.0, min(first_end, 6.0))  # 开场只放“说到素材关键词”的
        t0 = s["start"] + 0.1
        j, t1 = i, s["end"]
        while t1 - t0 < lo and j + 1 < len(subs):
            j += 1
            t1 = subs[j]["end"]
        if t1 - t0 < lo * 0.8:
            continue
        t1 = min(t1, t0 + hi, D - 0.3)
        txt = "".join(x["text"].replace(" ", "") for x in subs[max(0, i - 1): j + 2])  # 关键词可能被断句拆开
        kw = next((m for m in mats if any(k and k.replace(" ", "") in txt for k in m.get("keywords", []))), None)
        if early and kw is None:
            continue
        cands.append({"t0": t0, "t1": t1, "kw": kw, "score": (10 if kw else 0) - cost(t0, t1)})
    def hits_prog(c):  # 和列表/卡片/数据重叠：会把它们截掉或让条目出不来，尽量不选
        return any(g["t0"] - 0.3 < c["t1"] and c["t0"] < g["t1"] + 0.3 and g.get("items") for g in graphics)
    pool = [c for c in cands if not hits_prog(c)] or cands
    picked = []
    for c in sorted(pool, key=lambda c: (-c["score"], c["t0"])):
        if len(picked) >= want:
            break
        if all(c["t0"] >= p["t1"] + gap or c["t1"] <= p["t0"] - gap for p in picked):
            picked.append(c)
    if not picked and pool:  # 太短的视频也至少放一个
        picked = [max(pool, key=lambda c: c["score"])]
    picked.sort(key=lambda c: c["t0"])
    used, rr, overlays = {}, 0, []
    for c in picked:
        m = c["kw"] if c["kw"] is not None and used.get(c["kw"]["path"], 0) < 2 else None
        if m is None:
            gen = [x for x in mats if x.get("generic", True)] or mats
            m = min(gen, key=lambda x: (used.get(x["path"], 0), (gen.index(x) - rr) % len(gen)))
            rr += 1
        k = used.get(m["path"], 0)
        used[m["path"]] = k + 1
        overlays.append({"path": m["path"], "start": round(c["t0"], 3), "end": round(c["t1"], 3),
                         "offset": 4.0 * k})
    # 橙光 Plus：一部分素材改成“整屏大图 + 人物角落小窗”（素材放大展示），其余保持小窗
    stage_media = []
    if (style.get("hud") or {}).get("plus"):
        keep = []
        for k, ov in enumerate(overlays):
            if k % 2 == 0 and ov["end"] - ov["start"] >= 3.0:
                stage_media.append({"t0": ov["start"], "t1": ov["end"], "type": "media", "layout": "stage",
                                    "media": ov["path"], "offset": ov["offset"], "kicker": ["", ""]})
            else:
                keep.append(ov)
        overlays = keep
    # 信息层让位
    out = []
    for g in graphics:
        g = dict(g)
        for ov in overlays:
            a, b = ov["start"] - 0.3, ov["end"] + 0.3
            if not (g["t0"] < b and a < g["t1"]):
                continue
            last_at = max([float(it.get("at", 0) or 0) for it in g.get("items") or []] + [0.0])
            if a - g["t0"] >= max(1.5, last_at + 1.2):  # 素材之前还够看（逐条出现的都已出完）：截短
                g["t1"] = a
            elif g["t1"] - b >= 2.0 and not last_at:  # 素材之后还有 2 秒以上：后移开场
                shift = b - g["t0"]
                g["t0"] = b
                if g.get("items"):
                    g["items"] = [dict(it, at=max(0.0, float(it.get("at", 0) or 0) - shift)) for it in g["items"]]
            else:
                g = None
                break
        if g and g["t1"] - g["t0"] >= 1.5:
            out.append(g)
    if stage_media:  # 整屏素材也要让信息层让位（和上面同样规则）
        res = []
        for g in out:
            for sm in stage_media:
                a, b = sm["t0"] - 0.3, sm["t1"] + 0.3
                if g is not None and g["t0"] < b and a < g["t1"]:
                    if a - g["t0"] >= 1.5 and not g.get("items"):
                        g = dict(g, t1=a)
                    else:
                        g = None
            if g is not None:
                res.append(g)
        out = sorted(res + stage_media, key=lambda g: g["t0"])
    log(f"画中画素材 {len(overlays)} 个：" + "，".join(f"{os.path.basename(o['path'])}@{o['start']:.1f}s"
                                               for o in overlays) + (f"；整屏大图 {len(stage_media)} 个" if stage_media else "") + f"（可用素材 {len(mats)} 个）")
    return overlays, out


# ---------------- AI_UP：只在重点句推近 ----------------
def _key_zooms(subs, graphics, emph, pops, pc):
    """平时不缩放；只在重点句（大数字出现、模型标出的强调句、带数字/感叹的句子）推近一次，说完拉回。"""
    score = {}
    def add(i, v):
        if 0 <= i < len(subs):
            score[i] = max(score.get(i, 0), v)
    for g in graphics:
        if g.get("type") == "hero":
            add(g["i0"], 3)
    for i in emph:
        add(int(i), 3)
    for i, s in enumerate(subs):
        t = s["text"]
        if _find_number(t) and not re.match(r"\s*第\s*[一二三四五六七八九十0-9]+\s*[步点条个招]", t):
            add(i, 2)
        if t.rstrip().endswith(("！", "!")):
            add(i, 1)
        if any(p in t for p in pops[:6]):
            add(i, 1)
    gap = pc.get("zoom_gap", 7.0)
    z = pc.get("zoom_key", 1.12)
    picked = []
    for i in sorted(score, key=lambda i: (-score[i], subs[i]["start"])):
        st = subs[i]["start"]
        if score[i] < 2 or st < 1.5 or subs[i]["end"] - st < 0.8:  # 只有感叹/关键词的句子不够“重点”
            continue
        if all(abs(st - subs[j]["start"]) >= gap for j in picked):
            picked.append(i)
    zooms = []
    for i in sorted(picked):
        zooms.append({"t": round(subs[i]["start"], 3), "z": z})
        zooms.append({"t": round(subs[i]["end"], 3), "z": 1.0})
    log(f"重点推近 {len(picked)} 次：", [subs[i]["text"][:10] for i in sorted(picked)])
    return zooms


# ---------------- 主编排 ----------------
def _style_plan(subs, chapters, n_styles, D):
    """多风格混剪：每条字幕用哪个风格。章节够多时按章节轮换；否则按时长均分成若干段轮换。
    返回 (每条字幕的风格序号, 风格切换处的字幕序号集合)。"""
    n = len(subs)
    if n_styles <= 1 or n == 0:
        return [0] * n, set()
    if len(chapters) >= n_styles:
        of, ci = [], 0
        for k in range(n):
            while ci + 1 < len(chapters) and k >= chapters[ci + 1]["idx"]:
                ci += 1
            of.append(ci % n_styles)
    else:
        parts = max(n_styles, int(D // 40) + 1)
        of = [min(parts - 1, int(s["start"] / max(1e-3, D) * parts)) % n_styles for s in subs]
    switches = {k for k in range(1, n) if of[k] != of[k - 1]}
    return of, switches


def build_timeline(subs_src, vad_segs, src_info, style, brand_in, face, out_dir=None, broll_dir="", styles=None):
    styles = styles or [style]
    fps = style["canvas"]["fps"]
    segs = plan_cuts(subs_src, vad_segs, src_info["duration"], style["cut"])
    tm = TimeMap(segs)
    D = tm.duration
    log(f"去停顿：{src_info['duration']:.1f}s → {D:.1f}s，保留 {len(segs)} 段")
    subs = []
    for s in subs_src:
        a, b = tm.src2out(s["start"]), tm.src2out(s["end"])
        if b - a > 0.05:
            subs.append({"start": round(a, 3), "end": round(b, 3), "text": s["text"]})
    # 字幕连续化：间隙很小时延长上一条，避免字幕闪烁
    for i in range(len(subs) - 1):
        if subs[i + 1]["start"] - subs[i]["end"] < 0.35:
            subs[i]["end"] = subs[i + 1]["start"]

    # ---- 语义：章节 / 关键词 ----
    terms = T.extract_terms(subs, top=80)
    chapters = T.detect_chapters(subs, terms)
    pops = T.pop_terms(terms[:40], [c["label"] for c in chapters])
    sem = llm.plan_semantics(subs)
    if sem:
        if sem.get("chapters"):
            chapters = [{"idx": int(c["idx"]), "label": T._cut(str(c["label"]), 6)} for c in sem["chapters"]
                        if 0 <= int(c["idx"]) < len(subs)]
            if not chapters or chapters[0]["idx"] != 0:
                chapters.insert(0, {"idx": 0, "label": "开篇"})
        if sem.get("pop_terms"):
            pops = [p for p in sem["pop_terms"] if any(p in s["text"] for s in subs)] or pops
    name, role = T.find_name(subs)
    brand = dict(style["brand"])
    brand.update({k: v for k, v in (brand_in or {}).items() if v})
    if sem and sem.get("slogan") and not (brand_in or {}).get("slogan"):
        brand["slogan"] = sem["slogan"]
    if not brand.get("title"):
        brand["title"] = (name + role) if name else ""
    if not brand.get("logo_text") and name:
        brand["logo_text"] = name
    if not brand.get("seal_text") and name:
        brand["seal_text"] = name[0] + "印"
    if not brand.get("name"):
        brand["name"] = name or ""
    log("章节:", [c["label"] for c in chapters], "| 弹字词:", pops[:10])

    style_of, switches = _style_plan(subs, chapters, len(styles), D)
    if len(styles) > 1:
        log("多风格混剪：", " → ".join(str(style_of[k]) for k in [0] + sorted(switches)))
    ch_starts = {c["idx"] for c in chapters} | switches
    outro = T.outro_start(subs)
    pc = style["pacing"]

    # ---- 版式分块 ----
    blocks = []
    i, n = 0, len(subs)
    # 开场钩子：全屏
    j = i
    while j < n and (subs[j]["end"] < pc["hook_sec"]) and (j + 1 not in ch_starts or j == 0):
        j += 1
    j = max(j, 0)
    blocks.append({"i0": 0, "i1": j, "mode": "full"})
    i = j + 1
    cyc = 1  # 开场全屏之后先切画中画
    while i < n:
        if outro is not None and i >= outro:
            blocks.append({"i0": i, "i1": n - 1, "mode": "full", "outro": True})
            break
        pc = styles[style_of[i]]["pacing"]
        if i in ch_starts:
            mode = pc.get("chapter_mode", "pip_circle")
            lo, hi = pc["block_sec"].get(mode, [5.0, 9.0])
            cyc = 0
        else:
            mode = pc["cycle"][cyc % len(pc["cycle"])]
            lo, hi = pc["block_sec"].get(mode, [5.0, 9.0])
            cyc += 1
        j = i
        while j + 1 < n:
            dur = subs[j]["end"] - subs[i]["start"]
            nxt_dur = subs[j + 1]["end"] - subs[i]["start"]
            if (j + 1) in ch_starts or (outro is not None and j + 1 >= outro):
                break
            if dur >= lo and nxt_dur > hi:
                break
            if mode == pc.get("chapter_mode", "pip_circle") and i in ch_starts and dur >= lo:
                break
            j += 1
        blocks.append({"i0": i, "i1": j, "mode": mode})
        i = j + 1
    # 合并过短的普通块
    # 写入时间与背景
    chap_of = []
    ci = 0
    for k in range(n):
        while ci + 1 < len(chapters) and k >= chapters[ci + 1]["idx"]:
            ci += 1
        chap_of.append(ci)
    files = []
    if broll_dir and os.path.isdir(broll_dir):
        files = [os.path.join(broll_dir, f) for f in os.listdir(broll_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov"))
            and not f.startswith("_")]
    seen_round = set()
    n_card = 0
    used_media = {}
    out_blocks = []
    for b in blocks:
        start = 0.0 if not out_blocks else subs[b["i0"]]["start"]
        txt = "".join(s["text"] for s in subs[b["i0"]: b["i1"] + 1])
        ch = chap_of[b["i0"]]
        label = chapters[ch]["label"]
        term = next((p for p in pops if p in txt), label)
        bg = None
        if b["mode"] in ("pip_round", "vignette"):
            media = next((f for f in files if os.path.splitext(os.path.basename(f))[0] in txt + label
                          and used_media.get(f, 0) < 2), None)
            if media:
                used_media[media] = used_media.get(media, 0) + 1
            if b["mode"] == "pip_round" and ch not in seen_round and not media:
                bg = {"type": "blur"}
                seen_round.add(ch)
            elif media:
                bg = {"type": "media", "path": media}
            else:
                bg = {"type": "card", "label": term[:5], "seed": ch}
            if b["mode"] == "vignette":
                bg["dark"] = True
        elif b["mode"] == "pip_circle":
            bg = {"type": "blur", "light": True}
        elif b["mode"] == "card":
            n_card += 1
            texts = [s["text"] for s in subs[b["i0"]: b["i1"] + 1]]
            sty = styles[style_of[b["i0"]]] if subs else style
            kind = "list" if b["i0"] in {c["idx"] for c in chapters[1:]} else "auto"
            if n_card == 1 and sty["pacing"].get("title_card") and brand.get("title") \
                    and not sty["keyword_pop"].get("hook"):
                kind = "title"
            bg = {"type": "motion", "card": make_card(kind, texts, label, terms, pops, chapters, ch, n_card,
                                                       brand.get("title", ""))}
        out_blocks.append({"start": round(start, 3), "mode": b["mode"], "bg": bg, "i0": b["i0"], "i1": b["i1"],
                           "chapter": ch, "style": style_of[b["i0"]] if subs else 0})
    for k in range(len(out_blocks)):
        out_blocks[k]["end"] = out_blocks[k + 1]["start"] if k + 1 < len(out_blocks) else round(D, 3)
    _plan_overlays(out_blocks, subs, chapters, chap_of, broll_dir, style)
    graphics, emph, overlays = [], [], []
    if (style.get("hud") or {}).get("enable"):
        from . import hudplan
        graphics, en, emph = hudplan.plan(subs, chapters, brand.get("title", ""), D, style, _find_number)
        for s_, e_ in zip(subs, en):
            s_["en"] = e_
        # 画中画素材优先：先给素材留出时段，信息层在这些时段让位（截短/后移/去掉）
        for b_ in out_blocks:
            b_.pop("overlay", None)
        overlays, graphics = _plan_pip_slots(subs, graphics, out_blocks, broll_dir, style, D)
    for b in out_blocks:
        for s in subs[b["i0"]: b["i1"] + 1]:
            s["mode"] = b["mode"]

    # ---- 推镜头（全屏块内逐句跳切缩放）----
    zooms = []
    if style["pacing"].get("zoom_mode") == "key":
        zooms = _key_zooms(subs, graphics, emph, pops, style["pacing"])
    else:
        for b in out_blocks:
            if b["mode"] != "full":
                continue
            zl = styles[b["style"]]["pacing"]["zoom_levels"]
            for k, s in enumerate(subs[b["i0"]: b["i1"] + 1]):
                zooms.append({"t": s["start"] if k else b["start"], "z": zl[k % len(zl)]})

    # ---- 数字人素材切换点：配一次景别变化（近景↔中景），像换机位 ----
    cz = style["pacing"].get("cut_zoom")
    cf = os.environ.get("AICLIP_CUTS")
    if cz and cf and os.path.exists(cf):
        try:
            import json as _json
            cuts = [float(c) for c in (_json.load(open(cf, encoding="utf-8")) or {}).get("cuts") or []]
            outs = sorted({round(tm.src2out(c), 3) for c in cuts if 0.5 < tm.src2out(c) < D - 0.5})
            zooms = [{"t": 0.0, "z": float(cz[0]), "cut": True}] + \
                [{"t": t_, "z": float(cz[(i + 1) % len(cz)]), "cut": True} for i, t_ in enumerate(outs)]
            log(f"数字人切换 {len(outs)} 次，切换处换景别", [round(t_, 1) for t_ in outs])
        except Exception as e:  # noqa
            log("切换点读取失败，忽略：", e)

    # ---- 关键词弹字 ----
    pop_ev, used, last = [], set(), -999
    per_min = {}
    blk_start, blk_style = {}, {}
    for b in out_blocks:
        for k in range(b["i0"], b["i1"] + 1):
            blk_start[k] = b["start"]
            blk_style[k] = b["style"]
    for si, s in enumerate(subs):
        if s.get("mode") != "full" or s["start"] < 5.0:
            continue
        kp = styles[blk_style.get(si, 0)]["keyword_pop"]
        pc = styles[blk_style.get(si, 0)]["pacing"]
        if kp.get("max_per_min", 4) <= 0:
            continue
        kin = kp.get("kind") == "kinetic"
        for term in pops:
            if term in used or term not in s["text"]:
                continue
            if kin and (len(term) < 3 and not re.search(r"[A-Za-z0-9]", term) or term.startswith("要点")):
                continue
            t = s["start"] + (s["end"] - s["start"]) * s["text"].find(term) / max(1, len(s["text"]))
            t = max(t, blk_start.get(si, 0) + pc["transition_sec"] + 0.1)
            if t > s["end"] - 0.3:
                continue
            m = int(t // 60)
            if t - last < kp["min_gap"] or per_min.get(m, 0) >= kp["max_per_min"]:
                continue
            dur = min(kp["dur"], max(1.0, s["end"] - t + 0.4))
            pop_ev.append({"t": round(t, 3), "dur": round(dur, 3), "text": term})
            used.add(term)
            last = t
            per_min[m] = per_min.get(m, 0) + 1
            break

    # 开场钩子大字（kinetic 风格）：前两句里最有分量的词
    kp0 = styles[out_blocks[0]["style"]]["keyword_pop"] if out_blocks else {}
    if kp0.get("hook") and subs and out_blocks and out_blocks[0]["mode"] == "full":
        head = " ".join(s["text"] for s in subs[:2])
        # 有标题就用标题当开场大字（最能概括这条视频），否则用前两句里的关键词
        cand = ([brand["title"]] if brand.get("title") and len(brand["title"]) <= 10 else []) + \
            [p for p in pops if p in head and len(p) >= 3] + \
            [g for g in terms if 3 <= len(g) <= 5 and g in head and g not in T.GENERIC
             and not any(ch in T.STOP for ch in g)]
        if cand:
            t0 = round(subs[0]["start"] + 0.25, 3)
            if all(abs(p["t"] - t0) > 3 for p in pop_ev):
                pop_ev.insert(0, {"t": t0, "dur": min(kp0.get("dur", 1.6), max(1.0, out_blocks[0]["end"] - t0 - 0.2)),
                                  "text": cand[0]})

    # ---- 章节时间 / 黑场 / 音效 ----
    chap_out = []
    for k, c in enumerate(chapters):
        st = 0.0 if k == 0 else subs[c["idx"]]["start"]
        en = subs[chapters[k + 1]["idx"]]["start"] if k + 1 < len(chapters) else D
        chap_out.append({"start": round(st, 3), "end": round(en, 3), "label": c["label"]})
    dips = [c["start"] for c in chap_out[1:]]
    # 风格切换处也加黑场，遮住包装层的变化
    for k in range(1, len(out_blocks)):
        if out_blocks[k]["style"] != out_blocks[k - 1]["style"] and \
                all(abs(out_blocks[k]["start"] - d) > 0.5 for d in dips):
            dips.append(out_blocks[k]["start"])
    dips.sort()
    sfx = []
    prev = ("full", 0)
    for b in out_blocks[1:]:
        if (b["mode"], b["style"]) != prev or b["mode"] != "full":
            sfx.append({"t": max(0.0, b["start"] - 0.12), "kind": "whoosh"})
        prev = (b["mode"], b["style"])
    sfx += [{"t": p["t"], "kind": "pop"} for p in pop_ev]
    sfx += [{"t": d, "kind": "ding"} for d in dips]
    sfx += [{"t": max(0.0, g["t0"] - 0.08), "kind": "whoosh"} for g in graphics]

    nc = style["namecard"]
    namecard = None
    if nc.get("show") and brand.get("name"):
        lines = brand.get("namecard_lines") or [brand.get("slogan", "")]
        namecard = {"start": nc["start"], "dur": min(nc["dur"], max(2.5, out_blocks[0]["end"] - 0.3)),
                    "name": brand["name"], "lines": [l for l in lines if l]}

    # 手动设置的视频标题（网页“标题与封面”面板）：替换自动的开场标题
    title_fx = None
    tf = os.environ.get("AICLIP_TITLE_FX")
    if tf and os.path.exists(tf):
        try:
            import json as _json
            v = (_json.load(open(tf, encoding="utf-8")) or {}).get("video") or {}
            if v.get("enable") and any(str(x).strip() for x in v.get("lines") or []):
                title_fx = v
                if graphics and graphics[0].get("i0", 1) == 0 and graphics[0].get("type") in ("headline", "hero"):
                    graphics = graphics[1:]
                log("视频标题：使用手动设置", v.get("lines"))
        except Exception as e:  # noqa
            log("视频标题设置读取失败，忽略：", e)
    law = None
    if (style.get("law") or {}).get("enable"):
        from . import law as law_mod
        law = law_mod.plan(subs, brand, style, pops, terms)

    tl = {
        "law": law, "title_fx": title_fx,
        "fps": fps, "canvas": [style["canvas"]["w"], style["canvas"]["h"]], "duration": round(D, 3),
        "frames": int(round(D * fps)),
        "segments": [[round(a, 4), round(b, 4)] for a, b in segs],
        "subs": subs, "chapters": chap_out, "blocks": out_blocks, "zooms": zooms, "pops": pop_ev,
        "dips": dips, "sfx": sfx, "namecard": namecard, "graphics": graphics, "overlays": overlays, "brand": brand, "face": face,
        "src": {"w": src_info["w"], "h": src_info["h"], "fps": src_info["fps"]},
        "styles": [s.get("name") or "" for s in styles],
    }
    stat = {}
    for b in out_blocks:
        stat[b["mode"]] = stat.get(b["mode"], 0) + 1
    log(f"分块 {len(out_blocks)} 个 {stat}，弹字 {len(pop_ev)} 个，推镜 {len(zooms)} 次，音效 {len(sfx)} 个")
    return tl



# ---------------- AI_UP：画中画素材（素材小窗叠在口播画面上） ----------------
def _load_materials(broll_dir):
    """broll_dir/materials.json：[{"path":..., "keywords":[...]}]；没有就按文件名当关键词。"""
    if not broll_dir or not os.path.isdir(broll_dir):
        return []
    mj = os.path.join(broll_dir, "materials.json")
    if os.path.exists(mj):
        try:
            import json
            items = json.load(open(mj, encoding="utf-8"))
            out = []
            for m in items:
                if m.get("enabled", True) is False:
                    continue
                p = m.get("path") or m.get("file") or ""
                if p and not os.path.isabs(p):
                    p = os.path.join(broll_dir, p)
                if os.path.exists(p):
                    out.append(dict(m, path=p))
            return out
        except Exception:
            pass
    return [{"path": os.path.join(broll_dir, f), "keywords": [os.path.splitext(f)[0]]}
            for f in sorted(os.listdir(broll_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov"))
            and not f.startswith("_")]


def _plan_overlays(blocks, subs, chapters, chap_of, broll_dir, style):
    """在全屏口播段落上叠素材小窗：先按关键词匹配，其余用通用素材轮流；
    开场钩子和结尾不放，两个素材之间至少隔 gap 秒，每段素材只出现在句子边界上。"""
    cfg = style.get("media_pip") or {}
    if not cfg.get("enable", True):
        return
    mats = _load_materials(broll_dir)
    if not mats:
        return
    gap = float(cfg.get("min_gap", 8.0))
    lo, hi = cfg.get("dur", [2.5, 5.0])
    used, rr, last_end = {}, 0, -1e9
    for k, b in enumerate(blocks):
        if k == 0 or b["mode"] != "full" or b.get("outro"):
            continue
        dur = b["end"] - b["start"]
        if dur < lo or b["start"] - last_end < gap:
            continue
        txt = "".join(s["text"] for s in subs[b["i0"]: b["i1"] + 1])
        label = chapters[chap_of[b["i0"]]]["label"]
        m = next((m for m in mats if used.get(m["path"], 0) < 2 and any(
            kw and kw in txt + label for kw in m.get("keywords", []))), None)
        if m is None:
            gen = [x for x in mats if x.get("generic", True)]
            if not gen:
                continue
            m = gen[rr % len(gen)]
            rr += 1
        off = 4.0 * used.get(m["path"], 0)  # 同一段素材再次出现时从后面开始放，不重复同一画面
        used[m["path"]] = used.get(m["path"], 0) + 1
        # 素材窗从段落开头稍后出现，持续 lo~hi 秒，落在句子结束处
        st = b["start"] + 0.25
        en = min(b["end"] - 0.1, st + hi)
        for s_ in subs[b["i0"]: b["i1"] + 1]:
            if s_["end"] - st >= lo:
                en = min(en, s_["end"])
                break
        b["overlay"] = {"path": m["path"], "start": round(st, 3), "end": round(en, 3), "offset": off}
        last_end = en
