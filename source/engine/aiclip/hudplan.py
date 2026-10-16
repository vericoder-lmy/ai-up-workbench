"""AI_UP：信息层（HUD）编排。优先让大模型按文案生成“眉标 + 大字/数字 + 列表/卡片/数据”和英文字幕；
大模型不可用时用规则版兜底（开场点题、数字放大、章节标题、结尾总结）。"""
import hashlib
import json
import os
import re

from . import llm
from .utils import log

TYPES = ("hero", "headline", "list", "cards", "stats")


def _clip(s, n):
    """按视觉宽度截断：英文字母算半个字，不截断英文单词。"""
    s = str(s or "").strip()
    w, out = 0.0, []
    for tok in re.findall(r"[A-Za-z0-9.%+\-']+|.", s):
        tw = len(tok) * 0.55 if re.match(r"[A-Za-z0-9]", tok) else (0.4 if tok == " " else 1.0)
        if w + tw > n + 0.01:
            break
        out.append(tok)
        w += tw
    return "".join(out).strip() or s[:n]


def _soft(s, n=9, hard=14):
    """标题行收短但不拦腰截断：超过 n 字先在标点处断开；还长就整行保留（画面会自动缩小字号），超过 hard 才截。"""
    s = str(s or "").strip()
    if _vis(s) <= n:
        return s
    for m in re.finditer(r"[，,、：:；;？?！!。]", s):
        head = s[: m.end()].rstrip("，,、：:；;")
        if 2 <= _vis(head) <= hard:
            cut = head
    try:
        return cut  # noqa
    except NameError:
        return s if _vis(s) <= hard else _clip(s, hard)


def split_title(t):
    """标题里的“分行空格”：两侧都不是英文/数字的空格（“AI 配音”里的空格只是中英文间距）。
    也认 | ／ 和逗号。找到就返回两段（取最居中的那个），否则 None。"""
    t = str(t or "").strip()
    cands = []
    for m in re.finditer(r"\s*[|｜/／，,]\s*|\s+", t):
        a, b = t[: m.start()].strip(), t[m.end():].strip()
        if not a or not b:
            continue
        sep = m.group(0).strip()
        if not sep and (re.match(r"[A-Za-z0-9]", a[-1]) or re.match(r"[A-Za-z0-9]", b[0])):
            continue
        if _vis(a) >= 2.5 and _vis(b) >= 2.5:
            cands.append((abs(_vis(a) - _vis(b)), a, b))
    if not cands:
        return None
    _, a, b = min(cands)
    return [a, b]


def short_title(t, n=12):
    """开场标题：取第一句（到问号/感叹号/句号为止），太长再取第一个分句。"""
    t = re.split(r"(?<=[？?！!。])", str(t or "").strip())[0].strip()
    if _vis(t) > n:
        first = re.split(r"[，,；;：:]", t)[0].strip()
        t = first if _vis(first) >= 4 else _clip(t, n)
    return t


def _norm(g, n_subs):
    """校验并收紧一条大模型输出，不合格返回 None。"""
    try:
        i0, i1 = int(g["i0"]), int(g.get("i1", g["i0"]))
    except Exception:  # noqa
        return None
    if not (0 <= i0 < n_subs):
        return None
    i1 = max(i0, min(n_subs - 1, i1))
    typ = g.get("type") if g.get("type") in TYPES else "headline"
    out = {"i0": i0, "i1": i1, "type": typ}
    k = g.get("kicker") or []
    if isinstance(k, str):
        k = [k, ""]
    out["kicker"] = [_clip(k[0] if k else "", 34), _clip(k[1] if len(k) > 1 else "", 16)]
    if typ == "hero":
        h = g.get("hero") or {}
        val = _clip(h.get("val", ""), 8)
        if not val:
            return None
        out["hero"] = {"pre": _clip(h.get("pre", ""), 2), "val": val, "suf": _clip(h.get("suf", ""), 10),
                       "sub": _clip(h.get("sub", ""), 8)}
    lines = [_soft(x) for x in (g.get("lines") or []) if str(x).strip()][:2]
    if typ == "headline":
        if not lines:
            return None
        out["lines"] = lines
    for f, n in (("en", 32), ("note", 26)):
        if g.get(f):
            out[f] = _clip(g[f], n)
    items = []
    for it in (g.get("items") or [])[:4]:
        if not isinstance(it, dict):
            continue
        item = {kk: _clip(vv, 18) for kk, vv in it.items() if kk in ("n", "en", "zh", "tag", "label", "val", "note")}
        try:
            if it.get("i") is not None and i0 <= int(it["i"]) < n_subs:
                item["i"] = int(it["i"])  # 这一条开始讲的句子：画面随讲解逐条出现
        except (TypeError, ValueError):
            pass
        items.append(item)
    if typ in ("list", "cards", "stats"):
        if len(items) < 2:
            return None
        out["items"] = items
    return out


def _times(gs, subs, D, cfg):
    """按字幕序号落时间；去重叠、限时长、留间隔。"""
    lo, hi = cfg.get("dur", [3.0, 7.5])
    gap = cfg.get("gap", 1.2)
    res, last = [], -1e9
    for g in sorted(gs, key=lambda x: x["i0"]):
        t0 = subs[g["i0"]]["start"] + (0.15 if g["i0"] else 0.25)
        t1 = subs[g["i1"]]["end"]
        top = hi
        its = [it for it in g.get("items") or [] if "i" in it]
        if its:  # 逐条出现的列表：一直留到最后一条讲完
            last_i = max(it["i"] for it in its)
            nxt = min(len(subs) - 1, last_i + 1)
            t1 = max(t1, subs[nxt]["end"] if nxt > last_i else subs[last_i]["end"] + 1.0)
            top = max(hi, cfg.get("dur_list", 18.0))
        if g["type"] in ("list", "cards", "stats"):
            t1 = max(t1, t0 + 4.0)
        t1 = min(max(t1, t0 + lo), t0 + top, D - 0.2)
        if t0 < last + gap:
            t0 = last + gap
        if t1 - t0 < 2.0:
            continue
        g = dict(g, t0=round(t0, 3), t1=round(t1, 3))
        if its:
            g["items"] = [dict(it, at=round(max(0.0, subs[it["i"]]["start"] - t0), 3)) if "i" in it else it
                          for it in g["items"]]
        res.append(g)
        last = t1
    return res


_STEP = re.compile(r"^(第\s*([一二三四五六七八九十0-9]+)\s*[步点条个招]|首先|其次|再次|然后|最后)")
_CN = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cover_steps(gs, subs):
    """讲到“第X步 / 首先 / 最后”的句子必须有信息块：没被覆盖的补一个步骤标题。"""
    def covered(i):
        for g in gs:
            its = [it["i"] for it in g.get("items") or [] if "i" in it]
            if g["i0"] <= i <= max([g["i1"]] + its):
                return True
        return False
    steps = [i for i, s in enumerate(subs) if _STEP.match(s["text"].strip())]
    # 列表没标 i 时：区间里的“第X步”句子数正好等于条目数，就按顺序对上，做成逐条出现
    for g in gs:
        its = g.get("items") or []
        if g["type"] in ("list", "cards") and its and not any("i" in it for it in its):
            inside = [i for i in steps if g["i0"] <= i <= max(g["i1"], g["i0"] + 8)]
            if len(inside) == len(its):
                g["items"] = [dict(it, i=i) for it, i in zip(its, inside)]
    out = list(gs)
    n_step = 0
    for i, s in enumerate(subs):
        m = _STEP.match(s["text"].strip())
        if not m:
            continue
        n_step += 1
        if covered(i):
            continue
        num = m.group(2)
        k = int(num) if num and num.isdigit() else _CN.get(num or "", n_step)
        rest = re.sub(r"^[，,：:、\s]+", "", s["text"].strip()[m.end():])
        head = re.split(r"[，,。！？!?；;：:]", rest)[0] or rest
        lines = _split2(_clip(head, 12)) if head else [m.group(1)]
        out.append({"i0": i, "i1": min(len(subs) - 1, i + 1), "type": "headline",
                    "kicker": [f"STEP {k:02d}", m.group(1)], "lines": lines})
    return out


def _vis(text):
    return sum(0.55 if re.match(r"[A-Za-z0-9 ]", ch) else 1.0 for ch in str(text))


def _plus_layouts(gs):
    """橙光 Plus：没指定版式的信息块自动分配——
    短句标题、大数字 → bigword（人物移到一侧，另一侧超大字）；
    列表/卡片/数据且时长够 → stage（人物缩到角落小窗，整屏信息卡）；其余保持 side。"""
    out, last = [], None
    for g in gs:
        g = dict(g)
        if not g.get("layout"):
            dur = g["t1"] - g["t0"]
            lines = g.get("lines") or []
            if g["type"] == "hero":
                g["layout"] = "bigword"
            elif g["type"] == "headline" and lines and max(_vis(l) for l in lines) <= 6 and last != "bigword":
                g["layout"] = "bigword"
            elif g["type"] in ("list", "cards", "stats") and dur >= 4.5:
                g["layout"] = "stage"
            else:
                g["layout"] = "side"
        last = g["layout"]
        out.append(g)
    return out


def _split2(text, per=7):
    """长标题拆成两行（不拆英文单词），第二行作为橙色重点。"""
    toks = re.findall(r"[A-Za-z0-9.%+\-']+|.", str(text).strip())
    ws = [len(t) * 0.55 if re.match(r"[A-Za-z0-9]", t) else (0.4 if t == " " else 1.0) for t in toks]
    tot = sum(ws)
    if tot <= per:
        return ["".join(toks)]
    # 先找“安全断点”（虚词/标点之后、英文单词或空格两侧），落在 30%~70% 之间就用离中间最近的那个；
    # 没有才退回按长度对半（中文没有分词，硬拆容易把词拆开）
    acc, cands = 0.0, []
    for i in range(1, len(toks)):
        acc += ws[i - 1]
        a, b = toks[i - 1], toks[i]
        safe = a in "了的是：:，,？?！!、 " or b == " " or bool(re.match(r"[A-Za-z0-9]", a)) != bool(re.match(r"[A-Za-z0-9]", b))
        if b in "了的个们吗呢吧，,？?！!、。":
            safe = False
        cands.append((i, acc, safe))
    ok = [c for c in cands if c[2] and 0.25 * tot <= c[1] <= 0.75 * tot]
    pool = ok or cands
    bi = min(pool, key=lambda c: abs(c[1] - tot / 2) + (0 if ok else (1.5 if toks[c[0]] in "了的个们吗呢吧" else 0)))[0]
    return ["".join(toks[:bi]).strip(), "".join(toks[bi:]).strip()]


def _rules(subs, chapters, title, find_number):
    gs = []
    if subs:
        j = 0
        while j + 1 < len(subs) and subs[j]["end"] < 3.5:
            j += 1
        two = split_title(title)
        head = short_title(title or subs[0]["text"])
        gs.append({"i0": 0, "i1": j, "type": "headline", "kicker": ["EPISODE · 01", "本期主题"],
                   "lines": two or _split2(head)})
    steps = [i for i, s in enumerate(subs) if _STEP.match(s["text"].strip())]
    if len(steps) >= 2:
        items = []
        for k, i in enumerate(steps[:4]):
            m = _STEP.match(subs[i]["text"].strip())
            rest = re.sub(r"^[，,：:、\s]+", "", subs[i]["text"].strip()[m.end():])
            items.append({"n": f"{k + 1:02d}", "zh": _clip(rest or subs[i]["text"], 12), "i": i})
        gs.append({"i0": steps[0], "i1": steps[min(3, len(steps) - 1)], "type": "list",
                   "kicker": [f"{len(items)} STEPS", f"{len(items)} 步"], "items": items})
    skip = set(range(steps[0], steps[min(3, len(steps) - 1)] + 2)) if len(steps) >= 2 else set()
    starts = {c["idx"]: k for k, c in enumerate(chapters) if k > 0 and c["idx"] not in skip}
    for i, s in enumerate(subs):
        if i in starts:
            k = starts[i]
            lab = chapters[k]["label"]
            gs.append({"i0": i, "i1": min(len(subs) - 1, i + 1), "type": "headline",
                       "kicker": [f"STEP {k:02d}" if lab.startswith("第") else f"CHAPTER {k:02d}", lab],
                       "lines": [lab]})
            continue
        num = find_number(s["text"]) if i not in skip else None
        if num:
            v, dec, u, more = num
            val = f"{v:,.{dec}f}" if v >= 1000 else f"{v:.{dec}f}"
            gs.append({"i0": i, "i1": i, "type": "hero", "kicker": ["KEY NUMBER", "关键数据"],
                       "hero": {"pre": "", "val": val + ("+" if more else ""), "suf": "", "sub": u},
                       "note": _clip(s["text"], 22)})
    labels = [c["label"] for c in chapters[1:]]
    if len(labels) >= 3 and len(subs) > 6 and not all(_STEP.match(l) for l in labels):
        i0 = max(0, len(subs) - 4)
        gs.append({"i0": i0, "i1": len(subs) - 2, "type": "list", "kicker": ["RECAP", "总结"],
                   "items": [{"n": f"{k + 1:02d}", "zh": l} for k, l in enumerate(labels[:4])]})
    return gs


def plan(subs, chapters, title, D, style, find_number):
    """返回 (graphics, en, emph)；en 为逐句英文字幕（可能为空列表），emph 为需要推近强调的句序号。"""
    cfg = style.get("hud") or {}
    raw, en, emph = None, [], []
    path = os.environ.get("AICLIP_HUD_JSON")
    if path and os.path.exists(path):
        raw = json.load(open(path, encoding="utf-8"))
        log("信息层：使用", path)
    else:
        # 同一份字幕的规划结果存在工程目录：“重新剪辑”时内容保持不变（删掉这个文件即可让大模型重新规划）
        cache = os.environ.get("AICLIP_HUD_CACHE")
        sig = hashlib.sha1("\n".join(s["text"] for s in subs).encode("utf-8")).hexdigest()[:16]
        if cache and os.path.exists(cache):
            try:
                c = json.load(open(cache, encoding="utf-8"))
                if c.get("sig") == sig:
                    raw = c.get("raw")
                    log("信息层：复用上次的规划", cache)
            except Exception:  # noqa
                raw = None
        if raw is None and llm.enabled():
            raw = llm.plan_hud(subs, title)
            if raw and cache:
                try:
                    json.dump({"sig": sig, "raw": raw}, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                except OSError:
                    pass
    if raw:
        gs = [x for x in (_norm(g, len(subs)) for g in raw.get("graphics") or []) if x]
        en = [str(x) for x in raw.get("en") or []]
        emph = [int(x) for x in raw.get("emph") or [] if str(x).isdigit() and int(x) < len(subs)]
        if len(en) != len(subs):
            en = []
    else:
        gs = []
    if not gs:
        gs = [x for x in (_norm(g, len(subs)) for g in _rules(subs, chapters, title, find_number)) if x]
        log("信息层：规则版")
    gs = _cover_steps(gs, subs)
    gs = _times(gs, subs, D, cfg)
    if cfg.get("plus"):
        gs = _plus_layouts(gs)
    cover = sum(g["t1"] - g["t0"] for g in gs) / max(1.0, D)
    log(f"信息层 {len(gs)} 个（{', '.join(g['type'] for g in gs)}），覆盖 {cover:.0%}，英文字幕 {'有' if en else '无'}")
    return gs, en, emph
