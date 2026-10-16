"""文本理解（无需联网的规则版）：关键词、章节、标签、断行、人名提取。
配置 LLM 后（见 llm.py）会用大模型结果覆盖这里的启发式结果。"""
import re
from collections import Counter

STOP = set("的了啊呢吧呀吗么嘛哦哈是我你他她它们这那就也都会不很有在和与及或一个种大家说去给要能让被把还再才又对从就到为着过得地么些其该每各啥咋怎么么样什")
FILLERS = ["那么", "那个", "然后", "就是说", "啊", "呢", "吧", "呀", "嘛", "哦", "哈"]
CN_NUM = "一二三四五六七八九十"
MARK_RE = re.compile(r"^(?:那|好|啊|嗯)?\s*(第[一二三四五六七八九十\d]+[种个点条类步招]|首先|其次|再次|最后|另外)")
SUMMARY_RE = re.compile(r"^(?:所以|总之|总结一下|总的来说|最后提醒|所以我提醒)")
OUTRO_RE = re.compile(r"(点赞|关注|直播间|评论区|转发|收藏)")


_LAT = re.compile(r"[A-Za-z0-9]")


def _w(g):
    """显示宽度：英文字母/数字按半个字算（Claude 约等于 3 个汉字宽）。"""
    lat = len(_LAT.findall(g))
    return (len(g) - lat) + lat * 0.5


def _cuts_word(t, i, j):
    """t[i:j] 是否把英文单词从中间切开（AI_UP 修正：避免 Claude → de、Codex → ode 这类碎片）。"""
    if i > 0 and _LAT.match(t[i]) and _LAT.match(t[i - 1]):
        return True
    if j < len(t) and _LAT.match(t[j - 1]) and _LAT.match(t[j]):
        return True
    return False


def _cut(t, n):
    """取前 n 个字宽，不切断英文单词。"""
    out = ""
    for m in re.finditer(r"[A-Za-z0-9]+|.", t):
        if _w(out + m.group()) > n:
            break
        out += m.group()
    return out or t[:n]


def clean(t):
    t = re.sub(r"[，。！？、,.!?；;：:“”\"'（）()\s]", "", t)
    return t


def strip_fillers(t):
    t = clean(t)
    changed = True
    while changed and t:
        changed = False
        for f in FILLERS:
            if t.startswith(f) and len(t) > len(f):
                t = t[len(f):]
                changed = True
            if t.endswith(f) and len(t) > len(f):
                t = t[: -len(f)]
                changed = True
    return t


def extract_terms(subs, top=30, user_terms=None):
    """统计 2~5 字高频片段作为候选关键词（不依赖分词库）。"""
    cnt = Counter()
    for s in subs:
        t = clean(s["text"])
        seen = set()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9]{1,15}", t):  # 英文单词整体算一个词
            if w not in seen:
                seen.add(w)
                cnt[w] += 1
        for n in range(2, 6):
            for i in range(len(t) - n + 1):
                g = t[i: i + n]
                if _cuts_word(t, i, i + n) or (_LAT.match(g[0]) and _LAT.match(g[-1]) and not re.fullmatch(r"[A-Za-z0-9]+", g)):
                    continue
                if g[0] in STOP or g[-1] in STOP:
                    continue
                if sum(ch in STOP for ch in g) > n // 2:
                    continue
                if g in seen:
                    continue
                seen.add(g)
                cnt[g] += 1
    cand = {g: c for g, c in cnt.items() if c >= 2}
    # 去掉被更长且同频片段包含的子串
    keys = sorted(cand, key=len, reverse=True)
    for g in list(cand):
        for h in keys:
            if len(h) > len(g) and g in h and cand.get(h, 0) >= cand[g] * 0.8:
                cand.pop(g, None)
                break
    scored = sorted(cand.items(), key=lambda kv: kv[1] * (len(kv[0]) ** 1.2), reverse=True)
    terms = [g for g, _ in scored[:top]]
    if user_terms:
        terms = list(dict.fromkeys(list(user_terms) + terms))
    return terms


def _label_from(text, terms, maxlen=5):
    t = strip_fillers(text)
    t = MARK_RE.sub("", t)
    t = strip_fillers(t)
    for sep in ("就是", "是"):
        if sep in t and len(t.split(sep)[-1]) >= 2:
            t = t.split(sep)[-1]
            break
    t = re.sub(r"(性的|的)$", "", t)
    t = strip_fillers(t)
    if _w(t) <= maxlen:
        return t
    hits = [(t.find(g), g) for g in terms if g in t and _w(g) <= maxlen]
    hits.sort()
    lab = ""
    for _, g in hits:
        if g in lab:
            continue
        if _w(lab) + _w(g) <= maxlen:
            lab += g
    return lab or _cut(t, 4)


def _dict_terms():
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dict", "医学词库.txt")
    out = []
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#"):
                    out += line.split()
    return out


def detect_chapters(subs, terms, max_chapters=6):
    """返回 [{'idx': 起始字幕序号, 'label': 标签}]，第 0 章为开场。"""
    marks = []
    for i, s in enumerate(subs):
        t = clean(s["text"])
        if MARK_RE.match(t):
            marks.append(i)
    chapters = [{"idx": 0, "label": ""}]
    dict_terms = _dict_terms()
    for i in marks:
        rest = strip_fillers(MARK_RE.sub("", strip_fillers(subs[i]["text"])))
        # 标记句剩余部分 + 后续句，凑够约 16 字作为“本章开头”
        ctx = rest
        j = i + 1
        while len(ctx) < 16 and j < len(subs) and j not in marks:
            ctx += clean(subs[j]["text"])
            j += 1
        ctx = re.sub(r"^(最常见的|常见的)?(原因|情况)?(就是|是)", "", ctx)
        hits = []
        for g in sorted(set(dict_terms) | set(terms), key=len, reverse=True):
            if 2 <= len(g) and _w(g) <= 5 and g in ctx and not any(g in h for _, h in hits):
                hits.append((ctx.find(g), g))
        hits.sort()
        lab = ""
        for _, g in hits:
            if _w(lab) + _w(g) <= 5:
                lab += g
            if _w(lab) >= 4:
                break
        if len(lab) < 2:
            src = subs[i]["text"] if len(rest) >= 2 or i + 1 >= len(subs) else subs[i + 1]["text"]
            lab = _label_from(src, terms)
            if len(lab) <= 2 and i + 1 < len(subs):
                nxt = clean(subs[i + 1]["text"])
                extra = next((g for g in terms if g in nxt and _w(g) <= 5 - _w(lab) and g not in lab), "")
                lab += extra
        if len(lab) < 2:  # 视频被截短或句子太碎时取不到像样的词，用序号兜底
            lab = f"要点{CN_NUM[min(len(chapters), len(CN_NUM) - 1)]}"
        chapters.append({"idx": i, "label": lab})
    # 总结段
    if len(chapters) < max_chapters:
        for i, s in enumerate(subs):
            if i > (marks[-1] if marks else 0) and SUMMARY_RE.match(clean(s["text"])):
                chapters.append({"idx": i, "label": "总结提醒"})
                break
    # 没有“第一/第二/最后”等结构词：按时长均分为 3~5 段，用各段高频词做标签
    if len(chapters) == 1 and len(subs) >= 12:
        total = subs[-1]["end"] - subs[0]["start"]
        k = int(max(3, min(5, round(total / 50))))
        used = set()
        chapters = []
        for c in range(k):
            t0 = subs[0]["start"] + total * c / k
            idx = next(i for i, s in enumerate(subs) if s["start"] >= t0 - 1e-6) if c else 0
            if chapters and idx <= chapters[-1]["idx"]:
                continue
            nxt_t = subs[0]["start"] + total * (c + 1) / k
            seg = "".join(clean(s["text"]) for s in subs if t0 <= s["start"] < nxt_t)
            cnt = sorted(((seg.count(g) * len(g), g) for g in terms
                          if 2 <= len(g) and _w(g) <= 5 and g not in used and g not in GENERIC and seg.count(g) >= 2
                          and not any(ch in STOP for ch in g)), reverse=True)
            lab = cnt[0][1] if cnt else f"要点{CN_NUM[c]}"
            used.add(lab)
            chapters.append({"idx": idx, "label": lab})
        return chapters
    # 开场标签：取开场段最高频术语
    intro_end = chapters[1]["idx"] if len(chapters) > 1 else len(subs)
    intro_txt = "".join(clean(s["text"]) for s in subs[:intro_end])
    best = next((g for g in terms if g in intro_txt and 2 <= len(g) and _w(g) <= 5), "")
    chapters[0]["label"] = ("认识" + best if _w(best) <= 3 else best) if best else "开篇"
    if len(chapters) > max_chapters:
        chapters = chapters[:max_chapters]
    # AI_UP 修正：标签重复时改用这一章的结构词（如“第三步”），避免列表里出现两个一样的标签
    seen = set()
    for k, c in enumerate(chapters):
        if c["label"] in seen:
            m = MARK_RE.match(clean(subs[c["idx"]]["text"])) if c["idx"] < len(subs) else None
            c["label"] = m.group(1) if m else f"要点{CN_NUM[min(k, len(CN_NUM) - 1)]}"
        seen.add(c["label"])
    return chapters


GENERIC = set("自己 导致 原因 基本上 多人 很多 大家 时候 问题 情况 数值 我们 他们 今天 一下 现在 帮助 晚上 白天 时间 东西 方法 一些 这样 那样 因为 所以 但是 如果 可以 已经 还是 就是".split())


def pop_terms(terms, chapter_labels=()):
    """挑选适合做"大字弹出"的关键词：领域名词、3~5 字、无虚词。"""
    out = []
    for g in terms:
        if g in GENERIC or any(ch in STOP for ch in g) or g[0] in "性查多很":
            continue
        if len(g) < 3 and g not in chapter_labels:
            continue
        if any(g in o or o in g for o in out):
            continue
        out.append(g)
    for lab in chapter_labels:
        if lab and lab not in out and 2 <= len(lab) <= 5:
            out.append(lab)
    return out


def find_name(subs):
    for s in subs:
        m = re.search(r"我是([一-龥]{2,4}?)(医生|大夫|主任|老师|教授|律师)", s["text"])
        if m:
            return m.group(1), m.group(2)
    return None, None


def outro_start(subs):
    for i, s in enumerate(subs):
        if i > len(subs) * 0.8 and (OUTRO_RE.search(s["text"]) or re.search(r"我是.{2,4}(医生|老师)", s["text"])):
            return i
    return None


def wrap(text, max_chars):
    """按最大字数断行（优先在标点处断）。"""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    n_lines = (len(text) + max_chars - 1) // max_chars
    per = (len(text) + n_lines - 1) // n_lines
    lines, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= per:
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    return lines
