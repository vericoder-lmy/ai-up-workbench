"""文案切分：整段文案 → 句子（逐句合成配音）→ 字幕行（每行不超过 14 字）。"""
import re

SENT_END = "。！？!?；;…\n"


def split_sentences(script):
    script = re.sub(r"[ \t　]+", " ", script.replace("\r", "")).strip()
    out, buf = [], ""
    for ch in script:
        if ch == "\n":
            if buf.strip():
                out.append(buf.strip())
            buf = ""
            continue
        buf += ch
        if ch in SENT_END:
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    # 太长的句子按逗号再拆，避免单句超过约 60 字（合成更稳，也方便单句重做）
    res = []
    for s in out:
        while len(s) > 60:
            cut = max(s.rfind(c, 0, 60) for c in "，,、：:")
            cut = cut + 1 if cut > 10 else 60
            res.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            res.append(s)
    return [s for s in res if re.search(r"\w", s)]


def _visible_len(s):
    """字幕宽度：中文算 1，英文字母和数字约半个字宽算 0.55，标点和空格不算。"""
    s = re.sub(r"[，。！？!?；;、：:,\.\s“”\"'…]", "", s)
    latin = len(re.findall(r"[A-Za-z0-9+\-%]", s))
    return (len(s) - latin) + latin * 0.55


def _tokens(s):
    """英文单词/数字整体算一个块，不从中间断开；中文按字。"""
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9.+\-%]*\s?|\s|.", s)


def _hard_split(p, max_chars):
    """超长片段按字数均分成几行（避免最后剩一个字）。"""
    toks = _tokens(p)
    total = _visible_len(p)
    n = -(-total // max_chars)
    target = total / n
    lines, cur = [], ""
    for t in toks:
        if cur and _visible_len(cur + t) > target + 0.5 and len(lines) < n - 1:
            lines.append(cur)
            cur = ""
        cur += t
    lines.append(cur)
    return [l.strip() for l in lines if l.strip()]


def subtitle_lines(sentence, max_chars=14):
    """一句话 → 字幕行：先按标点断，再把超长的均分；行尾标点去掉（字幕习惯）。"""
    parts = [p for p in re.split(r"(?<=[，,、：:；;。！？!?])", sentence) if p.strip()]
    lines, cur = [], ""
    for p in parts:
        if _visible_len(cur + p) <= max_chars:
            cur += p
            continue
        if cur:
            lines.append(cur)
        cur = ""
        if _visible_len(p) > max_chars:
            lines += _hard_split(p, max_chars)
        else:
            cur = p
    if cur:
        lines.append(cur)
    clean = [re.sub(r"[，。,；;、：:]+$", "", l).strip() for l in lines]
    return [l for l in clean if l]
