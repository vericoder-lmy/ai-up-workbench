"""读音纠正（AI_UP）：配音前把容易读错的词换成“读法”，字幕仍显示原文。
词典文件：00_素材库/读音纠正.txt，每行“原文=读法”，# 开头是注释。长的词优先替换。
读法的写法：
- 有同音同调的字就换字：行长=航长（háng）、重新=虫新（chóng）；
- 没有同音字就换成不会读错的说法：藏房产=隐藏房产（cáng）。
文案里的原文不变，只有送去配音的文字会被替换；配音缓存按替换后的文字算，改了词典会自动重新合成相关句子。"""
import os
import re

from .pipeline import ROOT

FILE = os.path.join(ROOT, "00_素材库", "读音纠正.txt")
DEFAULT = """# 读音纠正词典：每行“原文=读法”，配音时把左边换成右边，字幕仍显示原文。
# 有同音同调的字就换字；没有同音字就换成不会读错的说法。长的词优先替换。
# 改完点“保存”，再重新试听或生成，相关句子会自动重新配音。
藏房产=隐藏房产
藏财产=隐藏财产
藏钱=把钱藏起来
"""
_cache = {"mtime": None, "pairs": []}


def text():
    if not os.path.exists(FILE):
        return DEFAULT
    return open(FILE, encoding="utf-8").read()


def save(content):
    os.makedirs(os.path.dirname(FILE), exist_ok=True)
    with open(FILE, "w", encoding="utf-8") as f:
        f.write(content.replace("\r\n", "\n"))
    _cache["mtime"] = None
    return len(pairs())


def pairs():
    if not os.path.exists(FILE):
        save(DEFAULT)
    mt = os.path.getmtime(FILE)
    if _cache["mtime"] != mt:
        out = []
        for ln in open(FILE, encoding="utf-8").read().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            m = re.match(r"^(.+?)\s*(?:=|→|->)\s*(.+)$", ln)
            if not m:
                continue
            a, b = m.group(1).strip(), m.group(2).strip()
            if a and b and a != b:
                out.append((a, b))
        out.sort(key=lambda p: -len(p[0]))
        _cache.update(mtime=mt, pairs=out)
    return _cache["pairs"]


def apply(s):
    """返回送去配音的文字（替换后）。"""
    s = str(s or "")
    if not s:
        return s
    ps = pairs()
    if not ps:
        return s
    # 一次扫描、长词优先，避免替换结果被再次替换
    pat = re.compile("|".join(re.escape(a) for a, _ in ps))
    table = dict(ps)
    return pat.sub(lambda m: table[m.group(0)], s)


def add(new_pairs):
    """追加词条（原文已存在就更新读法）。返回词条总数。"""
    cur = text().rstrip("\n").split("\n")
    have = {}
    for k, ln in enumerate(cur):
        m = re.match(r"^\s*([^#].*?)\s*(?:=|→|->)\s*(.+)$", ln)
        if m:
            have[m.group(1).strip()] = k
    for a, b in new_pairs:
        a, b = a.strip(), b.strip()
        if not a or not b or a == b:
            continue
        if a in have:
            cur[have[a]] = f"{a}={b}"
        else:
            cur.append(f"{a}={b}")
            have[a] = len(cur) - 1
    return save("\n".join(cur) + "\n")
