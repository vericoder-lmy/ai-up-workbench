"""SRT 读写。"""
import re

_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _t(g):
    h, m, s, ms = g
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


def parse_srt(path):
    for enc in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            text = open(path, encoding=enc).read()
            break
        except UnicodeError:
            continue
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    subs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        ti = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ti is None:
            continue
        m = _TS.findall(lines[ti])
        if len(m) < 2:
            continue
        txt = "".join(lines[ti + 1:]).strip()
        if not txt:
            continue
        subs.append({"start": _t(m[0]), "end": _t(m[1]), "text": txt})
    subs.sort(key=lambda s: s["start"])
    return subs


def fmt_ts(t):
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(subs, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(subs, 1):
            f.write(f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s['text']}\n\n")
