"""标题提炼（AI_UP）：根据整条文案内容，让大模型提炼一个抖音标题（不是照抄第一句）。
标题用空格分成两段：第一段抓眼球（痛点/悬念/身份），第二段点明核心观点或收获。
顶部大字、开场大字、律师金色标题、封面、成片文件名都用它。大模型不可用时退回第一句。"""
import hashlib
import json
import os
import re

from .pipeline import load_cfg
from .text import split_sentences


def _vis(s):
    return sum(0.55 if re.match(r"[A-Za-z0-9]", c) else (0.4 if c == " " else 1.0) for c in str(s))


def _clean(t):
    t = re.sub(r"[《》“”\"'「」【】。…]+", "", str(t or "")).strip()
    t = re.sub(r"\s+", " ", t)
    return t


def suggest(script, style=""):
    """返回 (标题, 来源说明)。"""
    from .emotion_ai import _chat
    first = (split_sentences(script) or [script[:20]])[0][:24]
    llm = load_cfg().get("llm", {})
    if not (llm.get("base_url") and llm.get("api_key")):
        return first, "第一句（没有配置大模型）"
    law = "law" in (style or "")
    prompt = (
        ("你是律师普法类抖音账号的运营。" if law else "你是 AI 科技类抖音博主的运营。") +
        "请通读下面整条口播文案，提炼这条视频的标题，输出 JSON：{\"title\":\"第一段 第二段\",\"why\":\"≤15字\"}\n"
        "要求：\n"
        "1. 概括整条视频真正讲的主题和结论，不要照抄或截取第一句；\n"
        "2. 用一个空格分成两段：第一段 4~8 字，抓眼球（痛点、反常识、悬念或身份，如“刑案办多了”“AI 配音翻车了”）；"
        "第二段 5~10 字，点明核心观点或观众能得到什么（如“深刻感悟少交朋友”“一句话就能修好”）；\n"
        "3. 口语、有冲击力，可以用问号或感叹号，不用书名号、引号、句号；\n"
        "4. 只能用文案里有的事实和数字，不许编造" + ("法条、罪名、案例结果" if law else "数据和功能") + "；\n"
        "5. 只输出 JSON。\n\n文案：\n" + script[:3000])
    err = "返回格式不对"
    for k in range(2):
        try:
            res = _chat(llm, prompt, timeout=90)
            t = _clean(res.get("title"))
            if t and _vis(t) <= 22:
                return t, "AI 提炼"
        except Exception as e:  # noqa
            err = str(e)[:60]
    return first, f"第一句（大模型调用失败：{err}）"


def for_job(project_dir, script, style, log=print):
    """任务用：同一份文案只提炼一次（缓存在工程目录 标题提炼.json，删掉可重新提炼）。"""
    sig = hashlib.sha1(script.encode("utf-8")).hexdigest()[:16]
    cache = os.path.join(project_dir, "标题提炼.json")
    try:
        c = json.load(open(cache, encoding="utf-8"))
        if c.get("sig") == sig and c.get("title"):
            return c["title"]
    except Exception:  # noqa
        pass
    t, src = suggest(script, style)
    log(f"标题：{t}（{src}）")
    if src == "AI 提炼":
        try:
            json.dump({"sig": sig, "title": t}, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        except OSError:
            pass
        return t
    return None
