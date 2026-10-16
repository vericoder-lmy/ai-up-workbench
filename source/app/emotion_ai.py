"""文案自动识别情绪（AI_UP）：把逐句文案交给大模型（config.local.json 里的中转站），
返回每句的情绪（默认/8 种预设/自定义一句语气）和建议语速；大模型不可用时用规则兜底。"""
import json
import re
import ssl
import urllib.error
import urllib.request

from .pipeline import load_cfg

PRESETS = ["平静", "开心", "激动", "温柔", "严肃", "悲伤", "惊讶", "生气"]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI_UP/0.1"  # 中转站前面有 Cloudflare，默认 UA 会被拦（1010）


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa
        return ssl.create_default_context()


def _chat(llm, prompt, timeout=120):
    key = llm["api_key"]
    base = llm["base_url"].rstrip("/")
    if llm.get("api_format") == "anthropic":
        root = base[:-3] if base.endswith("/v1") else base
        url = root + "/v1/messages"
        body = {"model": llm.get("model", ""), "max_tokens": 4096, "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}]}
        headers = {"x-api-key": key, "Authorization": "Bearer " + key, "anthropic-version": "2023-06-01"}
    else:
        url = base + "/chat/completions"
        body = {"model": llm.get("model", ""), "temperature": 0.3, "messages": [{"role": "user", "content": prompt}]}
        headers = {"Authorization": "Bearer " + key}
    headers.update({"Content-Type": "application/json", "User-Agent": UA})
    handlers = [urllib.request.HTTPSHandler(context=_ctx())]
    if llm.get("proxy", "direct") == "direct":
        handlers.insert(0, urllib.request.ProxyHandler({}))
    elif llm.get("proxy", "").startswith("http"):
        handlers.insert(0, urllib.request.ProxyHandler({"http": llm["proxy"], "https": llm["proxy"]}))
    op = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with op.open(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"大模型接口 HTTP {e.code}")
    if llm.get("api_format") == "anthropic":
        txt = "".join(c.get("text", "") for c in data.get("content", []) if c.get("type") == "text")
    else:
        txt = data["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise RuntimeError("大模型没有返回 JSON")
    return json.loads(m.group(0))


def _rules(sentences):
    """兜底：按标点和常见口播套路粗略判断。"""
    out = []
    for i, s in enumerate(sentences):
        emo, rate = "default", 1.0
        if re.search(r"(关注|点赞|评论区|收藏|下期见|我们一起)", s):
            emo = "开心"
        elif s.endswith(("！", "!")):
            emo = "激动"
        elif s.endswith(("？", "?")) and i < 3:
            emo = "惊讶"
        elif re.search(r"(注意|关键|一定要|千万|重点)", s):
            emo = "严肃"
        if i == 0 and emo == "default":
            rate = 1.1  # 开场稍快一点更抓人
        out.append({"emo": emo, "custom": "", "rate": rate, "why": ""})
    return out


def detect(sentences, title=""):
    """返回 (items, source)。items 与 sentences 一一对应：{"emo", "custom", "rate", "why"}。"""
    sentences = [str(s) for s in sentences][:200]
    llm = load_cfg().get("llm", {})
    if not (llm.get("base_url") and llm.get("api_key")):
        return _rules(sentences), "规则（没有配置大模型）"
    lines = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
    prompt = (
        "你是抖音知识类口播的配音导演。下面是一条口播的逐句文案（序号: 文本）" + (f"，标题：{title}" if title else "") + "。\n"
        "请为每一句决定配音的情绪和语速，同时检查错别字和容易被配音读错的多音字，输出 JSON：\n"
        '{"items":[{"i":序号,"emo":"default|平静|开心|激动|温柔|严肃|悲伤|惊讶|生气|custom",'
        '"custom":"emo=custom 时写一句≤20字的语气要求，如“带点无奈地自嘲”，否则留空",'
        '"rate":语速倍数0.8~1.3,"why":"≤8字理由",'
        '"fix":"这句有错别字或明显用词错误时，写出改正后的整句（只改错处，其余一字不动）；没有就留空",'
        '"pron":[{"w":"句中容易被语音合成读错的多音字所在的词（原文片段，2~6字）","say":"不会读错的读法：有同音同调的字就换字，没有就换成同义的常见说法","why":"≤12字，如 藏读cáng"}]}]}\n'
        "要求：\n"
        "1. 像真人博主一样有起伏，但不要句句都加戏：大约 1/3~1/2 的句子用 default（自然、亲切的口播语气）；\n"
        "2. 开场钩子、反转、金句用 激动/惊讶 或 custom；讲痛点可用 严肃 或 custom（无奈/吐槽）；讲步骤、方法用 default 或 平静；结尾引导关注用 开心 或 温柔；\n"
        "3. 只有确实需要细腻语气时才用 custom；预设能表达的就用预设；\n"
        "4. 语速大多 1.0~1.1，开场钩子可 1.1~1.2，强调的金句可 0.9；不要出现 0.8 以下或 1.3 以上；\n"
        "5. fix 只纠正真正的错别字（的地得、同音别字、错误用词），不要润色、不要改口语表达；拿不准就留空；\n"
        "6. pron 只标真正有歧义、语音合成很可能读错的多音字（如 藏房产 的藏应读 cáng、银行行长 的行长、重新 的重），"
        "常见词里读音明确的不要标；say 要保持原意，读出来自然；没有就给空数组；\n"
        "7. items 必须覆盖每一句，i 与序号一致；只输出 JSON。\n\n" + lines)
    try:
        res = _chat(llm, prompt)
    except Exception as e:  # noqa
        return _rules(sentences), f"规则（大模型调用失败：{str(e)[:60]}）"
    by_i = {}
    for it in res.get("items") or []:
        try:
            by_i[int(it.get("i"))] = it
        except (TypeError, ValueError):
            continue
    out = []
    fallback = _rules(sentences)
    try:
        from . import pron as _pron
        known = {a for a, _ in _pron.pairs()}  # 读音词典里已有的不再提示
    except Exception:  # noqa
        known = set()
    for i in range(len(sentences)):
        it = by_i.get(i)
        if not it:
            out.append(fallback[i])
            continue
        emo = str(it.get("emo") or "default")
        custom = str(it.get("custom") or "").strip()[:30]
        if emo not in PRESETS + ["default", "custom"]:
            emo, custom = ("custom", emo[:30]) if len(emo) > 1 else ("default", "")
        if emo == "custom" and not custom:
            emo = "default"
        try:
            rate = round(min(1.3, max(0.8, float(it.get("rate") or 1.0))), 1)
        except (TypeError, ValueError):
            rate = 1.0
        s_ = sentences[i]
        fix = str(it.get("fix") or "").strip()
        if not fix or fix == s_ or abs(len(fix) - len(s_)) > max(4, len(s_) // 4):
            fix = ""  # 改动太大的不当错别字处理（可能是在润色）
        pron = []
        for p in it.get("pron") or []:
            try:
                w, say = str(p.get("w") or "").strip(), str(p.get("say") or "").strip()
            except AttributeError:
                continue
            if w and say and w != say and w in s_ and len(w) <= 8 and len(say) <= 12 and w not in known:
                pron.append({"w": w, "say": say, "why": str(p.get("why") or "")[:16]})
        out.append({"emo": emo, "custom": custom if emo == "custom" else "", "rate": rate,
                    "why": str(it.get("why") or "")[:12], "fix": fix, "pron": pron[:3]})
    return out, "大模型"
