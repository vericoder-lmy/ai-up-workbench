"""可选的大模型接口（OpenAI 兼容协议：DeepSeek / 通义千问 DashScope / Kimi / 智谱 / OpenAI / 本地 LM Studio 等）。

环境变量：
  AICLIP_LLM_BASE_URL  例如 https://api.deepseek.com/v1 、https://dashscope.aliyuncs.com/compatible-mode/v1
  AICLIP_LLM_API_KEY
  AICLIP_LLM_MODEL     文本模型，例如 deepseek-chat / qwen-plus
  AICLIP_VLM_MODEL     视觉模型（分析参考截图），例如 qwen-vl-max / gpt-4o
未配置时系统自动使用规则版（textproc.py / analyze.py），不影响出片。
"""
import base64
import json
import os
import re

from .utils import log


def enabled(vision=False):
    if not (os.environ.get("AICLIP_LLM_BASE_URL") and os.environ.get("AICLIP_LLM_API_KEY")):
        return False
    return bool(os.environ.get("AICLIP_VLM_MODEL" if vision else "AICLIP_LLM_MODEL"))


# AI_UP：中转站前面有 Cloudflare，requests 默认的 “python-requests” 标识会被拦截（error code 1010）
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI_UP/0.1"


def _post(messages, model, timeout=300):
    import requests
    base = os.environ["AICLIP_LLM_BASE_URL"].rstrip("/")
    key = os.environ["AICLIP_LLM_API_KEY"]
    if os.environ.get("AICLIP_LLM_FORMAT") == "anthropic":
        # AI_UP：中转站只提供 Claude 原生格式（cc-switch / Claude Code 用的 /v1/messages）
        root = base[:-3] if base.endswith("/v1") else base
        r = requests.post(root + "/v1/messages", timeout=timeout,
                          headers={"x-api-key": key, "Authorization": "Bearer " + key,
                                   "anthropic-version": "2023-06-01", "User-Agent": UA},
                          json={"model": model, "max_tokens": 8192, "temperature": 0.2, "messages": messages})
        r.raise_for_status()
        txt = "".join(c.get("text", "") for c in r.json().get("content", []) if c.get("type") == "text")
    else:
        r = requests.post(base + "/chat/completions", timeout=timeout,
                          headers={"Authorization": "Bearer " + key, "User-Agent": UA},
                          json={"model": model, "messages": messages, "temperature": 0.2})
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {}


def plan_semantics(subs):
    """让 LLM 给出章节划分、短标签、弹字关键词。失败返回 None。"""
    if not enabled():
        return None
    lines = "\n".join(f"{i}: {s['text']}" for i, s in enumerate(subs))
    prompt = (
        "你是短视频口播剪辑师。下面是逐句字幕（序号: 文本）。请输出 JSON：\n"
        '{"chapters":[{"idx":起始序号,"label":"2~5字章节名"}],'
        '"pop_terms":["适合做大字弹出强调的3~5字关键词",...],'
        '"slogan":"8~10字的账号标语"}\n'
        "要求：chapters 第一项 idx=0 为开场，总数 4~6 个，按内容自然分段（如“第一种/第二种/最后”）；"
        "pop_terms 6~12 个，必须原文出现。只输出 JSON。\n\n" + lines)
    try:
        res = _post([{"role": "user", "content": prompt}], os.environ["AICLIP_LLM_MODEL"])
        log("LLM 语义规划完成")
        return res
    except Exception as e:  # noqa
        log("LLM 调用失败，使用规则版:", e)
        return None


def plan_hud(subs, title=""):
    """AI_UP：让 LLM 规划画面信息层（眉标/大字/数字/列表/卡片/数据）并逐句翻译英文字幕。失败返回 None。"""
    if not enabled():
        return None
    lines = "\n".join(f"{i}: {s['text']}" for i, s in enumerate(subs))
    prompt = (
        "你是抖音知识类口播视频的动效包装设计师。人物始终全屏，信息以文字图形叠在画面一侧。"
        "下面是逐句字幕（序号: 文本）" + (f"，视频标题：{title}" if title else "") + "。请输出 JSON：\n"
        '{"en":["与每句字幕一一对应的简短英文翻译", ...],\n'
        ' "emph":[最值得镜头推近强调的句序号，2~5 个，彼此间隔至少 6 秒，挑金句/关键结论/大数字],\n'
        ' "graphics":[{"i0":起始句序号,"i1":结束句序号,"type":"hero|headline|list|cards|stats",\n'
        '   "kicker":["2~4个英文单词的大写小标，如 STEP 01 · TOPIC RESEARCH","4~10字中文小标"],\n'
        '   "hero":{"pre":"可选前缀如$或+","val":"大数字或2~4字大词","suf":"可选英文后缀如 DEALS","sub":"可选中文单位如 个商单"},\n'
        '   "lines":["headline 用：每行≤6字，最多2行，第2行是重点"],\n'
        '   "en":"可选：一句大写英文副标题","note":"可选：≤20字中文补充说明",\n'
        '   "items":[list/cards 用 {"n":"01","en":"英文小标","zh":"≤12字中文"}；stats 用 {"label":"GROWTH · 涨粉","val":"98%","note":"补充"}]}]}\n'
        "规则：\n"
        "1. 第一个 graphics 从 i0=0 开始，用 headline 或 hero 点题；\n"
        "2. 大约每 6~10 秒一个信息块，总时长覆盖 50%~70%，两个块之间至少隔一句；每块持续 1~3 句；\n"
        "3. 说到具体数字用 hero（val 必须是原文数字，不许编造）；讲到并列的几点用 list 或 cards；"
        "有多个百分比/数据时用 stats；结尾总结用 list 列出前面讲过的要点；其余用 headline 提炼观点；\n"
        "4. list/cards 如果是边讲边列的（如第一步、第二步、第三步），每个 item 加 \"i\":这一条开始讲的句序号，"
        "画面会随讲解逐条出现，i1 写到最后一条讲完；凡是讲到“第X步/首先/最后”的句子都必须被某个信息块覆盖；\n"
        "5. 中文都要短，来自原文或忠实概括；字幕断句可能把一个词拆在两句里，理解时把相邻句连起来读；\n"
        "6. 只输出 JSON，不要解释。\n\n" + lines)
    for k in range(2):  # 中转站偶尔会中途断开连接：再试一次
        try:
            res = _post([{"role": "user", "content": prompt}], os.environ["AICLIP_LLM_MODEL"])
            log("LLM 信息层规划完成")
            return res
        except Exception as e:  # noqa
            log(f"LLM 信息层规划失败（第 {k + 1} 次）:", e)
    log("LLM 信息层规划两次都失败，使用规则版")
    return None


def plan_law(subs, title=""):
    """AI_UP 律师动效库：两行金色标题、开场红色大字、每句红色重点词。失败返回 None。"""
    if not enabled():
        return None
    lines = "\n".join(f"{i}: {s['text']}" for i, s in enumerate(subs))
    prompt = (
        "你是律师普法类抖音口播的剪辑包装师。下面是逐句字幕（序号: 文本）" + (f"，视频标题：{title}" if title else "") +
        "。请输出 JSON：\n"
        '{"title":["顶部标题第一行，≤7字，抓眼球","第二行，≤9字，点明观点"],\n'
        ' "hook":"开场砸进画面的红色大字，2~4字，必须是标题或前两句里原样出现的词，如 深刻感悟",\n'
        ' "kw":{"句序号":"这句里最该标红的词"}}\n'
        "规则：\n"
        "1. title 忠实概括视频主题，口语、有冲击力，不要书名号；如果给了视频标题，按它拆成两行；\n"
        "2. kw 的词必须是那一句里连续出现的原文片段，2~8 个字；优先标观点/结论/风险/法律后果/罪名/数字，"
        "像“第一不要多交朋友”这种列点句标“不要多交朋友”；大约 60%~80% 的句子有重点词，没有合适的就不写这句；\n"
        "3. 不许编造法条、罪名、数字，只能从原文里选；\n"
        "4. 只输出 JSON，不要解释。\n\n" + lines)
    for k in range(2):
        try:
            res = _post([{"role": "user", "content": prompt}], os.environ["AICLIP_LLM_MODEL"])
            log("LLM 律师包装规划完成")
            return res
        except Exception as e:  # noqa
            log(f"LLM 律师包装规划失败（第 {k + 1} 次）:", e)
    log("LLM 律师包装规划两次都失败，使用规则版")
    return None


def cover_text(title, subs, n=6):
    """AI_UP 封面大字：两行、每行不超过 n 个字，抓眼球、忠实于主题。失败返回 None。"""
    if not enabled():
        return None
    body = "\n".join(s["text"] for s in subs[:40])
    prompt = (f"给抖音口播视频做封面大字。视频标题：{title}\n文案：\n{body}\n\n"
              f"请输出 JSON：{{\"lines\":[\"第一行\",\"第二行（重点，会用橙色）\"]}}。每行不超过 {n} 个汉字（英文字母算半个），"
              "一共不超过 11 个字；口语、有冲击力、点明主题，不要编造文案里没有的事实，不要书名号和句号。只输出 JSON。")
    for k in range(2):
        try:
            res = _post([{"role": "user", "content": prompt}], os.environ["AICLIP_LLM_MODEL"], timeout=120)
            lines = [str(x).strip() for x in (res or {}).get("lines") or [] if str(x).strip()]
            if lines:
                return lines[:2]
        except Exception as e:  # noqa
            log(f"LLM 封面大字失败（第 {k + 1} 次）:", e)
    return None


def analyze_images(paths):
    """视觉模型读取参考截图，返回布局/配色/特效描述（用于覆盖风格参数）。"""
    if not enabled(vision=True):
        return None
    content = [{"type": "text", "text": (
        "这些是一个竖屏口播短视频的截图。请分析其剪辑包装风格并输出 JSON："
        '{"title_text":"顶部大标题文字","slogan_text":"副标题","chapter_labels":["进度条上的章节名"],'
        '"title_color":"#RRGGBB","accent_color":"#RRGGBB","subtitle_style":"outline|pill|box",'
        '"pip_shapes":["circle|round_rect|soft_circle"],"effects":["typewriter","keyword_pop","dip_to_black",...],'
        '"disclaimer_text":"底部免责声明"}。只输出 JSON。')}]
    for p in paths[:6]:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}})
    try:
        return _post([{"role": "user", "content": content}], os.environ["AICLIP_VLM_MODEL"])
    except Exception as e:  # noqa
        log("VLM 调用失败:", e)
        return None


def fix_subtitles(lines, terms=()):
    """大模型字幕纠错：只改语音识别造成的错别字/同音字，不改写句子、不合并拆分。失败返回 None。"""
    if not enabled() or not lines:
        return None
    body = "\n".join(f"{i}\t{t}" for i, t in enumerate(lines))
    prompt = ("下面是语音识别生成的中文口播字幕，每行“序号\\t文本”。请只修正识别错误（同音错字、专业术语、人名），"
              "不要润色、不要增删内容、不要合并或拆分行、不要加标点。可能出现的专业词：" + "、".join(list(terms)[:120]) +
              "\n请输出 JSON：{\"lines\":[\"第0行修正后\",\"第1行修正后\",...]}，行数必须与输入完全一致。\n\n" + body)
    try:
        res = _post([{"role": "user", "content": prompt}], os.environ["AICLIP_LLM_MODEL"], timeout=300)
        out = res.get("lines")
        if isinstance(out, list) and len(out) == len(lines):
            return [str(x) for x in out]
    except Exception as e:  # noqa
        log("LLM 字幕纠错失败，保留原识别结果：", e)
    return None
