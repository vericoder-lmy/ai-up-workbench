"""测试中转站大模型：先直连，不通再走系统代理；把能用的方式记进 config.local.json，并试一次分章。
不会打印 API Key。双击「测试大模型.bat」运行。"""
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("SSLKEYLOGFILE", None)

LOCAL = os.path.join(ROOT, "config.local.json")


def openers():
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    yield "direct", urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=ctx))
    sp = {k: "http://" + v.split("://", 1)[-1] for k, v in urllib.request.getproxies().items() if k in ("http", "https") and v}
    if sp:
        yield sp.get("https") or sp.get("http"), urllib.request.build_opener(urllib.request.ProxyHandler(sp), urllib.request.HTTPSHandler(context=ctx))


# 中转站前面有 Cloudflare：默认的 “Python-urllib” 标识会被当成机器人拦截（error code 1010）
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI_UP/0.1"


def _root(base):
    base = base.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def chat(op, llm, prompt, timeout=120, fmt="openai"):
    """fmt=openai：/v1/chat/completions；fmt=anthropic：/v1/messages（cc-switch / Claude Code 用的格式）。"""
    key = llm["api_key"]
    if fmt == "anthropic":
        url = _root(llm["base_url"]) + "/v1/messages"
        body = {"model": llm["model"], "max_tokens": 2048, "messages": [{"role": "user", "content": prompt}]}
        headers = {"x-api-key": key, "Authorization": "Bearer " + key, "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json", "User-Agent": UA}
    else:
        url = _root(llm["base_url"]) + "/v1/chat/completions"
        body = {"model": llm["model"], "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": UA}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
    with op.open(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    if fmt == "anthropic":
        return "".join(c.get("text", "") for c in d.get("content", []) if c.get("type") == "text")
    return d["choices"][0]["message"]["content"]


def main():
    cfg = json.load(open(LOCAL, encoding="utf-8"))
    llm = cfg.get("llm", {})
    print(f"中转站：{llm.get('base_url')}　模型：{llm.get('model')}")
    ok_mode, ok_fmt = None, None
    for mode, op in openers():
        name = "直连" if mode == "direct" else f"系统代理 {mode}"
        for fmt in ("anthropic", "openai"):
            fname = "Claude 格式 /v1/messages" if fmt == "anthropic" else "OpenAI 格式 /v1/chat/completions"
            t0 = time.time()
            print(f"  … 正在试 {name} + {fname}（最长等 3 分钟，请别关窗口）", flush=True)
            try:
                txt = chat(op, llm, "只回复两个字：收到", timeout=180, fmt=fmt)
                print(f"  ✓ {name} + {fname} 可用（{time.time() - t0:.1f} 秒），模型回复：{txt.strip()[:20]}")
                ok_mode, ok_fmt = mode, fmt
                break
            except urllib.error.HTTPError as e:
                print(f"  ✗ {name} + {fname}：HTTP {e.code} {e.read().decode('utf-8', 'replace')[:200]}")
            except Exception as e:
                print(f"  ✗ {name} + {fname}：{getattr(e, 'reason', e)}")
                break  # 连接层面的错误，换另一种连接方式
        if ok_mode:
            break
    if not ok_mode:
        print("\n大模型暂时用不了，剪辑会自动用规则版起章节名，不影响出片。")
        return
    cfg["llm"]["proxy"] = ok_mode
    cfg["llm"]["api_format"] = ok_fmt
    json.dump(cfg, open(LOCAL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("  已记住：" + ("直连" if ok_mode == "direct" else ok_mode) + " · " + ok_fmt + " 格式")

    print("\n试一次分章（用测试文案）…")
    subs = ["我是一个程序员", "最近在做一件事", "让 AI 帮我剪视频", "第一步，我把需求讲给 Claude",
            "让它帮我拆成模块", "第二步，让 Codex 写代码跑测试", "出了错它自己改", "第三步，我只做一件事",
            "看结果，提意见", "关注我，我们一起搞"]
    prompt = ("你是短视频口播剪辑师。下面是逐句字幕（序号: 文本）。请输出 JSON："
              '{"chapters":[{"idx":起始序号,"label":"2~5字章节名"}],"pop_terms":["关键词"]}。只输出 JSON。\n'
              + "\n".join(f"{i}: {s}" for i, s in enumerate(subs)))
    op = dict(openers())[ok_mode]
    t0 = time.time()
    print(chat(op, llm, prompt, timeout=300, fmt=ok_fmt)[:600])
    print(f"（用时 {time.time() - t0:.1f} 秒）")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("测试出错：", e)
    input("\n按回车键关闭…")
