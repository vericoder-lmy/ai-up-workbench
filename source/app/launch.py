"""启动入口：检查依赖（缺了自动安装）、检查端口（已经开着就直接打开网页），再启动工作台。"""
import importlib
import os
import socket
import subprocess
import sys
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEED = {"numpy": "numpy", "cv2": "opencv-python", "PIL": "pillow", "requests": "requests", "imageio_ffmpeg": "imageio-ffmpeg"}


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print(f"Python {sys.version.split()[0]}：{sys.executable}")
    missing = []
    for mod, pkg in NEED.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        print("第一次使用，正在安装：" + "、".join(missing) + " …")
        r = subprocess.run([sys.executable, "-m", "pip", "install", *missing])
        if r.returncode != 0:
            print("安装失败。可以手动运行：python -m pip install " + " ".join(missing))
            return
    from app import server
    port = server.PORT
    if _busy(port) and not _replace_old(port, server.BUILD):
        webbrowser.open(f"http://127.0.0.1:{port}")
        return
    server.main()


def _busy(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _api(port, path, post=False):
    import json
    import urllib.request
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 本机地址不走系统代理
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b"{}" if post else None,
                                 headers={"Content-Type": "application/json"})
    with op.open(req, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def _kill_listener(port):
    """旧版本没有关机接口：找到占用端口的进程直接结束（只结束这个端口的监听进程）。"""
    if os.name != "nt":
        return
    out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True, errors="ignore").stdout
    pids = set()
    for line in out.splitlines():
        p = line.split()
        # 本地地址是 :8790、远端地址是 0.0.0.0:0 的那一行就是监听进程（不依赖中英文的状态文字）
        if len(p) >= 5 and p[1].endswith(f":{port}") and p[2] in ("0.0.0.0:0", "[::]:0", "*:*"):
            pids.add(p[-1])
    for pid in pids:
        if pid.isdigit() and int(pid) != os.getpid():
            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)


def _replace_old(port, build):
    """端口已被占用：同一版本就直接打开网页；旧版本且没有任务在跑，就关掉旧的、启动新的。返回 True 表示继续启动。"""
    import time
    try:
        info = _api(port, "/api/meta")
    except Exception:
        info = {}
    if info.get("build") == build:
        print(f"工作台已经在运行（最新版本），直接打开 http://127.0.0.1:{port}")
        return False
    try:
        running = any(j.get("state") == "处理中" for j in _api(port, "/api/jobs").get("jobs", []))
    except Exception:
        running = False
    if running:
        print("开着的是旧版本工作台，但有任务正在处理，先不重启。等任务完成后关掉旧窗口，再双击启动工作台。")
        return False
    print("检测到开着的是旧版本工作台，正在自动关闭并启动新版本……")
    try:
        _api(port, "/api/shutdown", post=True)
    except Exception:
        pass
    for _ in range(10):
        time.sleep(0.5)
        if not _busy(port):
            return True
    _kill_listener(port)
    for _ in range(10):
        time.sleep(0.5)
        if not _busy(port):
            return True
    print(f"旧工作台没能自动关闭：请关掉之前那个黑色命令行窗口，再双击启动工作台。")
    return False


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
        print("\n工作台启动失败，请把上面的文字截图发给 Claude。")
