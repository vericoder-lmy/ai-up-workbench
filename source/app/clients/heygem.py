"""HeyGem（Duix.Avatar）本机 Docker 服务的客户端：声音克隆、语音合成、对口型。

HeyGem 的接口读的是容器里的文件路径，所以要先把文件复制到容器挂载的 Windows 目录，
再把对应的容器内路径传给接口。挂载目录通过 `docker inspect` 自动查找，也可以在
config.json 的 heygem.tts_data_dir / video_data_dir 里手动指定。
"""
import glob
import http.client
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid

_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 本机服务不走代理


class HeyGemError(RuntimeError):
    pass


def _http(method, url, body=None, timeout=600, raw=False, log=None):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with _NO_PROXY.open(req, timeout=timeout) as r:
            content, ctype, status = r.read(), r.headers.get("Content-Type", ""), r.status
    except urllib.error.HTTPError as e:
        content, ctype, status = e.read(), e.headers.get("Content-Type", ""), e.code
    except urllib.error.URLError as e:
        raise HeyGemError(f"连不上 {url}：{e.reason}（HeyGem 的 Docker 容器启动了吗？）")
    except (ConnectionError, OSError, http.client.HTTPException) as e:
        raise HeyGemError(f"HeyGem 连接中断：{e}")
    if log:
        preview = content[:600].decode("utf-8", "replace") if "json" in ctype or content[:1] in (b"{", b"[") else f"<{len(content)} 字节 {ctype}>"
        log(f"{method} {url} → HTTP {status}\n    请求：{json.dumps(body, ensure_ascii=False) if body is not None else ''}\n    返回：{preview}")
    if raw:
        return status, ctype, content
    try:
        return json.loads(content.decode("utf-8"))
    except Exception:
        raise HeyGemError(f"{url} 返回的不是 JSON（HTTP {status}）：{content[:300]!r}")


# ---------------- 挂载目录 ----------------

def _host_path(src):
    """Docker Desktop 有时把 D:\\x 写成 /run/desktop/mnt/host/d/x 或 /d/x。"""
    m = re.match(r"^/(?:run/desktop/mnt/host/)?([a-zA-Z])/(.*)$", src.replace("\\", "/"))
    if m and os.name == "nt":
        return f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\")
    return src


DOCKER_NOTE = ""  # 读不到容器时的原因，给测试报告用


def docker_containers():
    """返回 [{name, ports:'...', mounts:[(host, container)]}]；读不到时返回 []，原因写进 DOCKER_NOTE。"""
    global DOCKER_NOTE
    try:
        r = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        DOCKER_NOTE = "这个 Python 环境里找不到 docker 命令（不影响使用，会改用默认数据目录）"
        return []
    except Exception as e:
        DOCKER_NOTE = f"docker 命令出错：{e}"
        return []
    if r.returncode != 0:
        DOCKER_NOTE = "docker ps 失败：" + (r.stderr or "").strip()[:300]
        return []
    ids = r.stdout.split()
    if not ids:
        DOCKER_NOTE = "docker ps 没有列出任何运行中的容器"
    out = []
    for cid in ids:
        try:
            info = json.loads(subprocess.run(["docker", "inspect", cid], capture_output=True, text=True,
                                             encoding="utf-8", timeout=20).stdout)[0]
        except Exception:
            continue
        ports = json.dumps(info.get("NetworkSettings", {}).get("Ports") or {}) + json.dumps(info.get("HostConfig", {}).get("PortBindings") or {})
        mounts = [(_host_path(m.get("Source", "")), m.get("Destination", "")) for m in info.get("Mounts", [])]
        out.append({"name": info.get("Name", "").lstrip("/"), "image": info.get("Config", {}).get("Image", ""),
                    "ports": ports, "mounts": mounts})
    return out


def find_mount(port, override="", containers=None):
    """找到占用该端口的容器的数据目录 → (Windows 路径, 容器内路径)。"""
    if override:
        return override, "/code/data"
    for c in containers if containers is not None else docker_containers():
        if f'"{port}' in c["ports"] or f":{port}" in c["ports"] or f"{port}/tcp" in c["ports"]:
            ms = [m for m in c["mounts"] if m[1].rstrip("/") == "/code/data"] or c["mounts"]
            if ms:
                return ms[0]
    for guess in [r"D:\duix_avatar_data", r"D:\heygem_data"]:
        sub = "voice\\data" if port == 18180 else "face2face"
        p = os.path.join(guess, sub)
        if os.path.isdir(p):
            return p, "/code/data"
    return "", "/code/data"


class HeyGem:
    def __init__(self, cfg, log=print):
        h = cfg.get("heygem", {})
        self.tts_url = h.get("tts_url", "http://127.0.0.1:18180").rstrip("/")
        self.video_url = h.get("video_url", "http://127.0.0.1:8383").rstrip("/")
        self.poll = float(h.get("poll_sec", 3))
        self.container = h.get("container", "heygem-f2f")
        self.timeout_min = float(h.get("lipsync_timeout_min", 60))
        self.log = log
        cs = docker_containers()
        self.containers = cs
        self.tts_host, self.tts_ctr = find_mount(18180, h.get("tts_data_dir", ""), cs)
        self.vid_host, self.vid_ctr = find_mount(8383, h.get("video_data_dir", ""), cs)

    # ---------- 声音 ----------
    def clone_voice(self, wav_path, name):
        """预处理参考音频，返回 {asr_format_audio_url, reference_audio_text}。"""
        if not self.tts_host:
            raise HeyGemError("找不到 HeyGem 语音服务的数据目录，请在 config.json 填 heygem.tts_data_dir")
        rel = f"origin_audio/{name}.wav"
        dst = os.path.join(self.tts_host, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(wav_path, dst)
        last = None
        for ref in [rel, f"{self.tts_ctr.rstrip('/')}/{rel}"]:
            res = _http("POST", f"{self.tts_url}/v1/preprocess_and_tran",
                        {"format": "wav", "reference_audio": ref, "lang": "zh"}, log=self.log)
            last = res
            if isinstance(res, dict) and res.get("asr_format_audio_url"):
                return {"asr_format_audio_url": res["asr_format_audio_url"],
                        "reference_audio_text": res.get("reference_audio_text", ""), "reference_audio_sent": ref}
        raise HeyGemError(f"声音预处理失败：{last}")

    def tts(self, voice, text, out_wav, seed_fixed=True):
        body = {"speaker": str(uuid.uuid4()), "text": text, "format": "wav", "topP": 0.7,
                "max_new_tokens": 1024, "chunk_length": 100, "repetition_penalty": 1.2,
                "temperature": 0.7, "need_asr": False, "streaming": False,
                "is_fixed_seed": 1 if seed_fixed else 0, "is_norm": 0,
                "reference_audio": voice["asr_format_audio_url"],
                "reference_text": voice["reference_audio_text"]}
        status, ctype, content = _http("POST", f"{self.tts_url}/v1/invoke", body, raw=True, log=self.log)
        if status != 200 or content[:4] != b"RIFF":
            raise HeyGemError(f"语音合成失败（HTTP {status}, {ctype}）：{content[:300]!r}")
        os.makedirs(os.path.dirname(out_wav), exist_ok=True)
        with open(out_wav, "wb") as f:
            f.write(content)
        return out_wav

    # ---------- 对口型 ----------
    def lipsync(self, audio_wav, video_mp4, out_mp4, on_progress=None):
        if not self.vid_host:
            raise HeyGemError("找不到 HeyGem 对口型服务的数据目录，请在 config.json 填 heygem.video_data_dir")
        code = str(uuid.uuid4())
        tmp = os.path.join(self.vid_host, "temp")
        os.makedirs(tmp, exist_ok=True)
        a_name, v_name = f"{code}.wav", f"{code}.mp4"
        shutil.copyfile(audio_wav, os.path.join(tmp, a_name))
        shutil.copyfile(video_mp4, os.path.join(tmp, v_name))
        ctr = self.vid_ctr.rstrip("/")
        tried = []
        for a, v in [(f"{ctr}/temp/{a_name}", f"{ctr}/temp/{v_name}"), (f"temp/{a_name}", f"temp/{v_name}")]:
            res = _http("POST", f"{self.video_url}/easy/submit",
                        {"audio_url": a, "video_url": v, "code": code, "chaofen": 0, "watermark_switch": 0, "pn": 1},
                        log=self.log)
            tried.append(res)
            if isinstance(res, dict) and res.get("code") == 10001:
                # 忙碌/队列满：HeyGem 长时间运行后 GPU 队列会卡死，重启容器后再提交一次
                self.log(f"HeyGem 返回 10001（忙碌），正在重启容器 {self.container} …")
                subprocess.run(["docker", "restart", self.container], capture_output=True, text=True, timeout=180)
                self._wait_ready()
                res = _http("POST", f"{self.video_url}/easy/submit",
                            {"audio_url": a, "video_url": v, "code": code, "chaofen": 0, "watermark_switch": 0, "pn": 1},
                            log=self.log)
                tried.append(res)
            if isinstance(res, dict) and (res.get("success") is True or res.get("code") in (0, 10000, 200)):
                return self._wait(code, out_mp4, on_progress, total_frames=self._frames(audio_wav))
        raise HeyGemError(f"对口型任务提交失败：{tried}")

    @staticmethod
    def _frames(wav_path):
        """HeyGem 固定按 25fps 出片：总帧数 ≈ 配音秒数 × 25。"""
        try:
            import wave
            with wave.open(wav_path, "rb") as w:
                return int(w.getnframes() / float(w.getframerate()) * 25)
        except Exception:
            return 0

    def _current_frame(self, code):
        """从容器日志里读当前处理到第几帧（HeyGem 日志会打印 '<code> ... current_idx: N'）。"""
        try:
            r = subprocess.run(["docker", "logs", "--tail", "400", self.container], capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=15)
            hits = re.findall(re.escape(code) + r".*?current_idx: (\d+)", (r.stdout or "") + (r.stderr or ""))
            return int(hits[-1]) if hits else 0
        except Exception:
            return 0

    def restart_container(self):
        """取消任务后 HeyGem 里那个任务还会继续跑，重启容器把它清掉，避免和下一条任务抢显卡。"""
        self.log(f"重启容器 {self.container}，清掉 HeyGem 里未完成的任务")
        try:
            subprocess.run(["docker", "restart", self.container], capture_output=True, text=True, timeout=180)
        except Exception as e:
            self.log(f"重启容器失败：{e}")
        self._wait_ready()

    def _wait_ready(self, sec=120):
        t0 = time.time()
        while time.time() - t0 < sec:
            try:
                _http("GET", self.video_url + "/", timeout=5, raw=True)
                return
            except HeyGemError:
                time.sleep(3)

    def _wait(self, code, out_mp4, on_progress, total_frames=0):
        """HeyGem 的 progress 只在几个阶段跳（5→20→80→100），中间最久的“视频处理”阶段
        用容器日志里的帧号算真实进度；帧号和画面文件都 10 分钟不动才判定卡死。"""
        t0, last_pct = time.time(), -1
        tfile = os.path.join(self.vid_host, "temp", f"{code}-t.mp4")
        last_size, last_frame, last_move = -1, 0, time.time()
        frame_t0, frame_n0 = None, 0
        fails = 0
        while time.time() - t0 < self.timeout_min * 60:
            try:
                res = _http("GET", f"{self.video_url}/easy/query?code={code}", timeout=30, log=None)
                fails = 0
            except HeyGemError as e:
                # 查询偶尔会被断开（显卡满负荷时 HeyGem 响应慢）：连续失败 2 分钟才算真的出错
                fails += 1
                self.log(f"查询进度失败（第 {fails} 次）：{e}")
                if fails * self.poll > 120:
                    raise HeyGemError(f"HeyGem 2 分钟没有响应，容器可能崩溃了：{e}。"
                                      f"请运行 docker ps 看 {self.container} 是否在运行，必要时 docker restart {self.container} 后重试")
                time.sleep(self.poll)
                continue
            d = res.get("data") if isinstance(res, dict) and isinstance(res.get("data"), dict) else res
            status = d.get("status") if isinstance(d, dict) else None
            pct = d.get("progress") if isinstance(d, dict) else None
            msg = (d.get("msg") if isinstance(d, dict) else "") or ""
            size = os.path.getsize(tfile) if os.path.exists(tfile) else 0
            frame = self._current_frame(code) if status == 1 else 0
            if size != last_size or frame > last_frame:
                last_move = time.time()
            if frame and frame_t0 is None:
                frame_t0, frame_n0 = time.time(), frame
            last_size, last_frame = size, max(last_frame, frame)
            if pct != last_pct:
                self.log(f"对口型进度：status={status} progress={pct} msg={msg} 原始={json.dumps(res, ensure_ascii=False)[:300]}")
                last_pct = pct
            info = {"msg": msg, "frame": frame, "total": total_frames, "spf": None, "eta": None}
            if frame and frame_t0 and frame > frame_n0:
                spf = (time.time() - frame_t0) / (frame - frame_n0)
                info["spf"] = spf
                if total_frames:
                    info["eta"] = max(0, total_frames - frame) * spf + 20
            if on_progress:
                try:
                    on_progress(pct, time.time() - t0, size, info)
                except TypeError:
                    on_progress(pct, time.time() - t0, size)
            if status == 1 and time.time() - last_move > 600 and (size > 0 or frame > 0):
                raise HeyGemError(f"对口型卡住了：10 分钟没有新画面（帧 {frame}/{total_frames}）。"
                                  f"请在命令行运行 docker restart {self.container} 后重试")
            if status == 2 or (isinstance(d, dict) and d.get("result") and status not in (1, 3)):
                return self._collect(code, d.get("result", ""), out_mp4)
            if status == 3:
                raise HeyGemError(f"对口型失败：{json.dumps(res, ensure_ascii=False)}")
            time.sleep(self.poll)
        raise HeyGemError(f"对口型超过 {self.timeout_min} 分钟还没完成")

    def _collect(self, code, result, out_mp4):
        cands = []
        if result:
            r = result.replace("\\", "/")
            ctr = self.vid_ctr.rstrip("/")
            if r.startswith(ctr):
                r = r[len(ctr):]
            r = r.lstrip("/")
            cands += [os.path.join(self.vid_host, *r.split("/")), os.path.join(self.vid_host, "temp", os.path.basename(r))]
        cands += sorted(glob.glob(os.path.join(self.vid_host, "**", f"*{code}*-r.mp4"), recursive=True), key=os.path.getmtime, reverse=True)
        cands += sorted(glob.glob(os.path.join(self.vid_host, "**", f"*{code}*.mp4"), recursive=True), key=os.path.getmtime, reverse=True)
        for p in cands:
            if os.path.isfile(p) and not p.endswith(f"{code}.mp4"):
                os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
                shutil.copyfile(p, out_mp4)
                return out_mp4
        raise HeyGemError(f"对口型说完成了，但找不到结果文件。result={result}，查找位置：{self.vid_host}")
