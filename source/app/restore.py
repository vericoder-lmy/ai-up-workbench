"""对口型画质还原（AI_UP）：HeyGem 会把整张脸重新生成一遍，眼睛、眼镜、头发都会变糊。
这里只保留它生成的“嘴巴到下巴”区域，其余部分用原始素材的对应帧替换。

v2：输出用原片的分辨率（例如 1080p），HeyGem 只需要处理 720p 的小视频（快、省显存），
嘴部区域从 720p 放大后贴回 1080p 原片上。

做法：
1. 原片按 25fps 解码（与送进 HeyGem 的视频逐帧对应）；
2. 用“去掉嘴部区域后的小图”把 HeyGem 输出的每一帧对回原片的第几帧（Viterbi 平滑，兼容正放/倒放循环）；
3. 每帧用 YuNet 找嘴角和鼻尖，生成羽化椭圆遮罩，只把遮罩内换成 HeyGem 的嘴；遮罩边缘做颜色匹配；
4. 用高质量参数重新编码，音频直接拷贝。
原片帧顺序单调（混排素材不循环）时边读边写，不占内存；有倒放循环时才把原片帧缓存成高质量 JPEG。
"""
import os
import re
import subprocess

import cv2
import numpy as np

SMALL = (64, 36)
DET_W = 640  # 人脸检测在缩小图上做，快


def _ffmpeg():
    try:
        from .media import ffmpeg_bin  # AI_UP 里运行
        return ffmpeg_bin()
    except Exception:  # noqa
        return "ffmpeg"


def _size(path, ff):
    """视频显示尺寸（考虑旋转）。"""
    r = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True, errors="ignore")
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", r.stderr)
    if not m:
        raise RuntimeError(f"读不出视频尺寸：{path}")
    w, h = int(m.group(1)), int(m.group(2))
    if re.search(r"rotate\s*:\s*-?(90|270)|rotation of -?(90|270)", r.stderr):
        w, h = h, w
    return w, h


def _decode(path, ff, fps=None, size=None, start=None, end=None):
    """逐帧读取（BGR uint8）。fps 给定时先用 ffmpeg 转帧率，保证和送进 HeyGem 的视频取帧方式一致。"""
    w, h = size or _size(path, ff)
    vf = []
    if fps:
        vf.append(f"fps={fps}")
    vf.append(f"scale={w}:{h}:flags=lanczos")
    cut = (["-ss", str(start)] if start is not None else []) + (["-to", str(end)] if end is not None else [])
    cmd = [ff, "-v", "error"] + cut + ["-i", path, "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def _small(f):
    return cv2.cvtColor(cv2.resize(f, SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)


def _viterbi(cost, max_step=3, jump_pen=0.4):
    """cost: (n_out, n_src)。每步只允许 |Δj|<=max_step，Δj 越大罚越多；返回每个输出帧对应的源帧。"""
    n, m = cost.shape
    steps = np.arange(-max_step, max_step + 1)
    pen = jump_pen * np.maximum(0, np.abs(steps) - 1)
    acc = cost[0].copy()
    back = np.zeros((n, m), np.int16)
    idx = np.arange(m)
    for i in range(1, n):
        best = np.full(m, np.inf)
        arg = np.zeros(m, np.int16)
        for s, p in zip(steps, pen):
            src = idx - s
            ok = (src >= 0) & (src < m)
            v = np.full(m, np.inf)
            v[ok] = acc[src[ok]] + p
            better = v < best
            best[better] = v[better]
            arg[better] = s
        acc = best + cost[i]
        back[i] = arg
    path = np.zeros(n, np.int64)
    path[-1] = int(np.argmin(acc))
    for i in range(n - 1, 0, -1):
        path[i - 1] = path[i] - back[i, path[i]]
    return path


def _detector(model):
    if not model or not os.path.exists(model) or not hasattr(cv2, "FaceDetectorYN"):
        return None
    return cv2.FaceDetectorYN.create(model, "", (320, 320), 0.6, 0.3, 5)


def _mouth_ellipse(det, frame):
    """返回原图坐标下的 (cx, cy, ax, ay) 或 None：覆盖鼻翼下方到下巴、两侧到嘴角外侧。"""
    if det is None:
        return None
    h, w = frame.shape[:2]
    k = DET_W / w if w > DET_W else 1.0
    img = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame
    det.setInputSize((img.shape[1], img.shape[0]))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    f = max(faces, key=lambda r: r[2] * r[3]) / k
    x, y, fw, fh = f[:4]
    nose = np.array(f[8:10])
    mr, ml = np.array(f[10:12]), np.array(f[12:14])
    mc = (mr + ml) / 2
    mw = max(np.linalg.norm(ml - mr), fw * 0.3)
    top = nose[1] + (mc[1] - nose[1]) * 0.25          # 鼻翼下方
    chin = min(y + fh * 1.04, mc[1] + mw * 1.15)       # 下巴
    cy = (top + chin) / 2
    return float(mc[0]), float(cy), float(mw * 1.05), float((chin - top) / 2)


def restore(lip_path, orig_path, out_path, model=None, log=print, crf=14, start=None, end=None, on_progress=None):
    ff = _ffmpeg()
    cap = cv2.VideoCapture(lip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n_lip = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()
    LW, LH = _size(lip_path, ff)
    OW, OH = _size(orig_path, ff)
    if abs(OW / OH - LW / LH) > 0.02:
        raise RuntimeError(f"原片 {OW}x{OH} 和对口型 {LW}x{LH} 画面比例不同")
    OW, OH = OW // 2 * 2, OH // 2 * 2
    rfps = round(fps)
    orig = lambda: _decode(orig_path, ff, fps=rfps, size=(OW, OH), start=start, end=end)  # noqa: E731
    det = _detector(model)

    # 1. 小图：原片 + 对口型
    S, mid = [], None
    for f in orig():
        S.append(_small(f))
        if mid is None and len(S) == 12:
            mid = f.copy()
    if len(S) < 10:
        raise RuntimeError("原片解码失败")
    S = np.stack(S)
    mask_small = np.ones(SMALL[::-1], np.float32)
    el = _mouth_ellipse(det, mid) if mid is not None else None
    if el:
        cx, cy, ax, ay = el
        sx, sy = SMALL[0] / OW, SMALL[1] / OH
        cv2.ellipse(mask_small, (int(cx * sx), int(cy * sy)), (int(ax * sx * 1.6) + 2, int(ay * sy * 1.6) + 2),
                    0, 0, 360, 0, -1)
    else:
        mask_small[int(SMALL[1] * .4):int(SMALL[1] * .7), int(SMALL[0] * .38):int(SMALL[0] * .7)] = 0
    Sm = (S * mask_small).reshape(len(S), -1)
    costs = []
    for f in _decode(lip_path, ff, size=(LW, LH)):
        if on_progress and len(costs) % 50 == 0:
            on_progress(30 * len(costs) / n_lip)
        costs.append(np.abs(Sm - (_small(f) * mask_small).reshape(-1)).mean(1))
    cost = np.stack(costs).astype(np.float32)
    path = _viterbi(cost)
    fit = float(cost[np.arange(len(path)), path].mean())
    mono = bool(np.all(np.diff(path) >= 0))
    log(f"画质还原：对口型 {len(path)} 帧 {LW}x{LH} ↔ 原片 {len(S)} 帧 {OW}x{OH}，匹配误差 {fit:.2f}，"
        f"{'顺序播放' if mono else '有循环/倒放'}")
    if fit > 6.0:
        raise RuntimeError(f"对口型和原片对不上（误差 {fit:.1f}），跳过画质还原")

    # 2. 原片帧来源：单调时边读边走；否则缓存成高质量 JPEG
    if mono:
        it, cur, cur_f = orig(), -1, None

        def get(j):
            nonlocal cur, cur_f
            while cur < j:
                cur_f = next(it)
                cur += 1
            return cur_f
    else:
        jpg = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 97])[1] for f in orig()]

        def get(j):
            return cv2.imdecode(jpg[j], cv2.IMREAD_COLOR)

    # 3. 合成
    enc = subprocess.Popen([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
                            "-r", str(fps), "-i", "-", "-i", lip_path, "-map", "0:v", "-map", "1:a?",
                            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                            "-c:a", "copy", "-shortest", out_path], stdin=subprocess.PIPE)
    kx, ky = LW / OW, LH / OH
    ell_cache, last_el = {}, el
    try:
        for i, lf in enumerate(_decode(lip_path, ff, size=(LW, LH))):
            if on_progress and i % 25 == 0:
                on_progress(30 + 70 * i / max(1, len(path)))
            j = int(path[min(i, len(path) - 1)])
            base = get(j)
            e = ell_cache.get(j)
            if e is None:
                e = ell_cache[j] = _mouth_ellipse(det, base) or last_el
            last_el = e or last_el
            out = base.copy()
            if e is None:
                out = cv2.resize(lf, (OW, OH), interpolation=cv2.INTER_CUBIC)
                enc.stdin.write(out.tobytes())
                continue
            cx, cy, ax, ay = e
            x0, x1 = int(max(0, cx - ax * 1.6)), int(min(OW, cx + ax * 1.6))
            y0, y1 = int(max(0, cy - ay * 1.6)), int(min(OH, cy + ay * 1.6))
            # 对口型里对应区域（按比例换算坐标），放大到原片尺寸
            lx0, lx1 = int(x0 * kx), max(int(x0 * kx) + 2, int(round(x1 * kx)))
            ly0, ly1 = int(y0 * ky), max(int(y0 * ky) + 2, int(round(y1 * ky)))
            l = cv2.resize(lf[ly0:ly1, lx0:lx1], (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            d = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
            m = np.clip((1.25 - d) / 0.5, 0, 1)  # 椭圆内=1，向外羽化
            m = (m * m * (3 - 2 * m))[..., None]
            b = base[y0:y1, x0:x1].astype(np.float32)
            ring = (d > 1.0) & (d < 1.6)
            if ring.sum() > 50:  # 嘴部区域的整体颜色向原片靠拢，避免“补丁感”
                l += (b[ring].mean(0) - l[ring].mean(0))
            out[y0:y1, x0:x1] = np.clip(b * (1 - m) + l * m, 0, 255).astype(np.uint8)
            enc.stdin.write(out.tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    if on_progress:
        on_progress(100)
    if enc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError("画质还原编码失败")
    return out_path


if __name__ == "__main__":
    import sys
    restore(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
