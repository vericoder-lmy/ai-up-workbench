"""人脸/主体位置估计：用于画中画裁切与推镜头中心。
优先使用 OpenCV YuNet（把 face_detection_yunet_2023mar.onnx 放到 models/ 即可启用），
否则用肤色分割 + 最大连通域的轻量方法（无需模型、速度快）。"""
import os

import cv2
import numpy as np

from .utils import log, probe, read_frames

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _yunet():
    for name in ("face_detection_yunet_2023mar.onnx", "face_detection_yunet.onnx"):
        p = os.path.join(_HERE, "models", name)
        if os.path.exists(p) and hasattr(cv2, "FaceDetectorYN"):
            return p
    return None


def _skin_face(img):
    h, w = img.shape[:2]
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (40, 135, 85), (255, 175, 130))
    m[int(h * 0.62):] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    best, bs = None, 0
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < w * h * 0.004:
            continue
        ar = hh / max(1, ww)
        score = area * (1.0 if 0.8 < ar < 2.2 else 0.4) * (1.2 - abs((x + ww / 2) / w - 0.5))
        if score > bs:
            bs, best = score, (x + ww / 2, y + min(hh, ww * 1.3) / 2, ww)
    return best


def _motion_head(frames):
    """基于背景差分的主体定位：固定机位/虚拟背景下，用多帧中值作为背景，逐帧求前景，
    取前景最上方的头部轮廓估计脸部中心，再取各帧中位数。"""
    # 按原比例缩小（横屏素材不能压成竖条，否则头宽换算到纵向会严重偏小）
    h0, w0 = frames[0].shape[:2]
    sc = 320.0 / max(h0, w0)
    W_, H_ = max(32, int(w0 * sc)), max(32, int(h0 * sc))
    g = np.stack([cv2.GaussianBlur(cv2.resize(f, (W_, H_)), (0, 0), 1.2).astype(np.float32) for f in frames])
    bg = np.median(g, 0)
    res = []
    for f in g:
        m = (np.abs(f - bg).max(2) > 28).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mx = int(W_ * 0.11)
        rows = m[:, mx: W_ - mx].mean(1)
        top = next((y for y in range(int(H_ * 0.94)) if rows[y] > 0.05), None)
        if top is None:
            continue
        band = m[top + 8: top + 30]
        cols = np.where(band.mean(0) > 0.3)[0]
        if len(cols) < 5:
            continue
        hw = cols[-1] - cols[0] + 1
        if hw > min(W_, H_) * 0.62:
            continue
        res.append(((cols[0] + cols[-1]) / 2, top + 0.68 * hw, hw))
    if len(res) < 3:
        return None
    a = np.median(np.array(res, np.float32), 0)
    return a[0] / W_, a[1] / H_, a[2] / W_


def face_track(video_path, duration, samples=36):
    info = probe(video_path)
    W, H = info["w"], info["h"]
    frames = read_frames(video_path, [duration * (k + 0.5) / samples for k in range(samples)])
    det = None
    yp = _yunet()
    pts = []
    if yp:
        for img in frames:
            k = 640.0 / max(img.shape[:2])
            if k < 1:  # 缩小检测，速度快几倍
                img = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
            try:
                if det is None:
                    det = cv2.FaceDetectorYN.create(yp, "", (img.shape[1], img.shape[0]), 0.7)
                _, faces = det.detect(img)
            except Exception as e:  # 老版本 OpenCV 不支持该模型时退回轻量方法
                log("YuNet 人脸检测不可用，改用轻量方法：", e)
                pts = []
                break
            if faces is not None and len(faces):
                f = max(faces, key=lambda r: r[2] * r[3])
                ih, iw = img.shape[:2]
                pts.append(((f[0] + f[2] / 2) / iw, (f[1] + f[3] / 2) / ih, f[2] / iw))
    res = None
    if pts:
        a = np.array(pts, np.float32)
        res = {"cx": float(np.median(a[:, 0])), "cy": float(np.median(a[:, 1])), "fw": float(np.median(a[:, 2])), "method": "yunet"}
    elif len(frames) >= 8:
        r = _motion_head(frames)
        if r:
            res = {"cx": float(r[0]), "cy": float(r[1]), "fw": float(r[2]), "method": "motion"}
    if res is None:
        res = {"cx": 0.5, "cy": 0.4, "fw": 0.3, "method": "default"}
    log(f"主体位置({res['method']}): cx={res['cx']:.2f} cy={res['cy']:.2f} 脸宽={res['fw']:.2f}")
    return res
