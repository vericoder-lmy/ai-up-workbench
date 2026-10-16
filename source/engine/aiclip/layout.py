"""画幅：竖屏(1080x1920) / 横屏(1920x1080)。
风格模板统一按竖屏编写；横屏时用 to_landscape() 换成一套横屏排版（只改位置/尺寸，配色、字体、开关、节奏不变）。"""
import copy

PORTRAIT = (1080, 1920)
LANDSCAPE = (1920, 1080)
NAMES = {"auto": "跟随原片", "portrait": "竖屏", "landscape": "横屏"}

# 横屏排版：标题栏紧凑放顶部，画中画放左侧、右侧留给 B-roll/打字机，字幕在下方
_LAND = {
    "canvas": {"w": 1920, "h": 1080},
    # 横屏画面矮，标题栏压缩成两行：标题居中、副标题同一行靠右，进度条紧贴其下（总高约 190px）
    "header": {"haze_height": 210, "title_size": 80, "title_y": 12, "title_cx": 960,
               "slogan_size": 42, "slogan_y": 34, "slogan_right": 1880,
               "logo_size": 34, "logo_x": 28, "logo_y": 12},
    "progress": {"y": 124, "thickness": 8, "margin": 150, "node_r": 15, "label_y": 142, "label_size": 28},
    "subtitle": {"y": 962, "size": 64, "max_width": 1560},
    "typewriter": {"x": 1290, "y": 600, "size": 80, "max_w": 940},
    "panel": {"x": 40, "y": 262, "w": 1840, "h": 660, "radius": 60},
    "keyword_pop": {"size": 150, "y": 760},
    "namecard": {"x": 90, "y": 360, "name_size": 112, "line_size": 42, "line_gap": 10},
    "disclaimer": {"size": 34, "right": 1880, "y1": 1004, "y2": 1042},
    "modes": {
        "pip_round": {"x": 90, "y": 270, "w": 600, "h": 640, "radius": 52, "crop_w": 0.66},
        "pip_circle": {"x": 120, "y": 300, "w": 580, "h": 580, "radius": 290, "crop_w": 0.62},
        "vignette": {"x": 40, "y": 236, "w": 700, "h": 700, "radius": 350, "feather": 70, "crop_w": 0.66},
    },
}


def resolve(pref, src_w, src_h):
    """pref: auto|portrait|landscape（也接受 竖屏/横屏/跟随原片）→ portrait|landscape。"""
    p = {"竖屏": "portrait", "横屏": "landscape", "跟随原片": "auto", "": "auto", None: "auto"}.get(pref, pref)
    if p in ("portrait", "landscape"):
        return p
    return "landscape" if src_w and src_h and src_w > src_h * 1.05 else "portrait"


def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        elif isinstance(v, dict):
            if k == "modes":
                continue  # 风格里没有的版式不新增
            dst[k] = copy.deepcopy(v)
        else:
            dst[k] = v


def to_landscape(style):
    st = copy.deepcopy(style)
    if st.get("canvas", {}).get("w", 1080) > st.get("canvas", {}).get("h", 1920):
        return st  # 已是横屏模板
    old_sub_y = st.get("subtitle", {}).get("y", 1320)
    _merge(st, _LAND)
    # 对标视频本身是横屏时，保留测得的字幕高度
    an = st.get("analysis") or {}
    if an.get("aspect", 0) > 1 and 0.6 < an.get("subtitle_y_frac", 0) < 0.93:
        st["subtitle"]["y"] = int(an["subtitle_y_frac"] * 1080)
    elif old_sub_y > 1500:  # 竖屏模板把字幕放得很低 → 横屏也贴底
        st["subtitle"]["y"] = 990
    if st.get("subtitle", {}).get("size", 64) > 64:
        st["subtitle"]["size"] = 64
    if isinstance(st.get("landscape"), dict):  # 风格自带的横屏微调，最后覆盖
        _merge(st, st["landscape"])
    st["orientation"] = "landscape"
    return st


def apply(style, orientation):
    if orientation == "landscape":
        return to_landscape(style)
    st = copy.deepcopy(style)
    st["orientation"] = "portrait"
    return st
