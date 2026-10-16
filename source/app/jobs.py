"""任务队列：SQLite 记状态，一个后台线程按顺序处理（GPU 一次只跑一个）。
服务关掉再打开，没做完的任务会从头排队，已完成的阶段靠工程目录里的缓存直接跳过。"""
import datetime
import json
import os
import sqlite3
import threading
import time
import traceback
import uuid

from .pipeline import ROOT, Cancelled, Job, safe_name

DB = os.path.join(ROOT, "06_日报", "tasks.db")
FIELDS = ["id", "title", "script", "avatar", "voice", "style", "outputs", "state", "stage", "stage_pct",
          "detail", "port_pct", "land_pct", "files", "error", "project_dir", "created", "updated", "started"]


class Queue:
    def __init__(self):
        os.makedirs(os.path.dirname(DB), exist_ok=True)
        self.lock = threading.Lock()
        self.cancel_ids = set()
        with self._db() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, title TEXT, script TEXT, avatar TEXT,
                voice TEXT, style TEXT, outputs TEXT, state TEXT, stage INTEGER, stage_pct REAL, detail TEXT,
                port_pct REAL, land_pct REAL, files TEXT, error TEXT, project_dir TEXT, created REAL, updated REAL,
                started REAL)""")
            cols = {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
            if "use_materials" not in cols:
                c.execute("ALTER TABLE jobs ADD COLUMN use_materials INTEGER DEFAULT 1")
            if "emotions" not in cols:
                c.execute("ALTER TABLE jobs ADD COLUMN emotions TEXT DEFAULT '{}'")
            if "lipsync_res" not in cols:  # 对口型清晰度：720 / 1080
                c.execute("ALTER TABLE jobs ADD COLUMN lipsync_res INTEGER DEFAULT 720")
            if "title_auto" not in cols:  # 没填标题：开跑时让 AI 根据文案提炼
                c.execute("ALTER TABLE jobs ADD COLUMN title_auto INTEGER DEFAULT 0")
            if "brand" not in cols:  # 律师动效库：姓名/职称/律所
                c.execute("ALTER TABLE jobs ADD COLUMN brand TEXT DEFAULT '{}'")
            c.execute("UPDATE jobs SET state='排队中', detail='服务重启后继续' WHERE state='处理中'")
        self.wake = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def _db(self):
        c = sqlite3.connect(DB, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def _logfile(self, job):
        return os.path.join(job["project_dir"], "任务日志.txt")

    # ---------- 对外 ----------
    def create(self, d):
        now = time.time()
        jid = uuid.uuid4().hex[:10]
        from .text import split_sentences
        first = (split_sentences(d["script"]) or [d["script"][:20]])[0]
        title = safe_name(d.get("title") or first[:24])  # 没填标题：用第一句（不再按 16 个字硬截）
        pdir = os.path.join(ROOT, "04_工程", f"{datetime.date.today():%Y-%m-%d}_{title}_{jid[:4]}")
        row = {"id": jid, "title": title, "script": d["script"], "avatar": d["avatar"], "voice": d["voice"],
               "style": d.get("style") or "aiup_orange", "title_auto": 0 if (d.get("title") or "").strip() else 1,
               "lipsync_res": 1080 if str(d.get("lipsync_res")) == "1080" else 720,
               "use_materials": 0 if d.get("use_materials") is False else 1,
               "emotions": json.dumps(d.get("emotions") or {}, ensure_ascii=False),
               "brand": json.dumps({k: str(v)[:40] for k, v in (d.get("brand") or {}).items() if k in ("name", "role", "firm")},
                                   ensure_ascii=False), "outputs": json.dumps(d.get("outputs") or ["portrait", "landscape"]),
               "state": "排队中", "stage": 0, "stage_pct": 0, "detail": "等待开始", "port_pct": 0, "land_pct": 0,
               "files": "{}", "error": "", "project_dir": pdir, "created": now, "updated": now, "started": 0}
        with self._db() as c:
            c.execute(f"INSERT INTO jobs ({','.join(row)}) VALUES ({','.join('?' * len(row))})", list(row.values()))
        self.wake.set()
        return jid

    def list(self):
        with self._db() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 50")]
        for r in rows:
            r["outputs"] = json.loads(r["outputs"] or "[]")
            r["files"] = json.loads(r["files"] or "{}")
            for f in r["files"].values():  # 给网页用的相对路径
                for k in ("video", "srt", "cover", "cover_port", "cover_land"):
                    if f.get(k):
                        f[k + "_rel"] = os.path.relpath(f[k], ROOT).replace("\\", "/")
            r.pop("script", None)
        return rows

    def get(self, jid):
        with self._db() as c:
            r = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return dict(r) if r else None

    def update(self, jid, **f):
        if "files" in f:
            f["files"] = json.dumps(f["files"], ensure_ascii=False)
        f["updated"] = time.time()
        with self.lock, self._db() as c:
            c.execute(f"UPDATE jobs SET {','.join(k + '=?' for k in f)} WHERE id=?", list(f.values()) + [jid])

    def cancel(self, jid):
        self.cancel_ids.add(jid)
        j = self.get(jid)
        if j and j["state"] == "排队中":
            self.update(jid, state="已取消", detail="已取消")

    def retry(self, jid):
        self.cancel_ids.discard(jid)
        self.update(jid, state="排队中", error="", detail="重新排队（已完成的阶段会直接复用）")
        self.wake.set()

    # ---------- 后台 ----------
    def _next(self):
        with self._db() as c:
            r = c.execute("SELECT * FROM jobs WHERE state='排队中' ORDER BY created LIMIT 1").fetchone()
        return dict(r) if r else None

    def _worker(self):
        while True:
            j = self._next()
            if not j:
                self.wake.wait(3)
                self.wake.clear()
                continue
            jid = j["id"]
            j["outputs"] = json.loads(j["outputs"] or "[]")
            os.makedirs(j["project_dir"], exist_ok=True)
            lf = open(self._logfile(j), "a", encoding="utf-8")

            def log(msg):
                lf.write(f"[{datetime.datetime.now():%H:%M:%S}] {msg}\n")
                lf.flush()

            self.update(jid, state="处理中", started=time.time(), error="")
            try:
                # 每个任务开始前重新载入流程代码：更新了 app/ 里的代码不用重启工作台
                import importlib
                from . import avatar_mix, media, pipeline, restore
                from .clients import heygem, qwen_voice
                for m in (media, restore, avatar_mix, heygem, qwen_voice, pipeline):
                    importlib.reload(m)
                pipeline.Job(j, lambda **f: self.update(jid, **f), lambda: jid in self.cancel_ids, log).run()
                self.update(jid, state="完成", stage=8, stage_pct=100)
                log("完成")
            except Exception as e:  # 重新载入后 Cancelled 是新类，按名字判断
                if type(e).__name__ == "Cancelled":
                    self.update(jid, state="已取消", detail="已取消")
                    log("已取消")
                else:
                    log(traceback.format_exc())
                    self.update(jid, state="失败", error=str(e)[:1500])
            finally:
                lf.close()
                self.cancel_ids.discard(jid)
