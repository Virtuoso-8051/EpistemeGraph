"""
Background-job protocol shared by the web server and the CLI scripts.

Every long task (KG build, KG precision audit, batch of queries, benchmark generation, benchmark
run, retraining) runs as a separate low-priority process and writes a heartbeat to
  <workspace>/jobs/<job_id>/status.json   +   log.txt
so the website can show live progress, survive a page reload or a server restart, and re-open
the last results from cache.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kg_audit.utils.main_utils import read_json, write_json_atomic

TERMINAL = ("completed", "failed", "cancelled")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStatus:
    """Writer used inside a job process."""

    def __init__(self, job_dir: Optional[str]):
        self.job_dir = job_dir
        self.path = os.path.join(job_dir, "status.json") if job_dir else None
        self.state: Dict[str, Any] = (read_json(self.path, default={}) or {}) if self.path else {}

    def update(self, stage: str, done: int = 0, total: int = 0, note: str = "", **extra) -> None:
        if not self.path:
            return
        self.state.update({"state": "running", "stage": stage, "done": int(done), "total": int(total),
                           "note": note, "updated_at": _now(), "pid": os.getpid(), **extra})
        write_json_atomic(self.path, self.state)

    def finish(self, state: str = "completed", **extra) -> None:
        if not self.path:
            return
        self.state.update({"state": state, "updated_at": _now(), "finished_at": _now(), **extra})
        write_json_atomic(self.path, self.state)


def jobs_dir(ws_dir: str) -> str:
    return os.path.join(ws_dir, "jobs")


def new_job(ws_dir: str, job_type: str, params: Dict[str, Any]) -> str:
    job_id = f"{job_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    d = os.path.join(jobs_dir(ws_dir), job_id)
    os.makedirs(d, exist_ok=True)
    write_json_atomic(os.path.join(d, "status.json"), {
        "id": job_id, "type": job_type, "state": "queued", "stage": "queued", "done": 0, "total": 0,
        "params": params, "created_at": _now(), "updated_at": _now(), "workspace_dir": ws_dir,
    })
    return job_id


def launch(ws_dir: str, job_id: str, argv: List[str], root: str) -> int:
    """Start `python <argv>` detached at below-normal priority, logging to the job folder."""
    d = os.path.join(jobs_dir(ws_dir), job_id)
    log = open(os.path.join(d, "log.txt"), "w", encoding="utf-8")
    flags = 0
    if os.name == "nt":
        flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           "EPISTEME_JOB_DIR": d, "EPISTEME_WORKSPACE": os.path.basename(ws_dir.rstrip("/\\"))}
    proc = subprocess.Popen([sys.executable] + argv, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=flags, env=env)
    st = read_json(os.path.join(d, "status.json"), default={}) or {}
    st.update({"state": "running", "pid": proc.pid, "started_at": _now(), "updated_at": _now()})
    write_json_atomic(os.path.join(d, "status.json"), st)
    return proc.pid


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        import psutil
        p = psutil.Process(int(pid))
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def get_job(ws_dir: str, job_id: str) -> Optional[Dict[str, Any]]:
    d = os.path.join(jobs_dir(ws_dir), job_id)
    st = read_json(os.path.join(d, "status.json"), default=None)
    if not st:
        return None
    # a job whose process died without writing a terminal state is reported as failed
    if st.get("state") == "running" and st.get("pid") and not pid_alive(st["pid"]):
        st["state"] = "failed"
        st.setdefault("error", "process exited unexpectedly - see log")
    log = os.path.join(d, "log.txt")
    if os.path.exists(log):
        with open(log, "r", encoding="utf-8", errors="ignore") as f:
            st["log_tail"] = "".join(f.readlines()[-30:])
    return st


def list_jobs(ws_dir: str, job_type: Optional[str] = None) -> List[Dict[str, Any]]:
    d = jobs_dir(ws_dir)
    if not os.path.isdir(d):
        return []
    out = []
    for j in sorted(os.listdir(d), reverse=True):
        if job_type and not j.startswith(job_type + "_"):
            continue
        st = get_job(ws_dir, j)
        if st:
            st.pop("log_tail", None)
            out.append(st)
    return out


def cancel(ws_dir: str, job_id: str) -> bool:
    st = get_job(ws_dir, job_id)
    if not st or st.get("state") in TERMINAL:
        return False
    try:
        import psutil
        p = psutil.Process(int(st["pid"]))
        for ch in p.children(recursive=True):
            ch.kill()
        p.kill()
    except Exception:
        pass
    st.pop("log_tail", None)
    st.update({"state": "cancelled", "updated_at": _now()})
    write_json_atomic(os.path.join(jobs_dir(ws_dir), job_id, "status.json"), st)
    return True
