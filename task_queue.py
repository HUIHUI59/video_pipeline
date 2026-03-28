#!/usr/bin/env python3
"""
task_queue.py  v2.0
════════════════════════════════════════════════════════════════
修复：
  - init_queue：残留的 claimed 任务重置为 pending（防止新一轮启动
    时机器看到"全部 claimed"而退出）
  - claim_next：worker 启动时先等待片刻让所有机器都就绪再抢任务
    → 改用"有 pending 才认领，没有就等待重试"，不再在启动时立即退出
  - is_all_done：claimed 且超时才算 pending，否则要等
════════════════════════════════════════════════════════════════
"""

import os, json, time, fcntl, socket, logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("queue")

HEARTBEAT_TIMEOUT = 600   # 秒：claimed 超过此时间无心跳，视为死亡


class TaskQueue:
    def __init__(self, queue_dir: str, worker_id: str = "",
                 queue_name: str = "pipeline_queue"):
        self.queue_dir  = Path(queue_dir)
        self.queue_file = self.queue_dir / f"{queue_name}.json"
        self.lock_file  = self.queue_dir / f"{queue_name}.lock"
        self.worker_id  = worker_id or socket.gethostname()
        self._lock_fd   = None

    # ── 文件锁 ──────────────────────────────────────────────────
    def _acquire(self):
        self._lock_fd = open(self.lock_file, "w")
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)

    def _release(self):
        if self._lock_fd:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None

    def _load(self) -> dict:
        if self.queue_file.exists():
            try:
                return json.loads(self.queue_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"tasks": {}}

    def _save(self, data: dict):
        tmp = self.queue_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.queue_file)

    # ── 初始化队列 ──────────────────────────────────────────────
    def init_queue(self, src_files: list[str], start_idx: int = 1, force: bool = False):
        """
        初始化或刷新队列。
        - force=True：全部重置为 pending
        - force=False（续传）：
            * done → 保留
            * claimed → 重置为 pending（上次运行中断，本次重新分配）
            * pending → 保留
            * 新文件 → 添加为 pending
        """
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._acquire()
        try:
            existing = self._load()
            old_tasks = {} if force else existing.get("tasks", {})
            tasks = {}
            added = reset = kept = 0

            for i, src in enumerate(src_files):
                old = old_tasks.get(src)
                if force or old is None:
                    # 全新任务
                    tasks[src] = {
                        "status": "pending", "dst": "", "worker": "",
                        "claimed_at": 0.0, "done_at": 0.0,
                        "output_idx": start_idx + i, "retries": 0,
                    }
                    added += 1
                elif old["status"] == "done":
                    tasks[src] = old   # 已完成，保留
                    kept += 1
                elif old["status"] in ("claimed", "pending", "error"):
                    # claimed/pending → 重置为 pending，让本次重新分配
                    tasks[src] = {**old,
                        "status": "pending", "worker": "",
                        "claimed_at": 0.0,
                    }
                    reset += 1

            self._save({"tasks": tasks, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
            log.info(
                f"队列初始化：total={len(tasks)}  "
                f"新增={added}  重置={reset}  保留完成={kept}"
            )
        finally:
            self._release()

    # ── 认领任务 ────────────────────────────────────────────────
    def claim_next(self) -> Optional[tuple[str, str, int]]:
        """原子认领一个 pending 或超时 claimed 的任务。"""
        self._acquire()
        try:
            data  = self._load()
            tasks = data.get("tasks", {})
            now   = time.time()

            for src, t in tasks.items():
                s = t["status"]
                if s == "pending":
                    pass   # 直接认领
                elif s == "claimed":
                    age = now - t.get("claimed_at", 0)
                    if age < HEARTBEAT_TIMEOUT:
                        continue   # 还活着，跳过
                    log.info(
                        f"超时接手 [{t.get('worker','')}→{self.worker_id}]: "
                        f"{Path(src).name} ({age:.0f}s)"
                    )
                else:
                    continue   # done / error，跳过

                t.update({"status": "claimed", "worker": self.worker_id,
                           "claimed_at": now})
                self._save(data)
                return src, t.get("dst", ""), t["output_idx"]

            return None
        finally:
            self._release()

    # ── 心跳 ────────────────────────────────────────────────────
    def heartbeat(self, src: str):
        self._acquire()
        try:
            data = self._load()
            t = data.get("tasks", {}).get(src)
            if t and t["status"] == "claimed" and t.get("worker") == self.worker_id:
                t["claimed_at"] = time.time()
                self._save(data)
        finally:
            self._release()

    # ── 标记完成 ────────────────────────────────────────────────
    def mark_done(self, src: str, dst: str):
        self._acquire()
        try:
            data = self._load()
            if src in data["tasks"]:
                data["tasks"][src].update({
                    "status": "done", "dst": dst, "done_at": time.time(),
                })
                self._save(data)
        finally:
            self._release()

    # ── 标记失败（释放回 pending，最多重试 3 次）────────────────
    def mark_failed(self, src: str, error: str = ""):
        self._acquire()
        try:
            data = self._load()
            t = data.get("tasks", {}).get(src)
            if t:
                retries = t.get("retries", 0) + 1
                if retries >= 3:
                    t.update({"status": "error", "error": error[:200], "retries": retries})
                    log.warning(f"放弃（重试3次）: {Path(src).name}")
                else:
                    t.update({"status": "pending", "worker": "",
                               "claimed_at": 0.0, "retries": retries})
                    log.info(f"释放回队列（第{retries}次）: {Path(src).name}")
                self._save(data)
        finally:
            self._release()

    # ── 统计 ────────────────────────────────────────────────────
    def stats(self) -> dict:
        data  = self._load()
        tasks = data.get("tasks", {})
        cnt   = {"pending": 0, "claimed": 0, "done": 0, "error": 0}
        now   = time.time()
        for t in tasks.values():
            s = t["status"]
            if s == "claimed" and now - t.get("claimed_at", 0) >= HEARTBEAT_TIMEOUT:
                cnt["pending"] += 1   # 超时视为 pending
            else:
                cnt[s] = cnt.get(s, 0) + 1
        return cnt

    def is_all_done(self) -> bool:
        s = self.stats()
        return s.get("pending", 0) == 0 and s.get("claimed", 0) == 0

    def pending_count(self) -> int:
        return self.stats().get("pending", 0)