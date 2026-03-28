#!/usr/bin/env python3
"""
task_queue.py  —  基于共享存储的分布式任务队列
════════════════════════════════════════════════════════════════
设计：
  - 队列文件存在共享存储上（/mnt/movies/.pipeline_queue.json）
  - 每台机器用 fcntl 文件锁原子认领任务，避免重复处理
  - 每个认领的任务带 worker_id + claimed_at 时间戳
  - 心跳超时（默认 10 分钟）：超时的任务自动释放回队列
    → 4090 被 kill 后，它的任务最多 10 分钟后被其他机器接手
  - 任务三种状态：pending / claimed / done

队列文件结构：
  {
    "tasks": {
      "/mnt/movies/film.mkv": {
        "status": "pending" | "claimed" | "done",
        "dst": "",
        "worker": "",
        "claimed_at": 0.0,
        "done_at": 0.0,
        "output_idx": 1
      }
    },
    "created_at": "...",
    "total": 100
  }
════════════════════════════════════════════════════════════════
"""

import os
import json
import time
import fcntl
import socket
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("queue")

QUEUE_FILENAME    = ".pipeline_queue.json"
LOCK_FILENAME     = ".pipeline_queue.lock"
HEARTBEAT_TIMEOUT = 600   # 秒：超过此时间未心跳视为死亡，任务释放
HEARTBEAT_FILE    = ".pipeline_heartbeat_{worker}.json"


class TaskQueue:
    """
    线程安全 + 进程安全的分布式任务队列。
    使用 fcntl 文件锁保证跨进程原子操作。
    """

    def __init__(self, queue_dir: str, worker_id: str = ""):
        self.queue_dir  = Path(queue_dir)
        self.queue_file = self.queue_dir / QUEUE_FILENAME
        self.lock_file  = self.queue_dir / LOCK_FILENAME
        self.worker_id  = worker_id or f"{socket.gethostname()}_{os.getpid()}"
        self._hb_file   = self.queue_dir / HEARTBEAT_FILE.format(worker=self.worker_id.replace("/", "_"))
        self._lock_fd   = None

    # ──────────────────────────────────────────────────────────
    # 文件锁（跨进程互斥）
    # ──────────────────────────────────────────────────────────

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
        return {"tasks": {}, "total": 0}

    def _save(self, data: dict):
        tmp = self.queue_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.queue_file)

    # ──────────────────────────────────────────────────────────
    # 初始化队列（主控调用一次）
    # ──────────────────────────────────────────────────────────

    def init_queue(self, src_files: list[str], start_idx: int = 1, force: bool = False):
        """
        创建队列。force=False 时保留已 done 的任务（续传）。
        """
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._acquire()
        try:
            existing = self._load()
            tasks    = existing.get("tasks", {}) if not force else {}

            added = 0
            for i, src in enumerate(src_files):
                if src in tasks and tasks[src]["status"] == "done" and not force:
                    continue   # 已完成，保留
                if src not in tasks or force:
                    tasks[src] = {
                        "status":     "pending",
                        "dst":        "",
                        "worker":     "",
                        "claimed_at": 0.0,
                        "done_at":    0.0,
                        "output_idx": start_idx + i,
                    }
                    added += 1

            data = {
                "tasks":      tasks,
                "total":      len(tasks),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._save(data)
            log.info(f"队列初始化：{len(tasks)} 个任务（新增 {added}）")
        finally:
            self._release()

    # ──────────────────────────────────────────────────────────
    # 认领任务
    # ──────────────────────────────────────────────────────────

    def claim_next(self) -> Optional[tuple[str, str, int]]:
        """
        原子认领一个 pending 任务（或超时的 claimed 任务）。
        返回 (src_path, dst_name, output_idx) 或 None（队列空）。
        """
        self._acquire()
        try:
            data  = self._load()
            tasks = data.get("tasks", {})
            now   = time.time()

            for src, t in tasks.items():
                status = t["status"]
                # 认领 pending
                if status == "pending":
                    pass
                # 接手超时的 claimed（原 worker 已死）
                elif status == "claimed":
                    age = now - t.get("claimed_at", 0)
                    if age < HEARTBEAT_TIMEOUT:
                        continue   # 还在跑，跳过
                    log.info(
                        f"任务超时释放 [{t.get('worker','')}→{self.worker_id}]: "
                        f"{Path(src).name} (超时 {age:.0f}s)"
                    )
                else:
                    continue   # done，跳过

                # 认领
                t["status"]     = "claimed"
                t["worker"]     = self.worker_id
                t["claimed_at"] = now
                self._save(data)
                return src, t["dst"], t["output_idx"]

            return None   # 没有可认领的任务
        finally:
            self._release()

    # ──────────────────────────────────────────────────────────
    # 标记完成
    # ──────────────────────────────────────────────────────────

    def mark_done(self, src: str, dst: str):
        self._acquire()
        try:
            data = self._load()
            if src in data["tasks"]:
                data["tasks"][src].update({
                    "status":  "done",
                    "dst":     dst,
                    "done_at": time.time(),
                })
                self._save(data)
        finally:
            self._release()

    # ──────────────────────────────────────────────────────────
    # 标记失败（释放回 pending，允许其他机器重试）
    # ──────────────────────────────────────────────────────────

    def mark_failed(self, src: str, error: str = ""):
        self._acquire()
        try:
            data = self._load()
            if src in data["tasks"]:
                t = data["tasks"][src]
                retry = t.get("retries", 0) + 1
                if retry >= 3:
                    # 重试 3 次仍失败，标记 error 永久跳过
                    t.update({"status": "error", "error": error, "retries": retry})
                    log.warning(f"任务重试 3 次失败，永久跳过: {Path(src).name}")
                else:
                    t.update({"status": "pending", "worker": "", "retries": retry})
                    log.info(f"任务失败，释放回队列（第 {retry} 次）: {Path(src).name}")
                self._save(data)
        finally:
            self._release()

    # ──────────────────────────────────────────────────────────
    # 心跳（每 N 秒更新 claimed_at，防止超时释放）
    # ──────────────────────────────────────────────────────────

    def heartbeat(self, src: str):
        """更新任务的 claimed_at，证明 worker 还活着"""
        self._acquire()
        try:
            data = self._load()
            if src in data["tasks"] and data["tasks"][src]["status"] == "claimed":
                data["tasks"][src]["claimed_at"] = time.time()
                self._save(data)
        finally:
            self._release()

    # ──────────────────────────────────────────────────────────
    # 统计
    # ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        data  = self._load()
        tasks = data.get("tasks", {})
        cnt   = {"pending": 0, "claimed": 0, "done": 0, "error": 0}
        now   = time.time()
        for t in tasks.values():
            s = t["status"]
            # 超时的 claimed 计为 pending（事实上可被接手）
            if s == "claimed" and now - t.get("claimed_at", 0) >= HEARTBEAT_TIMEOUT:
                cnt["pending"] += 1
            else:
                cnt[s] = cnt.get(s, 0) + 1
        return cnt

    def is_all_done(self) -> bool:
        s = self.stats()
        return s["pending"] == 0 and s["claimed"] == 0