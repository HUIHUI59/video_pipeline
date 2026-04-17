#!/usr/bin/env python3
"""
task_queue.py  v3.0
════════════════════════════════════════════════════════════════
v3.0 关键改进（解决共享网络文件系统上的队列数据丢失问题）：

  问题：fcntl.flock 在 SMB/NFS 跨机挂载上不提供真正的排他锁，
  3 台机器同时对同一个 670KB JSON 文件做 read-modify-write 会产生
  竞争，导致 JSON 损坏。损坏后 auto-repair 返回空队列，重启时
  init_queue 看不到任何 done 记录，重新处理所有任务。

  修复1: done-log（追加写，不替换文件）
    - mark_done() 同时向 {queue}.done.log 追加一条 JSON 行
    - append 操作在大多数文件系统上比 replace 更原子
    - init_queue() 在加载主 JSON 后，用 done-log 补全/恢复 done 状态
    - 即使主 JSON 彻底损坏，done-log 保留了所有完成记录

  修复2: 心跳改用 per-worker 文件
    - heartbeat() 只写 {queue}.hb.{worker_id}（每台机器独立文件）
    - 不再每60秒重写整个大 JSON，极大降低主队列文件写入频率
    - claim_next() / stats() 通过读心跳文件判断任务是否仍存活
════════════════════════════════════════════════════════════════
"""

import os, json, time, fcntl, socket, logging, hashlib
from pathlib import Path
from typing import Optional

log = logging.getLogger("queue")

HEARTBEAT_TIMEOUT = 600   # 秒：心跳超过此时间无更新，视为死亡


class TaskQueue:
    def __init__(self, queue_dir: str, worker_id: str = "",
                 queue_name: str = "pipeline_queue"):
        self.queue_dir  = Path(queue_dir)
        self.queue_file = self.queue_dir / f"{queue_name}.json"
        self.lock_file  = self.queue_dir / f"{queue_name}.lock"
        self._done_log  = self.queue_dir / f"{queue_name}.done.log"
        self.worker_id  = worker_id or socket.gethostname()
        self._lock_fd   = None

    # ── 文件锁（本地进程间有效；SMB/NFS 下尽力而为）────────────────
    def _acquire(self):
        self._lock_fd = open(self.lock_file, "w")
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)

    def _release(self):
        if self._lock_fd:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None

    # ── 主队列 JSON I/O ──────────────────────────────────────────────
    def _load(self) -> dict:
        if not self.queue_file.exists():
            return {"tasks": {}}
        raw = self.queue_file.read_text(encoding="utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 尝试取首个完整的 JSON 对象
            depth = 0
            for i, ch in enumerate(raw):
                if ch == "{":
                    depth += 1
                elif ch == "}" and depth > 0:
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(raw[: i + 1])
                            log.warning(
                                f"队列文件 JSON 损坏已自动修复（截取前 {i+1} 字节）: "
                                f"{self.queue_file}"
                            )
                            return result
                        except json.JSONDecodeError:
                            break
            log.error(f"队列文件无法解析，重置为空: {self.queue_file}")
            return {"tasks": {}}

    def _save(self, data: dict):
        """
        原子写主 JSON。SMB/CIFS 上 write-behind 缓存会导致 write_text 返回后
        .tmp 在 NAS 上还没真正落地，紧接着 rename 会 ENOENT。三层防护：
          1. 用 open+fsync 强制 flush 到 NAS 后再 rename
          2. rename 失败 → 直接 write_text 到 .json
          3. rename 后 .json 若仍不存在 → 用 payload 补写
        done-log 是最终保险（append-only，不经 rename）。
        """
        tmp = self.queue_file.with_name(
            f"{self.queue_file.stem}.{os.getpid()}.tmp"
        )
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        # fsync 强制把 SMB 客户端的脏页推到服务器
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            try: os.fsync(f.fileno())
            except OSError: pass
        try:
            tmp.replace(self.queue_file)
        except Exception as e:
            log.warning(f"_save rename 失败 ({e})，改为直接覆盖写 .json")
            try:
                self.queue_file.write_text(payload, encoding="utf-8")
            finally:
                try: tmp.unlink(missing_ok=True)
                except Exception: pass
            return
        if not self.queue_file.exists():
            log.warning("_save rename 后 .json 不存在，用 payload 补写")
            try:
                self.queue_file.write_text(payload, encoding="utf-8")
            except Exception:
                pass

    # ── done-log：追加写，不替换文件 ────────────────────────────────
    def _done_log_append(self, src: str, dst: str):
        """将完成记录追加到 done-log。即使主 JSON 损坏，此文件仍可恢复进度。"""
        try:
            line = json.dumps({"src": src, "dst": dst, "t": time.time()},
                               ensure_ascii=False) + "\n"
            with open(self._done_log, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        except Exception:
            pass  # 非关键路径，失败不影响主流程

    def _done_log_read(self) -> dict:
        """从 done-log 读取所有完成记录，返回 {src: dst}。"""
        done: dict = {}
        if not self._done_log.exists():
            return done
        try:
            for line in self._done_log.read_text(
                    encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[rec["src"]] = rec.get("dst", "")
                except (json.JSONDecodeError, KeyError):
                    pass
        except Exception:
            pass
        return done

    # ── per-task claim 文件（跨机原子，通过 O_CREAT|O_EXCL）──────────
    #
    # fcntl.flock 在 SMB/CIFS 跨机挂载上不可靠；但 O_CREAT|O_EXCL
    # 映射到 SMB CreateDisposition=FILE_CREATE，是跨机原子的。
    # 每个 task 用 sha1(src)[:16] 做个独立的 claim 文件，存在就代表被认领。
    def _claim_path(self, src: str) -> Path:
        h = hashlib.sha1(src.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return self.queue_dir / f"{self.queue_file.stem}.claim.{h}"

    def _try_claim_file(self, src: str) -> bool:
        """
        尝试原子创建 claim 文件。成功返回 True，已存在返回 False。
        存在但 owner 死亡（心跳超时）→ 抢占式删除后重试。
        """
        cp = self._claim_path(src)
        for attempt in range(2):
            try:
                fd = os.open(str(cp),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                try:
                    os.write(fd, json.dumps({
                        "src": src, "worker": self.worker_id,
                        "t": time.time(),
                    }, ensure_ascii=False).encode("utf-8"))
                    try: os.fsync(fd)
                    except OSError: pass
                finally:
                    os.close(fd)
                return True
            except FileExistsError:
                # 检查现有 claim 是不是已死
                try:
                    old = json.loads(cp.read_text(encoding="utf-8"))
                    if not self._hb_is_alive(old.get("worker", ""),
                                             old.get("t", 0)):
                        try: cp.unlink(missing_ok=True)
                        except Exception: pass
                        continue   # 重试一次
                except Exception:
                    pass
                return False
            except Exception as e:
                log.warning(f"_try_claim_file 异常 ({e})")
                return False
        return False

    def _release_claim(self, src: str) -> None:
        try: self._claim_path(src).unlink(missing_ok=True)
        except Exception: pass

    # ── per-worker 心跳文件 ──────────────────────────────────────────
    def _hb_path(self, worker_id: str) -> Path:
        safe = worker_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.queue_dir / f"{self.queue_file.stem}.hb.{safe}"

    def _hb_is_alive(self, worker_id: str, claimed_at: float) -> bool:
        """判断 worker 是否仍在存活（心跳文件优先，回退到 claimed_at）。"""
        try:
            data = json.loads(
                self._hb_path(worker_id).read_text(encoding="utf-8"))
            return time.time() - data.get("t", 0) < HEARTBEAT_TIMEOUT
        except Exception:
            # 心跳文件不存在或损坏 → 回退到 claimed_at（兼容旧版 worker）
            return time.time() - claimed_at < HEARTBEAT_TIMEOUT

    # ── 清理所有 claim 文件（init_queue 调用时扫走孤儿）────────────
    def _purge_claims(self) -> int:
        cnt = 0
        for cp in self.queue_dir.glob(f"{self.queue_file.stem}.claim.*"):
            try:
                cp.unlink()
                cnt += 1
            except Exception:
                pass
        return cnt

    # ── 初始化队列 ──────────────────────────────────────────────────
    def init_queue(self, src_files: list[str], start_idx: int = 1,
                   force: bool = False):
        """
        初始化或刷新队列。
        - force=True：全部重置为 pending（忽略 done-log）
        - force=False（续传）：
            1. 加载主 JSON
            2. 用 done-log 补全/恢复 done 状态（主 JSON 损坏时的救援机制）
            3. done → 保留；claimed/pending/error → 重置为 pending；新文件 → 添加
        """
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._acquire()
        try:
            existing  = self._load()
            old_tasks = {} if force else existing.get("tasks", {})

            # 用 done-log 恢复被主 JSON 损坏丢失的 done 记录
            restored = 0
            if not force:
                for src, dst in self._done_log_read().items():
                    if (src not in old_tasks
                            or old_tasks[src].get("status") != "done"):
                        old_tasks[src] = {
                            "status": "done", "dst": dst, "worker": "",
                            "claimed_at": 0.0, "done_at": 0.0,
                            "output_idx": 0, "retries": 0,
                        }
                        restored += 1

            tasks: dict = {}
            added = reset = kept = 0

            for i, src in enumerate(src_files):
                old = old_tasks.get(src)
                if force or old is None:
                    tasks[src] = {
                        "status": "pending", "dst": "", "worker": "",
                        "claimed_at": 0.0, "done_at": 0.0,
                        "output_idx": start_idx + i, "retries": 0,
                    }
                    added += 1
                elif old["status"] == "done":
                    tasks[src] = old
                    kept += 1
                else:
                    # claimed / pending / error → 重置为 pending
                    tasks[src] = {**old,
                        "status": "pending", "worker": "",
                        "claimed_at": 0.0,
                    }
                    reset += 1

            self._save({"tasks": tasks,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})

            msg = (f"队列初始化：total={len(tasks)}  "
                   f"新增={added}  重置={reset}  保留完成={kept}")
            if restored:
                msg += f"  日志恢复={restored}"
            log.info(msg)

            # 清理上一轮遗留的心跳文件 + claim 文件
            for hb in self.queue_dir.glob(f"{self.queue_file.stem}.hb.*"):
                try:
                    hb.unlink()
                except Exception:
                    pass
            purged = self._purge_claims()
            if purged:
                log.info(f"清理遗留 claim 文件 {purged} 个")

        finally:
            self._release()

    # ── 认领任务 ────────────────────────────────────────────────────
    def claim_next(self) -> Optional[tuple[str, str, int]]:
        """
        原子认领一个 pending 或超时 claimed 的任务。

        三重屏障（从外到内）：
          1. fcntl.flock（同机多进程有效，SMB 跨机不可靠）
          2. _try_claim_file 用 O_CREAT|O_EXCL（跨机原子，SMB 可靠）
          3. done-log 查询，防止已 done 的又被重新认领
        """
        self._acquire()
        try:
            data  = self._load()
            tasks = data.get("tasks", {})
            # done-log 是最权威的 "已完成" 视图，防止主 JSON 落后导致重跑
            done_set = set(self._done_log_read().keys())
            now   = time.time()

            for src, t in tasks.items():
                if src in done_set:
                    continue
                s = t["status"]
                if s == "pending":
                    pass
                elif s == "claimed":
                    if self._hb_is_alive(t.get("worker", ""),
                                         t.get("claimed_at", 0)):
                        continue
                    log.info(
                        f"超时接手 [{t.get('worker','')}→{self.worker_id}]: "
                        f"{Path(src).name} "
                        f"({now - t.get('claimed_at', 0):.0f}s)"
                    )
                else:
                    continue  # done / error

                # 跨机原子 claim
                if not self._try_claim_file(src):
                    continue  # 被别的 worker 抢走了

                t.update({"status": "claimed", "worker": self.worker_id,
                           "claimed_at": now})
                self._save(data)
                return src, t.get("dst", ""), t["output_idx"]

            return None
        finally:
            self._release()

    # ── 心跳：写 per-worker 文件，不碰主 JSON ────────────────────────
    def heartbeat(self, src: str):
        """
        更新心跳。写到独立的 per-worker 文件，避免每60秒重写整个队列 JSON。
        这是减少共享文件系统竞争写的关键优化。
        """
        try:
            self._hb_path(self.worker_id).write_text(
                json.dumps({"src": src, "worker": self.worker_id,
                             "t": time.time()}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── 标记完成 ────────────────────────────────────────────────────
    def mark_done(self, src: str, dst: str):
        # 先追加 done-log（不受主 JSON 损坏影响）
        self._done_log_append(src, dst)
        self._release_claim(src)
        self._acquire()
        try:
            data = self._load()
            if src in data.get("tasks", {}):
                data["tasks"][src].update({
                    "status": "done", "dst": dst, "done_at": time.time(),
                })
                self._save(data)
        finally:
            self._release()

    # ── 标记失败（释放回 pending，最多重试 3 次）────────────────────
    def mark_failed(self, src: str, error: str = ""):
        self._release_claim(src)
        self._acquire()
        try:
            data = self._load()
            t = data.get("tasks", {}).get(src)
            if t:
                retries = t.get("retries", 0) + 1
                if retries >= 3:
                    t.update({"status": "error", "error": error[:200],
                               "retries": retries})
                    log.warning(f"放弃（重试3次）: {Path(src).name}")
                else:
                    t.update({"status": "pending", "worker": "",
                               "claimed_at": 0.0, "retries": retries})
                    log.info(f"释放回队列（第{retries}次）: {Path(src).name}")
                self._save(data)
        finally:
            self._release()

    # ── 统计 ────────────────────────────────────────────────────────
    def stats(self) -> dict:
        data  = self._load()
        tasks = data.get("tasks", {})
        cnt   = {"pending": 0, "claimed": 0, "done": 0, "error": 0}
        for t in tasks.values():
            s = t["status"]
            if s == "claimed":
                if self._hb_is_alive(t.get("worker", ""),
                                     t.get("claimed_at", 0)):
                    cnt["claimed"] += 1
                else:
                    cnt["pending"] += 1  # 心跳超时，视为 pending
            else:
                cnt[s] = cnt.get(s, 0) + 1
        return cnt

    def is_all_done(self) -> bool:
        """
        要两次连续的 "pending=0 且 claimed=0" 才认定结束。
        中间 sleep 2s 让 SMB rename 瞬态不要误伤。
        """
        s1 = self.stats()
        if s1.get("pending", 0) > 0 or s1.get("claimed", 0) > 0:
            return False
        time.sleep(2)
        s2 = self.stats()
        return s2.get("pending", 0) == 0 and s2.get("claimed", 0) == 0

    def pending_count(self) -> int:
        return self.stats().get("pending", 0)
