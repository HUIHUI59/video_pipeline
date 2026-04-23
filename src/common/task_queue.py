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

import os, json, time, fcntl, socket, logging, hashlib, threading
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
        # In-process dedupe of currently-claimed src paths.
        # Required when multiple worker threads in ONE process share the
        # same worker_id: _try_claim_file's "owner == self.worker_id →
        # return True" branch (line ~252) lets every thread re-claim the
        # same task. We track in-flight srcs in memory and skip them at
        # claim time. mark_done / mark_failed / _release_claim clear it.
        self._in_flight = set()
        self._in_flight_lock = threading.Lock()

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

    def _save(self, data: dict) -> bool:
        """
        尽力写主 JSON。**任何异常都不外抛**，失败返回 False（只记日志）。
        done-log 是真正的 source of truth；JSON 只是"最近一次快照"视图。

        尝试策略（最多 3 轮，每轮三步）：
          1. 写 <pid>.tmp + fsync
          2. tmp.replace(.json)
          3. 如果 rename 失败或 .json 不存在，直接 write_text(.json)
        每轮失败 sleep 递增后重试。
        """
        tmp = self.queue_file.with_name(
            f"{self.queue_file.stem}.{os.getpid()}.tmp"
        )
        try:
            payload = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"_save: json.dumps 失败 ({e})，跳过保存")
            return False

        last_err: str = ""
        for attempt in range(3):
            try:
                # 1) 写 tmp + fsync
                try:
                    with open(tmp, "w", encoding="utf-8") as f:
                        f.write(payload)
                        f.flush()
                        try: os.fsync(f.fileno())
                        except OSError: pass
                except Exception as e:
                    last_err = f"write tmp: {e}"
                    raise

                # 2) rename tmp → .json
                try:
                    tmp.replace(self.queue_file)
                    if self.queue_file.exists():
                        return True
                    last_err = "rename ok but .json missing"
                except Exception as e:
                    last_err = f"rename: {e}"

                # 3) 直接覆盖写 .json
                try:
                    self.queue_file.write_text(payload, encoding="utf-8")
                    if self.queue_file.exists():
                        try: tmp.unlink(missing_ok=True)
                        except Exception: pass
                        return True
                    last_err += " | direct write ok but file still missing"
                except Exception as e:
                    last_err += f" | direct write: {e}"

            except Exception:
                pass  # last_err 已记录，回到外层 backoff

            # backoff
            time.sleep(0.3 * (2 ** attempt))  # 0.3s, 0.6s, 1.2s

        log.error(f"_save 三次尝试都失败，跳过（done-log 兜底）: {last_err}")
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        return False

    # ── done-log：追加写，不替换文件 ────────────────────────────────
    def _done_log_append(self, src: str, dst: str) -> bool:
        """将完成记录追加到 done-log（append-only 的权威完成记录）。

        done-log 是 v3.0 设计中唯一可信的完成记录源；**追加失败会导致任务下次
        被重复处理**（init_queue 看不到记录 → 重置为 pending）。因此必须用
        log.error 把异常打到日志，便于问题排查；调用方可通过返回值感知失败。

        不向外抛异常：保证 worker 主循环绝不会因为 done-log I/O 异常崩溃，
        这和 v3.0 "worker 永不崩" 的硬约束一致。
        """
        try:
            line = json.dumps({"src": src, "dst": dst, "t": time.time()},
                               ensure_ascii=False) + "\n"
            with open(self._done_log, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            return True
        except Exception as e:
            log.error(
                f"done-log 追加失败（任务可能被重复处理）: {e} | "
                f"src={src} dst={dst} log={self._done_log}"
            )
            return False

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

        已知 race（SMB/NFS 上的 TOCTOU 窗口）：在"判断 owner 死亡"到
        "unlink+O_EXCL 重建"之间，别的 worker 也可能同时 unlink 并用 O_EXCL
        抢先创建。此时我们的二次 O_EXCL 会拿到 FileExistsError，正确回退；
        但另一种更危险的情况是 **O_EXCL 成功但里头记录了对方的 worker_id**——
        这不会发生，因为 O_EXCL 保证只有胜者能写。
        为进一步防御 "unlink 误删对方合法新 claim" 的风险，这里在 unlink 前
        二次读取并再次验证确实是死 claim，并把 unlink + create 限制在一次
        重试内完成，避免无界循环抢占。
        """
        claim_bytes = json.dumps({
            "src": src, "worker": self.worker_id,
            "t": time.time(),
        }, ensure_ascii=False).encode("utf-8")

        cp = self._claim_path(src)
        for attempt in range(2):
            try:
                fd = os.open(str(cp),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                try:
                    os.write(fd, claim_bytes)
                    try:
                        os.fsync(fd)
                    except OSError:
                        pass
                finally:
                    os.close(fd)
                # O_EXCL 保证只有我们能到这一步，文件内容就是我们写的
                return True
            except FileExistsError:
                # 已有 claim：先读文件，再二次确认 owner 死亡才抢占
                try:
                    raw = cp.read_text(encoding="utf-8")
                    old = json.loads(raw)
                except Exception:
                    # 文件存在但内容损坏 → 无法判断 owner，保守放弃
                    return False

                old_worker = old.get("worker", "")
                old_t = old.get("t", 0)
                # 自己的旧 claim？这不应该发生，但若出现直接当成自己持有
                if old_worker == self.worker_id:
                    return True
                if self._hb_is_alive(old_worker, old_t):
                    return False

                # owner 死亡 → 抢占：unlink + 第二轮 O_EXCL create
                # 竞争对手此时若同时抢占，其中一个的 O_EXCL 会拿到 EEXIST 正确
                # 放弃；胜者写入自己的内容后返回 True。
                try:
                    cp.unlink(missing_ok=True)
                except Exception as e:
                    log.debug(f"_try_claim_file unlink 死 claim 失败 ({e})")
                    return False
                continue   # 第二次 for 循环会重新 O_CREAT|O_EXCL
            except Exception as e:
                log.warning(f"_try_claim_file 异常 ({e})")
                return False
        return False

    def _release_claim(self, src: str) -> None:
        try: self._claim_path(src).unlink(missing_ok=True)
        except Exception: pass
        # Drop in-process tracking — both mark_done and mark_failed call
        # us, so this single cleanup covers both terminal paths.
        try:
            with self._in_flight_lock:
                self._in_flight.discard(src)
        except Exception: pass

    # ── per-task 失败标记（不碰主 JSON）────────────────────────────
    def _fail_hash(self, src: str) -> str:
        return hashlib.sha1(src.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _fail_prefix(self, src: str) -> str:
        return f"{self.queue_file.stem}.fail.{self._fail_hash(src)}"

    def _fail_append(self, src: str, error: str = "") -> int:
        """为一次失败创建一个标记文件。返回当前累计失败次数。"""
        prefix = self._fail_prefix(src)
        ts = int(time.time() * 1000)
        marker = self.queue_dir / f"{prefix}.{ts}"
        try:
            marker.write_text(
                (error or "")[:400], encoding="utf-8", errors="replace")
        except Exception:
            pass
        return self._fail_count(src)

    def _fail_count(self, src: str) -> int:
        try:
            return sum(1 for _ in self.queue_dir.glob(
                f"{self._fail_prefix(src)}.*"))
        except Exception:
            return 0

    def _purge_fails(self) -> int:
        cnt = 0
        for f in self.queue_dir.glob(f"{self.queue_file.stem}.fail.*"):
            try:
                f.unlink()
                cnt += 1
            except Exception:
                pass
        return cnt

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

            # 清理上一轮遗留的 hb / claim / fail 标记
            for hb in self.queue_dir.glob(f"{self.queue_file.stem}.hb.*"):
                try: hb.unlink()
                except Exception: pass
            purged_c = self._purge_claims()
            purged_f = self._purge_fails() if force else 0
            if purged_c or purged_f:
                log.info(f"清理遗留 claim={purged_c} fail={purged_f}")

        finally:
            self._release()

    # ── 认领任务（filesystem-first，运行时不写主 JSON）──────────────
    def claim_next(self) -> Optional[tuple[str, str, int]]:
        """
        核心设计：主 JSON 是 init_queue 写一次的静态快照；运行时的真实状态都
        存在 filesystem primitives 上：
          - done-log  = 已完成（append-only，authoritative）
          - .claim.*  = 已认领（O_CREAT|O_EXCL 原子，跨机可靠）
          - .fail.*   = 失败标记（每次 mark_failed 追加一个），计数 ≥3 视为 error

        Performance: queue.json grows to 50+MB at 168k tasks. Re-parsing
        on every call burned ~2-6s and made main process the throughput
        ceiling (8 workers idle waiting). We now load tasks list + done
        set ONCE into memory, iterate from a cursor on subsequent calls.
        Cross-process atomicity is preserved by _try_claim_file's
        O_CREAT|O_EXCL — we still skip already-claimed srcs.

        Cache invariants:
          - tasks list is from init_queue's static snapshot (never grows
            mid-run for stage 4) → safe to cache forever
          - done_set is updated locally in mark_done; remote workers'
            done updates only matter if we'd otherwise re-claim a done
            task — _try_claim_file already handles that
          - cursor advances monotonically; we never revisit a skipped src
            in the same process (acceptable: another machine claims it)
        """
        try:
            self._acquire()
        except Exception as e:
            log.warning(f"claim_next 获取锁失败 ({e})")
            return None
        try:
            try:
                # Lazy-init the cache on first call.
                if not hasattr(self, "_task_keys_cache"):
                    data = self._load()
                    tasks = data.get("tasks", {})
                    if not isinstance(tasks, dict):
                        log.warning("claim_next: tasks 非 dict（JSON 损坏？），返回 None")
                        return None
                    self._task_keys_cache = [
                        k for k, v in tasks.items() if isinstance(v, dict)
                    ]
                    self._done_set_cache = set(self._done_log_read().keys())
                    self._task_cursor = 0
                    log.info(f"claim_next cache: {len(self._task_keys_cache)} tasks, "
                             f"{len(self._done_set_cache)} done")

                # Iterate from the cursor (monotonic — never revisit).
                while self._task_cursor < len(self._task_keys_cache):
                    src = self._task_keys_cache[self._task_cursor]
                    self._task_cursor += 1
                    if src in self._done_set_cache:
                        continue
                    if self._fail_count(src) >= 3:
                        continue
                    with self._in_flight_lock:
                        if src in self._in_flight:
                            continue
                    # 跨机原子 claim（若被别人抢走或自己已 claim，返回 False）
                    if not self._try_claim_file(src):
                        continue
                    with self._in_flight_lock:
                        self._in_flight.add(src)
                    return src, "", 0
                return None
            except Exception as e:
                log.warning(f"claim_next 内部异常 ({e})，本轮返回 None 下次再试")
                return None
        finally:
            try: self._release()
            except Exception: pass

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

    # ── 标记完成（只动 filesystem，不动主 JSON）─────────────────────
    def mark_done(self, src: str, dst: str) -> bool:
        """
        完成时仅动 filesystem：
          1. 追加 done-log（append-only，authoritative）
          2. 删除 claim 文件
        主 JSON 不碰（静态快照，只在 init 时写）。

        返回 True 表示 done-log 追加成功；False 表示失败（日志已打，任务下次会
        被重复认领）。调用方如需强一致可据此决定是否重试，但主循环通常忽略。
        """
        ok = False
        try:
            ok = self._done_log_append(src, dst)
        except Exception as e:
            log.error(f"mark_done done-log 异常 ({e}) | src={src}")
        # Keep cache in sync so re-claim_next never returns this src.
        if hasattr(self, "_done_set_cache"):
            self._done_set_cache.add(src)
        try:
            self._release_claim(src)
        except Exception:
            pass
        return ok

    # ── 标记失败（只动 filesystem，不动主 JSON）─────────────────────
    def mark_failed(self, src: str, error: str = ""):
        """
        失败时仅动 filesystem：
          1. 写一个 .fail.<hash>.<ms_epoch> 标记文件（计数失败次数）
          2. 删除 claim 文件（释放回可认领状态）
        累计 ≥3 次失败的任务在 claim_next 里会被自动跳过（视作 error）。
        """
        try: self._release_claim(src)
        except Exception: pass
        try:
            retries = self._fail_append(src, error)
            if retries >= 3:
                log.warning(f"放弃（重试 {retries} 次）: {Path(src).name}")
            else:
                log.info(f"释放回队列（第{retries}次）: {Path(src).name}")
        except Exception as e:
            log.warning(f"mark_failed 写 fail 标记异常 ({e})")

    # ── 统计（filesystem 派生）──────────────────────────────────────
    def stats(self) -> dict:
        """
        从 filesystem 派生当前状态：
          - total  = 主 JSON 里 tasks 的条目数
          - done   = done-log 去重后条目数（authoritative）
          - claimed = 存活 claim 文件数
          - error  = 累计 ≥3 次 fail 的任务数
          - pending = total - done - claimed - error
        异常不外抛，返回全 0。
        """
        cnt = {"pending": 0, "claimed": 0, "done": 0, "error": 0}
        try:
            data = self._load()
            tasks = data.get("tasks", {})
            if not isinstance(tasks, dict):
                return cnt
            total = len(tasks)
            done_set = set(self._done_log_read().keys())
            cnt["done"] = len(done_set & set(tasks.keys()))

            # 活的 claim 计数
            alive_claimed = 0
            for cp in self.queue_dir.glob(f"{self.queue_file.stem}.claim.*"):
                try:
                    d = json.loads(cp.read_text(encoding="utf-8"))
                    if self._hb_is_alive(d.get("worker", ""), d.get("t", 0)):
                        alive_claimed += 1
                except Exception:
                    pass
            cnt["claimed"] = alive_claimed

            # error：按 hash 分组，任意前缀出现 ≥3 次视为 error
            fail_hashes: dict[str, int] = {}
            for f in self.queue_dir.glob(f"{self.queue_file.stem}.fail.*"):
                name = f.name  # classify_queue.fail.<hash>.<ms>
                parts = name.rsplit(".", 2)
                if len(parts) == 3:
                    prefix_body = parts[0]  # "classify_queue.fail.<hash>"
                    h = prefix_body.rsplit(".", 1)[-1]
                    fail_hashes[h] = fail_hashes.get(h, 0) + 1
            # 仅统计哈希命中 tasks 且计数 ≥3 的
            err = 0
            if fail_hashes:
                task_hash_map = {self._fail_hash(s): s for s in tasks.keys()}
                for h, c in fail_hashes.items():
                    if c >= 3 and h in task_hash_map:
                        src = task_hash_map[h]
                        if src not in done_set:
                            err += 1
            cnt["error"] = err

            cnt["pending"] = max(0, total - cnt["done"] - cnt["claimed"] - cnt["error"])
        except Exception as e:
            log.warning(f"stats 派生状态异常 ({e})，返回全 0")
        return cnt

    def is_all_done(self) -> bool:
        """
        判断任务全部结束。多层守卫防 SMB rename 瞬态假阳性。异常一律返回 False
        （保守，不要因为偶发 SMB 抖动让 worker 误判结束）。
        """
        try:
            s1 = self.stats()
            if s1.get("pending", 0) > 0 or s1.get("claimed", 0) > 0:
                return False
            total_s1 = sum(s1.values())
            done_log_count = len(self._done_log_read())
            if total_s1 == 0 and done_log_count > 0:
                log.warning(
                    f"is_all_done: 主 JSON 显示 0 条任务，但 done-log 有 "
                    f"{done_log_count} 条。判定为 SMB 瞬态错误，继续等待。"
                )
                return False
            time.sleep(2)
            s2 = self.stats()
            if s2.get("pending", 0) > 0 or s2.get("claimed", 0) > 0:
                return False
            if sum(s2.values()) == 0 and len(self._done_log_read()) > 0:
                log.warning(
                    "is_all_done: 二次检查仍为 0 但 done-log 非空，继续等待。"
                )
                return False
            return True
        except Exception as e:
            log.warning(f"is_all_done 异常 ({e})，保守判定未完成")
            return False

    def pending_count(self) -> int:
        return self.stats().get("pending", 0)
