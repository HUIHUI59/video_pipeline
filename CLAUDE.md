# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Distributed video preprocessing pipeline that converts arbitrary video formats/resolutions to a unified spec (1920×auto, 24fps, H.264/AAC) across three GPU servers (RTX 4090, A6000, A8000). Uses a shared file-based task queue to coordinate workers via SSH.

## Common Commands

**Run distributed pipeline (daily):**
```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml \
  --git-pull
```

**First-time run (deploy scripts to remotes):**
```bash
python distributed_dispatch.py --input-dir /mnt/movies/Films --output-dir /mnt/movies/Films/output --servers servers.yaml --git-pull --deploy
```

**Dry-run (preview tasks, no processing):**
```bash
python distributed_dispatch.py --input-dir /mnt/movies/Films --output-dir /mnt/movies/Films/output --servers servers.yaml --dry-run
```

**Check SSH connectivity:**
```bash
python distributed_dispatch.py --servers servers.yaml --check
```

**Emergency stop all workers:**
```bash
python distributed_dispatch.py --servers servers.yaml --stop
```

**Single-machine local mode (no coordinator):**
```bash
python process_videos.py ./input ./output
python process_videos.py ./input ./output --workers 1 --no-nvenc  # CPU-only, single worker
```

**Queue status inspection:**
```python
from task_queue import TaskQueue
q = TaskQueue('/path/to/output/.queue')
print(q.stats())
```

## Architecture

Three Python scripts communicate through a shared file-based task queue:

**`distributed_dispatch.py`** — Orchestrator running on control machine (RTX 4090):
1. Tests SSH reachability of all servers in `servers.yaml`
2. Optionally git-pulls on all machines (`--git-pull`)
3. Scans input directory and initializes the shared task queue
4. Optionally deploys scripts to remote machines (`--deploy`)
5. Launches `process_videos.py` on all reachable machines (SSH for remotes, subprocess for local)
6. Tails logs from all workers and waits for completion
7. On Ctrl+C: broadcasts SIGTERM → waits 3s → SIGKILL to all worker process groups

**`process_videos.py`** — Worker process (runs on each GPU machine):
- Calls `os.setsid()` at module load time (before argument parsing) — this makes the process a group leader so the dispatcher can kill the entire group (Python + all ffmpeg children) with `kill -TERM/-KILL -- -$PGID`
- In **queue mode**: atomically claims tasks from shared queue, sends heartbeat every 60s, marks done/failed
- In **local mode**: scans input dir directly, uses local `.pipeline_state.json` for checkpointing
- Encodes with NVENC (GPU H.264) and auto-falls back to libx264 on failure
- Output naming: `{first6chars_of_filename}_{uuid8}.mp4`

**`task_queue.py`** — Shared file-based queue with `fcntl` locking:
- States: `pending` → `claimed` → `done`/`error`
- `init_queue()` without `--force`: resets only `claimed` tasks to `pending` (preserves `done` — safe to re-run after interruption)
- `init_queue()` with `--force`: resets all tasks including completed ones
- Tasks claimed >10 minutes without a heartbeat are automatically reassigned
- Max 3 retries per task before marking as `error`

**`servers.yaml`** — Machine registry with SSH credentials, CUDA device IDs, conda env name, and compute weight (1.0–1.5).

## Key Implementation Details

**Process group kill:** `os.setsid()` in `process_videos.py` is intentional and critical — it ensures PGID == PID so the dispatcher can kill the entire ffmpeg process tree atomically. Do not remove this call or add `start_new_session=True` to subprocess calls.

**Queue recovery:** After an interrupted run, simply re-run without `--force` — completed tasks are preserved and `claimed` tasks are reset to `pending`.

**NVENC fallback:** If GPU encoding fails, the worker transparently retries with libx264. Verify NVENC availability with `ffmpeg -encoders | grep h264_nvenc`.

**Dynamic bitrate:** 4K→8Mbps, 1080p→3.5Mbps, 720p→2Mbps, capped at 2GB per output file.

## Dependencies

Uses conda environment `movietest` on all machines. Install with:
```bash
conda activate movietest
pip install -r requirements.txt   # rich, pyyaml, ultralytics, opencv, pydantic, mediapipe, fastapi, ...
sudo apt install -y ffmpeg openssh-server
```

**Stage 4 face detector tier**: MediaPipe Tasks `FaceDetector` (blaze_face_full_range, conf≥0.5) is tier-0 and auto-downloads the `.tflite` model to `~/.cache/mediapipe-models/` on first use. Override via `MEDIAPIPE_FACE_MODEL_PATH` env var. Falls back to YOLOv8-face then OpenCV Haar if MediaPipe import fails.

**Stage 4 camera-motion gate**: Farneback dense optical flow on the middle 5 frames (downsampled to 480 wide); avg L2 magnitude > `QUALITY_MAX_CAMERA_MOTION` (default 6.0 px/frame) adds `"camera_shake"` to `quality_metrics.issues`. Skip with `--skip-motion-detect`.
