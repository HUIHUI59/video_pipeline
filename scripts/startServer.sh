#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/Baiduyun \
  --output-dir /mnt/movies/Films/output \
  --target RTX4090_local
