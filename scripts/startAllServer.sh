#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/forCloudKor \
  --output-dir /mnt/movies/Films/forCloudKorOutput \
  --stage 4 --workers 16 --exec-mode process
