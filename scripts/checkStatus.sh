#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python distributed_dispatch.py \
  --output-dir /mnt/movies/Films/forCloudKorOutput \
  --status
