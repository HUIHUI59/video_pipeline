#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python distributed_dispatch.py --stop --target RTX4090_local
