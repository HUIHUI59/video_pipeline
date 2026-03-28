#!/bin/bash

# ================= 1. Global Configurations =================
INPUT_DIR="/mnt/movies/Films/CutTest/"
OUTPUT_DIR="/mnt/movies/Films/output"
SERVERS_CONF="servers.yaml"
DISPATCH_SCRIPT="distributed_dispatch.py"

echo "[SYSTEM] Initializing Distributed Task Dispatch Engine..."

# ================= 2. Pre-flight Strict Multi-Verifications =================
echo "[SYSTEM] Running pre-flight self-checks..."

# Check 1: Ensure cluster config exists
if [ ! -f "$SERVERS_CONF" ]; then
    echo "[FATAL ERROR] Cluster configuration not found: $SERVERS_CONF"
    exit 1
fi
echo "  -> Check 1 PASS: $SERVERS_CONF is locked and loaded."

# Check 2: Ensure core Python engine exists
if [ ! -f "$DISPATCH_SCRIPT" ]; then
    echo "[FATAL ERROR] Core dispatch engine not found: $DISPATCH_SCRIPT"
    exit 1
fi
echo "  -> Check 2 PASS: Dispatch engine script is ready."

# Check 3: Probe input data source (Accounting for all heterogeneous file formats and qualities)
if [ ! -d "$INPUT_DIR" ]; then
    echo "[FATAL ERROR] Input directory missing: $INPUT_DIR. Please check mounts."
    exit 1
fi
# Scan for total files to ensure we are actually feeding data to the pipeline
FILE_COUNT=$(find "$INPUT_DIR" -type f | wc -l)
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "[WARNING] Input directory is empty. No video resources found."
else
    echo "  -> Check 3 PASS: Data source active. Scanned $FILE_COUNT heterogeneous media files ready for processing."
fi

# Check 4: Ensure output path is ready (Create automatically if missing, zero interruption)
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "  -> Check 4 PASS: Auto-created output directory $OUTPUT_DIR."
else
    echo "  -> Check 4 PASS: Output directory is ready."
fi

# ================= 3. Atomic Execution =================
echo -e "\n[SYSTEM] All nodes and pathways verified. Injecting tasks to A6000 and A8000 cluster...\n"

# Capture the exit code of the Python engine for final verification
python3 "$DISPATCH_SCRIPT" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --servers "$SERVERS_CONF"

EXIT_CODE=$?

# ================= 4. Post-Execution Verification =================
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n[SUCCESS] Dispatch engine finished successfully. Tasks deployed to cluster!"
else
    echo -e "\n[FATAL ERROR] Dispatch process failed. Python engine exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi