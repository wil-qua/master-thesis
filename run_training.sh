#!/usr/bin/env bash

set -o pipefail
mkdir -p logs

LOG="logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "Training startet. Logdatei: $LOG"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -u train.py 2>&1 | tee "$LOG"

RC=${PIPESTATUS[0]}

echo ""
echo "================================"
echo "TRAINING EXIT CODE: $RC"
echo "LOGDATEI: $LOG"
echo "================================"

echo "TRAINING EXIT CODE: $RC" >> "$LOG"

exit "$RC"
