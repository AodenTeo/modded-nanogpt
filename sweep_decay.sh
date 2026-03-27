#!/bin/bash
# Sweep v5: Decay floor test
# muon=0.010, adam=0.003, cooldown=0.70 (best from v3)
# decay_floor=0.4 (vs baseline 0.1)
# Usage: ./sweep_decay.sh

set -e

MUON_LR="0.010"
ADAM_LR="0.003"
COOLDOWN_FRAC="0.70"
DECAY_FLOORS="0.20 0.25"

RESULTS_DIR="sweep_results_v5_decay"
SUMMARY="$RESULTS_DIR/summary.txt"
mkdir -p "$RESULTS_DIR"

echo "========================================" | tee "$SUMMARY"
echo "Decay Sweep started: $(date)"            | tee -a "$SUMMARY"
echo "Muon LR: $MUON_LR"                       | tee -a "$SUMMARY"
echo "Adam LR: $ADAM_LR"                       | tee -a "$SUMMARY"
echo "Cooldown: $COOLDOWN_FRAC"                | tee -a "$SUMMARY"
echo "Decay Floors: $DECAY_FLOORS"             | tee -a "$SUMMARY"
echo "========================================" | tee -a "$SUMMARY"

run_count=0
total_runs=1

for decay in $DECAY_FLOORS; do
    run_count=$((run_count + 1))
    run_id="sweep_muon${MUON_LR}_decay${decay}"
    log_file="$RESULTS_DIR/${run_id}.log"

    echo ""
    echo ">>> [$run_count/$total_runs] decay_floor=$decay"
    echo ">>> Started: $(date)"
    echo ">>> Log: $log_file"

    MUON_LR=$MUON_LR ADAM_LR=$ADAM_LR COOLDOWN_FRAC=$COOLDOWN_FRAC DECAY_FLOOR=$decay RUN_ID=$run_id \
        torchrun --standalone --nproc_per_node=8 \
        train_gpt_medium.py 2>&1 | tee "$log_file"

    echo ">>> Finished: $(date)"
done

echo ""
echo "========================================"
echo "Run complete: $(date)"
echo "========================================"
