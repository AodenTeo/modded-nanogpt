#!/bin/bash
# LR sweep for medium track: 4 muon_lr × 3 adam_lr = 12 runs
# Each run takes ~17 minutes; total ~3.5 hours.
# Usage: ./sweep_lr.sh

set -e

MUON_LRS="0.015 0.017 0.019 0.021"
ADAM_LRS="0.004 0.006 0.008"

RESULTS_DIR="sweep_results"
SUMMARY="$RESULTS_DIR/summary.txt"
mkdir -p "$RESULTS_DIR"

echo "========================================" | tee "$SUMMARY"
echo "LR Sweep started: $(date)"               | tee -a "$SUMMARY"
echo "Muon LRs: $MUON_LRS"                     | tee -a "$SUMMARY"
echo "Adam LRs: $ADAM_LRS"                      | tee -a "$SUMMARY"
echo "========================================" | tee -a "$SUMMARY"

run_count=0
total_runs=12

for muon_lr in $MUON_LRS; do
    for adam_lr in $ADAM_LRS; do
        run_count=$((run_count + 1))
        run_id="sweep_muon${muon_lr}_adam${adam_lr}"
        log_file="$RESULTS_DIR/${run_id}.log"

        echo ""
        echo ">>> [$run_count/$total_runs] muon_lr=$muon_lr adam_lr=$adam_lr"
        echo ">>> Started: $(date)"
        echo ">>> Log: $log_file"

        MUON_LR=$muon_lr ADAM_LR=$adam_lr RUN_ID=$run_id \
            torchrun --standalone --nproc_per_node=8 \
            train_gpt_medium.py 2>&1 | tee "$log_file"

        echo ">>> Finished: $(date)"
    done
done

echo ""
echo "========================================"
echo "All $total_runs runs complete: $(date)"
echo "========================================"
echo ""
echo "Parsing results..."
python3 parse_sweep.py "$RESULTS_DIR"
