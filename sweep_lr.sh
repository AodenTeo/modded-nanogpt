#!/bin/bash
# Sweep v4: cooldown_frac × muon_lr with adam_lr fixed at 0.003
# 3 muon_lr × 4 cooldown_frac = 12 runs ≈ 3.5 hours
# Usage: ./sweep_lr.sh

set -e

MUON_LRS="0.008 0.010 0.012"
COOLDOWN_FRACS="0.40 0.50 0.55 0.60"
ADAM_LR="0.003"

RESULTS_DIR="sweep_results_v4_cooldown"
SUMMARY="$RESULTS_DIR/summary.txt"
mkdir -p "$RESULTS_DIR"

echo "========================================" | tee "$SUMMARY"
echo "Cooldown Sweep started: $(date)"         | tee -a "$SUMMARY"
echo "Muon LRs: $MUON_LRS"                    | tee -a "$SUMMARY"
echo "Cooldown fracs: $COOLDOWN_FRACS"         | tee -a "$SUMMARY"
echo "Adam LR: $ADAM_LR (fixed)"               | tee -a "$SUMMARY"
echo "========================================" | tee -a "$SUMMARY"

run_count=0
total_runs=12

for muon_lr in $MUON_LRS; do
    for cd_frac in $COOLDOWN_FRACS; do
        run_count=$((run_count + 1))
        run_id="sweep_muon${muon_lr}_cd${cd_frac}"
        log_file="$RESULTS_DIR/${run_id}.log"

        echo ""
        echo ">>> [$run_count/$total_runs] muon_lr=$muon_lr cd_frac=$cd_frac adam_lr=$ADAM_LR"
        echo ">>> Started: $(date)"
        echo ">>> Log: $log_file"

        MUON_LR=$muon_lr ADAM_LR=$ADAM_LR COOLDOWN_FRAC=$cd_frac RUN_ID=$run_id \
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
