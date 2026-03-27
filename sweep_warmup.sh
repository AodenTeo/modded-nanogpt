#!/bin/bash
# Sweep v7: delayed ramp + flat / gentler late peak
# Hypothesis: keep early LR low for stability, but avoid a sharp late LR jump.
# 2 muon_lr × 3 ramp_stretch × 3 peak_mult = 18 runs
# Usage: ./sweep_warmup.sh

set -e

MUON_LRS="0.010 0.012"
ADAM_LR="0.004"
COOLDOWN_FRAC="0.70"
DECAY_FLOOR="0.1"
RAMP_STRETCHS="1.5 2.0 2.5"
PEAK_MULTS="1.73 1.9 2.0"

RESULTS_DIR="sweep_results_v7_soft_peak"
SUMMARY="$RESULTS_DIR/summary.txt"
mkdir -p "$RESULTS_DIR"

echo "========================================" | tee "$SUMMARY"
echo "Warmup Sweep started: $(date)"            | tee -a "$SUMMARY"
echo "Muon LRs: $MUON_LRS"                      | tee -a "$SUMMARY"
echo "Adam LR: $ADAM_LR"                       | tee -a "$SUMMARY"
echo "Cooldown: $COOLDOWN_FRAC"                | tee -a "$SUMMARY"
echo "Decay floor: $DECAY_FLOOR"               | tee -a "$SUMMARY"
echo "Ramp stretch: $RAMP_STRETCHS"            | tee -a "$SUMMARY"
echo "Peak mults: $PEAK_MULTS"                 | tee -a "$SUMMARY"
echo "========================================" | tee -a "$SUMMARY"

run_count=0
total_runs=18

for muon_lr in $MUON_LRS; do
    for stretch in $RAMP_STRETCHS; do
        for peak in $PEAK_MULTS; do
            run_count=$((run_count + 1))
            run_id="sweep_muon${muon_lr}_stretch${stretch}_peak${peak}"
            log_file="$RESULTS_DIR/${run_id}.log"

            echo ""
            echo ">>> [$run_count/$total_runs] muon_lr=$muon_lr stretch=$stretch peak=$peak"
            echo ">>> Started: $(date)"
            echo ">>> Log: $log_file"

            MUON_LR=$muon_lr ADAM_LR=$ADAM_LR COOLDOWN_FRAC=$COOLDOWN_FRAC DECAY_FLOOR=$DECAY_FLOOR \
            LR_RAMP_STRETCH=$stretch LR_PEAK_MULT=$peak RUN_ID=$run_id \
                torchrun --standalone --nproc_per_node=8 \
                train_gpt_medium.py 2>&1 | tee "$log_file"

            echo ">>> Finished: $(date)"
        done
    done
done

echo ""
echo "========================================"
echo "All $total_runs runs complete: $(date)"
echo "========================================"
echo ""
echo "Parsing results..."
python3 parse_sweep.py "$RESULTS_DIR"
