#!/usr/bin/env python3
"""Plot decay floor run vs baseline winner."""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def parse_loss_curve(filepath):
    text = Path(filepath).read_text()
    pattern = r"step:(\d+)/\d+ val_loss:([\d.]+)"
    return [(int(s), float(l)) for s, l in re.findall(pattern, text)]

def main():
    runs = [
        ("sweep_results_v2_yarn_fix/sweep_muon0.015_adam0.004.log", "Baseline (0.015/0.004)"),
        ("sweep_results_v3_low_lr/sweep_muon0.010_adam0.003.log", "Low LR (0.010/0.003)"),
        ("sweep_results_v5_decay/sweep_muon0.010_decay0.4.log", "Decay Floor 0.4 (0.010)"),
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for path, label in runs:
        if not Path(path).exists(): continue
        curve = parse_loss_curve(path)
        steps, losses = zip(*curve)
        ax.plot(steps, losses, label=label, linewidth=2)
        
    ax.axhline(y=2.92, color='red', linestyle='--', linewidth=1, label='target')
    ax.set_xlabel("Step")
    ax.set_ylabel("Val Loss")
    ax.set_title("Decay Floor Experiment vs Baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2.90, 3.20)
    
    fig.savefig("analysis_decay.png", dpi=150)
    print("Saved: analysis_decay.png")

if __name__ == "__main__":
    main()
