#!/usr/bin/env python3
"""Plot all completed runs across all sweeps against the baseline winner."""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

WINNER = "sweep_results_v2_yarn_fix/sweep_muon0.015_adam0.004.log"
SWEEP_DIRS = [
    "sweep_results_v6_warmup",
    "sweep_results_v7_soft_peak",
]


def parse_loss_curve(filepath):
    """Extract (step, val_loss) pairs from a log file."""
    text = Path(filepath).read_text()
    pattern = r"step:(\d+)/\d+ val_loss:([\d.]+)"
    return [(int(s), float(l)) for s, l in re.findall(pattern, text)]


def label_from_path(filepath):
    """Build a readable label from the log filename."""
    name = Path(filepath).stem
    return name.replace("sweep_", "").replace("_", " ")


def main():
    """Plot all runs, highlighting the baseline winner."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Collect all unique runs (deduplicate across sweep dirs)
    seen = set()
    for d in SWEEP_DIRS:
        for f in sorted(Path(d).glob("sweep_*.log")):
            curve = parse_loss_curve(f)
            if len(curve) < 2:
                continue
            label = label_from_path(f)
            if label in seen:
                continue
            seen.add(label)
            steps, losses = zip(*curve)
            ax.plot(steps, losses, label=label,
                    linewidth=1, alpha=0.5)

    # Plot winner on top with emphasis
    curve = parse_loss_curve(WINNER)
    steps, losses = zip(*curve)
    ax.plot(steps, losses, label="muon0.015 adam0.004 (winner)",
            linewidth=2.5, color='black', zorder=10)

    ax.axhline(y=2.92, color='red', linestyle='--',
               linewidth=1, label='target (2.92)')
    ax.set_xlabel("Step")
    ax.set_ylabel("Val Loss")
    ax.set_title("All Sweep Runs vs Baseline Winner")
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2.90, 3.15)
    fig.savefig("comparison_all.png", dpi=150, bbox_inches='tight')
    print("Saved: comparison_all.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
