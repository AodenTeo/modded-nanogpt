#!/usr/bin/env python3
"""Plot val_loss curves from sweep log files."""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def parse_loss_curve(filepath):
    """Extract (step, val_loss) pairs from a log file."""
    text = filepath.read_text()
    pattern = r"step:(\d+)/\d+ val_loss:([\d.]+)"
    return [(int(s), float(l)) for s, l in re.findall(pattern, text)]


def plot_sweep(results_dir, title, output_path):
    """Plot all loss curves from a sweep directory and save."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for f in sorted(Path(results_dir).glob("sweep_*.log")):
        curve = parse_loss_curve(f)
        if not curve:
            continue
        steps, losses = zip(*curve)
        label = f.stem.replace("sweep_", "").replace("_", " ")
        ax.plot(steps, losses, label=label, linewidth=1.5)

    ax.axhline(y=2.92, color='red', linestyle='--', linewidth=1, label='target (2.92)')
    ax.set_xlabel("Step")
    ax.set_ylabel("Val Loss")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2.90, 3.10)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    """Generate plots for all available sweep directories."""
    sweeps = [
        ("sweep_results", "Sweep v1: LR only (no YaRN fix)", "sweep_v1_curves.png"),
        ("sweep_results_v2_yarn_fix", "Sweep v2: LR + YaRN fix", "sweep_v2_curves.png"),
        ("sweep_results_v3_low_lr", "Sweep v3: Low LR + YaRN fix", "sweep_v3_curves.png"),
        ("sweep_results_v4_cooldown", "Sweep v4: Cooldown x Low LR", "sweep_v4_curves.png"),
        ("sweep_results_v5_decay", "Sweep v5: Decay Floor", "sweep_v5_curves.png"),
        ("sweep_results_v6_warmup", "Sweep v6: Delayed Ramp + Higher Peak", "sweep_v6_curves.png"),
        ("sweep_results_v7_soft_peak", "Sweep v7: Delayed Ramp + Soft Peak", "sweep_v7_curves.png"),
    ]
    for results_dir, title, output in sweeps:
        if Path(results_dir).exists():
            plot_sweep(results_dir, title, output)
        else:
            print(f"Skipping {results_dir} (not found)")


if __name__ == "__main__":
    main()
