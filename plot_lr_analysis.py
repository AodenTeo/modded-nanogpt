#!/usr/bin/env python3
"""Plot val_loss and effective learning rate over time."""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Constants from train_gpt_medium.py
NUM_STEPS = 4740
SCHEDULED_STEPS = 4700

SWEEP_DIRS = [
    "sweep_results",
    "sweep_results_v2_yarn_fix",
    "sweep_results_v3_low_lr",
    "sweep_results_v4_cooldown",
    "sweep_results_v5_decay",
]

def get_lr_schedule(step, peak_lr, cooldown_frac, decay_floor=0.1):
    """Reconstruct the LR schedule logic from train_gpt_medium.py."""
    if step > SCHEDULED_STEPS:
        return peak_lr * decay_floor
    
    x = step / SCHEDULED_STEPS
    if x > 3/12:
        lr_mult = 2.0
    elif x > 2/12:
        lr_mult = 1.73
    elif x > 1/12:
        lr_mult = 1.52
    else:
        lr_mult = 1.0
        
    current_lr = peak_lr * lr_mult
    
    # Cooldown logic
    if x >= 1 - cooldown_frac:
        w = (1 - x) / cooldown_frac
        decayed_mult = lr_mult * w + (1 - w) * decay_floor
        return peak_lr * decayed_mult
        
    return current_lr

def parse_loss_curve(filepath):
    text = Path(filepath).read_text()
    pattern = r"step:(\d+)/\d+ val_loss:([\d.]+)"
    return [(int(s), float(l)) for s, l in re.findall(pattern, text)]

def extract_config(filename):
    muon = 0.015 # default
    cd = 0.70    # default
    decay = 0.1  # default
    
    m = re.search(r"muon([\d.]+)", filename)
    if m: muon = float(m.group(1))
    
    m = re.search(r"cd([\d.]+)(?:\.log)?$", filename)
    if m: cd = float(m.group(1))
    
    m = re.search(r"decay([\d.]+)(?:\.log)?$", filename)
    if m: decay = float(m.group(1))
    
    return muon, cd, decay

def main():
    # Find the global winner to highlight
    all_runs = []
    for d in SWEEP_DIRS:
        for f in sorted(Path(d).glob("sweep_*.log")):
            curve = parse_loss_curve(f)
            if not curve: continue
            final_loss = curve[-1][1]
            all_runs.append((final_loss, f))
    
    all_runs.sort()
    winner_path = all_runs[0][1]
    print(f"Global winner: {winner_path} ({all_runs[0][0]})")

    targets = [
        winner_path,
        Path("sweep_results_v2_yarn_fix/sweep_muon0.015_adam0.004.log"),
        Path("sweep_results_v3_low_lr/sweep_muon0.010_adam0.003.log"),
        Path("sweep_results_v5_decay/sweep_muon0.010_decay0.25.log"),
        Path("sweep_results_v5_decay/sweep_muon0.010_decay0.20.log"),
    ]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    for path in targets:
        if not path.exists(): continue
        
        curve = parse_loss_curve(path)
        steps, losses = zip(*curve)
        
        muon, cd, decay = extract_config(path.name)
        label = f"muon={muon} cd={cd} decay={decay}"
        if path == winner_path: label += " (WINNER)"
        
        # Plot Loss
        ax1.plot(steps, losses, label=label, linewidth=2)
        
        # Plot LR
        lrs = [get_lr_schedule(s, muon, cd, decay) for s in steps]
        ax2.plot(steps, lrs, label=label, linewidth=2)

    # Add vertical line at step 3500 (approx where green starts losing lead)
    for ax in (ax1, ax2):
        ax.axvline(x=3500, color='gray', linestyle=':', alpha=0.8, label='Gap starts closing (3500)')

    ax1.set_ylabel("Val Loss")
    ax1.set_ylim(2.90, 3.20)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title("Validation Loss vs Effective Learning Rate")
    
    ax2.set_ylabel("Effective Muon LR")
    ax2.set_xlabel("Step")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    fig.savefig("analysis_lr_loss.png", dpi=150)
    print("Saved: analysis_lr_loss.png")

if __name__ == "__main__":
    main()
