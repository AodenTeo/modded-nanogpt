#!/usr/bin/env python3
"""Parse sweep log files and print a sorted summary table."""
import re
import sys
from pathlib import Path


def parse_log(filepath: Path):
    """Extract final val_loss and train_time from a sweep log."""
    text = filepath.read_text()
    pattern = r"step:\d+/\d+ val_loss:([\d.]+) train_time:([\d.]+)ms"
    matches = re.findall(pattern, text)
    if not matches:
        return None, None
    val_loss, train_time_ms = matches[-1]
    return float(val_loss), float(train_time_ms)


def extract_config(filename: str):
    """Extract muon_lr and adam_lr from log filename."""
    match = re.match(
        r"sweep_muon([\d.]+)_adam([\d.]+)\.log", filename
    )
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def main():
    """Read all sweep logs, sort by val_loss, and print table."""
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "sweep_results")
    results = []

    for f in sorted(results_dir.glob("sweep_*.log")):
        muon_lr, adam_lr = extract_config(f.name)
        val_loss, train_time_ms = parse_log(f)
        if muon_lr is not None and val_loss is not None:
            results.append((val_loss, train_time_ms, muon_lr, adam_lr))

    if not results:
        print("No results found. Check sweep_results/ for log files.")
        return

    # Sort by val_loss (best first)
    results.sort()

    header = f"{'RANK':>4} {'MUON_LR':>9} {'ADAM_LR':>9} {'VAL_LOSS':>10} {'TIME(s)':>9} {'PASS':>5}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for i, (val_loss, time_ms, muon_lr, adam_lr) in enumerate(results, 1):
        time_s = time_ms / 1000
        passed = "YES" if val_loss < 2.92 else "NO"
        print(
            f"{i:>4} {muon_lr:>9.3f} {adam_lr:>9.3f} "
            f"{val_loss:>10.4f} {time_s:>9.1f} {passed:>5}"
        )
    print(sep)

    best = results[0]
    print(f"\nBest: muon_lr={best[2]}, adam_lr={best[3]} "
          f"-> val_loss={best[0]:.4f}, time={best[1]/1000:.1f}s")


if __name__ == "__main__":
    main()
