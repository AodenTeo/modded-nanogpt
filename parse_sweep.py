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
    """Extract config values from log filename."""
    config = {}
    # Match muon LR
    m = re.search(r"muon([\d.]+?)(?:_|\.log|$)", filename)
    if m:
        config["muon_lr"] = float(m.group(1))
    # Match adam LR
    m = re.search(r"adam([\d.]+?)(?:_|\.log|$)", filename)
    if m:
        config["adam_lr"] = float(m.group(1))
    # Match cooldown frac
    m = re.search(r"cd([\d.]+?)(?:_|\.log|$)", filename)
    if m:
        config["cd_frac"] = float(m.group(1))
    # Match delayed ramp factor
    m = re.search(r"stretch([\d.]+?)(?:_|\.log|$)", filename)
    if m:
        config["stretch"] = float(m.group(1))
    # Match peak multiplier
    m = re.search(r"peak([\d.]+?)(?:_|\.log|$)", filename)
    if m:
        config["peak"] = float(m.group(1))
    return config


def main():
    """Read all sweep logs, sort by val_loss, and print table."""
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "sweep_results")
    results = []

    for f in sorted(results_dir.glob("sweep_*.log")):
        config = extract_config(f.name)
        val_loss, train_time_ms = parse_log(f)
        if val_loss is not None:
            results.append((val_loss, train_time_ms, config, f.name))

    if not results:
        print("No completed results found.")
        return

    # Sort by val_loss (best first)
    results.sort()

    # Build header from available config keys
    all_keys = set()
    for _, _, config, _ in results:
        all_keys.update(config.keys())
    col_order = [k for k in ["muon_lr", "adam_lr", "cd_frac", "stretch", "peak"] if k in all_keys]

    header_parts = [f"{'RANK':>4}"]
    for k in col_order:
        header_parts.append(f"{k.upper():>10}")
    header_parts += [f"{'VAL_LOSS':>10}", f"{'TIME(s)':>9}", f"{'PASS':>5}"]
    header = " ".join(header_parts)
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for i, (val_loss, time_ms, config, _) in enumerate(results, 1):
        time_s = time_ms / 1000
        passed = "YES" if val_loss < 2.92 else "NO"
        parts = [f"{i:>4}"]
        for k in col_order:
            parts.append(f"{config.get(k, 0):>10.3f}")
        parts += [f"{val_loss:>10.4f}", f"{time_s:>9.1f}", f"{passed:>5}"]
        print(" ".join(parts))
    print(sep)

    best = results[0]
    desc = ", ".join(f"{k}={best[2][k]}" for k in col_order)
    print(f"\nBest: {desc} -> val_loss={best[0]:.4f}, time={best[1]/1000:.1f}s")


if __name__ == "__main__":
    main()
