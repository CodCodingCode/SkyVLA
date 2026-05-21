"""Merge per-benchmark JSON/txt into one summary table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    rows = []

    for jf in sorted(out_dir.glob("huge_*.json")):
        d = json.loads(jf.read_text())
        rows.append({
            "benchmark": d.get("benchmark", "HUGE"),
            "run": jf.stem,
            "split": d.get("split"),
            "mse_normalized": d.get("mse_normalized"),
            "mse_raw": d.get("mse_raw"),
            "n_samples": d.get("n_samples"),
        })

    for sf in ("citynav_val_seen.txt", "citynav_val_unseen.txt"):
        fp = out_dir / sf
        if fp.exists():
            text = fp.read_text()
            for line in text.splitlines():
                if line.strip().startswith("NE:") or line.strip().startswith("SR:"):
                    pass
            rows.append({"benchmark": "CityNav-oracle", "run": sf.replace(".txt", ""), "log": text[-500:]})

    summary = {"results": rows, "out_dir": str(out_dir)}
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
