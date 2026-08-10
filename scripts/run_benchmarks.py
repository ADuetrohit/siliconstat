"""Measure real Monte Carlo throughput for the README / docs tables.

Every number this prints comes from an actual run on the machine executing it.
"""
from __future__ import annotations

import json
import os
import sys

from siliconstat.bench import benchmark_environment, run_benchmark
from siliconstat.core import parse_netlist_file
from siliconstat.variation import default_mismatch_model

GRID = (
    ("current_mirror", [100, 1000, 10000], [1]),
    ("current_mirror", [2000], [1, 2, 4]),
    ("diff_pair", [100, 1000], [1]),
    ("two_stage_opamp", [50, 200], [1]),
    ("rc_divider", [1000], [1]),
    ("inverter_transient", [25], [1]),
)


def main() -> int:
    os.makedirs("out", exist_ok=True)
    results = []
    for name, samples, workers in GRID:
        circuit = parse_netlist_file(f"examples/{name}.net")
        model = default_mismatch_model(circuit)
        print(f"benchmarking {name}: samples={samples} workers={workers}", flush=True)
        rows = run_benchmark(circuit, model, sample_counts=samples,
                             worker_counts=workers, seed=12345)
        for row in rows:
            row["example"] = name
            print(f"   {name:20s} n={row['samples']:>6} w={row['workers']} "
                  f"{row['wall_s']:8.3f} s  {row['samples_per_s']:8.1f} samples/s  "
                  f"ok={row['successful']} fail={row['failed']}"
                  + ("  [fell back to sequential]"
                     if row["fell_back_to_sequential"] else ""), flush=True)
        results.extend(rows)

    with open("out/benchmarks.json", "w", encoding="utf-8") as fh:
        json.dump({"environment": benchmark_environment(), "results": results},
                  fh, indent=2)
    print("\nwrote out/benchmarks.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
