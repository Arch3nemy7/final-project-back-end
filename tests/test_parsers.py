"""Dependency-free tests for the telemetry parsers.

Run from the server/ directory with plain Python (no pytest needed):
    python tests/test_parsers.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner import parse_stats_line, parse_metric_line, kimg_from_name, build_train_args  # noqa: E402


def check(name, cond):
    print(("PASS" if cond else "FAIL") + "  " + name)
    if not cond:
        check.failed += 1
check.failed = 0


# A realistic line from StyleGAN2-ADA's stats.jsonl (values are nested dicts).
STATS = ('{"Progress/tick": {"num": 1, "mean": 20.0, "std": 0.0}, '
         '"Progress/kimg": {"mean": 80.0}, "Loss/G/loss": {"mean": 2.31}, '
         '"Loss/D/loss": {"mean": 0.88}, "Timing/sec_per_kimg": {"mean": 5.0}, '
         '"timestamp": 1700000000.0}')
row = parse_stats_line(STATS)
check("stats: tick parsed", row and row["tick"] == 20)
check("stats: kimg parsed", row and row["kimg"] == 80.0)
check("stats: loss_g parsed", row and abs(row["loss_g"] - 2.31) < 1e-9)
check("stats: sec_per_kimg parsed", row and row["sec_per_kimg"] == 5.0)
check("stats: blank line -> None", parse_stats_line("   ") is None)
check("stats: junk line -> None", parse_stats_line("not json") is None)

# A realistic line from metric-fid50k_full.jsonl.
METRIC = ('{"results": {"fid50k_full": 12.34}, "metric": "fid50k_full", '
          '"snapshot_pkl": "network-snapshot-000080.pkl", "timestamp": 1700000000.0}')
m = parse_metric_line(METRIC)
check("metric: fid parsed", m and abs(m["fid"] - 12.34) < 1e-9)
check("metric: kimg from pkl name", m and m["kimg"] == 80)
check("metric: empty results -> None", parse_metric_line('{"results": {}}') is None)

check("kimg_from_name fakes png", kimg_from_name("fakes000200.png") == 200)
check("kimg_from_name no digits", kimg_from_name("reals.png") is None)

args = build_train_args(
    {"cfg": "auto", "ticks": "1000", "snap": "50", "batch": "32", "gpus": "1",
     "resume": "ffhq256", "mirror": True, "fp32": False, "tf32": False},
    "/data/ds.zip", "/runs/r1/gn")
check("train args: kimg mapped from ticks", "--kimg" in args and args[args.index("--kimg") + 1] == "1000")
check("train args: data + outdir present", "--data" in args and "--outdir" in args)
check("train args: mirror true", args[args.index("--mirror") + 1] == "true")
check("train args: resume passed", args[args.index("--resume") + 1] == "ffhq256")

print()
if check.failed:
    print(f"{check.failed} test(s) FAILED")
    sys.exit(1)
print("All parser tests passed.")
